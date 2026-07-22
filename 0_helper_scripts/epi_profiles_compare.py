#!/usr/bin/env python3
"""
epi_profiles_compare.py
------------------------
Compare deeptools enrichment profiles across loop classes on the same axes.

API usage:
    from epi_profiles_compare import run_comparison
    run_comparison(
        loop_classes={
            '5-5':         df_55,
            '10-10':       df_1010,
            '5-10 | 10-5': df_510,
        },
        out_dir='deeptools_compare/c5v10',
        tag='cluster5v10',
        class_colors={
            '5-5':         '#e41a1c',
            '10-10':       '#377eb8',
            '5-10 | 10-5': '#4daf4a',
        },
    )

CLI usage:
    python epi_profiles_compare.py \
        --wide_tsv loops_clustered.tsv \
        --classes "5-5:5_5,10-10:10_10,5-10or10-5:5_10+10_5" \
        --cluster_col_a1 cluster_anchor1 \
        --cluster_col_a2 cluster_anchor2 \
        --out_dir ./out --tag c5v10
"""

import os, json, gzip, subprocess, warnings, argparse
warnings.filterwarnings('ignore')
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({
    'font.size':     6,
    'axes.linewidth': 0.25,
    'axes.labelsize': 6,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'svg.fonttype':  'none',       # keeps text as real text, not outlines, in Illustrator
})

from matplotlib.patches import Patch  # add to imports
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

############################################################
# Config
############################################################

BIGWIG_PAT = '/mnt/florenceshares/Masahiro/Saitoulab_data/saitoulab_chip/bws/{cell_line}_{prefix}_{mark}_pool.bw'
PREFIXES   = ['BDF121', 'BDF121', 'BDF121', 'AAG']
CELL_LINES = ['ESC', 'EpiLC', 'd4c7PGCLC', 'GSC']

# per-cell-line line styles (used when show_cell_lines=True)
CL_STYLES  = {
    'ESC':       '-',
    'EpiLC':     '--',
    'd4c7PGCLC': '-.',
    'GSC':       ':',
}

MARKS = [
    'K27ac',  'K4me1',  'K4me3',
    'CTCF',   'Rad21',  'Stag1',  'Stag2',
    'Ring1b', 'H2Aub',
    'K9me2',  'K9me3',
    'K36me2', 'K36me3',
    'K27me3', 'Laminb1',
]

FLANK     = 20_000
BIN_SIZE  = 50
SMOOTH    = 5
NPROC     = 4
N_WORKERS = 4

_PALETTE = [
    '#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
    '#ff7f00', '#a65628', '#f781bf', '#999999',
    '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3',
    '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3',
    '#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#cccccc'
    '#66a61e', '#e6ab02', '#a6761d', '#666666',
]

############################################################
# 1)  Anchor BED
############################################################

def make_anchor_bed(
    df: pd.DataFrame,
    out_path: str,
    chrom_cols: tuple = ('chrom1', 'chrom2'),
    start_cols: tuple = ('start1', 'start2'),
    end_cols:   tuple = ('end1',   'end2'),
    half: int = 500,
) -> int:
    """
    Union of anchor1 + anchor2 midpoints ±half bp, deduped.
    Works on any loop DataFrame — caller is responsible for subsetting.
    """
    parts = []
    for chrom_col, start_col, end_col in zip(chrom_cols, start_cols, end_cols):
        mid = ((df[start_col] + df[end_col]) // 2).astype(int)
        parts.append(pd.DataFrame({
            'chrom': df[chrom_col].values,
            'start': (mid - half).clip(lower=0).values,
            'end':   (mid + half).values,
        }))

    anchors = (
        pd.concat(parts)
        .drop_duplicates()
        .query("chrom.str.match('^chr[0-9XYM]+$')", engine='python')
        .sort_values(['chrom', 'start'])
    )
    anchors.to_csv(out_path, sep='\t', header=False, index=False)
    return len(anchors)


############################################################
# 2)  deeptools computeMatrix
############################################################

def _run_computematrix(bed: str, bw_files: list,
                        labels: list, out_gz: str) -> None:
    if os.path.exists(out_gz):
        return
    cmd = [
        'computeMatrix', 'reference-point',
        '--referencePoint',          'center',
        '--beforeRegionStartLength', str(FLANK),
        '--afterRegionStartLength',  str(FLANK),
        '--binSize',                 str(BIN_SIZE),
        '--scoreFileName',           *bw_files,
        '--regionsFileName',         bed,
        '--outFileName',             out_gz,
        '--samplesLabel',            *labels,
        '--numberOfProcessors',      str(NPROC),
        '--sortRegions',             'keep',
        '--missingDataAsZero',
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"computeMatrix failed:\n{r.stderr[:1000]}")


############################################################
# 3)  Parse matrix
############################################################

def _parse_matrix(gz: str):
    with gzip.open(gz, 'rt') as f:
        hdr  = json.loads(f.readline().lstrip('@'))
        rows = [
            [float(x) if x not in ('nan', 'NaN', '') else np.nan
             for x in line.rstrip('\n').split('\t')[6:]]
            for line in f
        ]
    labels = hdr['sample_labels']
    data   = np.array(rows, dtype=np.float32)
    n_bins = data.shape[1] // len(labels)
    return data, n_bins, labels


def _mean_profiles(data: np.ndarray, n_bins: int,
                    labels: list) -> dict:
    """Returns {cell_line: mean_profile (n_bins,)}"""
    out = {}
    for i, lbl in enumerate(labels):
        profile = np.nanmean(data[:, i*n_bins:(i+1)*n_bins], axis=0)
        if SMOOTH > 1:
            profile = uniform_filter1d(profile, size=SMOOTH)
        out[lbl] = profile
    return out


############################################################
# 4)  Worker  (one class × one mark)
############################################################

def _worker(args: tuple):
    class_label, mark, bed_path, class_dir, replot = args

    bw_files, bw_labels = [], []
    for i, cl in enumerate(CELL_LINES):
        bw = BIGWIG_PAT.format(cell_line=cl, prefix=PREFIXES[i], mark=mark)
        if os.path.exists(bw):
            bw_files.append(bw)
            bw_labels.append(cl)
        else:
            print(f"    WARN: missing bigwig {bw}")

    if not bw_files:
        return class_label, mark, {}

    gz = os.path.join(class_dir, f'matrix_{mark}.gz')
    try:
        if not replot:
            _run_computematrix(bed_path, bw_files, bw_labels, gz)
        if not os.path.exists(gz):
            return class_label, mark, {}
        data, n_bins, labels = _parse_matrix(gz)
        profs = _mean_profiles(data, n_bins, labels)
    except Exception as e:
        print(f"    ERROR {class_label}/{mark}: {e}")
        return class_label, mark, {}

    return class_label, mark, profs


############################################################
# 5)  Plot
############################################################

def plot_comparison(
    all_profiles:  dict,    # {class_label: {mark: {cell_line: np.ndarray}}}
    class_colors:  dict,    # {class_label: hex}
    n_anchors:     dict,    # {class_label: int}
    out_dir:       str,
    tag:           str,
    ncols:         int = 4,
    show_cell_lines: bool = True,
    alphas: dict = None,
    linestyles: dict = {},
) -> None:
    """
    cols  = marks (independent y-scales)
    lines = loop_class × cell_line   (show_cell_lines=True)
         OR loop_class pooled        (show_cell_lines=False)
    """
    alphas = alphas or dict()
    marks_ok = [m for m in MARKS
                if any(bool(all_profiles.get(cl, {}).get(m))
                       for cl in all_profiles)]
    if not marks_ok:
        print("  No profiles to plot.")
        return

    n_bins = FLANK * 2 // BIN_SIZE
    x_kb   = np.linspace(-FLANK / 1000, FLANK / 1000, n_bins)
    LETTER_SIZE = (8.5, 11)   # inches, portrait. Use (11, 8.5) for landscape.

    fig, axes = plt.subplots(
        (len(marks_ok) + ncols - 1) // ncols, ncols,
        figsize=LETTER_SIZE,
        sharey=False,
        squeeze=False,
    )
    axes = axes.flatten() 
    
    if len(marks_ok) == 1:
        axes = [axes]

    for ax, mark in zip(axes, marks_ok):
        for class_label, mark_profiles in all_profiles.items():
            profs = mark_profiles.get(mark, {})
            color = class_colors[class_label]
            alpha = alphas.get(class_label, 0.9)

            if show_cell_lines:
                # one line per class × cell_line
                # color = class,  linestyle = cell_line
                for cl in CELL_LINES:
                    prof = profs.get(cl)
                    if prof is None or np.all(np.isnan(prof)):
                        continue
                    ax.plot(x_kb, prof,
                            color=color,
                            linestyle=CL_STYLES[cl],
                            linewidth=1,
                            alpha=alpha)
            else:
                # pool cell lines: mean of all available profiles
                available = [p for p in profs.values()
                             if p is not None and not np.all(np.isnan(p))]
                if not available:
                    continue
                pooled = np.nanmean(np.stack(available), axis=0)
                ax.plot(x_kb, pooled,
                        color=color,
                        linewidth=1,
                        alpha=alpha,
                        label=class_label,
                        linestyle=linestyles.get(class_label, '-'))

        ax.axvline(0, color='#888', lw=0.25, ls='--', zorder=0)
        ax.set_title(mark, fontsize=7, fontweight='bold', pad=4)
        ax.tick_params(labelsize=6)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_linewidth(0.25)
        if ax is axes[0]:
            ax.set_xlabel('Distance from anchor', fontsize=6)
            ax.set_ylabel('Mean signal log₂(O/E)', fontsize=6)
        ax.set_xticks([-FLANK/1000, 0, FLANK/1000], width=0.25, length=2.5)
        ax.set_xticklabels([f'-{FLANK//1000} kb', '0', f'+{FLANK//1000} kb'],
                           fontsize=6)

    fig.subplots_adjust(
        left=0.08, right=0.98,
        top=0.95,      # leaves room for suptitle
        bottom=0.2,   # leaves room for legend below last row
        hspace=0.45,   # vertical gap between rows — this is what's colliding now
        wspace=0.35,   # horizontal gap between columns
    )
        
    for ax in axes[len(marks_ok):]:
        ax.set_visible(False)
    # ── legend ──────────────────────────────────────────────────────────────
    handles = []
    # class entries (solid line in class color)
    for class_label, color in class_colors.items():
        n = n_anchors.get(class_label, 0)
        ls = linestyles.get(class_label, '-')   # pull the class's actual linestyle
        handles.append(plt.Line2D([0], [0], color=color,
                                linestyle=ls,
                                lw=1.0,        # matches your data-line stroke spec
                                label=f'{class_label}  (n={n:,})'))
    if show_cell_lines:
        # cell line entries (grey line in cell-line style)
        handles.append(plt.Line2D([0], [0], color='none', label=''))  # spacer
    for cl in CELL_LINES:
        handles.append(plt.Line2D([0], [0], color='#555',
                                   linestyle=CL_STYLES[cl],
                                   lw=1.0, label=cl))

    fig.legend(handles=handles,
           ncol=1,      # grid layout like the screenshot
           frameon=False,                  # no box around legend
           fontsize=6,
           prop={'size': 6},
           handlelength=1.4,
           handleheight=1.4,
           handletextpad=0.6,
           columnspacing=1.4,
           labelspacing=0.8,
           loc='lower center',
           bbox_to_anchor=(0.5, -0.06))

    fig.suptitle(f'Loop class comparison — {tag}',
                 fontsize=7, fontweight='bold')
    # plt.tight_layout(rect=[0, 0.1, 1, 1])

    out = Path(out_dir) / f'profiles_compare.{tag}'
    fig.set_size_inches(LETTER_SIZE)
    plt.savefig(f'{out}.svg', format='svg')
    plt.savefig(f'{out}.png', dpi=300)
    plt.close()
    print(f"  → Saved: {out}.svg / .png")


############################################################
# 6)  Public API
############################################################

def run_comparison(
    loop_classes:    dict,           # {label: df}
    out_dir:         str,
    tag:             str,
    class_colors:    dict  = None,   # {label: hex}  — auto if None
    replot:          bool  = False,
    show_cell_lines: bool  = True,
    linestyles:      dict  = {},
    alphas:          dict  = None,
    anchor_half:     int   = 500,
) -> dict:
    """
    Parameters
    ----------
    loop_classes : {'5-5': df_55, '10-10': df_1010, '5-10 | 10-5': df_510}
                   Each df needs chrom1/start1/end1/chrom2/start2/end2 columns.
    class_colors : optional color override; missing classes auto-assigned.
    show_cell_lines : True  → color=class, linestyle=cell_line
                      False → one pooled line per class

    Returns
    -------
    all_profiles : {class_label: {mark: {cell_line: np.ndarray}}}
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # fill missing colors
    provided = class_colors or {}
    class_colors = {
        label: provided.get(label, _PALETTE[i % len(_PALETTE)])
        for i, label in enumerate(loop_classes)
    }

    # build anchor BEDs
    n_anchors   = {}
    worker_args = []
    for class_label, df in loop_classes.items():
        safe      = (class_label.replace(' ', '_')
                                .replace('|', 'or')
                                .replace('-', '_'))
        class_dir = out_dir / safe
        class_dir.mkdir(exist_ok=True)

        bed_path = class_dir / 'anchors.bed'
        n = make_anchor_bed(df, str(bed_path), half=anchor_half)
        n_anchors[class_label] = n
        print(f"  {class_label}: {n:,} anchors  → {bed_path}")

        if n == 0:
            print(f"  WARNING: no anchors for {class_label}, skipping")
            continue

        for mark in MARKS:
            worker_args.append(
                (class_label, mark, str(bed_path), str(class_dir), replot)
            )

    # parallel computeMatrix
    all_profiles: dict[str, dict] = {cl: {} for cl in loop_classes}

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futs = {pool.submit(_worker, a): (a[0], a[1]) for a in worker_args}
        for fut in as_completed(futs):
            class_label, mark, profs = fut.result()
            all_profiles[class_label][mark] = profs
            n_cls = sum(1 for p in profs.values() if p is not None)
            print(f"    ✓ {class_label:20s}  {mark:12s}  ({n_cls} cell lines)")

    plot_comparison(
        all_profiles, class_colors, n_anchors,
        str(out_dir), tag,
        show_cell_lines=show_cell_lines,
        linestyles=linestyles or {},
        alphas=alphas or {},
    )
    return all_profiles


############################################################
# 7)  CLI
############################################################

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--wide_tsv',       required=True)
    p.add_argument('--cluster_col_a1', default='cluster_anchor1')
    p.add_argument('--cluster_col_a2', default='cluster_anchor2')
    p.add_argument('--classes',        required=True,
                    help=(
                        'Comma-separated specs: "label:pair1+pair2+..."  '
                        'where each pair is "c1_c2" (cluster_a1_cluster_a2).  '
                        'Example: "5-5:5_5,10-10:10_10,5-10or10-5:5_10+10_5"'
                    ))
    p.add_argument('--out_dir',        required=True)
    p.add_argument('--tag',            required=True)
    p.add_argument('--replot',         action='store_true')
    p.add_argument('--pool_cell_lines', action='store_true',
                    help='Average across cell lines (default: show per cell line)')
    return p.parse_args()


def main():
    args = _parse_args()
    df   = pd.read_csv(args.wide_tsv, sep='\t')
    print(f"Loaded {len(df):,} loops from {args.wide_tsv}")

    loop_classes = {}
    for spec in args.classes.split(','):
        label, pairs_str = spec.split(':', 1)
        mask = pd.Series(False, index=df.index)
        for pair in pairs_str.split('+'):
            c1, c2 = pair.split('_', 1)
            mask |= (
                (df[args.cluster_col_a1].astype(str) == c1) &
                (df[args.cluster_col_a2].astype(str) == c2)
            )
        sub = df[mask]
        print(f"  {label}: {len(sub):,} loops")
        loop_classes[label] = sub

    run_comparison(
        loop_classes=loop_classes,
        out_dir=args.out_dir,
        tag=args.tag,
        replot=args.replot,
        show_cell_lines=not args.pool_cell_lines,
        alphas=args.alphas or {},
        linestyles=args.linestyles or {},
    )
    print('[done]')


if __name__ == '__main__':
    main()
