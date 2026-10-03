#!/usr/bin/env python3
# conda activate umap_env

"""
Epigenomic enrichment at loop anchors — deeptools computeMatrix pipeline.

For each loop category:
  1. Build merged anchor BED (anchor1 + anchor2 from all cell lines, deduped)
  2. Per mark (parallel): computeMatrix reference-point ±20kb, 50bp bins
  3. Parse matrices → mean profiles per cell line
  4. Plot: one figure per category
           cols=marks (independent y-scales), lines=cell lines

Usage:
    python epi_profiles.py               # all categories
    python epi_profiles.py --cats CRE,CTCF
    python epi_profiles.py --replot      # skip computeMatrix, just replot
"""

import os, sys, json, gzip, subprocess, warnings, argparse
warnings.filterwarnings('ignore')
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d   # optional smoothing

############################################################
# Config — EDIT PATHS
############################################################

WIDE_TSV   = 'clustering/epigenomics/loops_clustered_wide_labeled.tsv'
OUT_BASE   = 'deeptools_profiles'

# {cell_line} and {mark} are filled at runtime
BIGWIG_PAT = '/mnt/florenceshares/Masahiro/Saitoulab_data/saitoulab_chip/bws/{cell_line}_{prefix}_{mark}_pool.bw'
PREFIXES=['BDF121', 'BDF121', 'BDF121', 'AAG']
CELL_LINES = ['ESC', 'EpiLC', 'd4c7PGCLC', 'GSC']
CL_COLORS = {
    'ESC':       '#e41a1c',
    'EpiLC':     '#4daf4a',
    'd4c7PGCLC': '#377eb8',
    'GSC':       '#dede00',
}

MARKS = [
    'K4me3', 'K27ac', 'K4me1', 'K36me3', 
    'CTCF', 'Rad21', 'Stag1', 'Stag2',
    'Ring1b', 'H2Aub', 'K27me3',
    'K9me2', 'K9me3', 'K36me2',
    'Laminb1',
]

FLANK     = 20_000   # bp each side
BIN_SIZE  = 50       # bp per bin 
SMOOTH    = 5        # bins for smoothing
NPROC     = 1        # threads per computeMatrix call
N_WORKERS = 12       # parallel mark workers

############################################################
# Args
############################################################

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cats',   default=None,
                   help='Comma-separated categories, e.g. CRE,CTCF')
    p.add_argument('--replot', action='store_true', default=False,
                   help='Skip computeMatrix, just replot existing matrices')
    p.add_argument('--mode',   default='both', choices=['broad','fine', 'both'],
                   help='Category granularity (default: both)')
    return p.parse_args()

# Load + annotate
def load_df(path, mode: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t')
    print(f"Loaded {len(df):,} loops from {path}")
    for cl in CELL_LINES:
        df[f'cat_{cl}'] = df[f'category_{mode}_{cl}']
    return df

# Anchor BED — merged anchor1 + anchor2 across all cell lines
def make_anchor_bed(df: pd.DataFrame, cat: str, out_path: str) -> int:
    """
    Union of anchor1 and anchor2 midpoints for loops of `cat`
    active in any cell line (AbLE > 0).
    Uses ±500 bp around each anchor midpoint as the BED interval.
    """
    HALF = 500
    parts = []
    for cl in CELL_LINES:
        cc, ac = f'cat_{cl}', f'AbLE_score_{cl}'
        if cc not in df.columns:
            continue
        mask = df[cc] == cat
        if ac in df.columns:
            mask &= df[ac] > 0
        sub = df[mask]
        if len(sub) == 0:
            continue

        for chrom_col, start_col, end_col in [
            ('chrom1','start1','end1'),
            ('chrom2','start2','end2'),
        ]:
            mid  = ((sub[start_col] + sub[end_col]) // 2).astype(int)
            part = pd.DataFrame({
                'chrom': sub[chrom_col].values,
                'start': (mid - HALF).clip(lower=0).values,
                'end':   (mid + HALF).values,
            })
            parts.append(part)

    if not parts:
        return 0

    anchors = (pd.concat(parts)
                 .drop_duplicates()
                 .query("chrom.str.match('^chr[0-9XYM]+$')", engine='python')
                 .sort_values(['chrom','start']))
    anchors.to_csv(out_path, sep='\t', header=False, index=False)
    return len(anchors)


def run_computematrix(bed: str, bw_files: list,
                      labels: list, out_gz: str):
    if os.path.exists(out_gz):
        return   # cached

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

def parse_matrix(gz: str):
    """
    Returns:
        data   : np.float32 array (n_regions, n_samples*n_bins)
        n_bins : int
        labels : list[str]
    """
    with gzip.open(gz, 'rt') as f:
        hdr  = json.loads(f.readline().lstrip('@'))
        rows = []
        for line in f:
            parts = line.rstrip('\n').split('\t')
            rows.append([
                float(x) if x not in ('nan','NaN','') else np.nan
                for x in parts[6:]   # skip BED columns
            ])

    labels = hdr['sample_labels']
    data   = np.array(rows, dtype=np.float32)
    n_samples = len(labels)
    n_bins    = data.shape[1] // n_samples
    return data, n_bins, labels


def mean_profiles(data: np.ndarray, n_bins: int,
                  labels: list) -> dict:
    """Returns {label: mean_profile (n_bins,)}"""
    out = {}
    for i, lbl in enumerate(labels):
        mat       = data[:, i*n_bins : (i+1)*n_bins]
        profile   = np.nanmean(mat, axis=0)
        if SMOOTH > 1:
            profile = uniform_filter1d(profile, size=SMOOTH)
        out[lbl] = profile
    return out

def _worker(args):
    cat, mark, bed_path, cat_dir, replot = args

    bw_files, bw_labels = [], []
    for i, cl in enumerate(CELL_LINES):
        prefix = PREFIXES[i]
        bw = BIGWIG_PAT.format(cell_line=cl, prefix=prefix, mark=mark)
        if os.path.exists(bw):
            bw_files.append(bw)
            bw_labels.append(cl)
        else:
            print(f"    WARN: missing bigwig {bw}")

    if not bw_files:
        return cat, mark, {}

    gz = os.path.join(cat_dir, f'matrix_{mark}.gz')

    try:
        if not replot:
            run_computematrix(bed_path, bw_files, bw_labels, gz)
        if not os.path.exists(gz):
            return cat, mark, {}
        data, n_bins, labels = parse_matrix(gz)
        profs = mean_profiles(data, n_bins, labels)   # {cl: array}
    except Exception as e:
        print(f"    ERROR {mark}: {e}")
        return cat, mark, {}

    return cat, mark, profs

def plot_profiles(cat: str, mark_profiles: dict,
                  n_anchors: int, out_dir: str):
    """
    mark_profiles : {mark: {cell_line: np.ndarray (n_bins,)}}
    Each subplot has its own y-scale (independent).
    All cell lines share the same x-axis and line colors.
    """
    marks_ok = [m for m in MARKS
                if m in mark_profiles and mark_profiles[m]]
    if not marks_ok:
        print(f"  No profiles to plot for {cat}")
        return

    n_bins = FLANK * 2 // BIN_SIZE          # 800
    x_kb   = np.linspace(-FLANK/1000, FLANK/1000, n_bins)

    fig, axes = plt.subplots(
        1, len(marks_ok),
        figsize=(2.5 * len(marks_ok), 3.5),
        sharey=False,    # independent y-scales per mark
    )
    if len(marks_ok) == 1:
        axes = [axes]

    for ax, mark in zip(axes, marks_ok):
        profs = mark_profiles[mark]

        for cl in CELL_LINES:
            prof = profs.get(cl)
            if prof is None or np.all(np.isnan(prof)):
                continue
            ax.plot(x_kb, prof,
                    color=CL_COLORS[cl],
                    linewidth=1.5,
                    label=cl,
                    alpha=0.9)

        ax.axvline(0, color='#888', lw=0.7, ls='--', zorder=0)
        ax.set_title(mark, fontsize=9, fontweight='bold', pad=4)
        ax.set_xlabel('Distance (kb)', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.spines[['top','right']].set_visible(False)

        # y-label only on leftmost
        if ax is axes[0]:
            ax.set_ylabel('Mean signal', fontsize=7)
        else:
            ax.set_ylabel('')

        # x-ticks: just ±FLANK and 0
        ax.set_xticks([-FLANK/1000, 0, FLANK/1000])
        ax.set_xticklabels([f'-{FLANK//1000}', '0', f'+{FLANK//1000}'],
                           fontsize=6)

    # shared legend below all panels
    handles = [
        plt.Line2D([0],[0], color=CL_COLORS[cl], lw=2, label=cl)
        for cl in CELL_LINES
    ]
    fig.legend(handles=handles, ncol=len(CELL_LINES), frameon=False,
               fontsize=8, loc='lower center', bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f'{cat}  (n={n_anchors:,} anchors)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0.08, 1, 1])

    out = os.path.join(out_dir, f'profiles_{cat}.svg')
    plt.savefig(out, dpi=None, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")

def run_mode(args, mode: str):
    os.makedirs(f'{OUT_BASE}/{mode}', exist_ok=True)
    df = load_df(WIDE_TSV, mode)
    cats = pd.unique(df[[f'category_{mode}_{cl}' for cl in CELL_LINES]].values.ravel())
    
    cluster_order = cats[~pd.isna(cats)]
    cats_requested = ([c.strip() for c in args.cats.split(',')]
                      if args.cats else cluster_order)
    cats = [c for c in cats_requested
            if any(c in df.get(f'cat_{cl}', pd.Series(dtype=str)).values
                   for cl in CELL_LINES)]
    print(f"Categories to process: {cats}")
    for cat in cats:
        print(f"\n{'='*55}")
        print(f"Category: {cat}")
        safe    = cat.replace('-','_').replace(' ','_')
        cat_dir = os.path.join(f'{OUT_BASE}/{mode}', safe)
        os.makedirs(cat_dir, exist_ok=True)
        bed_path  = os.path.join(cat_dir, 'anchors.bed')
        n_anchors = make_anchor_bed(df, cat, bed_path)
        print(f"  Anchors: {n_anchors:,}")
        if n_anchors == 0:
            continue

        worker_args = [
            (cat, mark, bed_path, cat_dir, args.replot)
            for mark in MARKS
        ]
        mark_profiles: dict[str, dict] = {}

        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            futs = {pool.submit(_worker, a): a[1] for a in worker_args}
            for fut in as_completed(futs):
                _, mark, profs = fut.result()
                mark_profiles[mark] = profs
                n_cls = sum(1 for p in profs.values() if p is not None)
                print(f"    ✓ {mark:12s}  ({n_cls} cell lines)")

        plot_profiles(cat, mark_profiles, n_anchors, f'{OUT_BASE}/{mode}')
    
def main():
    args = parse_args()
    os.makedirs(OUT_BASE, exist_ok=True)

    mode = args.mode
    both = False
    if mode == 'both':
        both = True
    if mode == 'broad' or both:
        print(f"\n{'='*55}\nMode: broad categories")
        run_mode(args, 'broad')
    if mode == 'fine' or both:
        print(f"\n{'='*55}\nMode: fine categories")
        run_mode(args, 'fine')
    print('\n[done]')

if __name__ == '__main__':
    main()
