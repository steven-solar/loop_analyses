#!/usr/bin/env python3
"""
apa_clusters.py
---------------
APA pileup pipeline for loop classes defined by anchor-cluster combinations.
Replaces 00_make_bedpes.py + 02_run_coolpup.sh + 03_plot_APA.py.

Pipeline
--------
  1. make_bedpes  — per-(class × cell_line) BEDPE files
  2. run_coolpup  — coolpup.py for every BEDPE (parallel)
  3. plot_apa     — class_rows or cellline_rows grid

API usage
---------
    from apa_clusters import run_apa_pipeline

    loop_classes = {
        '5-5':         df[mask_55],
        '10-10':       df[mask_1010],
        '5-10 | 10-5': df[mask_510],
    }
    run_apa_pipeline(loop_classes, out_dir='apa_c5v10', tag='c5v10')

CLI usage
---------
    python apa_clusters.py \
        --loops_tsv loops_clustered.tsv \
        --classes "5-5:5_5,10-10:10_10,5-10or10-5:5_10+10_5" \
        --cluster_col_a1 cluster_anchor1 \
        --cluster_col_a2 cluster_anchor2 \
        --out_dir apa_out --tag c5v10
"""
import resource
import gc

import os, subprocess, warnings, argparse
warnings.filterwarnings('ignore')
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from coolpuppy.lib import numutils
from coolpuppy.lib import io as cp_io

############################################################
# Config — edit paths
############################################################

MCOOL_PAT    = '/mnt/coldstorage/shares/Masahiro/microc/mcs/{cell_line}_WT_10B.mcool'
EXPECTED_PAT = 'expected/{cell_line}_cis_expected_res{res}.tsv'

CELL_LINES = ['ESC', 'EpiLC', 'd4c7PGCLC', 'GSC']
COORD_COLS = ['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2']

RES       = 500
FLANK_BP  = 10_000
MINDIST   = 10_000
MAXDIST   = 2_000_000

TOTAL_CPUS = 12 # int(os.environ.get('APA_TOTAL_CPUS', '8'))
N_WORKERS  = 12 # int(os.environ.get('APA_N_WORKERS', '2'))
NPROC      = 1 # max(1, TOTAL_CPUS // N_WORKERS)

MIN_LOOPS = 10

VMIN     = -1.5
VMAX     =  1.5
ENRICH_N = 3

_PALETTE = [
    '#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
    '#ff7f00', '#a65628', '#f781bf', '#999999',
]


############################################################
# Helpers
############################################################

def limit_memory_gb(gb):
    bytes_ = int(gb * 1024**3)
    resource.setrlimit(resource.RLIMIT_AS, (bytes_, bytes_))

def _safe(s: str) -> str:
    return (s.replace(' ', '_').replace('|', 'or')
             .replace('/', '_').replace('-', '_'))

def _get_subprocess_env(n_threads: int = NPROC) -> dict:
    env = os.environ.copy()
    for var in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                'NUMBA_NUM_THREADS']:
        env[var] = str(n_threads)
    return env

############################################################
# 1) BEDPEs
############################################################

def make_bedpes(
    loop_classes:  dict,
    out_dir:       str,
    cell_line_col: str = 'cell_line',
    able_col:      str = 'AbLE_score',
    min_loops:     int = MIN_LOOPS,
) -> dict:
    """
    Write per-(class × cell_line) BEDPE files.

    Parameters
    ----------
    loop_classes  : {label: df}  long-format, has cell_line_col + COORD_COLS
    out_dir       : root output dir; BEDPEs go in <out_dir>/bedpe/

    Returns
    -------
    bedpe_map : {(class_label, cell_line): path_str or None}
    """
    bedpe_dir = Path(out_dir) / 'bedpe'
    bedpe_dir.mkdir(parents=True, exist_ok=True)

    bedpe_map = {}

    for class_label, df in loop_classes.items():
        safe = _safe(class_label)

        for cl in CELL_LINES:
            sub = df[df[cell_line_col] == cl]
            if able_col in sub.columns:
                sub = sub[sub[able_col] > 0]

            sub = (sub[COORD_COLS]
                   .query("chrom1 == chrom2")
                   .drop_duplicates()
                   .sort_values(['chrom1', 'start1']))

            n = len(sub)
            out_path = bedpe_dir / f'{safe}__{cl}.bedpe'

            if n < min_loops:
                print(f"  [skip] {class_label} × {cl}: {n} loops < {min_loops}")
                bedpe_map[(class_label, cl)] = None
                continue

            sub.to_csv(out_path, sep='\t', header=False, index=False)
            print(f"  {class_label:25s} × {cl:12s}: {n:>7,} → {out_path.name}")
            bedpe_map[(class_label, cl)] = str(out_path)

    return bedpe_map


############################################################
# 2) coolpup workers
############################################################

def _coolpup_worker(args: tuple):
    print(f"Running coolpup worker for args: {args}")
    class_label, cl, bedpe_path, out_path, replot = args

    # use cached if exists (replot=True means skip coolpup entirely)
    if os.path.exists(out_path):
        return class_label, cl, out_path, 'cached'
    if replot:
        return class_label, cl, None, 'missing (replot=True)'

    mcool    = MCOOL_PAT.format(cell_line=cl)
    expected = EXPECTED_PAT.format(cell_line=cl, res=RES)
    cool     = f'{mcool}::resolutions/{RES}'

    if not os.path.exists(mcool):
        return class_label, cl, None, f'missing mcool: {mcool}'
    if not os.path.exists(expected):
        return class_label, cl, None, f'missing expected: {expected}'

    cmd = [
        'coolpup.py', cool, bedpe_path,
        '--features_format', 'bedpe',
        '--expected',        expected,
        '--flank',           str(FLANK_BP),
        '--mindist',         str(MINDIST),
        '--maxdist',         str(MAXDIST),
        '--clr_weight_name', 'weight',
        '--nshifts',         str(0),
        '--nproc',           str(NPROC),
        '-o',                out_path,
    ]
    # limit_memory_gb(8)
    r = subprocess.run(cmd, capture_output=True, text=True, env=_get_subprocess_env(NPROC),)
    if r.returncode != 0:
        return class_label, cl, None, f'coolpup error:\n{r.stderr[:400]}'

    return class_label, cl, out_path, 'ok'


def run_coolpup(
    bedpe_map: dict,
    out_dir:   str,
    replot:    bool = False,
    n_workers: int  = N_WORKERS,
) -> dict:
    """
    Run coolpup for every (class, cell_line) in bedpe_map.

    Returns
    -------
    npz_map : {(class_label, cell_line): npz_path or None}
    """
    npz_dir = Path(out_dir) / 'npz'
    npz_dir.mkdir(parents=True, exist_ok=True)

    worker_args = []
    for (class_label, cl), bedpe_path in bedpe_map.items():
        if bedpe_path is None:
            continue
        safe     = _safe(class_label)
        out_path = str(npz_dir / f'{safe}__{cl}_res{RES}_pad{FLANK_BP}')
        worker_args.append((class_label, cl, bedpe_path, out_path, replot))

    # initialise all to None
    npz_map = {k: None for k in bedpe_map}

    # for args in worker_args:
    #     class_label, cl, npz_path, status = _coolpup_worker(args)
    #     npz_map[(class_label, cl)] = npz_path
    #     icon = '✓' if npz_path else '✗'
    #     print(f"  {icon} {class_label:25s} × {cl:12s}  [{status}]")

    # with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
    #     futs = {pool.submit(_coolpup_worker, a): a for a in worker_args}
    #     for fut in as_completed(futs):
    #         class_label, cl, npz_path, status = fut.result()
    #         npz_map[(class_label, cl)] = npz_path
    #         icon = '✓' if npz_path else '✗'
    #         print(f"  {icon} {class_label:25s} × {cl:12s}  [{status}]")

    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = [
            pool.submit(_coolpup_worker, args)
            for args in worker_args
        ]

        for fut in as_completed(futures):
            class_label, cl, npz_path, status = fut.result()
            npz_map[(class_label, cl)] = npz_path
            icon = "✓" if npz_path else "✗"
            print(f"  {icon} {class_label:25s} × {cl:12s}  [{status}]")

    return npz_map


############################################################
# 3) Load pileup
############################################################

def _load_pileup(npz_path: str) -> tuple:
    """Returns (log2_OE_matrix, n_loops) or (None, 0)."""
    if not npz_path or not os.path.exists(npz_path):
        return None, 0

    try:
        df = cp_io.load_pileup_df(npz_path)
    except Exception as e:
        print(f"  [WARN] {npz_path}: {e}")
        return None, 0

    if df.shape[0] == 0:
        return None, 0

    if df.shape[0] == 1:
        pup = df['data'].iloc[0]
        n   = int(df['n'].iloc[0])
    else:
        mats    = np.stack(df['data'].to_numpy())
        ns      = df['n'].to_numpy().astype(float)
        weights = ns[:, None, None]
        pup     = (mats * weights).sum(axis=0) / weights.sum()
        n       = int(ns.sum())

    with np.errstate(divide='ignore', invalid='ignore'):
        log2_oe = np.log2(pup)

    return log2_oe, n


############################################################
# 4) Panel renderer
############################################################

def _plot_panel(ax, mat, n, title='',
                vmin=VMIN, vmax=VMAX,
                add_ylabel=False, ylabel=''):
    flank_kb = FLANK_BP / 1_000

    im = ax.imshow(
        mat, origin='upper', cmap='RdBu_r',
        vmin=vmin, vmax=vmax, interpolation='none',
        extent=[-flank_kb, flank_kb, flank_kb, -flank_kb],
    )
    ax.axhline(0, color='k', lw=0.5, ls='--', alpha=0.5)
    ax.axvline(0, color='k', lw=0.5, ls='--', alpha=0.5)

    enrich = numutils.get_enrichment(mat, ENRICH_N)
    ax.text(0.03, 0.97, f'{enrich:.2f}', transform=ax.transAxes,
            ha='left', va='top', fontsize=8,
            bbox=dict(fc='white', ec='none', alpha=0.6, pad=1))
    ax.text(0.97, 0.03, f'n={n:,}', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=7,
            bbox=dict(fc='white', ec='none', alpha=0.6, pad=1))

    ax.set_title(title, fontsize=9, fontweight='bold', pad=3)
    ticks = [-flank_kb, 0, flank_kb]
    tick_labels = [f'-{FLANK_BP//1000}kb', '0', f'+{FLANK_BP//1000}kb']
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=6)
    ax.set_yticks(ticks)

    if add_ylabel:
        ax.set_yticklabels(tick_labels, fontsize=6)
        ax.set_ylabel(ylabel, fontsize=9, fontweight='bold',
                      rotation=0, ha='right', va='center', labelpad=45)
    else:
        ax.set_yticklabels([])

    return im


def _empty_panel(ax, msg='no data'):
    ax.text(0.5, 0.5, msg, ha='center', va='center',
            transform=ax.transAxes, fontsize=8, color='#888')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)


def _add_colorbar(fig, last_im):
    if last_im is None:
        return
    fig.subplots_adjust(right=0.88)
    cax  = fig.add_axes([0.90, 0.15, 0.018, 0.70])
    cbar = fig.colorbar(last_im, cax=cax)
    cbar.set_label('log₂ O/E', fontsize=9)
    cbar.ax.tick_params(labelsize=7)


############################################################
# 5) Plot grid
############################################################

def plot_apa(
    npz_map:      dict,
    class_labels: list,
    out_dir:      str,
    tag:          str,
    mode:         str = 'class_rows',
) -> None:
    """
    mode = 'class_rows'    rows=loop classes,  cols=cell lines
    mode = 'cellline_rows' rows=cell lines,     cols=loop classes
    """
    out_dir = Path(out_dir) / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)

    # pre-load all matrices
    all_mats, all_ns = {}, {}
    for key, npz_path in npz_map.items():
        mat, n = _load_pileup(npz_path)
        all_mats[key] = mat
        all_ns[key]   = n

    # shared symmetric vrange from 99th percentile of |values|
    finite_vals = np.concatenate([
        m[np.isfinite(m)].ravel()
        for m in all_mats.values() if m is not None
    ]) if any(m is not None for m in all_mats.values()) else np.array([])

    if len(finite_vals):
        absmax = np.nanpercentile(np.abs(finite_vals), 99)
        vmin, vmax = -absmax, absmax
    else:
        vmin, vmax = VMIN, VMAX

    # layout
    if mode == 'class_rows':
        rows, cols  = class_labels, CELL_LINES
        key_fn      = lambda r, c: (r, c)
        row_lbl_fn  = lambda r: r.replace('_', ' ')
        col_lbl_fn  = lambda c: c
        suptitle    = f'APA  loop classes × cell lines — {tag}'
    else:
        rows, cols  = CELL_LINES, class_labels
        key_fn      = lambda r, c: (c, r)
        row_lbl_fn  = lambda r: r
        col_lbl_fn  = lambda c: c.replace('_', ' ')
        suptitle    = f'APA  cell lines × loop classes — {tag}'

    n_rows, n_cols = len(rows), len(cols)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.2 * n_cols + 0.8, 3.2 * n_rows),
        squeeze=False,
    )

    last_im = None
    for row_i, row in enumerate(rows):
        for col_i, col in enumerate(cols):
            ax  = axes[row_i, col_i]
            k   = key_fn(row, col)
            mat = all_mats.get(k)
            n   = all_ns.get(k, 0)

            col_title = col_lbl_fn(col) if row_i == 0 else ''
            row_label = row_lbl_fn(row)

            if mat is None or not np.any(np.isfinite(mat)):
                _empty_panel(ax)
                if row_i == 0:
                    ax.set_title(col_title, fontsize=9, fontweight='bold')
                if col_i == 0:
                    ax.set_ylabel(row_label, fontsize=9, fontweight='bold',
                                  rotation=0, ha='right', va='center',
                                  labelpad=45)
            else:
                im = _plot_panel(
                    ax, mat, n,
                    title=col_title,
                    vmin=vmin, vmax=vmax,
                    add_ylabel=(col_i == 0),
                    ylabel=row_label,
                )
                last_im = im

    _add_colorbar(fig, last_im)
    fig.suptitle(suptitle, fontsize=13, fontweight='bold', y=1.01)
    # plt.tight_layout(rect=[0, 0, 0.88, 1])

    out = out_dir / f'APA_{mode}.{tag}.svg'
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: {out}")
    del all_mats
    del all_ns
    gc.collect()


############################################################
# 6) Public API
############################################################

def run_apa_pipeline(
    loop_classes:  dict,
    out_dir:       str,
    tag:           str,
    cell_line_col: str  = 'cell_line',
    able_col:      str  = 'AbLE_score',
    replot:        bool = False,
    n_workers:     int  = N_WORKERS,
    modes:         list = ('class_rows', 'cellline_rows'),
) -> dict:
    """
    Full pipeline: BEDPEs → coolpup → plots.

    Parameters
    ----------
    loop_classes  : {label: df}  long-format loops (has cell_line_col,
                    chrom1/start1/end1/chrom2/start2/end2, optional able_col)
    replot        : skip coolpup, just plot from existing NPZ files
    modes         : which plot layouts to produce

    Returns
    -------
    npz_map : {(class_label, cell_line): npz_path or None}
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_labels = list(loop_classes.keys())

    print(f"\n{'='*55}")
    print(f"APA pipeline  tag={tag}  classes={class_labels}")
    for label, df in loop_classes.items():
        print(f"  {label}: {len(df):,} rows total")

    print(f"\n--- Step 1: BEDPEs ---")
    bedpe_map = make_bedpes(
        loop_classes, str(out_dir),
        cell_line_col=cell_line_col,
        able_col=able_col,
    )

    print(f"\n--- Step 2: coolpup ---")
    npz_map = run_coolpup(bedpe_map, str(out_dir),
                           replot=replot, n_workers=n_workers)

    print(f"\n--- Step 3: plots ---")
    for mode in modes:
        plot_apa(npz_map, class_labels, str(out_dir), tag, mode=mode)

    return npz_map


############################################################
# CLI
############################################################

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--loops_tsv',       required=True)
    p.add_argument('--cluster_col_a1',  default='cluster_anchor1')
    p.add_argument('--cluster_col_a2',  default='cluster_anchor2')
    p.add_argument('--cell_line_col',   default='cell_line')
    p.add_argument('--able_col',        default='AbLE_score')
    p.add_argument('--classes',         required=True,
                    help=(
                        'Comma-separated specs "label:pair1+pair2" '
                        'where pair is "a1_a2". '
                        'Example: "5-5:5_5,10-10:10_10,5-10or10-5:5_10+10_5"'
                    ))
    p.add_argument('--out_dir',         required=True)
    p.add_argument('--tag',             required=True)
    p.add_argument('--replot',          action='store_true')
    p.add_argument('--total_cpus', type=int, default=TOTAL_CPUS,
                  help='Total CPU threads to use (for BLAS/numba/etc.)')
    p.add_argument('--n_workers',       type=int, default=N_WORKERS,
                  help='Number of parallel worker processes to use')
    p.add_argument('--modes',           default='class_rows,cellline_rows')
    return p.parse_args()


def main():
    args = _parse_args()
    global TOTAL_CPUS, N_WORKERS, NPROC
    TOTAL_CPUS = args.total_cpus
    N_WORKERS  = args.n_workers
    NPROC      = max(1, TOTAL_CPUS // max(1, N_WORKERS))
    df = pd.read_csv(args.loops_tsv, sep='\t',
        dtype={
            args.cell_line_col: 'category',
            args.cluster_col_a1: 'category',
            args.cluster_col_a2: 'category',
            'start1': 'int32', 'end1': 'int32',
            'start2': 'int32', 'end2': 'int32',
        }
    )
    print(f"Loaded {len(df):,} loops from {args.loops_tsv}")

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

    run_apa_pipeline(
        loop_classes=loop_classes,
        out_dir=args.out_dir,
        tag=args.tag,
        cell_line_col=args.cell_line_col,
        able_col=args.able_col,
        replot=args.replot,
        n_workers=args.n_workers,
        modes=args.modes.split(','),
    )
    print('[done]')


if __name__ == '__main__':
    main()
