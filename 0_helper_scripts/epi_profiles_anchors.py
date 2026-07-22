#!/usr/bin/env python3
"""
epi_profiles_anchors.py
------------------------
Like epi_profiles_compare.py but takes anchor DataFrames directly.
Useful for comparing all cluster-X anchors regardless of their loop partner.

API usage:
    from epi_profiles_anchors import run_anchor_comparison

    anchors_c5  = anchor_df[anchor_df['cluster'] == 5]
    anchors_c10 = anchor_df[anchor_df['cluster'] == 10]

    run_anchor_comparison(
        anchor_classes={
            'cluster 5':  anchors_c5,
            'cluster 10': anchors_c10,
        },
        out_dir='profiles_anchors_c5v10',
        tag='c5v10',
    )

CLI usage:
    python epi_profiles_anchors.py \
        --anchors_tsv anchors_clustered.tsv \
        --cluster_col cluster \
        --clusters 5,10,15 \
        --out_dir profiles_out --tag c5v10
"""

import argparse
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# reuse all workers + plotting from the loop version
from epi_profiles_compare import (
    MARKS, CELL_LINES, PREFIXES, BIGWIG_PAT,
    FLANK, BIN_SIZE, SMOOTH, NPROC, N_WORKERS,
    CL_STYLES, _PALETTE,
    _run_computematrix, _parse_matrix, _mean_profiles,
    plot_comparison,
)


############################################################
# 1) BED from anchor df  (replaces make_anchor_bed in loop version)
############################################################

def make_anchor_bed(
    df: pd.DataFrame,
    out_path: str,
    chrom_col: str = 'chrom',
    mid_col: str = 'mid',
    half:      int = 500,
) -> int:
    """
    Write a BED file of anchor midpoints ± half bp.

    df may have any extra columns — only chrom/mid are used.
    Rows with non-standard chromosomes are dropped.
    """
    mid = df[mid_col].astype(int)

    anchors = pd.DataFrame({
        'chrom': df[chrom_col].values,
        'start': (mid - half).clip(lower=0).values,
        'end':   (mid + half).values,
    })
    anchors = (
        anchors
        .drop_duplicates()
        .query("chrom.str.match('^chr[0-9XYM]+$')", engine='python')
        .sort_values(['chrom', 'start'])
    )
    anchors.to_csv(out_path, sep='\t', header=False, index=False)
    return len(anchors)


############################################################
# 2) Worker  (same logic as loop version, different BED source)
############################################################

def _worker(args: tuple):
    class_label, mark, bed_path, class_dir, replot = args

    bw_files, bw_labels = [], []
    for i, cl in enumerate(CELL_LINES):
        bw = BIGWIG_PAT.format(cell_line=cl, prefix=PREFIXES[i], mark=mark)
        bw_path = Path(bw)
        if bw_path.exists():
            bw_files.append(bw)
            bw_labels.append(cl)
        else:
            print(f"    WARN: missing bigwig {bw}")

    if not bw_files:
        return class_label, mark, {}

    gz = str(Path(class_dir) / f'matrix_{mark}.gz')
    try:
        if not replot:
            _run_computematrix(bed_path, bw_files, bw_labels, gz)
        if not Path(gz).exists():
            return class_label, mark, {}
        data, n_bins, labels = _parse_matrix(gz)
        profs = _mean_profiles(data, n_bins, labels)
    except Exception as e:
        print(f"    ERROR {class_label}/{mark}: {e}")
        return class_label, mark, {}

    return class_label, mark, profs


############################################################
# 3) Public API
############################################################

def run_anchor_comparison(
    anchor_classes:  dict,          # {label: anchor_df}
    out_dir:         str,
    tag:             str,
    chrom_col:       str   = 'chrom',
    mid_col:         str   = 'mid',
    anchor_half:     int   = 500,
    colors:    dict  = None,
    replot:          bool  = False,
    show_cell_lines: bool  = True,
    n_workers:       int   = N_WORKERS,
    ncols:          int   = 4,
    alphas:    dict  = None,
    linestyles: dict  = {},
) -> dict:
    """
    Parameters
    ----------
    anchor_classes : {'cluster 5': df_c5, 'cluster 10': df_c10, ...}
                     Each df needs chrom_col / mid_col.
                     Any subsetting (by cluster, cell line, etc.) is done
                     by the caller before passing.
    colors         : {label: hex}  — auto-assigned from palette if None.
    show_cell_lines: True  → color=class, linestyle=cell_line per mark
                     False → one pooled line per class

    Returns
    -------
    all_profiles : {class_label: {mark: {cell_line: np.ndarray}}}
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # fill missing colors
    provided = colors or {}
    class_colors = {
        label: provided.get(label, _PALETTE[i % len(_PALETTE)])
        for i, label in enumerate(anchor_classes)
    }

    # ---- BED files ----
    n_anchors   = {}
    worker_args = []

    print(f"\n{'='*55}")
    print(f"Anchor comparison  tag={tag}")

    for class_label, df in anchor_classes.items():
        safe      = (class_label.replace(' ', '_')
                                .replace('|', 'or')
                                .replace('/', '_')
                                .replace('-', '_'))
        class_dir = out_dir / safe
        class_dir.mkdir(exist_ok=True)

        bed_path = class_dir / 'anchors.bed'
        n = make_anchor_bed(
            df, str(bed_path),
            chrom_col=chrom_col,
            mid_col=mid_col,
            half=anchor_half,
        )
        n_anchors[class_label] = n
        print(f"  {class_label}: {n:,} anchors  → {bed_path.name}")

        if n == 0:
            print(f"  WARNING: no anchors for {class_label}, skipping")
            continue

        for mark in MARKS:
            worker_args.append(
                (class_label, mark, str(bed_path), str(class_dir), replot)
            )

    # ---- computeMatrix (parallel) ----
    all_profiles = {cl: {} for cl in anchor_classes}

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_worker, a): (a[0], a[1]) for a in worker_args}
        for fut in as_completed(futs):
            class_label, mark, profs = fut.result()
            all_profiles[class_label][mark] = profs
            n_cls = sum(1 for p in profs.values() if p is not None)
            print(f"    ✓ {class_label:25s}  {mark:12s}  ({n_cls} cell lines)")

    # ---- plot ----  (reuses loop-comparison renderer unchanged)
    plot_comparison(
        all_profiles, class_colors, n_anchors,
        str(out_dir), tag,
        show_cell_lines=show_cell_lines,
        ncols=ncols,
        alphas=alphas,
        linestyles=linestyles,
    )

    return all_profiles


############################################################
# CLI
############################################################

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--anchors_tsv',  required=True,
                    help='TSV with chrom/start/end + cluster_col')
    p.add_argument('--cluster_col',  default='cluster')
    p.add_argument('--clusters',     required=True,
                    help='Comma-separated cluster ids, e.g. "5,10,15"')
    p.add_argument('--chrom_col',    default='chrom')
    p.add_argument('--mid_col',      default='mid')
    p.add_argument('--out_dir',      required=True)
    p.add_argument('--tag',          required=True)
    p.add_argument('--replot',       action='store_true')
    p.add_argument('--n_workers',    type=int, default=N_WORKERS)
    return p.parse_args()


def main():
    args = _parse_args()
    df = pd.read_csv(args.anchors_tsv, sep='\t')
    print(f"Loaded {len(df):,} anchors from {args.anchors_tsv}")

    cluster_ids = [c.strip() for c in args.clusters.split(',')]

    anchor_classes = {}
    for cid in cluster_ids:
        mask = df[args.cluster_col].astype(str) == cid
        sub  = df[mask]
        print(f"  cluster {cid}: {len(sub):,} anchors")
        anchor_classes[f'cluster {cid}'] = sub

    run_anchor_comparison(
        anchor_classes=anchor_classes,
        out_dir=args.out_dir,
        tag=args.tag,
        chrom_col=args.chrom_col,
        mid_col=args.mid_col,
        replot=args.replot,
        show_cell_lines=not args.pool_cell_lines,
        n_workers=args.n_workers,
        alphas=args.alphas or {},
        linestyles=args.linestyles or {},
    )
    print('[done]')


if __name__ == '__main__':
    main()
