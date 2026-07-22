#!/usr/bin/env python3
"""
loop_cluster_heatmap.py
-----------------------
For each cell line, plot a (cluster_left × cluster_right) count heatmap
showing which anchor cluster pairs form loops.

Input: long-format loop df — one row per (loop × cell_line),
       with cluster_left, cluster_right columns.

API:
    from loop_cluster_heatmap import plot_loop_cluster_heatmaps

    plot_loop_cluster_heatmaps(
        loop_df=loop_df,
        out_dir='plots/loop_cluster_heatmaps',
    )
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import sys
if 'ipykernel' not in sys.modules:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


TRAJECTORY = ['ESC', 'EpiLC', 'd4c7PGCLC', 'GSC']


############################################################
# Compute
############################################################

def compute_pair_matrix(
    df:               pd.DataFrame,
    cluster_a1_col:   str  = 'cluster_left',
    cluster_a2_col:   str  = 'cluster_right',
    normalize:        str  = 'row',    # 'row' | 'col' | 'total' | None
    able_col:         str  = None,     # if given, sum AbLE instead of count
    min_cluster_size: int  = 0,
) -> pd.DataFrame:
    """
    Build cluster_a1 × cluster_a2 matrix for a single cell line subset.

    normalize:
      'row'   : fraction of each left cluster's loops
      'col'   : fraction of each right cluster's loops
      'total' : fraction of all loops
       None   : raw counts / sum
    """
    sub = df[[cluster_a1_col, cluster_a2_col]].copy()
    if able_col and able_col in df.columns:
        sub['_val'] = df[able_col].values
    else:
        sub['_val'] = 1.0

    sub = sub[(sub[cluster_a1_col] <= 12) & (sub[cluster_a2_col] <= 12)]
    sub[cluster_a1_col] = sub[cluster_a1_col].astype(str)
    sub[cluster_a2_col] = sub[cluster_a2_col].astype(str)
    sub = sub.dropna()

    mat = sub.pivot_table(
        index=cluster_a1_col,
        columns=cluster_a2_col,
        values='_val',
        aggfunc='sum',
        fill_value=0,
    )

    # drop small clusters
    if min_cluster_size > 0:
        row_mask = mat.sum(axis=1) >= min_cluster_size
        col_mask = mat.sum(axis=0) >= min_cluster_size
        mat = mat.loc[row_mask, col_mask]

    # sort numerically
    def _sort(idx):
        try:
            return sorted(idx, key=lambda x: int(x))
        except (ValueError, TypeError):
            return sorted(idx, key=str)

    mat = mat.loc[_sort(mat.index), _sort(mat.columns)]

    if normalize == 'row':
        mat = mat.div(mat.sum(axis=1).replace(0, np.nan), axis=0)
    elif normalize == 'col':
        mat = mat.div(mat.sum(axis=0).replace(0, np.nan), axis=1)
    elif normalize == 'total':
        mat = mat / mat.values.sum()

    return mat


############################################################
# Single panel
############################################################

def _plot_panel(
    ax,
    mat:           pd.DataFrame,
    raw:           pd.DataFrame,
    title:         str,
    cmap:          str,
    vmin:          float,
    vmax:          float,
    cbar_label:    str,
    cluster_names: dict,
    annot:         bool,
    annot_fs:      int,
    add_cbar:      bool,
) -> object:

    def _labels(index):
        return [cluster_names.get(str(i), str(i)) for i in index]

    im = sns.heatmap(
        mat,
        ax=ax,
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        annot=raw.values.astype(int) if annot else False,
        fmt='d' if annot else '',
        annot_kws={'size': annot_fs},
        linewidths=0.3, linecolor='#dddddd',
        xticklabels=_labels(mat.columns),
        yticklabels=_labels(mat.index),
        cbar=add_cbar,
        cbar_kws={'label': cbar_label, 'shrink': 0.8} if add_cbar else {},
    )

    # diagonal highlight — same cluster on both anchors
    common = set(mat.index) & set(mat.columns)
    for cid in common:
        r = list(mat.index).index(cid)
        c = list(mat.columns).index(cid)
        ax.add_patch(plt.Rectangle(
            (c, r), 1, 1, fill=False,
            edgecolor='black', lw=1.5, zorder=3,
        ))

    ax.set_title(title, fontsize=11, fontweight='bold', pad=6)
    ax.set_xlabel('Anchor 2 cluster', fontsize=8)
    ax.set_ylabel('Anchor 1 cluster', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    return im


############################################################
# Public API
############################################################

def plot_loop_cluster_heatmaps(
    loop_df:          pd.DataFrame,
    out_dir:          str,
    cell_line_col:    str   = 'cell_line',
    cluster_a1_col:   str   = 'cluster_left',
    cluster_a2_col:   str   = 'cluster_right',
    able_col:         str   = None,
    trajectory:       list  = None,
    normalize:        str   = 'row',
    min_cluster_size: int   = 50,
    cluster_names:    dict  = None,
    annot:            bool  = True,
    annot_max:        int   = 20,
    cmap:             str   = 'YlOrRd',
) -> dict:
    """
    Parameters
    ----------
    loop_df           : long-format, one row per (loop × cell_line).
    able_col          : if given, cells show sum of AbLE scores
                        instead of loop counts.
    normalize         : 'row' | 'col' | 'total' | None
    min_cluster_size  : drop clusters with fewer total loops than this.
    cluster_names     : {cluster_id_str: display_name}
    annot_max         : disable per-cell annotation if n_clusters > this.

    Returns
    -------
    {cell_line: (norm_matrix, raw_matrix)}
    """
    trajectory    = trajectory or TRAJECTORY
    cluster_names = cluster_names or {}
    out_dir       = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cell_lines = [cl for cl in trajectory if cl in loop_df[cell_line_col].unique()]
    if not cell_lines:
        raise ValueError(f"No matching cell lines found. "
                         f"Available: {loop_df[cell_line_col].unique()}")

    cbar_label = {
        'row':   'Fraction of anchor-1 cluster loops',
        'col':   'Fraction of anchor-2 cluster loops',
        'total': 'Fraction of all loops',
        None:    'Loop count' if able_col is None else f'Sum {able_col}',
    }[normalize]

    # ---- compute all matrices first (shared vrange) ----
    results = {}
    for cl in cell_lines:
        sub  = loop_df[loop_df[cell_line_col] == cl]
        norm = compute_pair_matrix(sub, cluster_a1_col, cluster_a2_col,
                                    normalize=normalize,
                                    able_col=able_col,
                                    min_cluster_size=min_cluster_size)
        raw  = compute_pair_matrix(sub, cluster_a1_col, cluster_a2_col,
                                    normalize=None,
                                    able_col=None,          # always counts for annotation
                                    min_cluster_size=min_cluster_size)
        results[cl] = (norm, raw)
        n_loops = len(sub)
        print(f"  {cl:15s}: {n_loops:,} loops  "
              f"{norm.shape[0]}×{norm.shape[1]} cluster grid")

    # shared vmax from 99th percentile
    all_vals = np.concatenate([
        m.values[np.isfinite(m.values)].ravel()
        for m, _ in results.values()
    ])
    vmin = 0
    vmax = float(np.nanpercentile(all_vals, 99)) if len(all_vals) else 1.0

    # auto-disable annotation on large grids
    max_clusters = max(max(m.shape) for m, _ in results.values())
    if annot and max_clusters > annot_max:
        print(f"  NOTE: {max_clusters} clusters > {annot_max} "
              f"— disabling annotations")
        annot = False

    # ---- combined figure: one panel per cell line ----
    n      = len(cell_lines)
    dim    = max(4.0, 0.45 * max_clusters)
    fig, axes = plt.subplots(
        1, n,
        figsize=(dim * n + 1.5, dim + 1.0),
        squeeze=False,
    )
    axes = axes[0]

    last_im = None
    for ax, cl in zip(axes, cell_lines):
        norm, raw = results[cl]
        im = _plot_panel(
            ax=ax,
            mat=norm,
            raw=raw,
            title=cl,
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            cbar_label=cbar_label,
            cluster_names=cluster_names,
            annot=annot,
            annot_fs=max(5, 8 - max_clusters // 5),
            add_cbar=(ax is axes[-1]),
        )
        last_im = im

    norm_str = normalize or 'raw'
    fig.suptitle(f'Loop anchor cluster pairs  ({norm_str}-normalised)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    stem = out_dir / f'loop_cluster_pairs.{norm_str}'
    fig.savefig(f'{stem}.png', dpi=200, bbox_inches='tight')
    fig.savefig(f'{stem}.svg', bbox_inches='tight', format='svg')
    print(f"  → Saved: {stem}.png / .svg")

    # ---- individual per-cell-line figures ----
    for cl in cell_lines:
        norm, raw = results[cl]
        _dim = max(5, 0.45 * max(norm.shape) + 1.5)
        fig_s, ax_s = plt.subplots(figsize=(_dim, _dim))
        _plot_panel(
            ax=ax_s, mat=norm, raw=raw,
            title=cl, cmap=cmap,
            vmin=vmin, vmax=vmax,
            cbar_label=cbar_label,
            cluster_names=cluster_names,
            annot=annot,
            annot_fs=max(5, 8 - max_clusters // 5),
            add_cbar=True,
        )
        fig_s.suptitle(
            f'{cl}  —  loop anchor cluster pairs  ({norm_str}-normalised)',
            fontsize=11, fontweight='bold',
        )
        plt.tight_layout()
        out_s = out_dir / f'loop_cluster_pairs.{cl}.{norm_str}'
        fig_s.savefig(f'{out_s}.png', dpi=200, bbox_inches='tight')
        fig_s.savefig(f'{out_s}.svg', bbox_inches='tight', format='svg')
        plt.close(fig_s)
        print(f"  → Saved: {out_s}.png / .svg")

    if 'ipykernel' in sys.modules:
        from IPython.display import display
        display(fig)
    else:
        plt.close(fig)

    # ---- CSVs ----
    csv_dir = out_dir / 'csv'
    csv_dir.mkdir(exist_ok=True)
    for cl, (norm, raw) in results.items():
        raw.to_csv(csv_dir  / f'loop_pairs_raw.{cl}.csv')
        norm.to_csv(csv_dir / f'loop_pairs_{norm_str}.{cl}.csv')
    print(f"  → Saved CSVs: {csv_dir}")

    return results


############################################################
# CLI
############################################################

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--loop_tsv',          required=True)
    p.add_argument('--cell_line_col',     default='cell_line')
    p.add_argument('--cluster_a1_col',    default='cluster_left')
    p.add_argument('--cluster_a2_col',    default='cluster_right')
    p.add_argument('--able_col',          default=None)
    p.add_argument('--normalize',         default='row',
                    choices=['row', 'col', 'total', 'none'])
    p.add_argument('--min_cluster_size',  type=int, default=50)
    p.add_argument('--out_dir',           required=True)
    return p.parse_args()


def main():
    args    = _parse_args()
    loop_df = pd.read_csv(args.loop_tsv, sep='\t')
    print(f"Loaded {len(loop_df):,} rows from {args.loop_tsv}")

    plot_loop_cluster_heatmaps(
        loop_df=loop_df,
        out_dir=args.out_dir,
        cell_line_col=args.cell_line_col,
        cluster_a1_col=args.cluster_a1_col,
        cluster_a2_col=args.cluster_a2_col,
        able_col=args.able_col,
        normalize=None if args.normalize == 'none' else args.normalize,
        min_cluster_size=args.min_cluster_size,
    )
    print('[done]')


if __name__ == '__main__':
    main()
