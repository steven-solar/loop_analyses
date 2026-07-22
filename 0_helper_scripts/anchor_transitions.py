#!/usr/bin/env python3
"""
anchor_transitions.py
---------------------
Count and visualise anchor cluster transitions along a developmental trajectory.

Input: long-format anchor df — one row per (anchor × cell_line).
       Must have: anchor_id (or chrom/start/end), cell_line, cluster columns.

Trajectory: ESC → EpiLC → d4c7PGCLC → GSC

For each consecutive pair, anchors are matched by identity and a
(from_cluster × to_cluster) count matrix is built.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


TRAJECTORY = ['ESC', 'EpiLC', 'd4c7PGCLC', 'GSC']


############################################################
# 1)  Pivot long → wide
############################################################

def pivot_to_wide(
    df:            pd.DataFrame,
    cell_line_col: str = 'cell_line',
    cluster_col:   str = 'cluster',
    anchor_id_col: str = None,        # if None, uses chrom+start+end
    trajectory:    list = None,
) -> pd.DataFrame:
    """
    Pivot long-format df to one row per anchor with one cluster column
    per cell line.

    Returns
    -------
    wide df with columns: anchor_id, cluster_ESC, cluster_EpiLC, ...
    """
    trajectory = trajectory or TRAJECTORY

    # build anchor key
    if anchor_id_col and anchor_id_col in df.columns:
        df = df.copy()
        df['_anchor_key'] = df[anchor_id_col].astype(str)
    elif {'chrom', 'start', 'end'}.issubset(df.columns):
        df = df.copy()
        df['_anchor_key'] = (df['chrom'].astype(str) + ':' +
                              df['mid'].astype(str))
    else:
        raise ValueError(
            "Cannot identify anchors — pass anchor_id_col or ensure "
            "chrom/start/end columns are present."
        )

    # keep only trajectory cell lines
    sub = df[df[cell_line_col].isin(trajectory)][
        ['_anchor_key', cell_line_col, cluster_col]
    ].copy()
    sub[cluster_col] = sub[cluster_col].astype(str)

    wide = sub.pivot_table(
        index='_anchor_key',
        columns=cell_line_col,
        values=cluster_col,
        aggfunc='first',   # each anchor should appear once per cell line
    )
    wide.columns = [f'cluster_{cl}' for cl in wide.columns]
    wide = wide.reset_index().rename(columns={'_anchor_key': 'anchor_id'})

    print(f"  Pivoted: {len(wide):,} unique anchors × {len(wide.columns)-1} cell lines")
    for cl in trajectory:
        col = f'cluster_{cl}'
        if col in wide.columns:
            n_null = wide[col].isna().sum()
            print(f"    {cl:15s}: {wide[col].notna().sum():,} anchors, "
                  f"{n_null:,} missing")
    return wide


############################################################
# 2)  Compute transitions  (operates on wide df)
############################################################

def compute_transitions(
    wide:       pd.DataFrame,
    trajectory: list,
    normalize:  str = 'row',
) -> tuple:
    """
    Build from_cluster × to_cluster count matrix per step.

    Returns
    -------
    raw_counts  : {(from_cl, to_cl): DataFrame}  integer counts
    norm_counts : {(from_cl, to_cl): DataFrame}  normalized
    """
    raw_counts  = {}
    norm_counts = {}

    for from_cl, to_cl in zip(trajectory[:-1], trajectory[1:]):
        from_col = f'cluster_{from_cl}'
        to_col   = f'cluster_{to_cl}'

        if from_col not in wide.columns or to_col not in wide.columns:
            print(f"  [WARN] missing columns for {from_cl}→{to_cl}, skipping")
            continue

        # only anchors present in BOTH cell lines
        pairs = wide[[from_col, to_col]].dropna()
        n_total    = len(wide)
        n_both     = len(pairs)
        n_from_only = wide[from_col].notna().sum() - n_both
        n_to_only   = wide[to_col].notna().sum()   - n_both

        print(f"\n  {from_cl} → {to_cl}")
        print(f"    anchors in both    : {n_both:,}")
        print(f"    only in {from_cl:10s}: {n_from_only:,}")
        print(f"    only in {to_cl:10s}: {n_to_only:,}")

        counts = pd.crosstab(pairs[from_col], pairs[to_col])

        def _sort(idx):
            try:
                return sorted(idx, key=lambda x: int(x))
            except (ValueError, TypeError):
                return sorted(idx, key=str)

        counts = counts.loc[_sort(counts.index), _sort(counts.columns)]
        raw_counts[(from_cl, to_cl)] = counts

        if normalize == 'row':
            norm = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
        elif normalize == 'col':
            norm = counts.div(counts.sum(axis=0).replace(0, np.nan), axis=1)
        else:
            norm = counts.astype(float)

        norm_counts[(from_cl, to_cl)] = norm

    return raw_counts, norm_counts


############################################################
# 3)  Plotting  (unchanged from before)
############################################################

def _plot_step(ax, raw, norm, step, cmap, vmin, vmax,
               norm_label, cluster_names=None,
               annot=True, annot_fontsize=7, add_cbar=False):

    from_cl, to_cl = step

    def _labels(index):
        if cluster_names:
            return [cluster_names.get(str(i), str(i)) for i in index]
        return [str(i) for i in index]

    sns.heatmap(
        norm,
        ax=ax,
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        annot=raw.values if annot else False,
        fmt='d' if annot else '',
        annot_kws={'size': annot_fontsize},
        linewidths=0.3, linecolor='#dddddd',
        xticklabels=_labels(norm.columns),
        yticklabels=_labels(norm.index),
        cbar=add_cbar,
        cbar_kws={'label': norm_label, 'shrink': 0.7} if add_cbar else {},
    )

    # highlight diagonal — same cluster id before and after
    common = set(norm.index) & set(norm.columns)
    for cid in common:
        r = list(norm.index).index(cid)
        c = list(norm.columns).index(cid)
        ax.add_patch(plt.Rectangle(
            (c, r), 1, 1, fill=False,
            edgecolor='black', lw=1.5, zorder=3,
        ))

    ax.set_title(f'{from_cl}  →  {to_cl}',
                 fontsize=10, fontweight='bold', pad=6)
    ax.set_xlabel(f'{to_cl} cluster',   fontsize=8)
    ax.set_ylabel(f'{from_cl} cluster', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)


def _save(fig, stem: Path) -> None:
    fig.savefig(f'{stem}.png', dpi=200, bbox_inches='tight')
    fig.savefig(f'{stem}.svg', bbox_inches='tight', format='svg')
    plt.close(fig)
    print(f"  → Saved: {stem}.png / .svg")


def plot_transitions(raw_counts, norm_counts, out_dir, tag,
                     normalize='row', cluster_names=None,
                     annot=True, annot_fontsize=7):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = list(norm_counts.keys())
    if not steps:
        print("  No transitions to plot.")
        return

    cmap       = 'YlOrRd' if normalize in ('row', 'col') else 'Blues'
    vmin       = 0
    vmax       = 1.0 if normalize in ('row', 'col') else None
    norm_label = {
        'row': 'Fraction of from-cluster (fate)',
        'col': 'Fraction of to-cluster (origin)',
        'raw': 'Anchor count',
    }[normalize]

    n_steps   = len(steps)
    panel_dim = max(4.0, 0.45 * max(
        max(m.shape) for m in norm_counts.values()
    ))

    # ---- combined ----
    fig, axes = plt.subplots(
        1, n_steps,
        figsize=(panel_dim * n_steps + 1.5, panel_dim + 1.0),
        squeeze=False,
    )
    for ax, step in zip(axes[0], steps):
        _plot_step(
            ax=ax, raw=raw_counts[step], norm=norm_counts[step],
            step=step, cmap=cmap, vmin=vmin,
            vmax=vmax or float(norm_counts[step].max().max()),
            norm_label=norm_label, cluster_names=cluster_names,
            annot=annot, annot_fontsize=annot_fontsize,
            add_cbar=(ax is axes[0][-1]),
        )
    fig.suptitle(f'Anchor cluster transitions — {tag}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    _save(fig, out_dir / f'transitions_combined.{tag}')

    # ---- per-step ----
    for step in steps:
        from_cl, to_cl = step
        norm  = norm_counts[step]
        _vmax = vmax or float(norm.max().max())

        fig_s, ax_s = plt.subplots(
            figsize=(max(5, 0.45 * norm.shape[1] + 1.5),
                     max(5, 0.45 * norm.shape[0] + 1.0)),
        )
        _plot_step(
            ax=ax_s, raw=raw_counts[step], norm=norm,
            step=step, cmap=cmap, vmin=vmin, vmax=_vmax,
            norm_label=norm_label, cluster_names=cluster_names,
            annot=annot, annot_fontsize=annot_fontsize,
            add_cbar=True,
        )
        fig_s.suptitle(
            f'{from_cl} → {to_cl}  ({normalize}-normalised)  — {tag}',
            fontsize=11, fontweight='bold',
        )
        plt.tight_layout()
        _save(fig_s, out_dir / f'transition.{from_cl}_to_{to_cl}.{tag}')


############################################################
# 4)  Stability summary
############################################################

def stability_summary(raw_counts: dict) -> pd.DataFrame:
    rows = []
    for (from_cl, to_cl), counts in raw_counts.items():
        total  = counts.values.sum()
        common = set(counts.index) & set(counts.columns)
        stable = sum(counts.loc[c, c] for c in common)
        rows.append({
            'step':        f'{from_cl} → {to_cl}',
            'n_anchors':   int(total),
            'n_stable':    int(stable),
            'frac_stable': round(stable / total, 4) if total else np.nan,
        })
    return pd.DataFrame(rows)


############################################################
# 5)  Public API
############################################################

def run_transition_analysis(
    df:               pd.DataFrame,
    out_dir:          str,
    tag:              str,
    cell_line_col:    str  = 'cell_line',
    cluster_col:      str  = 'cluster',
    anchor_id_col:    str  = None,
    trajectory:       list = None,
    normalize:        str  = 'row',
    cluster_names:    dict = None,
    annot:            bool = True,
    annot_max_clusters: int = 20,
) -> dict:
    """
    Parameters
    ----------
    df            : long-format anchor df — one row per (anchor × cell_line).
    cell_line_col : column identifying the cell line  (default 'cell_line').
    cluster_col   : single cluster column shared across all cell lines
                    (default 'cluster').
    anchor_id_col : unique anchor identifier column. If None, uses
                    chrom+start+end.
    trajectory    : ordered cell lines. Defaults to TRAJECTORY.
    normalize     : 'row' (fate) | 'col' (origin) | 'raw'
    cluster_names : {cluster_id_str: display_name}
    """
    trajectory = trajectory or TRAJECTORY
    out_dir    = Path(out_dir)

    print(f"\n{'='*55}")
    print(f"Transition analysis  tag={tag}")
    print(f"Trajectory : {' → '.join(trajectory)}")
    print(f"Input      : {len(df):,} anchor×cell-line rows")
    for cl in trajectory:
        n = (df[cell_line_col] == cl).sum()
        print(f"  {cl:15s}: {n:,} rows")

    # long → wide
    wide = pivot_to_wide(
        df, cell_line_col=cell_line_col,
        cluster_col=cluster_col,
        anchor_id_col=anchor_id_col,
        trajectory=trajectory,
    )

    # auto-disable annotations on large grids
    n_clusters = max(
        wide[f'cluster_{cl}'].nunique()
        for cl in trajectory
        if f'cluster_{cl}' in wide.columns
    )
    if annot and n_clusters > annot_max_clusters:
        print(f"  NOTE: {n_clusters} clusters — disabling cell annotations")
        annot = False

    raw_counts, norm_counts = compute_transitions(wide, trajectory, normalize)

    # save CSVs
    csv_dir = out_dir / 'csv'
    csv_dir.mkdir(parents=True, exist_ok=True)

    wide.to_csv(csv_dir / f'anchors_wide.{tag}.tsv', sep='\t', index=False)

    for (from_cl, to_cl), counts in raw_counts.items():
        step_str = f'{from_cl}_to_{to_cl}'
        counts.to_csv(csv_dir / f'transitions_raw.{step_str}.{tag}.csv')
        norm_counts[(from_cl, to_cl)].to_csv(
            csv_dir / f'transitions_{normalize}norm.{step_str}.{tag}.csv'
        )

    stab = stability_summary(raw_counts)
    stab.to_csv(csv_dir / f'stability_summary.{tag}.csv', index=False)
    print(f"\n  Stability summary:\n{stab.to_string(index=False)}")

    plot_transitions(
        raw_counts, norm_counts,
        out_dir=str(out_dir), tag=tag,
        normalize=normalize, cluster_names=cluster_names,
        annot=annot,
    )

    return {'raw': raw_counts, 'norm': norm_counts, 'wide': wide}


############################################################
# CLI
############################################################

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--anchors_tsv',    required=True)
    p.add_argument('--cell_line_col',  default='cell_line')
    p.add_argument('--cluster_col',    default='cluster')
    p.add_argument('--anchor_id_col',  default=None)
    p.add_argument('--trajectory',     default=None,
                    help='Comma-separated ordered cell lines')
    p.add_argument('--normalize',      default='row',
                    choices=['row', 'col', 'raw'])
    p.add_argument('--out_dir',        required=True)
    p.add_argument('--tag',            required=True)
    p.add_argument('--no_annot',       action='store_true')
    return p.parse_args()


def main():
    args = _parse_args()
    df   = pd.read_csv(args.anchors_tsv, sep='\t')
    print(f"Loaded {len(df):,} rows from {args.anchors_tsv}")

    trajectory = (
        [c.strip() for c in args.trajectory.split(',')]
        if args.trajectory else None
    )

    run_transition_analysis(
        df=df,
        out_dir=args.out_dir,
        tag=args.tag,
        cell_line_col=args.cell_line_col,
        cluster_col=args.cluster_col,
        anchor_id_col=args.anchor_id_col,
        trajectory=trajectory,
        normalize=args.normalize,
        annot=not args.no_annot,
    )
    print('[done]')


if __name__ == '__main__':
    main()
