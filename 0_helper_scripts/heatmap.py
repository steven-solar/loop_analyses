#!/usr/bin/env python
"""
heatmap.py
-------------------
Generic, re-runnable heatmap utilities.

  make_cluster_heatmap   row-level data -> groupby mean -> heatmap
  make_weight_heatmap    already-aggregated data (e.g. PCA loadings) -> heatmap

Can also be run from the command line (see main() below).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---- defaults ----
DEFAULT_FEATURE_ORDER = [
    'ATAC',
    'K27ac', 'K4me1', 'K4me3',
    'CTCF', 'Rad21', 'Stag1', 'Stag2',
    'Ring1b', 'H2Aub',
    'K9me2', 'K9me3', 'K27me3', 'K36me2', 'K36me3', 'Laminb1',
]
DEFAULT_BORDERS = [1, 4, 8, 16]


# ---------------------------------------------------------------------------
# Private shared renderer
# ---------------------------------------------------------------------------

def _render_heatmap(df: pd.DataFrame, feature_cols: list, out_path: Path,
                     title: str, value_label: str, borders: list,
                     cmap: str = 'RdBu_r',
                     vmin: float = None, vmax: float = None,
                     row_labels: list = None) -> None:
    """
    Core heatmap renderer shared by all public functions.

    df          : DataFrame whose rows become y-axis entries.
                  Only feature_cols columns are plotted.
    row_labels  : Optional list of strings to override y-tick labels.
                  Must be same length as df. Defaults to df.index.
    vmin/vmax   : If both None, uses ±abs-max (symmetric auto-scale).
                  If only vmax given, vmin = -vmax.
    """
    if vmax is None:
        vmax = df[feature_cols].abs().to_numpy().max()
        vmax = vmax if vmax > 0 else 1.0
    if vmin is None:
        vmin = -vmax

    n_rows = len(df)
    fig, ax = plt.subplots(figsize=(16, max(3, 0.4 * n_rows)))

    sns.heatmap(
        df[feature_cols],
        cmap=cmap, center=0, vmin=vmin, vmax=vmax,
        linewidths=0.5, linecolor='white', annot=False, ax=ax,
        cbar_kws={'label': value_label, 'shrink': 0.8, 'pad': 0.01},
    )

    # y-tick labels
    labels = row_labels if row_labels is not None else list(df.index.astype(str))
    ax.set_yticklabels(labels, rotation=0, fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)

    # feature-group borders
    for b in borders:
        if b < len(feature_cols):
            ax.axvline(b, color='black', linewidth=1.5, alpha=0.6)

    fig.suptitle(title)
    plt.savefig(out_path, bbox_inches='tight', format='svg')
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_weight_heatmap(weights: pd.DataFrame, feature_cols: list,
                         out_path: str, tag: str,
                         borders: list = None, cmap: str = 'RdBu_r',
                         value_label: str = 'Rotation weight',
                         vmin: float = None, vmax: float = None,
                         title: str = None) -> None:
    """
    Heatmap for already-aggregated data (e.g. PCA loadings, motif scores).

    weights     : DataFrame shape (n_rows, n_features), e.g. (n_PCs, n_marks).
                  Index values become y-tick labels.
    feature_cols: ordered list of columns to plot (x-axis order).
    vmin/vmax   : optional fixed colour scale; if None, uses ±abs-max.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    borders = DEFAULT_BORDERS if borders is None else borders

    missing = [f for f in feature_cols if f not in weights.columns]
    if missing:
        raise ValueError(f"feature_cols not found in weights: {missing}")

    _render_heatmap(
        df=weights,
        feature_cols=feature_cols,
        out_path=out_path,
        title=title or tag,
        value_label=value_label,
        borders=borders,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )


def make_cluster_heatmap(df: pd.DataFrame, feature_cols: list,
                          out_path: str, tag: str,
                          cluster_col: str = 'cluster',
                          mapping: dict = None, squish: bool = False,
                          borders: list = None, cmap: str = 'RdBu_r',
                          value_label: str = 'Mean z-score',
                          min_cluster_size: int = 0,
                          vmin: float = None, vmax: float = None) -> pd.DataFrame:
    """
    Row-level data -> groupby mean -> heatmap.

    df            : one row per observation, with cluster_col + feature_cols.
    feature_cols  : ordered list of feature columns (x-axis order).
    mapping       : optional dict cluster_id(str) -> display name.
    squish        : if True + mapping given, merge same-named clusters
                    into one pooled row.
    min_cluster_size : drop groups smaller than this before plotting.

    Returns the aggregated cluster x feature DataFrame (also saved as CSV).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    borders = DEFAULT_BORDERS if borders is None else borders

    # ---- validation ----
    missing_feats = [f for f in feature_cols if f not in df.columns]
    if missing_feats:
        raise ValueError(f"feature_cols not found in df: {missing_feats}")
    if cluster_col not in df.columns:
        raise ValueError(f"cluster_col '{cluster_col}' not in df. "
                          f"Found: {list(df.columns)}")

    df = df.copy()
    df[cluster_col] = df[cluster_col].astype(str)

    # ---- grouping key ----
    if mapping is not None:
        unmapped = set(df[cluster_col].unique()) - set(mapping.keys())
        if unmapped:
            print(f"  WARNING: {len(unmapped)} cluster id(s) not in mapping, "
                  f"left unmapped: {sorted(unmapped)}", file=sys.stderr)
        df['_cluster_name'] = df[cluster_col].map(mapping).fillna(df[cluster_col])

        if squish:
            group_key = '_cluster_name'
        else:
            df['_row_label'] = np.where(
                df[cluster_col].isin(mapping.keys()),
                df['_cluster_name'] + ' (' + df[cluster_col] + ')',
                df[cluster_col],
            )
            group_key = '_row_label'
    else:
        group_key = cluster_col

    # ---- drop small groups ----
    group_sizes = df.groupby(group_key).size()
    small = group_sizes[group_sizes < min_cluster_size].index.tolist()
    if small:
        print(f"  Dropping {len(small)} group(s) with < {min_cluster_size} rows: "
              f"{small} (sizes: {group_sizes.loc[small].tolist()})")
    df = df[~df[group_key].isin(small)]
    if df.empty:
        raise ValueError(
            f"All groups had < min_cluster_size={min_cluster_size} rows — nothing to plot.")

    # ---- aggregate ----
    cluster_means  = df.groupby(group_key)[feature_cols].mean()
    cluster_counts = df.groupby(group_key).size()

    # sort: numeric ids sort numerically, else alphabetically
    try:
        row_order = sorted(
            cluster_means.index,
            key=lambda x: (int(x.split(' ')[0].split('(')[0])
                           if x.split(' ')[0].split('(')[0].lstrip('-').isdigit()
                           else x),
        )
    except Exception:
        row_order = sorted(cluster_means.index, key=str)

    cluster_means  = cluster_means.loc[row_order]
    cluster_counts = cluster_counts.loc[row_order]

    if squish:
        # ---- save CSV ----
        csv_path = out_path.parent / f'cluster_feature_means.{tag}.csv'
        out_csv = cluster_means.copy()
        out_csv.insert(0, 'n', cluster_counts.values)
        out_csv.to_csv(csv_path)
        print(f"  Saved cluster means: {csv_path}")

    # ---- row labels with counts ----
    row_labels = [f'{lbl}  (n={n})'
                  for lbl, n in zip(cluster_means.index, cluster_counts.values)]

    squish_note = ' (squished)' if (mapping is not None and squish) else ''


    _render_heatmap(
        df=cluster_means,
        feature_cols=feature_cols,
        out_path=out_path,
        title=f'Cluster feature profiles{squish_note} — {tag}',
        value_label=value_label,
        borders=borders,
        vmin=vmin, vmax=vmax,
        cmap=cmap,
        row_labels=row_labels,
    )

    if squish:
        return out_csv


# ---------------------------------------------------------------------------
# Optional: mapping loader (used by CLI)
# ---------------------------------------------------------------------------

def load_mapping(mapping_tsv: str, cluster_col: str = 'cluster',
                  name_col: str = 'name') -> dict:
    map_df = pd.read_csv(mapping_tsv, sep='\t', dtype=str)
    missing = {cluster_col, name_col} - set(map_df.columns)
    if missing:
        raise ValueError(f"mapping_tsv missing columns: {missing}. "
                          f"Found: {list(map_df.columns)}")
    map_df[cluster_col] = map_df[cluster_col].astype(str)
    return dict(zip(map_df[cluster_col], map_df[name_col]))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--values_tsv', required=True)
    p.add_argument('--cluster_col', default='cluster')
    p.add_argument('--feature_cols', nargs='*', default=None)
    p.add_argument('--feature_order_file', default=None)
    p.add_argument('--borders', default=None,
                    help='Comma-separated, e.g. "1,4,8,16"')
    p.add_argument('--mapping_tsv', default=None)
    p.add_argument('--mapping_cluster_col', default='cluster')
    p.add_argument('--mapping_name_col', default='name')
    p.add_argument('--squish', action='store_true')
    p.add_argument('--out_path', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--value_label', default='Mean z-score')
    p.add_argument('--cmap', default='RdBu_r')
    args = p.parse_args()

    df = pd.read_csv(args.values_tsv, sep='\t')
    print(f"Loaded {len(df):,} rows from {args.values_tsv}")

    if args.feature_order_file:
        with open(args.feature_order_file) as fh:
            feature_cols = [l.strip() for l in fh if l.strip()]
    elif args.feature_cols:
        feature_cols = args.feature_cols
    else:
        feature_cols = [f for f in DEFAULT_FEATURE_ORDER if f in df.columns]
        if not feature_cols:
            raise ValueError("No default features found — pass --feature_cols.")

    borders = ([int(x) for x in args.borders.split(',')]
               if args.borders else None)

    mapping = None
    if args.mapping_tsv:
        mapping = load_mapping(args.mapping_tsv,
                               args.mapping_cluster_col,
                               args.mapping_name_col)
        print(f"Loaded mapping for {len(mapping)} cluster(s)")
    elif args.squish:
        print("WARNING: --squish set but no --mapping_tsv given, ignoring.",
              file=sys.stderr)

    make_cluster_heatmap(
        df=df,
        feature_cols=feature_cols,
        out_path=args.out_path,
        tag=args.tag,
        cluster_col=args.cluster_col,
        mapping=mapping,
        squish=args.squish,
        borders=borders,
        cmap=args.cmap,
        value_label=args.value_label,
    )


if __name__ == '__main__':
    main()