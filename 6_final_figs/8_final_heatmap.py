import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')
plt.rcParams.update({
    "text.usetex": False})
plt.rcParams['svg.fonttype'] = 'none'
plt.rc('pdf', fonttype=42)
plt.rcParams["ps.useafm"] = True
matplotlib.rcParams['path.simplify'] = False        # don't simplify vector paths
matplotlib.rcParams['agg.path.chunksize'] = 0       # no chunking

matplotlib.rcParams.update({
    'font.size':     6,
    'axes.linewidth': 0.25,
    'axes.labelsize': 6,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'svg.fonttype':  'none',       # keeps text as real text, not outlines, in Illustrator
})

import sys
sys.path.append('/mnt/md0/sjsolar/loop_pred/loop_analyses/0_helper_scripts')  # wherever you save it
from heatmap import make_cluster_heatmap, make_weight_heatmap

FEATURE_ORDER = [
    'ATAC',
    'K27ac', 'K4me1', 'K4me3',
    'CTCF', 'Rad21', 'Stag1', 'Stag2',
    'Ring1b', 'H2Aub',
    'K9me2', 'K9me3', 'K27me3', 'K36me2', 'K36me3', 'Laminb1',
]
BORDERS = [1, 4, 8, 16]  # feature-axis group borders, fixed regardless of row grouping

# row order = these cluster_name_short values, in this order; display label is the value
CATEGORY_ORDER = ['CTCF', 'CRE', 'PRC', 'Weak']
CATEGORY_LABELS = {'CTCF': 'CTCF', 'CRE': 'CRE', 'PRC': 'PRC', 'Weak': 'Mixed'}

# fine-grained `cluster` ids to break out of their parent category and show as their own rows,
# appended after the four categories above, in this order
PULLOUT_CLUSTERS = {11: 'K9 Bivalent', 12: 'Stag2-CRE'}


def make_merged_cluster_heatmap(
    anchors_tsv,
    out_cluster_dir,
    tag='merged_clusters',
    vmin=-3, vmax=3,
    value_label='Mean z-score',
):
    """
    Load a reclustered anchors TSV (e.g. anchors_reclustered_per_cl.tsv),
    build epigenomic '_clip' heatmap rows for the four merged categories
    (CTCF, CRE, PRC, Weak->Mixed, in that order), plus two fine-grained
    `cluster` ids (11, 12) broken out as their own separate rows, and
    call make_cluster_heatmap() on the result.
    """
    df = pd.read_csv(anchors_tsv, sep='\t')
    print(f"[info] loaded {len(df)} rows from {anchors_tsv}")

    clip_cols = [f'{f}_clip' for f in FEATURE_ORDER]
    required = clip_cols + ['cluster', 'cluster_name_short']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    n0 = len(df)
    df = df.dropna(subset=required).copy()
    if len(df) < n0:
        print(f"[warn] dropped {n0 - len(df)} rows with NaNs in cluster/clip cols")

    unexpected_cats = set(df['cluster_name_short'].unique()) - set(CATEGORY_ORDER)
    if unexpected_cats:
        print(f"[warn] cluster_name_short values outside CATEGORY_ORDER (rows dropped): {unexpected_cats}")

    present_pullouts = set(df.loc[df['cluster'].isin(PULLOUT_CLUSTERS), 'cluster'].unique())
    missing_pullouts = set(PULLOUT_CLUSTERS) - present_pullouts
    if missing_pullouts:
        print(f"[warn] pullout cluster ids not found in `cluster` column: {missing_pullouts}")

    is_pullout = df['cluster'].isin(PULLOUT_CLUSTERS)

    # main rows: the four categories, excluding anything pulled out, labeled/ordered per CATEGORY_ORDER
    main = df[~is_pullout & df['cluster_name_short'].isin(CATEGORY_ORDER)].copy()
    main['plot_row'] = main['cluster_name_short'].map(CATEGORY_LABELS)

    # pullout rows: cluster 11 -> 'K9 Bivalent', cluster 12 -> 'Stag2-CRE', regardless of their category
    pullout = df[is_pullout].copy()
    pullout['plot_row'] = pullout['cluster'].map(PULLOUT_CLUSTERS)

    plot_df = pd.concat([main, pullout], ignore_index=True)

    row_order = [CATEGORY_LABELS[c] for c in CATEGORY_ORDER] + list(PULLOUT_CLUSTERS.values())
    plot_df['plot_row'] = pd.Categorical(plot_df['plot_row'], categories=row_order, ordered=True)
    plot_df = plot_df.sort_values('plot_row')
    print(f"[info] final row order: {row_order}")

    cluster_means = (
        plot_df.groupby('plot_row', observed=True)[clip_cols]
        .mean()
        .reindex(row_order)
    )
    means_path = f'{out_cluster_dir}/cluster_feature_means.{tag}.csv'
    cluster_means.to_csv(means_path)
    print(f"[info] wrote cluster means -> {means_path}")

    heatmap_path = f'{out_cluster_dir}/cluster_heatmap.{tag}.svg'
    make_cluster_heatmap(
        df=plot_df,
        feature_cols=clip_cols,
        out_path=heatmap_path,
        tag=tag,
        cluster_col='plot_row',
        borders=BORDERS,
        vmin=vmin, vmax=vmax,
        value_label=value_label,
    )
    print(f"[info] wrote heatmap -> {heatmap_path}")

    return cluster_means, row_order


if __name__ == '__main__':
    ANCHORS_TSV = '/mnt/md0/sjsolar/loop_pred/loop_analyses/3_anchor_clustering/clustering_recluster_per_cellline/anchors_reclustered_per_cl.tsv'
    OUT_CLUSTER_DIR = '.'  # set to your actual output dir

    make_merged_cluster_heatmap(ANCHORS_TSV, OUT_CLUSTER_DIR)