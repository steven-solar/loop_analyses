import pandas as pd
import importlib
import sys
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

sys.path.append('/mnt/md0/sjsolar/loop_pred/loop_analyses/0_helper_scripts')  # wherever you save it
import anchor_sankey
importlib.reload(anchor_sankey)
from anchor_sankey import plot_anchor_sankey

CELL_LINES = ['ESC', 'EpiLC', 'd4c7PGCLC', 'GSC']

# anchors_wide = pd.read_csv('../3_anchor_clustering/clustering/anchors_3kb_wide_labeled.tsv', sep='\t')
# anchors_wide = pd.read_csv('../3_anchor_clustering/clustering_subcluster_c6/anchors_reclustered_wide.tsv', sep='\t')
# anchors_wide = pd.read_csv('../3_anchor_clustering/clustering_recluster_gsc/anchors_recluster_gsc_wide.tsv', sep='\t')
# anchors_wide = pd.read_csv('../3_anchor_clustering/clustering_subcluster_cre_prc/anchors_cre_prc_reclustered_wide.tsv', sep='\t')
anchors_wide = pd.read_csv('../3_anchor_clustering/clustering_recluster_per_cellline/anchors_reclustered_per_cl_wide.tsv', sep='\t')

names_cats_colors_df = pd.read_csv('../3_anchor_clustering/clustering/anchor_names_cats_colors.tsv', sep='\t')
cluster_colors = {f"{row['cluster']}": row['cluster_color'] for _, row in names_cats_colors_df.iterrows()}
cluster_names = {f"{row['cluster']}": row['cluster_name_long'] for _, row in names_cats_colors_df.iterrows()}
category_colors = {row['category_name']: row['category_color'] for _, row in names_cats_colors_df.iterrows()}

sankey_data = plot_anchor_sankey(
    wide=anchors_wide,           # long-format, same as run_transition_analysis
    out_dir='anchor_sankey',
    trajectory=CELL_LINES,
    normalize=False,           # True = link width = fraction of from-cluster
    min_flow=50,               # hide tiny flows
	able_col_pat=None,
	cluster_colors=cluster_colors,
	cluster_names=cluster_names,
)