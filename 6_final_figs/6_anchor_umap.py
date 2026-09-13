import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({
    'font.size':     6,
    'axes.linewidth': 0.25,
    'axes.labelsize': 6,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'svg.fonttype':  'none',
    'pdf.fonttype':  42,
})

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd 

df = pd.read_csv('../3_anchor_clustering/clustering_recluster_per_cellline/anchors_q0.01_3kb_all_celllines_pca_clustered_umap.tsv', sep='\t')

def style_axes(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(0.25)
    ax.tick_params(width=0.25, length=2)
    sns.despine(ax=ax)

# By cluster
fig, ax = plt.subplots(figsize=(10, 8))

cluster_color_map = (
    df.drop_duplicates('cluster_name_long')
      [['cluster_name_long', 'cluster_color']]
      .set_index('cluster_name_long')['cluster_color']
      .to_dict()
)

for cluster, color in cluster_color_map.items():
    mask = df['cluster_name_long'] == cluster
    ax.scatter(
        df.loc[mask, 'UMAP1'],
        df.loc[mask, 'UMAP2'],
        c=color,
        s=2,
        alpha=0.2,
        linewidths=1,    # data stroke = 1 pt (marker edge)
        edgecolors=color,
        label=cluster,
        rasterized=True,
    )

patches = [mpatches.Patch(color=color, label=label) 
           for label, color in cluster_color_map.items()]
ax.legend(
    handles=patches,
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    frameon=False,   # no legend box
    markerscale=2,
    fontsize=6
)

ax.set_xlabel('UMAP1', fontsize=6)
ax.set_ylabel('UMAP2', fontsize=6)
ax.set_title('Anchor UMAP colored by cluster', fontsize=7)
ax.set_aspect('equal')
style_axes(ax)

plt.savefig('anchor_umap/anchor_umap_by_cluster.pdf')


# By category
fig, ax = plt.subplots(figsize=(10, 8))

category_color_map = (
    df.drop_duplicates('category_name')
      [['category_name', 'category_color']]
      .set_index('category_name')['category_color']
      .to_dict()
)

for category, color in category_color_map.items():
    mask = df['category_name'] == category
    ax.scatter(
        df.loc[mask, 'UMAP1'],
        df.loc[mask, 'UMAP2'],
        c=color,
        s=2,
        alpha=0.2,
        linewidths=1,
        edgecolors=color,
        label=category,
        rasterized=True
    )

patches = [mpatches.Patch(color=color, label=label) 
           for label, color in category_color_map.items()]
ax.legend(
    handles=patches,
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    frameon=False,
    markerscale=2,
    fontsize=6
)

ax.set_xlabel('UMAP1', fontsize=6)
ax.set_ylabel('UMAP2', fontsize=6)
ax.set_title('Anchor UMAP colored by category', fontsize=7)
ax.set_aspect('equal')
style_axes(ax)

plt.savefig('anchor_umap/anchor_umap_by_category.pdf')


# By cell line
cl_color_df = pd.read_csv('../data/cl_colors.tsv', sep='\t')
cl_colormap = dict(zip(cl_color_df['cell_line'], cl_color_df['color']))
print(cl_colormap)

fig, ax = plt.subplots(figsize=(10, 8))

for cl, color in cl_colormap.items():
    mask = df['cell_line'] == cl
    ax.scatter(
        df.loc[mask, 'UMAP1'],
        df.loc[mask, 'UMAP2'],
        c=color,
        s=2,
        alpha=0.2,
        linewidths=1,
        edgecolors=color,
        label=cl,
        rasterized=True
    )

patches = [mpatches.Patch(color=color, label=label) 
           for label, color in cl_colormap.items()]
ax.legend(
    handles=patches,
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    frameon=False,
    markerscale=2,
    fontsize=6
)

ax.set_xlabel('UMAP1', fontsize=6)
ax.set_ylabel('UMAP2', fontsize=6)
ax.set_title('Anchor UMAP colored by cell line', fontsize=7)
ax.set_aspect('equal')
style_axes(ax)

plt.savefig('anchor_umap/anchor_umap_by_cell_line.pdf')
