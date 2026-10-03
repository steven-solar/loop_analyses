# conda activate umap_env (or any)

'''
Overlap my clusters with peak based defs, have run this a few times on diff clustering because of PRC-CRE issue, so a few commented out files here
'''

import matplotlib
matplotlib.use('Agg')          # non-interactive, saves to file fine
import matplotlib.pyplot as plt
import pandas as pd

# CLUSTER_PATH = 'clustering'
# CLUSTER_PATH='clustering_subcluster_c6'
# CLUSTER_PATH='clustering_recluster_gsc'
# CLUSTER_PATH = 'clustering_subcluster_cre_prc'
CLUSTER_PATH = 'clustering_recluster_per_cellline'

ANN_PATH = '../2_epigenomics/anchor_epigenomics'

cluster_3kb = pd.read_csv(f'{CLUSTER_PATH}/anchors_reclustered_per_cl.tsv', sep='\t')

ann_3kb = pd.read_csv(f'{ANN_PATH}/anchors_q0.01_3kb_annotated.p10.tsv', sep='\t')
merge_3kb = pd.merge(cluster_3kb, ann_3kb, on=['anchor_id', 'chrom', 'mid', 'cell_line', 'anchor'], how='inner')
merge_3kb['is_CRE'] = merge_3kb['is_E'] | merge_3kb['is_P']
merge_3kb['is_none'] = ~(merge_3kb['is_CTCF'] | merge_3kb['is_CRE'] | merge_3kb['is_PRC_narrow'])

def add_excl_defs(df):
    df['is_only_CRE'] = df['is_CRE'] & ~df['is_CTCF'] & ~df['is_PRC_narrow'] & ~df['is_PRC_broad']
    df['is_only_CTCF'] = df['is_CTCF'] & ~df['is_CRE'] & ~df['is_PRC_narrow'] & ~df['is_PRC_broad']
    df['is_only_PRC_narrow'] = df['is_PRC_narrow'] & ~df['is_CRE'] & ~df['is_CTCF']
    df['is_none'] = ~(df['is_CTCF'] | df['is_CRE'] | df['is_PRC_narrow']) # | df['is_PRC_broad'])
    return df

merge_3kb = add_excl_defs(merge_3kb)

# anno_cols = ['is_only_CTCF', 'is_only_CRE', 'is_only_PRC_narrow', 'is_none']
# anno_colors = {
#     'is_only_CRE':  '#33BCEE',
#     'is_only_CTCF': '#CC3412',
#     'is_only_PRC_narrow': '#4EB265',
#     'is_none': '#777777'
# }

anno_cols = ['is_CTCF', 'is_E', 'is_PRC_narrow', 'is_none']
anno_colors = {
    'is_E':  '#33BCEE',
    'is_CTCF': '#CC3412',
    'is_PRC_narrow': '#4EB265',
    'is_none': '#777777'
}

counts = (merge_3kb
          .groupby('cluster')[anno_cols]
          .sum()                          # sum of 1s = count of positives
          .astype(int))

cluster_sizes = merge_3kb.groupby('cluster').size().rename('total')
proportions = counts.div(cluster_sizes, axis=0)

print(counts)
print(proportions)

fig, ax = plt.subplots(figsize=(10, 5))
proportions.plot(kind='bar', ax=ax, color=[anno_colors[col] for col in anno_cols], edgecolor='black', linewidth=0.5)

ax.set_xlabel('Cluster')
ax.set_ylabel('Proportion')
ax.set_title('Annotation frequency per cluster')
ax.legend(title='Annotation', bbox_to_anchor=(1.01, 1), loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.savefig(f'{CLUSTER_PATH}/manual_overlap.proportion.svg', bbox_inches='tight')


fig, ax = plt.subplots(figsize=(10, 5))
counts.plot(kind='bar', ax=ax, color=[anno_colors[col] for col in anno_cols], edgecolor='black', linewidth=0.5)

ax.set_xlabel('Cluster')
ax.set_ylabel('Count')
ax.set_title('Annotation frequency per cluster')
ax.legend(title='Annotation', bbox_to_anchor=(1.01, 1), loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.savefig(f'{CLUSTER_PATH}/manual_overlap.counts.svg', bbox_inches='tight')
