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
    df['is_CRE_not_PRC'] =  df['is_CRE'] & ~df['is_PRC_narrow']
    df['is_PRC_narrow_not_E'] = df['is_PRC_narrow'] & ~df['is_E']
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

anno_cols = ['is_CTCF', 'is_CRE', 'is_PRC_narrow', 'is_none']
anno_colors = {
    'is_CRE':  '#33BCEE',
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


def cluster_classification_metrics(df, name_col='cluster_name_short'):
    """
    Use the already-assigned cluster_name_short (already clean -- 'CTCF',
    'CRE', 'PRC', etc, no suffix to strip) directly as my clustering's
    predicted class per anchor, and score it against the per-anchor
    peak-based truth for each major class. Anything that isn't CTCF/CRE/PRC
    (e.g. 'Mixed') simply never matches any of the three classes below, so
    it correctly counts as a negative prediction for all of them -- no
    explicit 'none' mapping needed.
 
    Unlike a Venn diagram against an external set, the full universe is
    known here, so accuracy (which needs true negatives) is actually
    computable, unlike for the Fursova-style external overlap.
    """
    truth_cols = {'CTCF': 'is_CTCF', 'CRE': 'is_CRE', 'PRC': 'is_PRC_narrow'}
 
    rows = []
    for cls, truth_col in truth_cols.items():
        pred  = (df[name_col] == cls)
        truth = df[truth_col].astype(bool)
 
        TP = int((pred & truth).sum())
        FP = int((pred & ~truth).sum())
        FN = int((~pred & truth).sum())
        TN = int((~pred & ~truth).sum())
 
        jaccard   = TP / (TP + FP + FN) if (TP + FP + FN) else float('nan')
        precision = TP / (TP + FP)      if (TP + FP) else float('nan')
        recall    = TP / (TP + FN)      if (TP + FN) else float('nan')
        accuracy  = (TP + TN) / (TP + FP + FN + TN)
 
        rows.append({
            'class': cls, 'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
            'jaccard': jaccard, 'precision': precision,
            'recall': recall, 'accuracy': accuracy,
        })
 
    return pd.DataFrame(rows).set_index('class')
 
 
metrics = cluster_classification_metrics(merge_3kb)
print("\ncluster -> cluster_name_short (one row per unique cluster, for a sanity check):")
print(
    merge_3kb[['cluster', 'cluster_name_short']]
    .drop_duplicates()
    .sort_values('cluster')
    .to_string(index=False)
)
print("\nClassification metrics (my cluster-based class vs peak-based truth):")
print(metrics.round(4).to_string())
metrics.round(4).to_csv(f'{CLUSTER_PATH}/manual_overlap.classification_metrics.tsv', sep='\t')
 