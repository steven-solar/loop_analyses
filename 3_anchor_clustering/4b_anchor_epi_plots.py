# conda activate umap_env

import sys
sys.path.append('/mnt/md0/sjsolar/loop_pred/loop_analyses/0_helper_scripts')

# from cpu_limit import limit_cpus
# limit_cpus(16)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive, saves to file fine
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from IPython.display import SVG
from scipy.stats import mannwhitneyu
import itertools
import sys
sys.path.append('/mnt/md0/sjsolar/loop_pred/loop_analyses/0_helper_scripts')  # wherever you save it

import importlib
import epi_profiles_compare, epi_profiles_anchors
importlib.reload(epi_profiles_compare)
importlib.reload(epi_profiles_anchors)
from epi_profiles_compare import run_comparison
from epi_profiles_anchors import run_anchor_comparison
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)        
pd.set_option('display.width', None)           
pd.set_option('display.max_colwidth', None)    # full content in each cell

CELL_LINES = ['ESC', 'EpiLC', 'd4c7PGCLC', 'GSC']
MERGE_COLS = ['anchor_id', 'chrom', 'mid', 'cell_line', 'window']
OUT_DIR='deeptools_compare/all_anchors'
clusters = pd.read_csv('clustering/anchors_q0.01_3kb_pca_clustered_umap.tsv', sep='\t')
annotation = pd.read_csv('../2_epigenomics/anchor_epigenomics/anchors_q0.01_3kb_annotated.p10.tsv', sep='\t')
annotation = annotation[MERGE_COLS + [c for c in annotation.columns if c.startswith('is_')]]
merge = pd.merge(clusters, annotation, on=MERGE_COLS, how='inner')
merge['is_CRE'] = merge['is_P'] | merge['is_E']

def add_excl_defs(df):
    df['is_only_CRE'] = df['is_CRE'] & ~df['is_CTCF'] & ~df['is_PRC_narrow']
    df['is_only_CTCF'] = df['is_CTCF'] & ~df['is_CRE'] & ~df['is_PRC_narrow']
    df['is_only_PRC_narrow'] = df['is_PRC_narrow'] & ~df['is_CRE'] & ~df['is_CTCF']
    df['is_none'] = ~(df['is_CTCF'] | df['is_CRE'] | df['is_PRC_narrow'])
    return df

anchors = add_excl_defs(merge)
anchor_cluster_dfs = {i: anchors[anchors['cluster'] == i] for i in range(13)}
anchors_c0 = anchor_cluster_dfs[0]
anchors_c1  = anchor_cluster_dfs[1]
anchors_c2  = anchor_cluster_dfs[2]
anchors_c3 = anchor_cluster_dfs[3]
anchors_c4  = anchor_cluster_dfs[4]
anchors_c5  = anchor_cluster_dfs[5]
anchors_c6  = anchor_cluster_dfs[6]
anchors_c7 = anchor_cluster_dfs[7]
anchors_c8 = anchor_cluster_dfs[8]
anchors_c9 = anchor_cluster_dfs[9]
anchors_c10 = anchor_cluster_dfs[10]
anchors_c11 = anchor_cluster_dfs[11]
anchors_c12 = anchor_cluster_dfs[12]

# ctcf_1_3_7 = anchors_c1.append(anchors_c3).append(anchors_c7)
ctcf_1_3_7_8_9 = anchors_c1.append(anchors_c3).append(anchors_c7).append(anchors_c8).append(anchors_c9)
cre_2_4_6 = anchors_c2.append(anchors_c4).append(anchors_c6)
prc_5_10 = anchors_c5.append(anchors_c10)

for i in range(13):
    print(f'Cluster {i}: {len(anchor_cluster_dfs[i])} anchors')
    run_anchor_comparison(
        anchor_classes={
            'All': merge,
            'CTCF': ctcf_1_3_7_8_9,
            'CRE': cre_2_4_6,
            'PRC': prc_5_10,
            f'C{i}':  anchor_cluster_dfs[i],
        },
        alphas = {
            'All': 0.5,
            'CTCF': 0.5,
            'CRE': 0.5,
            'PRC': 0.5,
            f'C{i}': 0.9,
        },
        out_dir='deeptools_compare/all_anchors',
        tag=f'{i}_vs_all',
        show_cell_lines=False,
        replot=True,
    )


run_anchor_comparison(
    anchor_classes={
        'All': merge,
        'CRE': cre_2_4_6,
        'C2': anchors_c2,
        'C4': anchors_c4,
        'C6': anchors_c6,
    },
    alphas = {
        'C2': 0.9,
        'C4': 0.9,
        'C6': 0.9,
        'All': 0.5,
        'CRE': 0.5,
    },
    out_dir='deeptools_compare/all_anchors',
    tag='cre_clusters',
    show_cell_lines=False,
    replot=True,
)


run_anchor_comparison(
    anchor_classes={
        'All': merge,
        'CTCF': ctcf_1_3_7_8_9,
        'C1': anchors_c1,
        'C3': anchors_c3,
        'C7': anchors_c7,
        'C8': anchors_c8,
        'C9': anchors_c9,
    },
    alphas = {
        'C1': 0.9,
        'C3': 0.9,
        'C7': 0.9,
        'C8': 0.9,
        'C9': 0.9,
        'All': 0.5,
        'CTCF': 0.5,
    },
    out_dir='deeptools_compare/all_anchors',
    tag='ctcf_clusters',
    show_cell_lines=False,
    replot=True,
)

run_anchor_comparison(
    anchor_classes={
        'All': merge,
        'PRC': prc_5_10,
        'C5': anchors_c5,
        'C10': anchors_c10,
    },
    alphas = {
        'C5': 0.9,
        'C10': 0.9,
        'All': 0.5,
        'PRC': 0.5,
    },
    out_dir='deeptools_compare/all_anchors',
    tag='prc_clusters',
    show_cell_lines=False,
    replot=True,
)