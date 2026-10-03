# conda activate coolbox
# this is a very hungry script for cpu/ram, I typically would limit cpu with taskset

import sys
sys.path.append('/mnt/md0/sjsolar/loop_pred/loop_analyses/0_helper_scripts')
from cpu_limit import limit_cpus

limit_cpus(24)

import pandas as pd
import numpy as np

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

import sys
sys.path.append('/mnt/md0/sjsolar/loop_pred/loop_analyses/0_helper_scripts')  # wherever you save it

import importlib
import epi_profiles_anchors
importlib.reload(epi_profiles_anchors)
from epi_profiles_anchors import run_anchor_comparison

CELL_LINES = ['ESC', 'EpiLC', 'd4c7PGCLC', 'GSC']
MERGE_COLS = ['anchor_id', 'chrom', 'mid', 'cell_line', 'window']
OUT_DIR='anchor_deeptools/'

# clusters = pd.read_csv('../3_anchor_clustering/clustering/anchors_3kb_labeled.tsv', sep='\t')
# clusters = pd.read_csv('../3_anchor_clustering/clustering_subcluster_c6/anchors_reclustered.tsv', sep='\t')
clusters = pd.read_csv('../3_anchor_clustering/clustering_recluster_per_cellline/anchors_reclustered_per_cl.tsv', sep='\t')

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

anchors_ctcf_cluster = anchors[anchors['category_name'] == 'CTCF']
anchors_cre_cluster = anchors[anchors['category_name'] == 'CRE']
anchors_prc_cluster = anchors[anchors['category_name'] == 'PRC']
anchors_weak_cluster = anchors[anchors['category_name'] == 'Weak']
anchors_bivalent_cluster = anchors[anchors['category_name'] == 'Bivalent CRE']
anchors_cohesin_cre_cluster = anchors[anchors['category_name'] == 'Cohesin CRE']
anchors_other_cluster = anchors[anchors['category_name'] == 'Other']

anchors_ctcf_ann = anchors[anchors['is_only_CTCF']]
anchors_cre_ann = anchors[anchors['is_only_CRE']]
anchors_prc_ann = anchors[anchors['is_only_PRC_narrow']]
anchors_none_ann = anchors[anchors['is_none']]

names_cats_colors_df = pd.read_csv('../3_anchor_clustering/clustering/anchor_names_cats_colors.tsv', sep='\t')

cluster_colors = {row['cluster']: row['cluster_color'] for _, row in names_cats_colors_df.iterrows()}
category_colors = {row['category_name']: row['category_color'] for _, row in names_cats_colors_df.iterrows()}

run_anchor_comparison(
	anchor_classes={
    	'All': anchors,
        'CTCF-Clusters': anchors_ctcf_cluster,
        'CRE-Clusters': anchors_cre_cluster,
        'PRC-Clusters': anchors_prc_cluster,
		'Weak-Cluster': anchors_weak_cluster,
    },
    alphas = {
        'All': 0.75,
        'CTCF-Clusters': 0.75,
        'CRE-Clusters': 0.75,
        'PRC-Clusters': 0.75,
        'Weak-Cluster': 0.75,
    },
    colors = {
        'All': '#000000',
		'CTCF-Clusters': category_colors['CTCF'],
		'CRE-Clusters': category_colors['CRE'],
		'PRC-Clusters': category_colors['PRC'],
		'Weak-Cluster': category_colors['Weak'],
	},
    out_dir=f'{OUT_DIR}',
    tag='major_categories',
    show_cell_lines=False,
	replot=False,
)

run_anchor_comparison(
	anchor_classes={
    	'All': anchors,
        'CTCF-Clusters': anchors_ctcf_cluster,
        'CRE-Clusters': anchors_cre_cluster,
        'PRC-Clusters': anchors_prc_cluster,
		'Weak-Cluster': anchors_weak_cluster,
        'Bivalent-Cluster': anchors_bivalent_cluster,
        'Cohesin CRE-Cluster': anchors_cohesin_cre_cluster,
		'Other-Clusters': anchors_other_cluster,
    },
    alphas = {
        'All': 0.75,
        'CTCF-Clusters': 0.75,
        'CRE-Clusters': 0.75,
        'PRC-Clusters': 0.75,
        'Weak-Cluster': 0.75,
		'Bivalent-Cluster': 0.75,
        'Cohesin CRE-Cluster': 0.75,
		'Other-Clusters': 0.75,
    },
    colors = {
        'All': '#000000',
		'CTCF-Clusters': category_colors['CTCF'],
		'CRE-Clusters': category_colors['CRE'],
		'PRC-Clusters': category_colors['PRC'],
        'Weak-Cluster': category_colors['Weak'],
        'Bivalent-Cluster': category_colors['Bivalent CRE'],
        'Cohesin CRE-Cluster': category_colors['Cohesin CRE'],
		'Other-Clusters': category_colors['Other'],
	},
    out_dir=f'{OUT_DIR}',
    tag='all_categories',
    show_cell_lines=False,
	replot=False,
)


run_anchor_comparison(
	anchor_classes={
    	'All': anchors,
        'CTCF-Clusters': anchors_ctcf_cluster,
		'CTCF-Annotation': anchors_ctcf_ann,
        'CRE-Clusters': anchors_cre_cluster,
        'CRE-Annotation': anchors_cre_ann,
        'PRC-Clusters': anchors_prc_cluster,
        'PRC-Annotation': anchors_prc_ann,
		'Weak-Cluster': anchors_weak_cluster,
        'None-Annotation': anchors_none_ann,
    },
    alphas = {
        'All': 0.75,
        'CTCF-Clusters': 0.75,
        'CTCF-Annotation': 0.75,
        'CRE-Clusters': 0.75,
		'CRE-Annotation': 0.75,
        'PRC-Clusters': 0.75,
        'PRC-Annotation': 0.75,
        'Weak-Cluster': 0.75,
        'None-Annotation': 0.75,
    },
    colors = {
        'All': '#000000',
		'CTCF-Clusters': category_colors['CTCF'],
        'CTCF-Annotation': category_colors['CTCF'],
		'CRE-Clusters': category_colors['CRE'],
        'CRE-Annotation': category_colors['CRE'],
		'PRC-Clusters': category_colors['PRC'],
        'PRC-Annotation': category_colors['PRC'],
		'Weak-Cluster': category_colors['Weak'],
        'None-Annotation': category_colors['Weak'],
	},
	linestyles = {
        'CTCF-Annotation': '--',
		'CRE-Annotation': '--',
		'PRC-Annotation': '--',
		'None-Annotation': '--',
	},
    out_dir=f'{OUT_DIR}',
    tag='major_categories_ann',
    show_cell_lines=False,
	replot=False,
)

run_anchor_comparison(
	anchor_classes={
    	'All': anchors,
        'CTCF-Annotation': anchors_ctcf_ann,
        'CTCF-Clusters': anchors_ctcf_cluster,
        'C1: CTCF (ESC, EpiLC)': anchor_cluster_dfs[1],
        'C3: CTCF (d4c7PGCLC)': anchor_cluster_dfs[3],
		'C7: CTCF (GSC)': anchor_cluster_dfs[7],
        'C8: CTCF (Active region)': anchor_cluster_dfs[8],
        'C9: CTCF (Repressed region)': anchor_cluster_dfs[9],
    },
    alphas = {
        'All': 0.25,
        'CTCF-Annotation': 0.5,
        'CTCF-Clusters': 0.5,
        'C1: CTCF (ESC, EpiLC)': 0.75,
        'C3: CTCF (d4c7PGCLC)': 0.75,
		'C7: CTCF (GSC)': 0.75,
        'C8: CTCF (Active region)': 0.75,
        'C9: CTCF (Repressed region)': 0.75,
    },
    colors = {
        'All': '#000000',
		'CTCF-Annotation': category_colors['CTCF'],
		'CTCF-Clusters': category_colors['CTCF'],
		'C1: CTCF (ESC, EpiLC)': cluster_colors[1],
		'C3: CTCF (d4c7PGCLC)': cluster_colors[3],
		'C7: CTCF (GSC)': cluster_colors[7],
		'C8: CTCF (Active region)': cluster_colors[8],
		'C9: CTCF (Repressed region)': cluster_colors[9]
	},
    linestyles = {
        'CTCF-Annotation': '--',
    },
    out_dir=f'{OUT_DIR}',
    tag='ctcf_comparison',
    show_cell_lines=False,
	replot=False,
)

run_anchor_comparison(
	anchor_classes={
		'All': anchors,
		'CRE-Annotation': anchors_cre_ann,
		'CRE-Clusters': anchors_cre_cluster,
		'C2: CRE (ESC, EpiLC)': anchor_cluster_dfs[2],
		'C4: CRE (d4c7PGCLC)': anchor_cluster_dfs[4],
		'C6: CRE (GSC)': anchor_cluster_dfs[6],
	},
	alphas = {
		'All': 0.25,
		'CRE-Annotation': 0.5,
		'CRE-Clusters': 0.5,
		'C2: CRE (ESC, EpiLC)': 0.75,
		'C4: CRE (d4c7PGCLC)': 0.75,
		'C6: CRE (GSC)': 0.75,
	},
    colors = {
        'All': '#000000',
		'CRE-Annotation': category_colors['CRE'],
		'CRE-Clusters': category_colors['CRE'],
		'C2: CRE (ESC, EpiLC)': cluster_colors[2],
		'C4: CRE (d4c7PGCLC)': cluster_colors[4],
		'C6: CRE (GSC)': cluster_colors[6],
	},
    linestyles = {
        'CRE-Annotation': '--',
	},
	out_dir=f'{OUT_DIR}',
	tag='cre_comparison',
	show_cell_lines=False,
	replot=False,
)

run_anchor_comparison(
	anchor_classes={
		'All': anchors,
		'PRC-Annotation': anchors_prc_ann,
		'PRC-Clusters': anchors_prc_cluster,
		'C5: PRC (ESC, EpiLC)': anchor_cluster_dfs[5],
		'C10: PRC (d4c7PGCLC)': anchor_cluster_dfs[10],
	},
	alphas = {
		'All': 0.25,
		'PRC-Annotation': 0.5,
		'PRC-Clusters': 0.5,
		'C5: PRC (ESC, EpiLC)': 0.75,
		'C10: PRC (d4c7PGCLC)': 0.75,
	},
    colors = {
		'All': '#000000',
		'PRC-Annotation': category_colors['PRC'],
		'PRC-Clusters': category_colors['PRC'],
		'C5: PRC (ESC, EpiLC)': cluster_colors[5],
		'C10: PRC (d4c7PGCLC)': cluster_colors[10],
	},
    linestyles = {
        'PRC-Annotation': '--',
	},
	out_dir=f'{OUT_DIR}',
	tag='prc_comparison',
	show_cell_lines=False,
	replot=False,
)

# Major per cell line
run_anchor_comparison(
	anchor_classes={
		'CTCF-Clusters': anchors_ctcf_cluster,
	},
	alphas = {
		'CTCF-Clusters': 0.75,
	},
    colors = {
		'CTCF-Clusters': category_colors['CTCF'],
	},
	out_dir=f'{OUT_DIR}',
	tag='ctcf_cell_lines',
	show_cell_lines=True,
	replot=False,
)

run_anchor_comparison(
	anchor_classes={
		'CRE-Clusters': anchors_cre_cluster,
	},
	alphas = {
		'CRE-Clusters': 0.75,
	},
    colors = {
		'CRE-Clusters': category_colors['CRE'],
	},
	out_dir=f'{OUT_DIR}',
	tag='cre_cell_lines',
	show_cell_lines=True,
	replot=False,
)

run_anchor_comparison(
	anchor_classes={
		'PRC-Clusters': anchors_prc_cluster,
	},
	alphas = {
		'PRC-Clusters': 0.75,
	},
    colors = {
		'PRC-Clusters': category_colors['PRC'],
	},
	out_dir=f'{OUT_DIR}',
	tag='prc_cell_lines',
	show_cell_lines=True,
	replot=False,
)

run_anchor_comparison(
	anchor_classes={
		'Weak-Cluster': anchors_weak_cluster,
	},
	alphas = {
		'Weak-Cluster': 0.75,
	},
    colors = {
		'Weak-Cluster': category_colors['Weak'],
	},
	out_dir=f'{OUT_DIR}',
	tag='weak_cell_lines',
	show_cell_lines=True,
	replot=False,
)

for i in range(13):
    print(f'Cluster {i}: {len(anchor_cluster_dfs[i])} anchors')
    run_anchor_comparison(
        anchor_classes={
            'All': anchors,
            'CTCF-Clusters': anchors_ctcf_cluster,
            'CRE-Clusters': anchors_cre_cluster,
            'PRC-Clusters': anchors_prc_cluster,
            'Weak': anchors_weak_cluster,
            f'C{i}':  anchor_cluster_dfs[i],
        },
        alphas = {
            'All': 0.5,
            'CTCF-Clusters': 0.5,
            'CRE-Clusters': 0.5,
            'PRC-Clusters': 0.5,
            f'C{i}': 0.9,
        },
        colors = {
            'All': '#000000',
			'CTCF-Clusters': category_colors['CTCF'],
			'CRE-Clusters': category_colors['CRE'],
			'PRC-Clusters': category_colors['PRC'],
			'Weak': category_colors['Weak'],
            f'C{i}': cluster_colors[i],
		},
        out_dir=f'{OUT_DIR}',
        tag=f'{i}_vs_all',
        show_cell_lines=False,
        replot=False,
    )

    run_anchor_comparison(
        anchor_classes={
            f'C{i}':  anchor_cluster_dfs[i],
        },
        alphas = {
            f'C{i}': 0.75,
        },
        colors = {
            f'C{i}': cluster_colors[i],
		},
        out_dir=f'{OUT_DIR}',
        tag=f'{i}_cell_lines',
        show_cell_lines=True,
        replot=False,
    )
