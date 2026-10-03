# conda activate coolbox
# this is a very hungry script for cpu/ram, I typically would limit cpu with taskset

import sys
sys.path.append('/mnt/md0/sjsolar/loop_pred/loop_analyses/0_helper_scripts')
from cpu_limit import limit_cpus

limit_cpus(n=12)

import pandas as pd

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
import epi_profiles_compare
importlib.reload(epi_profiles_compare)
from epi_profiles_compare import run_comparison

OUT_DIR='loop_deeptools/'

loop_df = pd.read_csv('../4_loop_strength_prediction/loops/loop_df_3kb.tsv', sep='\t')

homotypic_cluster_loop_df = {
	i: loop_df[(loop_df['cluster_left'] == i) & (loop_df['cluster_right'] == i)] for i in range(13)
}

cluster_category_loops = {
	'CTCF-CTCF': loop_df[(loop_df['loop_name'] == 'CTCF-CTCF')],
	'CTCF-CRE': loop_df[(loop_df['loop_name'] == 'CTCF-CRE')],
	'CTCF-PRC': loop_df[(loop_df['loop_name'] == 'CTCF-PRC')],
	'CRE-CRE': loop_df[(loop_df['loop_name'] == 'CRE-CRE')],
	'CRE-PRC': loop_df[(loop_df['loop_name'] == 'CRE-PRC')],
	'PRC-PRC': loop_df[(loop_df['loop_name'] == 'PRC-PRC')],
	'Weak-related': loop_df[(loop_df['loop_name'] == 'Weak-related')],
	'Other-related': loop_df[(loop_df['loop_name'] == 'Other-related')],
}

ann_loops = {
	'CTCF-CTCF': loop_df[(loop_df['is_only_CTCF_right']) & (loop_df['is_only_CTCF_left'])],
	'CTCF-CRE': loop_df[((loop_df['is_only_CTCF_right']) & (loop_df['is_only_CRE_left'])) | ((loop_df['is_only_CTCF_left']) & (loop_df['is_only_CRE_right']))],
	'CTCF-PRC': loop_df[((loop_df['is_only_CTCF_right']) & (loop_df['is_only_PRC_narrow_left'])) | ((loop_df['is_only_CTCF_left']) & (loop_df['is_only_PRC_narrow_right']))],
	'CRE-CRE': loop_df[(loop_df['is_only_CRE_right']) & (loop_df['is_only_CRE_left'])],
	'CRE-PRC': loop_df[((loop_df['is_only_CRE_right']) & (loop_df['is_only_PRC_narrow_left'])) | ((loop_df['is_only_CRE_left']) & (loop_df['is_only_PRC_narrow_right']))],
	'PRC-PRC': loop_df[(loop_df['is_only_PRC_narrow_right']) & (loop_df['is_only_PRC_narrow_left'])],
	'None': loop_df[(loop_df['is_none_right']) | (loop_df['is_none_left'])],
}

loop_color_df = pd.read_csv('../4_loop_strength_prediction/loops/loop_colors.tsv', sep='\t')
loop_colors = {row['loop_name']: row['color'] for _, row in loop_color_df.iterrows()}
names_cats_colors_df = pd.read_csv('../3_anchor_clustering/clustering/anchor_names_cats_colors.tsv', sep='\t')
cluster_name_colors = {row['cluster_name_long']: row['cluster_color'] for _, row in names_cats_colors_df.iterrows()}
cluster_colors = {row['cluster']: row['cluster_color'] for _, row in names_cats_colors_df.iterrows()}

run_comparison(
	loop_classes={
    	'All': loop_df,
        'CTCF-CTCF Clusters': cluster_category_loops['CTCF-CTCF'],
        'CRE-CRE Clusters': cluster_category_loops['CRE-CRE'],
        'PRC-PRC Clusters': cluster_category_loops['PRC-PRC'],
		'Weak-related Clusters': cluster_category_loops['Weak-related'],
        'Other-related Clusters': cluster_category_loops['Other-related'],
    },
    alphas = {
        'All': 0.5,
        'CTCF-CTCF Clusters': 0.75,
        'CRE-CRE Clusters': 0.75,
        'PRC-PRC Clusters': 0.75,
		'Weak-related Clusters': 0.75,
		'Other-related Clusters': 0.75,
    },
    class_colors = {
        'All': '#000000',
		'CTCF-CTCF Clusters': loop_colors['CTCF-CTCF'],
		'CRE-CRE Clusters': loop_colors['CRE-CRE'],
		'PRC-PRC Clusters': loop_colors['PRC-PRC'],
		'Weak-related Clusters': loop_colors['Weak-related'],
		'Other-related Clusters': loop_colors['Other-related'],
	},
    out_dir=f'{OUT_DIR}',
    tag='all_homo_categories',
    show_cell_lines=False,
	replot=True
)

run_comparison(
	loop_classes={
    	'All': loop_df,
        'CTCF-CTCF Clusters': cluster_category_loops['CTCF-CTCF'],
        'CTCF-CRE Clusters': cluster_category_loops['CTCF-CRE'],
        'CTCF-PRC Clusters': cluster_category_loops['CTCF-PRC'],
        'CRE-CRE Clusters': cluster_category_loops['CRE-CRE'],
        'CRE-PRC Clusters': cluster_category_loops['CRE-PRC'],
        'PRC-PRC Clusters': cluster_category_loops['PRC-PRC'],
		'Weak-related Clusters': cluster_category_loops['Weak-related'],
        'Other-related Clusters': cluster_category_loops['Other-related'],
    },
    alphas = {
        'All': 0.5,
        'CTCF-CTCF Clusters': 0.75,
        'CTCF-CRE Clusters': 0.75,
        'CTCF-PRC Clusters': 0.75,
        'CRE-CRE Clusters': 0.75,
        'CRE-PRC Clusters': 0.75,
        'PRC-PRC Clusters': 0.75,
		'Weak-related Clusters': 0.75,
		'Other-related Clusters': 0.75,
    },
    class_colors = {
        'All': '#000000',
		'CTCF-CTCF Clusters': loop_colors['CTCF-CTCF'],
		'CTCF-CRE Clusters': loop_colors['CTCF-CRE'],
		'CTCF-PRC Clusters': loop_colors['CTCF-PRC'],
		'CRE-CRE Clusters': loop_colors['CRE-CRE'],
		'CRE-PRC Clusters': loop_colors['CRE-PRC'],
		'PRC-PRC Clusters': loop_colors['PRC-PRC'],
		'Weak-related Clusters': loop_colors['Weak-related'],
		'Other-related Clusters': loop_colors['Other-related'],
	},
    out_dir=f'{OUT_DIR}',
    tag='all_categories',
    show_cell_lines=False,
	replot=True
)

run_comparison(
	loop_classes={
    	'All': loop_df,
        'CTCF-CTCF Clusters': cluster_category_loops['CTCF-CTCF'],
		'CTCF-CTCF Annotation': ann_loops['CTCF-CTCF'],
        'CTCF-CRE Clusters': cluster_category_loops['CTCF-CRE'],
		'CTCF-CRE Annotation': ann_loops['CTCF-CRE'],
        'CTCF-PRC Clusters': cluster_category_loops['CTCF-PRC'],
		'CTCF-PRC Annotation': ann_loops['CTCF-PRC'],
        'CRE-CRE Clusters': cluster_category_loops['CRE-CRE'],
		'CRE-CRE Annotation': ann_loops['CRE-CRE'],
        'CRE-PRC Clusters': cluster_category_loops['CRE-PRC'],
		'CRE-PRC Annotation': ann_loops['CRE-PRC'],
        'PRC-PRC Clusters': cluster_category_loops['PRC-PRC'],
		'PRC-PRC Annotation': ann_loops['PRC-PRC'],
		'Weak-related Clusters': cluster_category_loops['Weak-related'],
        'Other-related Clusters': cluster_category_loops['Other-related'],
		'None Annotation': ann_loops['None'],
    },
    alphas = {
        'All': 0.5,
        'CTCF-CTCF Clusters': 0.75,
		'CTCF-CTCF Annotation': 0.75,
        'CTCF-CRE Clusters': 0.75,
		'CTCF-CRE Annotation': 0.75,
        'CTCF-PRC Clusters': 0.75,
		'CTCF-PRC Annotation': 0.75,
        'CRE-CRE Clusters': 0.75,
		'CRE-CRE Annotation': 0.75,
        'CRE-PRC Clusters': 0.75,
		'CRE-PRC Annotation': 0.75,
        'PRC-PRC Clusters': 0.75,
		'PRC-PRC Annotation': 0.75,
		'Weak-related Clusters': 0.75,
		'Other-related Clusters': 0.75,
		'None Annotation': 0.75,
    },
    class_colors = {
        'All': '#000000',
		'CTCF-CTCF Clusters': loop_colors['CTCF-CTCF'],
		'CTCF-CTCF Annotation': loop_colors['CTCF-CTCF'],
		'CTCF-CRE Clusters': loop_colors['CTCF-CRE'],
		'CTCF-CRE Annotation': loop_colors['CTCF-CRE'],
		'CTCF-PRC Clusters': loop_colors['CTCF-PRC'],
		'CTCF-PRC Annotation': loop_colors['CTCF-PRC'],
		'CRE-CRE Clusters': loop_colors['CRE-CRE'],
		'CRE-CRE Annotation': loop_colors['CRE-CRE'],
		'CRE-PRC Clusters': loop_colors['CRE-PRC'],
		'CRE-PRC Annotation': loop_colors['CRE-PRC'],
		'PRC-PRC Clusters': loop_colors['PRC-PRC'],
		'PRC-PRC Annotation': loop_colors['PRC-PRC'],
		'Weak-related Clusters': loop_colors['Weak-related'],
		'Other-related Clusters': loop_colors['Other-related'],
		'None Annotation': loop_colors['Other-related'],
	},
	linestyles = {
		'CTCF-CTCF Annotation': '--',
		'CTCF-CRE Annotation': '--',
		'CTCF-PRC Annotation': '--',
		'CRE-CRE Annotation': '--',
		'CRE-PRC Annotation': '--',
		'PRC-PRC Annotation': '--',
		'None Annotation': '--',
	},
    out_dir=f'{OUT_DIR}',
    tag='all_categories_vs_ann',
    show_cell_lines=False,
	replot=True
)


run_comparison(
	loop_classes={
    	'All': loop_df,
        'CTCF-CTCF Annotation': ann_loops['CTCF-CTCF'],
        'CTCF-CTCF Clusters': cluster_category_loops['CTCF-CTCF'],
        'C1-C1': homotypic_cluster_loop_df[1],
        'C3-C3': homotypic_cluster_loop_df[3],
		'C7-C7': homotypic_cluster_loop_df[7],
        'C8-C8': homotypic_cluster_loop_df[8],
        'C9-C9': homotypic_cluster_loop_df[9],
    },
    alphas = {
        'All': 0.25,
        'CTCF-CTCF Annotation': 0.5,
        'CTCF-CTCF Clusters': 0.5,
        'C1-C1': 0.75,
        'C3-C3': 0.75,
		'C7-C7': 0.75,
        'C8-C8': 0.75,
        'C9-C9': 0.75,
    },
    class_colors = {
        'All': '#000000',
		'CTCF-CTCF Annotation': loop_colors['CTCF-CTCF'],
		'CTCF-CTCF Clusters': loop_colors['CTCF-CTCF'],
		'C1-C1': cluster_colors[1],
		'C3-C3': cluster_colors[3],
		'C7-C7': cluster_colors[7],
		'C8-C8': cluster_colors[8],
		'C9-C9': cluster_colors[9]
	},
    linestyles = {
        'CTCF-CTCF Annotation': '--',
    },
    out_dir=f'{OUT_DIR}',
    tag='ctcf_comparison',
    show_cell_lines=False,
	replot=True
)

run_comparison(
	loop_classes={
		'All': loop_df,
		'CRE-CRE Annotation': ann_loops['CRE-CRE'],
		'CRE-CRE Clusters': cluster_category_loops['CRE-CRE'],
		'C2-C2': homotypic_cluster_loop_df[2],
		'C4-C4': homotypic_cluster_loop_df[4],
		'C6-C6': homotypic_cluster_loop_df[6],
	},
	alphas = {
		'All': 0.25,
		'CRE-CRE Annotation': 0.5,
		'CRE-CRE Clusters': 0.5,
		'C2-C2': 0.75,
		'C4-C4': 0.75,
		'C6-C6': 0.75,
	},
    linestyles = {
        'CRE-CRE Annotation': '--',
	},
	class_colors = {
		'All': '#000000',
		'CRE-CRE Annotation': loop_colors['CRE-CRE'],
		'CRE-CRE Clusters': loop_colors['CRE-CRE'],
		'C2-C2': cluster_colors[2],
		'C4-C4': cluster_colors[4],
		'C6-C6': cluster_colors[6],
	},
	out_dir=f'{OUT_DIR}',
	tag='cre_comparison',
	show_cell_lines=False,
	replot=True
)

run_comparison(
	loop_classes={
		'All': loop_df,
		'PRC-PRC Annotation': ann_loops['PRC-PRC'],
		'PRC-PRC Clusters': cluster_category_loops['PRC-PRC'],
		'C5-C5': homotypic_cluster_loop_df[5],
		'C10-C10': homotypic_cluster_loop_df[10],
	},
	alphas = {
		'All': 0.25,
		'PRC-PRC Annotation': 0.5,
		'PRC-PRC Clusters': 0.5,
		'C5-C5': 0.75,
		'C10-C10': 0.75,
	},
    linestyles = {
        'PRC-PRC Annotation': '--',
	},
	class_colors = {
		'All': '#000000',
		'PRC-PRC Annotation': loop_colors['PRC-PRC'],
		'PRC-PRC Clusters': loop_colors['PRC-PRC'],
		'C5-C5': cluster_colors[5],
		'C10-C10': cluster_colors[10],
	},
	out_dir=f'{OUT_DIR}',
	tag='prc_comparison',
	show_cell_lines=False,
	replot=True,
)

# Major per cell line
run_comparison(
	loop_classes={
		'CTCF-Clusters': cluster_category_loops['CTCF-CTCF'],
	},
	alphas = {
		'CTCF-Clusters': 0.75,
	},
	class_colors = {
		'CTCF-Clusters': loop_colors['CTCF-CTCF'],
	},
	out_dir=f'{OUT_DIR}',
	tag='ctcf_cell_lines',
	show_cell_lines=True,
	replot=True
)

run_comparison(
	loop_classes={
		'CRE-Clusters': cluster_category_loops['CRE-CRE'],
	},
	alphas = {
		'CRE-Clusters': 0.75,
	},
	class_colors = {
		'CRE-Clusters': loop_colors['CRE-CRE'],
	},
	out_dir=f'{OUT_DIR}',
	tag='cre_cell_lines',
	show_cell_lines=True,
	replot=True
)

run_comparison(
	loop_classes={
		'PRC-Clusters': cluster_category_loops['PRC-PRC'],
	},
	alphas = {
		'PRC-Clusters': 0.75,
	},
	class_colors = {
		'PRC-Clusters': loop_colors['PRC-PRC'],
	},
	out_dir=f'{OUT_DIR}',
	tag='prc_cell_lines',
	show_cell_lines=True,
	replot=True
)

run_comparison(
	loop_classes={
		'Weak-Cluster': cluster_category_loops['Weak-related'],
	},
	alphas = {
		'Weak-Cluster': 0.75,
	},
	class_colors = {
		'Weak-Cluster': loop_colors['Weak-related'],
	},
	out_dir=f'{OUT_DIR}',
	tag='weak_cell_lines',
	show_cell_lines=True,
	replot=True
)


for i in range(13):
	run_comparison(
		loop_classes={
			'All': loop_df,
			'CTCF-CTCF Clusters': cluster_category_loops['CTCF-CTCF'],
			'CRE-CRE Clusters': cluster_category_loops['CRE-CRE'],
			'PRC-PRC Clusters': cluster_category_loops['PRC-PRC'],
			'Weak-related Clusters': cluster_category_loops['Weak-related'],
			'Other-related Clusters': cluster_category_loops['Other-related'],
			f'C{i}-C{i}': homotypic_cluster_loop_df[i],
		},
		alphas = {
			'All': 0.5,
			'CTCF-CTCF Clusters': 0.75,
			'CRE-CRE Clusters': 0.75,
			'PRC-PRC Clusters': 0.75,
			'Weak-related Clusters': 0.75,
			'Other-related Clusters': 0.75,
			f'C{i}-C{i}': 0.9,
		},
		class_colors = {
			'All': '#000000',
			'CTCF-CTCF Clusters': loop_colors['CTCF-CTCF'],
			'CRE-CRE Clusters': loop_colors['CRE-CRE'],
			'PRC-PRC Clusters': loop_colors['PRC-PRC'],
			'Weak-related Clusters': loop_colors['Weak-related'],
			'Other-related Clusters': loop_colors['Other-related'],
			f'C{i}-C{i}': cluster_colors[i],
		},
		linestyles = {
			'All': '--',
			'CTCF-CTCF Clusters': '--',
			'CRE-CRE Clusters': '--',
			'PRC-PRC Clusters': '--',
			'Weak-related Clusters': '--',
			'Other-related Clusters': '--',
		},
		out_dir=f'{OUT_DIR}',
		tag=f'cluster_{i}',
		show_cell_lines=False,
		replot=True,
	)

	run_comparison(
		loop_classes={
			f'C{i}-C{i}': homotypic_cluster_loop_df[i],
		},
		alphas = {
			f'C{i}-C{i}': 0.75,
		},
		class_colors = {
			f'C{i}-C{i}': cluster_colors[i],
		},
		out_dir=f'{OUT_DIR}',
		tag=f'homotypic_C{i}_cell_lines',
		show_cell_lines=True,
		replot=True
	)
