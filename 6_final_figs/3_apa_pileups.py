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
from apa_clusters import run_apa_pipeline
from cpu_limit import limit_cpus
limit_cpus(n=12, start=24)
CELL_LINES = ['ESC', 'EpiLC', 'd4c7PGCLC', 'GSC']

# loop_df = pd.read_csv('../4_loop_strength_prediction/loops/loop_df_3kb.tsv', sep='\t')
loop_df = pd.read_csv('../4_loop_strength_prediction/loops/loop_df_cre_prc_reclustered.tsv', sep='\t')
# loop_df = pd.read_csv('../4_loop_strength_prediction/loops/loop_df_reclustered_per_cl.tsv', sep='\t')

loop_dfs = dict()

# loop_dfs = {
#     f"C{i}-C{i}": loop_df[
#         (loop_df["cluster_left"] == i) & (loop_df["cluster_right"] == i)
#     ]
#     for i in range(13)
# }

print (loop_df['loop_name'].unique())

for loop_name in loop_df['loop_name'].unique():
	if 'related' in loop_name:
		continue
	print(f"{loop_name}: {sum(loop_df['loop_name'] == loop_name)} loops")
	loop_dfs[loop_name] = loop_df[loop_df['loop_name'] == loop_name]

print(f"Loop classes: {list(loop_dfs.keys())}")
run_apa_pipeline(
    loop_classes=loop_dfs,
    out_dir='loop_apa_recluster_cre_prc',
    tag='all',
	replot=False,
    n_workers=12,
)
