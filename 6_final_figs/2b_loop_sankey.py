import pandas as pd
import importlib
import sys

sys.path.append('/mnt/md0/sjsolar/loop_pred/loop_analyses/0_helper_scripts')  # wherever you save it
import anchor_sankey
importlib.reload(anchor_sankey)
from anchor_sankey import plot_loop_sankey

CELL_LINES = ['ESC', 'EpiLC', 'd4c7PGCLC', 'GSC']

# loop_df_wide  = pd.read_csv('../4_loop_strength_prediction/loops/loop_df_3kb_wide.tsv', sep='\t')
# loop_df_wide  = pd.read_csv('../4_loop_strength_prediction/loops/loop_df_cre_prc_reclustered_wide.tsv', sep='\t')
loop_df_wide  = pd.read_csv('../4_loop_strength_prediction/loops/loop_df_reclustered_per_cl_wide.tsv', sep='\t')

loop_color_df = pd.read_csv('../4_loop_strength_prediction/loops/loop_colors.tsv', sep='\t')

loop_colors = {row['loop_name']: row['color']      for _, row in loop_color_df.iterrows()}
loop_names  = {row['loop_name']: row['loop_name']  for _, row in loop_color_df.iterrows()}

sankey_data = plot_loop_sankey(
    wide=loop_df_wide,           # long-format, same as run_transition_analysis
    out_dir='loop_sankey',
    trajectory=CELL_LINES,
    normalize=False,           # True = link width = fraction of from-cluster
    min_flow=50,               # hide tiny flows
	able_col_pat='AbLE_score_{cl}',
	loop_colors=loop_colors,
	loop_names=loop_names,
)
