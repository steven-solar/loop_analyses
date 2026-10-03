# conda activate umap_env

import pandas as pd
import matplotlib
matplotlib.use('Agg')          # non-interactive, saves to file fine
import sys
sys.path.append('/mnt/md0/sjsolar/loop_pred/loop_analyses/0_helper_scripts')  # wherever you save it
from apa_clusters import run_apa_pipeline

CELL_LINES = ['ESC', 'EpiLC', 'd4c7PGCLC', 'GSC']

loop_df = pd.read_csv('../4_loop_strength_prediction/loops/loop_df_3kb.tsv', sep='\t')
loop_dfs = {
    f"{i}-{i}": loop_df[
        (loop_df["cluster_left"] == i) & (loop_df["cluster_right"] == i)
    ]
    for i in range(13)
}

print(loop_dfs.keys())

ctcf_clusters = [1,3,7]
all_ctcf_clusters = [1,3,7,8,9]
cre_clusters = [2,4,6]
prc_clusters = [5,10]

loop_dfs['CTCF'] = loop_df[(loop_df['cluster_left'].isin(ctcf_clusters)) & (loop_df['cluster_right'].isin(ctcf_clusters))]
loop_dfs['All_CTCF'] = loop_df[(loop_df['cluster_left'].isin(all_ctcf_clusters)) & (loop_df['cluster_right'].isin(all_ctcf_clusters))]
loop_dfs['CRE'] = loop_df[(loop_df['cluster_left'].isin(cre_clusters)) & (loop_df['cluster_right'].isin(cre_clusters))]
loop_dfs['PRC'] = loop_df[(loop_df['cluster_left'].isin(prc_clusters)) & (loop_df['cluster_right'].isin(prc_clusters))]

run_apa_pipeline(
    loop_classes=loop_dfs,
    out_dir='apa_all',
    tag='all',
    n_workers=4,
)
