# conda activate umap_env

import pandas as pd
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
plt.rcParams.update({
    "text.usetex": False})
plt.rcParams['svg.fonttype'] = 'none'
plt.rc('pdf',fonttype = 42)
plt.rcParams["ps.useafm"] = True
matplotlib.rcParams['svg.fonttype'] = 'none' 
matplotlib.rcParams['path.simplify'] = False        # don't simplify vector paths
matplotlib.rcParams['agg.path.chunksize'] = 0       # no chunking
import gc

import sys
sys.path.append('/mnt/md0/sjsolar/loop_pred/loop_analyses/0_helper_scripts')  # wherever you save it
from umap_scatter import plot_umap_scatter

IN_DIR = 'clustering'
OUT_DIR = 'clustering'

windows = ['3kb', '10kb']
q_vals = ['0.01', '0.02', '0.05']
for window in windows:
    for q in q_vals:
        print(f"Plotting UMAP for anchors_q{q}_{window}...")
        df = pd.read_csv(f'{IN_DIR}/anchors_q{q}_{window}_pca_clustered_umap.tsv', sep='\t')
        plot_umap_scatter(
            df=df,
            out_path=f'{OUT_DIR}/anchors_q{q}_{window}_umap.svg',
            title=f'Anchors q={q}, window={window}',
            figsize=(6, 5),
        )
        print(f"Done plotting UMAP for anchors_q{q}_{window}.")
        plt.close('all')       # close all open figures
        del df                 # drop the dataframe
        gc.collect()           # force garbage collection
        