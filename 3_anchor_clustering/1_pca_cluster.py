# conda activate umap_env
import pandas as pd
import numpy as np
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad
import warnings
warnings.filterwarnings('ignore')

# i drop this all over the place for later handling of text and labels in adobe illustrator
plt.rcParams.update({"text.usetex": False})
matplotlib.rcParams['path.simplify'] = False
matplotlib.rcParams['agg.path.chunksize'] = 0
matplotlib.rcParams.update({
    'font.size':     6,
    'axes.linewidth': 0.25,
    'axes.labelsize': 6,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'svg.fonttype':  'none',       # keeps text as real text not shapes that look like text
})

import sys
sys.path.append('/mnt/md0/sjsolar/loop_pred/loop_analyses/0_helper_scripts')
from heatmap import make_cluster_heatmap, make_weight_heatmap

IN_DIR   = '../2_epigenomics/anchor_epigenomics'
OUT_PCA_DIR   = 'pca'
OUT_CLUSTER_DIR   = 'clustering'
os.makedirs(OUT_PCA_DIR, exist_ok=True)
os.makedirs(OUT_CLUSTER_DIR, exist_ok=True)

N_PCA_COMPS    = 15 # max PCA components to compute
N_PCA_USE      = int(sys.argv[1]) if len(sys.argv) > 1 else 4 # I end up using 4 based on viewing the elbow plot
LEIDEN_RES     = 0.5       # Leiden resolution (lower res = fewer clusters and vice versa. We wanted to err on side of too many clusters, and recombine as needed to pull out any interesting examples)
N_NEIGHBORS    = 15        # middle of the road param
RANDOM_STATE   = 42

FEATURE_ORDER = [
    'ATAC',
    'K4me3', 'K27ac', 'K4me1', 'K36me3', 
    'CTCF', 'Rad21', 'Stag1', 'Stag2',
    'Ring1b', 'H2Aub', 'K27me3',
    'K9me2', 'K9me3', 'K36me2',
    'Laminb1',
]
BORDERS = [1, 5, 9, 12, 15, 16]

def run_pca(df: pd.DataFrame, q: float, window: str) -> pd.DataFrame:
    X = df[FEATURE_ORDER].values
    obs = pd.DataFrame({'anchor_id': df['anchor_id'].values})
    obs.index = obs.index.astype(str)
    var   = pd.DataFrame(index=FEATURE_ORDER)    

    adata = ad.AnnData(X=X, obs=obs, var=var)
    print(f"\n  AnnData: {adata.n_obs:,} observations x {adata.n_vars} features")

    # z score each feature, clip at +-10 standard devs
    sc.pp.scale(adata, max_value=10)

    clip_cols = [f'{f}_clip' for f in FEATURE_ORDER]
    df[clip_cols] = pd.DataFrame(
        adata.X,
        columns=clip_cols,
        index=df.index,
    )

    n_comps = min(N_PCA_COMPS, adata.n_vars - 1)
    print(f"  PCA with n_comps={n_comps} ...")
    sc.tl.pca(adata, n_comps=n_comps, svd_solver='arpack')
    pcs = adata.obsm['X_pca']
    pc_cols = [f'PC{i+1}' for i in range(pcs.shape[1])]
    df[pc_cols] = pcs
    df.to_csv(f'{OUT_PCA_DIR}/anchors_q{q}_{window}_pca.tsv', sep='\t', index=False)

    var_ratio = adata.uns['pca']['variance_ratio']
    for i, v in enumerate(var_ratio, start=1):
        print(f"  PC{i}: {v:.3%}")

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(range(1, len(var_ratio) + 1), var_ratio, 'o-', markersize=4)
    ax.axvline(N_PCA_USE + 0.5, color='red', linestyle='--',
                label=f'Using {N_PCA_USE} PCs')
    ax.set_xlabel('PC')
    ax.set_ylabel('Variance explained')
    ax.set_title('PCA elbow (epigenomics mode)')
    ax.legend(frameon=False)
    ax.spines[['top','right']].set_visible(False)
    plt.savefig(os.path.join(OUT_PCA_DIR, f'pca_elbow.q{q}_{window}.svg'), bbox_inches='tight', format="svg")
    plt.close()
    print(f"PCA elbow saved. Using {N_PCA_USE} PCs for neighbor graph.")

    # save the PC weights
    weights = pd.DataFrame(
        adata.varm['PCs'].T,      # (n_vars, n_comps) -> (n_comps, n_vars) to match sklearn-style layout
        columns=clip_cols,
        index=pc_cols
    )
    weights.to_csv(f'{OUT_PCA_DIR}/pca_loadings.q{q}_{window}.csv')
    # and output it as heatmap
    make_weight_heatmap(
        weights=weights,
        feature_cols=clip_cols,
        out_path=f'{OUT_PCA_DIR}/pca_loadings.q{q}_{window}.svg',
        tag=f'pca_weights.q{q}_{window}',
        borders=BORDERS,
        value_label='Rotation weight',
        vmin=-0.6, vmax=0.6,
        title=f'PCA weights (n={n_comps}, FDR<={q}, {window} window)',
    )


    # then scale by variance explained for easier to interpret heatmap viz
    var_ratio = adata.uns['pca']['variance_ratio'][:n_comps]
    weights_scaled = weights.mul(np.sqrt(var_ratio), axis=0)
    weights_scaled.to_csv(f'{OUT_PCA_DIR}/pca_loadings.q{q}_{window}.scaled.csv')
    make_weight_heatmap(
        weights=weights_scaled,
        feature_cols=clip_cols,
        out_path=f'{OUT_PCA_DIR}/pca_loadings.q{q}_{window}.scaled.svg',
        tag=f'pca_weights.q{q}_{window}.scaled',
        borders=BORDERS,
        value_label='Rotation weight',
        # vmin=-0.6, vmax=0.6, let it autoscale
        title=f'PCA weights (n={n_comps}, FDR<={q}, {window} window)',
    )

    # run KNN using 4 PCs
    sc.pp.neighbors(adata, n_pcs=N_PCA_USE, n_neighbors=N_NEIGHBORS, random_state=RANDOM_STATE)
    print(f"  Leiden clustering (resolution={LEIDEN_RES}) ...")

    # partition the graph into clusters
    sc.tl.leiden(adata, resolution=LEIDEN_RES, key_added='leiden', random_state=RANDOM_STATE)
    n_clusters = adata.obs['leiden'].nunique()
    print(f"{n_clusters} clusters found")
    df['cluster'] = adata.obs['leiden'].values

    # get mean vals for heatmap
    cluster_means = df.groupby('cluster')[clip_cols].mean()
    cluster_means = cluster_means.loc[sorted(cluster_means.index, key=int)]
    cluster_means.to_csv(f'{OUT_CLUSTER_DIR}/cluster_feature_means.{q}_{window}.res{LEIDEN_RES}.csv')

    make_cluster_heatmap(
        df=df,
        feature_cols=clip_cols,
        out_path=f'{OUT_CLUSTER_DIR}/cluster_heatmap.q{q}_{window}.res{LEIDEN_RES}.svg',
        tag=f'q{q}_{window}.res{LEIDEN_RES}',
        cluster_col='cluster',
        borders=BORDERS,
        vmin=-3, vmax=3,
        value_label='Mean z-score',
    )

    print("UMAP for viz")
    sc.tl.umap(adata, random_state=RANDOM_STATE)
    df = df.copy()
    df['cluster'] = adata.obs['leiden'].values
    df['UMAP1']   = adata.obsm['X_umap'][:, 0]
    df['UMAP2']   = adata.obsm['X_umap'][:, 1]

    return df, adata


# Run PCA, clustering, and UMAP for each combination of window and q-value (eventually stick with 3kb, fdr<0.01)
windows = ['3kb'] #, '10kb']
q_vals = ['0.01'] #, '0.02', '0.05']
for window in windows:
    for q in q_vals:
        anchors_df = pd.read_csv(f'{IN_DIR}/anchors_q{q}_{window}.tsv', sep='\t')
        print(f"Running joint PCA for anchors_q{q}_{window}...")
        df, adata = run_pca(anchors_df, q, window)
        df.to_csv(f'{OUT_CLUSTER_DIR}/anchors_q{q}_{window}_pca_clustered_umap.tsv', sep='\t', index=False)
        print(f"Done joint PCA, KNN, UMAP for anchors_q{q}_{window}.")

print('done!')
