# conda activate umap_env
import pandas as pd
import numpy as np
import seaborn as sns
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad
import warnings
warnings.filterwarnings('ignore')
plt.rcParams.update({
    "text.usetex": False})
plt.rcParams['svg.fonttype'] = 'none'
plt.rc('pdf',fonttype = 42)
plt.rcParams["ps.useafm"] = True
matplotlib.rcParams['path.simplify'] = False        # don't simplify vector paths
matplotlib.rcParams['agg.path.chunksize'] = 0       # no chunking


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
from heatmap import make_cluster_heatmap, make_weight_heatmap

IN_TSV         =  'clustering/anchors_3kb_labeled.tsv'

OUT_PCA_DIR   = 'pca_recluster_per_cellline'
OUT_CLUSTER_DIR   = 'clustering_recluster_per_cellline'
os.makedirs(OUT_PCA_DIR, exist_ok=True)
os.makedirs(OUT_CLUSTER_DIR, exist_ok=True)

N_PCA_COMPS    = 15 # max PCA components to compute (elbow decides how many to use)
N_PCA_USE      = int(sys.argv[1]) if len(sys.argv) > 1 else 4
LEIDEN_RES     = 0.5       # Leiden resolution (overridden by CLI arg)
N_NEIGHBORS    = 15        # scanpy neighbor graph
RANDOM_STATE   = 42

CELL_LINES = ['ESC', 'EpiLC', 'd4c7PGCLC', 'GSC']

FEATURE_ORDER = [
    'ATAC',
    'K27ac', 'K4me1', 'K4me3',
    'CTCF', 'Rad21', 'Stag1', 'Stag2',
    'Ring1b', 'H2Aub',
    'K9me2', 'K9me3', 'K27me3', 'K36me2', 'K36me3', 'Laminb1',
]
BORDERS = [1, 4, 8, 16]

def run_pca(df: pd.DataFrame, q: str, window: str, cell_line: str) -> pd.DataFrame:
    """Run PCA -> neighbors -> Leiden -> UMAP for a single cell line's anchors."""
    X = df[FEATURE_ORDER].values
    obs = pd.DataFrame({'anchor_id': df['anchor_id'].values})
    obs.index = obs.index.astype(str)
    var   = pd.DataFrame(index=FEATURE_ORDER)

    adata = ad.AnnData(X=X, obs=obs, var=var)
    print(f"\n  AnnData [{cell_line}]: {adata.n_obs:,} observations × {adata.n_vars} features")

    # ---- scale to zero mean, clip at 10 sd ----
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
    df.to_csv(f'{OUT_PCA_DIR}/anchors_q{q}_{window}_{cell_line}_pca.tsv', sep='\t', index=False)

    var_ratio = adata.uns['pca']['variance_ratio']
    for i, v in enumerate(var_ratio, start=1):
        print(f"  PC{i}: {v:.3%}")

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(range(1, len(var_ratio) + 1), var_ratio, 'o-', markersize=4)
    ax.axvline(N_PCA_USE + 0.5, color='red', linestyle='--',
                label=f'Using {N_PCA_USE} PCs')
    ax.set_xlabel('PC')
    ax.set_ylabel('Variance explained')
    ax.set_title(f'PCA elbow ({cell_line})')
    ax.legend(frameon=False)
    ax.spines[['top','right']].set_visible(False)
    plt.savefig(os.path.join(OUT_PCA_DIR, f'pca_elbow.q{q}_{window}_{cell_line}.svg'), bbox_inches='tight', format="svg")
    plt.close()
    print(f"  PCA elbow saved. Using top {N_PCA_USE} PCs for neighbor graph.")

    weights = pd.DataFrame(
        adata.varm['PCs'].T,      # (n_vars, n_comps) -> (n_comps, n_vars) to match sklearn-style layout
        columns=clip_cols,
        index=pc_cols
    )
    weights.to_csv(f'{OUT_PCA_DIR}/pca_loadings.q{q}_{window}_{cell_line}.csv')
    make_weight_heatmap(
        weights=weights,
        feature_cols=clip_cols,
        out_path=f'{OUT_PCA_DIR}/pca_loadings.q{q}_{window}_{cell_line}.svg',
        tag=f'pca_weights.q{q}_{window}_{cell_line}',
        borders=BORDERS,
        value_label='Rotation weight',
        vmin=-0.6, vmax=0.6,
        title=f'PCA weights (n={n_comps}, FDR<={q}, {window} window, {cell_line})',
    )

    var_ratio_used = adata.uns['pca']['variance_ratio'][:n_comps]
    weights_scaled = weights.mul(np.sqrt(var_ratio_used), axis=0)
    weights_scaled.to_csv(f'{OUT_PCA_DIR}/pca_loadings.q{q}_{window}_{cell_line}.scaled.csv')
    make_weight_heatmap(
        weights=weights_scaled,
        feature_cols=clip_cols,
        out_path=f'{OUT_PCA_DIR}/pca_loadings.q{q}_{window}_{cell_line}.scaled.svg',
        tag=f'pca_weights.q{q}_{window}_{cell_line}.scaled',
        borders=BORDERS,
        value_label='Rotation weight',
        # vmin=-0.6, vmax=0.6,
        title=f'PCA weights (n={n_comps}, FDR<={q}, {window} window, {cell_line})',
    )

    sc.pp.neighbors(adata, n_pcs=N_PCA_USE, n_neighbors=N_NEIGHBORS, random_state=RANDOM_STATE)
    print(f"  Leiden clustering (resolution={LEIDEN_RES}) ...")
    sc.tl.leiden(adata, resolution=LEIDEN_RES, key_added='leiden', random_state=RANDOM_STATE)
    n_clusters = adata.obs['leiden'].nunique()
    print(f"  → {n_clusters} clusters found")

    # prefix cluster labels with cell line so they don't collide once rejoined
    df['cluster'] = [f'{cell_line}_{lbl}' for lbl in adata.obs['leiden'].values]
    cluster_means = df.groupby('cluster')[clip_cols].mean()
    cluster_means = cluster_means.loc[
        sorted(cluster_means.index, key=lambda c: int(c.rsplit('_', 1)[-1]))
    ]
    cluster_means.to_csv(f'{OUT_CLUSTER_DIR}/cluster_feature_means.{q}_{window}_{cell_line}.res{LEIDEN_RES}.csv')

    make_cluster_heatmap(
        df=df,
        feature_cols=clip_cols,
        out_path=f'{OUT_CLUSTER_DIR}/cluster_heatmap.q{q}_{window}_{cell_line}.res{LEIDEN_RES}.svg',
        tag=f'q{q}_{window}_{cell_line}.res{LEIDEN_RES}',
        cluster_col='cluster',
        borders=BORDERS,
        vmin=-3, vmax=3,
        value_label='Mean z-score',
    )

    # print("  Computing UMAP for visualization ...")
    # sc.tl.umap(adata, random_state=RANDOM_STATE)

    # # pull results back into df
    # df = df.copy()
    # df['UMAP1']   = adata.obsm['X_umap'][:, 0]
    # df['UMAP2']   = adata.obsm['X_umap'][:, 1]

    return df, adata


windows = ['3kb']
q_vals = ['0.01']
for window in windows:
    for q in q_vals:
        anchors_df = pd.read_csv(f'{IN_TSV}', sep='\t')

        per_cellline_results = []
        for cell_line in CELL_LINES:
            cl_df = anchors_df[anchors_df['cell_line'] == cell_line].copy()
            if cl_df.empty:
                print(f"  Skipping {cell_line}: no anchors found.")
                continue
            print(f"Running per-cell-line PCA for anchors_q{q}_{window}, {cell_line} only...")
            cl_result, _ = run_pca(cl_df, q, window, cell_line)
            cl_result.to_csv(
                f'{OUT_CLUSTER_DIR}/anchors_q{q}_{window}_{cell_line}_pca_clustered_umap.tsv',
                sep='\t', index=False,
            )
            per_cellline_results.append(cl_result)
            print(f"Done PCA, KNN, UMAP for {cell_line}.")

        # rejoin all cell lines into a single long dataframe
        combined_df = pd.concat(per_cellline_results, axis=0, ignore_index=True)
        combined_df.to_csv(
            f'{OUT_CLUSTER_DIR}/anchors_q{q}_{window}_all_celllines_pca_clustered_umap.tsv',
            sep='\t', index=False,
        )
        print(f"Done joint rejoin for anchors_q{q}_{window}: {len(combined_df):,} total anchors across {len(per_cellline_results)} cell lines.")

        # --------------------------------------------------------------
        # Merge the combined (per-cell-line-clustered) long df back onto
        # the original input df, then relabel anchors that share a
        # specific cross-cell-line cluster "signature" into a single
        # merged cluster (here: ESC=3, EpiLC=2, d4c7PGCLC=4, GSC=3 or 8
        # -> cluster 5 / "PRC"). Everything else is left untouched.
        # --------------------------------------------------------------
        # merged = anchors_df.merge(
        #     combined_df,
        #     on=['anchor_id', 'cell_line'],
        #     how='left',
        #     suffixes=('', '_dup'),
        # )

        # # strip the "{cell_line}_" prefix off the cluster label to get
        # # back the raw Leiden cluster number for that cell line
        # merged['cluster_num'] = merged['cluster'].str.rsplit('_', n=1).str[-1].astype('Int64')

        # # pivot to one row per anchor, one column per cell line's cluster
        # cluster_wide = merged.pivot_table(
        #     index='anchor_id', columns='cell_line', values='cluster_num', aggfunc='first'
        # )

        # required_cols = ['ESC', 'EpiLC', 'd4c7PGCLC', 'GSC']
        # missing = [c for c in required_cols if c not in cluster_wide.columns]
        # if missing:
        #     print(f"  Warning: missing cell line(s) {missing} in this run; skipping cross-cell-line relabeling.")
        # else:
        #     matched_anchors = cluster_wide[
        #         (cluster_wide['ESC'] == 3)
        #         & (cluster_wide['EpiLC'] == 2)
        #         & (cluster_wide['d4c7PGCLC'] == 4)
        #         & (cluster_wide['GSC'].isin([3, 8]))
        #     ].index

        #     mask = merged['anchor_id'].isin(matched_anchors)
        #     print(f"  {mask.sum():,} rows ({merged.loc[mask, 'anchor_id'].nunique():,} anchors) matched the cross-cell-line signature -> relabeling as cluster 5.")

        #     updates = {
        #         "cluster": 5,
        #         "cluster_color": "#4EB265",
        #         "cluster_name_long": "PRC_5",
        #         "cluster_name_short": "PRC",
        #         "category_name": "PRC",
        #         "category_color": "#4EB265",
        #     }

        #     for col, new_val in updates.items():
        #         if col not in merged.columns:
        #             merged[col] = np.nan
        #         merged[f"old_{col}"] = merged[col]  # backup original values
        #         merged.loc[mask, col] = new_val

        # merged = merged.drop(columns=['cluster_num'])

        # # cluster now mixes per-cell-line labels (e.g. "ESC_3") with the
        # # new merged label (int 5) for relabeled anchors -> cast to str
        # # so grouping/plotting treats them consistently
        # merged['cluster'] = merged['cluster'].astype(str)

        # clip_cols = [f'{f}_clip' for f in FEATURE_ORDER]
        # final_cluster_means = merged.groupby('cluster')[clip_cols].mean()
        # final_cluster_means.to_csv(
        #     f'{OUT_CLUSTER_DIR}/cluster_feature_means.{q}_{window}_final_merged.res{LEIDEN_RES}.csv'
        # )

        # make_cluster_heatmap(
        #     df=merged,
        #     feature_cols=clip_cols,
        #     out_path=f'{OUT_CLUSTER_DIR}/cluster_heatmap.q{q}_{window}_final_merged.res{LEIDEN_RES}.svg',
        #     tag=f'q{q}_{window}.final_merged.res{LEIDEN_RES}',
        #     cluster_col='cluster',
        #     borders=BORDERS,
        #     vmin=-3, vmax=3,
        #     value_label='Mean z-score',
        # )
        # print(f"  Regenerated cluster heatmap for final merged clusters.")

        # merged.to_csv(
        #     f'{OUT_CLUSTER_DIR}/anchors_q{q}_{window}_all_celllines_merged_relabeled.tsv',
        #     sep='\t', index=False,
        # )
        # print(f"Done merging + relabeling for anchors_q{q}_{window}.")

print('done')