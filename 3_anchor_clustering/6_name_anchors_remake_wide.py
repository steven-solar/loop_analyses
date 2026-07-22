import pandas as pd

def attach_cluster_metadata(anchor_df, cluster_meta_path):
    """
    Attach cluster name, cluster color, category name, and category color
    to an anchor dataframe by joining on the 'cluster' column.

    Parameters
    ----------
    anchor_df : pd.DataFrame
        DataFrame containing at least a 'cluster' column (int).
    cluster_meta_path : str
        Path to the TSV file with columns:
        cluster, cluster_name_long, cluster_name_short, cluster_color, category_name, category_color

    Returns
    -------
    pd.DataFrame
        anchor_df with six new columns appended:
        cluster_name_long, cluster_name_short, cluster_color, category_name, category_color
    """
    meta_cols = ['cluster', 'cluster_name_long', 'cluster_name_short', 'cluster_color', 'category_name', 'category_color']
    
    meta_df = pd.read_csv(cluster_meta_path, sep='\t', usecols=meta_cols)
    meta_df['cluster'] = meta_df['cluster'].astype(int)

    anchor_df = anchor_df.copy()
    anchor_df['cluster'] = anchor_df['cluster'].astype(int)

    # drop any of these cols if they already exist to avoid _x/_y suffixes on re-run
    cols_to_drop = [c for c in meta_cols[1:] if c in anchor_df.columns]
    if cols_to_drop:
        anchor_df = anchor_df.drop(columns=cols_to_drop)

    anchor_df = anchor_df.merge(meta_df, on='cluster', how='left')

    n_unmatched = anchor_df['cluster_name_long'].isna().sum()
    if n_unmatched:
        print(f"WARNING: {n_unmatched} rows had no matching cluster in metadata")

    return anchor_df

window='3kb'
anchors = pd.read_csv(f'clustering/anchors_q0.01_{window}_pca_clustered_umap.tsv', sep='\t')
anchors['anchor_id_nocl'] = anchors['chrom'] + ':' + anchors['mid'].astype(str)
anchors = attach_cluster_metadata(anchors, f'clustering/anchor_names_cats_colors.tsv')
anchors.to_csv(f'clustering/anchors_{window}_labeled.tsv', sep='\t', index=False)

static_cols = ['chrom', 'mid', 'anchor_id_nocl']
value_cols = [c for c in anchors.columns if c not in static_cols + ['cell_line', 'window']]

wide_df = anchors.pivot_table(
    index='anchor_id_nocl',
    columns='cell_line',
    values=value_cols,
    aggfunc='first'   # should be unique; 'first' guards against accidental dupes
)

wide_df.columns = [f'{val}_{cl}' for val, cl in wide_df.columns]
wide_df = wide_df.reset_index()

static_df = anchors.drop_duplicates(subset=['anchor_id_nocl'])[static_cols]
wide_df = static_df.merge(wide_df, on='anchor_id_nocl', how='left')

print(f"{window}: {len(anchors)} long rows -> {len(wide_df)} wide loops, {wide_df.shape[1]} columns")

out_path = f'clustering/anchors_{window}_wide_labeled.tsv'
wide_df.to_csv(out_path, sep='\t', index=False)
print(f'Saved wide {window} df to {out_path}')

del anchors, wide_df
