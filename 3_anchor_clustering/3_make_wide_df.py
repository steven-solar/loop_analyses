# conda activate umap_env

import pandas as pd

window='3kb'
anchors = pd.read_csv('clustering_recluster_per_cellline/anchors_reclustered_per_cl.tsv', sep='\t')

anchors['anchor_id_nocl'] = anchors['chrom'] + ':' + anchors['mid'].astype(str)

static_cols = ['chrom', 'mid', 'anchor_id_nocl']
value_cols = [c for c in anchors.columns if c not in static_cols + ['cell_line', 'window']]

# sanity check: static cols should be identical within a anchor_id_nocl across cell lines
n_inconsistent = anchors.groupby('anchor_id_nocl')[static_cols].nunique().gt(1).any(axis=1).sum()
if n_inconsistent:
    print(f"WARNING: {n_inconsistent} anchor_ids have inconsistent static columns across cell lines")

wide_df = anchors.pivot_table(
    index='anchor_id_nocl',
    columns='cell_line',
    values=value_cols,
    aggfunc='first'   # each (anchor_id_nocl, cell_line) should be unique; 'first' just guards against accidental dupes
)

# check for accidental duplicate (anchor_id_nocl, cell_line) pairs that aggfunc='first' would silently mask
# n_dupe_keys = anchors.duplicated(subset=['anchor_id_nocl', 'cell_line']).sum()
# if n_dupe_keys:
#     print(f"WARNING: {n_dupe_keys} duplicate (anchor_id_nocl, cell_line) rows collapsed by pivot_table")

# flatten MultiIndex columns: ('ATAC_left', 'ESC') -> 'ATAC_left_ESC'
wide_df.columns = [f'{val}_{cl}' for val, cl in wide_df.columns]
wide_df = wide_df.reset_index()

# reattach the static (cell-line-invariant) coordinate columns
static_df = anchors.drop_duplicates(subset=['anchor_id_nocl'])[static_cols]
wide_df = static_df.merge(wide_df, on='anchor_id_nocl', how='left')

print(f"{window}: {len(anchors)} long rows -> {len(wide_df)} wide loops, {wide_df.shape[1]} columns")

# out_path = f'clustering/anchors_{window}_wide.tsv'
# out_path = f'clustering_subcluster_c6/anchors_reclustered_wide.tsv'
# out_path = f'clustering_recluster_gsc/anchors_recluster_gsc_wide.tsv'
# out_path = f'clustering_subcluster_cre_prc/anchors_cre_prc_reclustered_wide.tsv'
out_path = f'clustering_recluster_per_cellline/anchors_reclustered_per_cl_wide.tsv'

wide_df.to_csv(out_path, sep='\t', index=False)
print(f'Saved wide {window} df to {out_path}')

del anchors, wide_df
