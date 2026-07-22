import pandas as pd
import gc
OUT_DIR = 'loops'

windows = ['3kb']

for window in windows:
    # loop_df = pd.read_csv(f'{OUT_DIR}/loop_df_{window}.tsv', sep='\t')    
    # loop_df = pd.read_csv(f'{OUT_DIR}/loop_df_reclustered.tsv', sep='\t')
    # loop_df = pd.read_csv(f'{OUT_DIR}/loop_df_recluster_gsc.tsv', sep='\t')
    # loop_df = pd.read_csv(f'{OUT_DIR}/loop_df_cre_prc_reclustered.tsv', sep='\t')
    loop_df = pd.read_csv(f'{OUT_DIR}/loop_df_reclustered_per_cl.tsv', sep='\t')

    # columns that are identical for a given loop_id regardless of cell_line
    # (i.e. don't vary by cell_line) stay un-suffixed; everything else gets pivoted wide
    static_cols = ['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2', 'loop_id', 'window', 'size', 'res']
    value_cols = [c for c in loop_df.columns if c not in static_cols + ['cell_line', 'window']]

    print(f"Pivoting {window}: {len(value_cols)} value columns across {loop_df['cell_line'].nunique()} cell lines")

    # sanity check: static cols should be identical within a loop_id across cell lines
    n_inconsistent = loop_df.groupby('loop_id')[static_cols].nunique().gt(1).any(axis=1).sum()
    if n_inconsistent:
        print(f"WARNING: {n_inconsistent} loop_ids have inconsistent static columns across cell lines")

    wide_df = loop_df.pivot_table(
        index='loop_id',
        columns='cell_line',
        values=value_cols,
        aggfunc='first'   # each (loop_id, cell_line) should be unique; 'first' just guards against accidental dupes
    )

    # check for accidental duplicate (loop_id, cell_line) pairs that aggfunc='first' would silently mask
    n_dupe_keys = loop_df.duplicated(subset=['loop_id', 'cell_line']).sum()
    if n_dupe_keys:
        print(f"WARNING: {n_dupe_keys} duplicate (loop_id, cell_line) rows collapsed by pivot_table")

    # flatten MultiIndex columns: ('ATAC_left', 'ESC') -> 'ATAC_left_ESC'
    wide_df.columns = [f'{val}_{cl}' for val, cl in wide_df.columns]
    wide_df = wide_df.reset_index()

    # reattach the static (cell-line-invariant) coordinate columns
    static_df = loop_df.drop_duplicates(subset=['loop_id'])[static_cols]
    wide_df = static_df.merge(wide_df, on='loop_id', how='left')

    print(f"{window}: {len(loop_df)} long rows -> {len(wide_df)} wide loops, {wide_df.shape[1]} columns")

    # out_path = f'{OUT_DIR}/loop_df_{window}_wide.tsv'
    # out_path = f'{OUT_DIR}/loop_df_reclustered_wide.tsv'
    # out_path = f'{OUT_DIR}/loop_df_recluster_gsc_wide.tsv'
    # out_path = f'{OUT_DIR}/loop_df_cre_prc_reclustered_wide.tsv'
    out_path = f'{OUT_DIR}/loop_df_reclustered_per_cl_wide.tsv'
    
    wide_df.to_csv(out_path, sep='\t', index=False)
    print(f'Saved wide {window} df to {out_path}')

    del loop_df, wide_df
    gc.collect()
