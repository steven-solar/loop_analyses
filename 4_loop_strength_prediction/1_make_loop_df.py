import pandas as pd
import os 
import gc

CELL_LINES = ['ESC', 'EpiLC', 'd4c7PGCLC', 'GSC']
windows = ['3kb']
q_vals = [0.01]
OUT_DIR = 'loops'
IN_LOOP_DIR = '../1_call_loops/loops/quantified_calls'
# IN_ANCHOR_DIR = '../3_anchor_clustering/clustering'
# IN_ANCHOR_DIR = '../3_anchor_clustering/clustering_subcluster_c6'
# IN_ANCHOR_DIR = '../3_anchor_clustering/clustering_recluster_gsc'
# IN_ANCHOR_DIR = '../3_anchor_clustering/clustering_subcluster_cre_prc'
IN_ANCHOR_DIR = '../3_anchor_clustering/clustering_recluster_per_cellline'

os.makedirs(OUT_DIR, exist_ok=True)

def categorize_loops(row):
    loop_name = 'Other-related'
    if row['category_name_left'] == 'CTCF' and row['category_name_right'] == 'CTCF':
        loop_name = 'CTCF-CTCF'
    elif row['category_name_left'] == 'CRE' and row['category_name_right'] == 'CRE':
        loop_name = 'CRE-CRE'
    elif row['category_name_left'] == 'PRC' and row['category_name_right'] == 'PRC':
        loop_name = 'PRC-PRC'
    elif (row['category_name_left'] == 'CTCF' and row['category_name_right'] == 'CRE') or (row['category_name_left'] == 'CRE' and row['category_name_right'] == 'CTCF'):
        loop_name = 'CTCF-CRE'
    elif (row['category_name_left'] == 'CTCF' and row['category_name_right'] == 'PRC') or (row['category_name_left'] == 'PRC' and row['category_name_right'] == 'CTCF'):
        loop_name = 'CTCF-PRC'
    elif (row['category_name_left'] == 'CRE' and row['category_name_right'] == 'PRC') or (row['category_name_left'] == 'PRC' and row['category_name_right'] == 'CRE'):
        loop_name = 'CRE-PRC'
    elif (row['category_name_left'] == 'Weak' or row['category_name_right'] == 'Weak'):
        loop_name = 'Weak-related'
    return loop_name

for window in windows:
    loop_df = None
    for cl in CELL_LINES:
        print(f"Making loop_df for {cl} {window}...")
        cl_loop_df = pd.read_csv(f'{IN_LOOP_DIR}/{cl}_q0.01_AbLE{window}.tsv', sep='\t')
        cl_loop_df['cell_line'] = cl
        cl_loop_df['window'] = window
        if loop_df is None:
            loop_df = cl_loop_df
        else:
            loop_df = pd.concat([loop_df, cl_loop_df], ignore_index=True)

    # anchor_df = pd.read_csv(f'{IN_ANCHOR_DIR}/anchors_{window}_labeled.tsv', sep='\t')
    # anchor_df = pd.read_csv(f'{IN_ANCHOR_DIR}/anchors_reclustered.tsv', sep='\t')
    # anchor_df = pd.read_csv(f'{IN_ANCHOR_DIR}/anchors_recluster_gsc.tsv', sep='\t')
    # anchor_df = pd.read_csv(f'{IN_ANCHOR_DIR}/anchors_cre_prc_reclustered.tsv', sep='\t')
    anchor_df = pd.read_csv(f'{IN_ANCHOR_DIR}/anchors_reclustered_per_cl.tsv', sep='\t')

    annotation = pd.read_csv('../2_epigenomics/anchor_epigenomics/anchors_q0.01_3kb_annotated.p10.tsv', sep='\t')
    MERGE_COLS = ['anchor_id', 'chrom', 'mid', 'cell_line', 'window']
    annotation = annotation[MERGE_COLS + [c for c in annotation.columns if c.startswith('is_')]]
    merge = pd.merge(anchor_df, annotation, on=MERGE_COLS, how='inner')
    merge['is_CRE'] = merge['is_P'] | merge['is_E']
    def add_excl_defs(df):
        df['is_only_CRE'] = df['is_CRE'] & ~df['is_CTCF'] & ~df['is_PRC_narrow']
        df['is_only_CTCF'] = df['is_CTCF'] & ~df['is_CRE'] & ~df['is_PRC_narrow']
        df['is_only_PRC_narrow'] = df['is_PRC_narrow'] & ~df['is_CRE'] & ~df['is_CTCF']
        df['is_none'] = ~(df['is_CTCF'] | df['is_CRE'] | df['is_PRC_narrow'])
        return df
    anchor_df = add_excl_defs(merge)
    
    # A given (chrom, mid, cell_line) bin's features shouldn't depend on which
    # side ('left'/'right') it happened to be labeled when it was first called
    # as a loop anchor -- the same bin can serve as a right anchor for one loop
    # and a left anchor for another. Dedupe on position instead of filtering
    # by the 'anchor' label, so every bin is available to match on both sides.
    dup_mask = anchor_df.duplicated(subset=['chrom', 'mid', 'cell_line'], keep=False)
    n_dupes = dup_mask.sum()
    if n_dupes:
        print(f"WARNING: {n_dupes} rows share (chrom, mid, cell_line) with a differing 'anchor' label or feature values.")
        # Sanity check: are the feature columns actually identical between duplicates?
        feat_check_cols = [c for c in anchor_df.columns if c not in ['chrom', 'mid', 'cell_line', 'anchor']]
        n_conflicting = (anchor_df[dup_mask]
                          .groupby(['chrom', 'mid', 'cell_line'])[feat_check_cols]
                          .nunique()
                          .gt(1)
                          .any(axis=1)
                          .sum())
        print(f"  -> {n_conflicting} of those groups have CONFLICTING feature values across duplicates (not just a left/right label difference).")

    anchor_df_dedup = anchor_df.drop_duplicates(subset=['chrom', 'mid', 'cell_line']).drop(columns=['anchor'])

    feature_cols = [c for c in anchor_df_dedup.columns if c not in ['chrom', 'mid', 'cell_line']]

    left_anchors = anchor_df_dedup.rename(columns={c: f'{c}_left' for c in feature_cols})
    right_anchors = anchor_df_dedup.rename(columns={c: f'{c}_right' for c in feature_cols})

    print(len(loop_df), len(left_anchors), len(right_anchors))

    loop_df = loop_df.merge(left_anchors, how='left',
                             left_on=['chrom1', 'left', 'cell_line'],
                             right_on=['chrom', 'mid', 'cell_line'])
    loop_df = loop_df.drop(columns=['chrom', 'mid'])

    loop_df = loop_df.merge(right_anchors, how='left',
                             left_on=['chrom2', 'right', 'cell_line'],
                             right_on=['chrom', 'mid', 'cell_line'])
    loop_df = loop_df.drop(columns=['chrom', 'mid'])
    loop_df['loop_name'] = loop_df.apply(categorize_loops, axis=1)
    # loop_df.to_csv(f'{OUT_DIR}/loop_df_{window}.tsv', sep='\t', index=False)
    # loop_df.to_csv(f'{OUT_DIR}/loop_df_reclustered.tsv', sep='\t', index=False)
    # loop_df.to_csv(f'{OUT_DIR}/loop_df_recluster_gsc.tsv', sep='\t', index=False)
    # loop_df.to_csv(f'{OUT_DIR}/loop_df_cre_prc_reclustered.tsv', sep='\t', index=False)
    loop_df.to_csv(f'{OUT_DIR}/loop_df_reclustered_per_cl.tsv', sep='\t', index=False)
    print(f'Done making loop_df for {window}. Saved to {OUT_DIR}/loop_df_reclustered_per_cl.tsv')
