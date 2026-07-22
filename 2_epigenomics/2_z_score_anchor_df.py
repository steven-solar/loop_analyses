import pandas as pd
import numpy as np

CELL_LINES = ["ESC", "EpiLC", "d4c7PGCLC", "GSC"]
windows = ["3kb", "10kb"]
q_vals = [0.01, 0.02, 0.05]
IN_DIR = 'loop_epigenomics'
OUT_DIR = 'anchor_epigenomics'
def get_anchor_df(df, anchor_num, cl, window):
    rename = {}
    anchor_cols = ['chrom']
    if anchor_num == 1:
        anchor_cols.append('left')
    elif anchor_num == 2:
        anchor_cols.append('right')
    anchor_cols += ['res', 'FDR']
    for col in df.columns:
        if col.startswith(f'anchor{anchor_num}_'):
            if 'ATAC' in col:
                rename[col] = col.replace(f'anchor{anchor_num}_', '')
                anchor_cols.append(col)
            elif 'Input' in col:
                continue
            elif '_norm' in col:
                rename[col] = col.replace(f'anchor{anchor_num}_', '').replace('_norm', '')
                anchor_cols.append(col)
    df = df[anchor_cols].copy()
    if anchor_num == 1:
        df = df.rename(columns={'left': 'mid'})
        df['anchor'] = 'left'
    elif anchor_num == 2:
        df = df.rename(columns={'right': 'mid'})
        df['anchor'] = 'right'
    df['cell_line'] = cl
    df['window'] = window
    df['anchor_id'] = df['chrom'] + ':' + df['mid'].astype(str) + ':' + df['cell_line']
    df.drop_duplicates(subset=['anchor_id'], inplace=True)
    df = df.rename(columns=rename)
    return df
    
def get_features(df):
    """
    Discover raw signal columns to use:
      - anchor1/2_*_norm  (input-normalized ChIP signal)
      - size              (raw loop size in bp)
    Excludes decile columns entirely.
    """
    norm_cols = [c for c in df.columns
                 if c.endswith('_norm') and 'Input' not in c
                 and ('anchor1_' in c or 'anchor2_' in c)] + ['ATAC']
    feature_names = [c.replace('anchor1_', '').replace('anchor2_', '').replace('_norm', '') for c in norm_cols]
    return sorted(list(set(feature_names)))

def compute_zscores(df, norm_feat_cols):
    """
    For each feature in norm_feat_cols, compute mean/std from the joint distribution,
    then z-score every cell line's values using those shared statistics.
    """
    print(f"Computing z-scores for {len(df)} anchors on {len(norm_feat_cols)} features...")
    stats = dict()
    for feat in norm_feat_cols:
        pooled = df[feat]
        pooled = pooled.replace([np.inf, -np.inf], np.nan).dropna()
        print(f"Feature: {feat}, pooled size: {len(pooled)}")
        mu  = pooled.mean()
        std = pooled.std()
        print(f"Feature: {feat}, mean: {mu:.4f}, std: {std:.4f}")
        stats[feat] = (mu, std)
        df[f'{feat}_z'] = (df[feat] - mu) / (std if std > 0 else 1.0)
    return df, stats


for window in windows:
    for q_val in q_vals:
        print(f"Processing anchors for window {window} and q-value {q_val}...")
        all_anchors = None
        for cl in CELL_LINES:
            fp = f"{IN_DIR}/{cl}_q{q_val}_AbLE{window}.tsv"
            df = pd.read_csv(fp, sep="\t")
            feature_names = get_features(df)
            df['chrom'] = df['chrom1']
            anchor1_df = get_anchor_df(df, 1, cl, window)
            anchor2_df = get_anchor_df(df, 2, cl, window)
            anchor_df = pd.concat([anchor1_df, anchor2_df], ignore_index=True)
            if all_anchors is None:
                all_anchors = anchor_df
            else:
                all_anchors = pd.concat([all_anchors, anchor_df], ignore_index=True)
        all_anchors = all_anchors.drop_duplicates(subset=['anchor_id'])
        all_anchors, stats = compute_zscores(all_anchors, feature_names)
        col_order = ['anchor_id', 'chrom', 'mid', 'cell_line', 'anchor', 'window'] + feature_names + [f'{feat}_z' for feat in feature_names]
        all_anchors = all_anchors[col_order]
        all_anchors.to_csv(f"{OUT_DIR}/anchors_q{q_val}_{window}.tsv", sep="\t", index=False)
        stats_df = pd.DataFrame([(k, v[0], v[1]) for k, v in stats.items()], columns=['feat', 'mean', 'std'])
        stats_df.to_csv(f"{OUT_DIR}/anchors_q{q_val}_{window}_stats.tsv", sep="\t", index=False)
        print(f"Saved {len(all_anchors)} anchors for window {window} and q-value {q_val} to {OUT_DIR}/anchors_q{q_val}_{window}.tsv")
