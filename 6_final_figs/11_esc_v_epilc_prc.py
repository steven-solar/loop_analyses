# conda activate coolbox

import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re

GENES_PATH = '../data/mm10_genes.bed'
OUT_BED    = '../data/mm10_prc_genes.bed'
 
FAMILY_PREFIXES = ['SOX', 'DLX', 'FOX', 'HOX', 'MNX', 'NKX']
EXACT_GENES     = ['SP9']
EXCLUDE_GENES = ['FOXRED1', 'FOXRED2']
STRIP_ANTISENSE = True
ANTISENSE_SUFFIX_RE = re.compile(r'(?:as|os|ot|it)\d*$', re.IGNORECASE)

cols = ['chrom', 'start', 'end', 'gene', 'score', 'strand',
        'thickStart', 'thickEnd', 'itemRgb', 'blockCount', 'blockSizes',
        'blockStarts', 'transcript_id']
genes = pd.read_csv(GENES_PATH, sep='\t', header=None, names=cols)
 
gene_upper = genes['gene'].str.upper()
is_family = pd.Series(False, index=genes.index)
for prefix in FAMILY_PREFIXES:
    is_family |= gene_upper.str.startswith(prefix)
is_exact = gene_upper.isin(EXACT_GENES)
 
hits = genes[is_family | is_exact].copy()

n_before_exclude = hits['gene'].nunique()
hits = hits[~hits['gene'].str.upper().isin(EXCLUDE_GENES)]
n_excluded = n_before_exclude - hits['gene'].nunique()
if n_excluded:
    print(f"dropped {n_excluded} gene(s) via EXCLUDE_GENES: {EXCLUDE_GENES}")
 
if STRIP_ANTISENSE:
    is_antisense = hits['gene'].str.contains(ANTISENSE_SUFFIX_RE)
    if is_antisense.any():
        print(f"dropped {is_antisense.sum()} antisense/overlapping transcript(s): "
              f"{sorted(hits.loc[is_antisense, 'gene'].unique())}")
    hits = hits[~is_antisense]

matched_genes = sorted(hits['gene'].unique())
print(f"matched {len(matched_genes)} unique gene symbols, {len(hits)} transcript rows:")
for g in matched_genes:
    print(' ', g)
 
hits[['chrom', 'start', 'end', 'gene', 'score', 'strand']].to_csv(
    OUT_BED, sep='\t', header=False, index=False
)
print(f"\nwrote {len(hits)} rows to {OUT_BED}")

LOOP_PATH = '../4_loop_strength_prediction/loops/loop_df_reclustered_per_cl_wide.tsv' 

STAGE_A, STAGE_B = 'ESC', 'EpiLC'
LOOP_CLASS_COL_A = f'loop_name_{STAGE_A}'
LOOP_CLASS_COL_B = f'loop_name_{STAGE_B}'

LOOP_CLASS     = 'PRC-PRC'

bed  = pd.read_csv(OUT_BED, sep='\t', header=None,
                    names=['chrom', 'start', 'end', 'gene', 'score', 'strand'])
wide = pd.read_csv(LOOP_PATH, sep='\t')


def overlaps_bed(chroms, starts, ends, bed):
    """True if (chrom, start, end) overlaps ANY interval in bed."""
    hit = pd.Series(False, index=chroms.index)
    for chrom, sub in bed.groupby('chrom'):
        mask = (chroms == chrom)
        if not mask.any():
            continue
        s, e = starts[mask].to_numpy(), ends[mask].to_numpy()
        any_overlap = np.zeros(mask.sum(), dtype=bool)
        for b_start, b_end in zip(sub['start'], sub['end']):
            any_overlap |= (s < b_end) & (e > b_start)
        hit.loc[mask] = any_overlap
    return hit


anchor1_hit = overlaps_bed(wide['chrom1'], wide['start1'], wide['end1'], bed)
anchor2_hit = overlaps_bed(wide['chrom2'], wide['start2'], wide['end2'], bed)
wide['overlaps_tf_locus'] = anchor1_hit | anchor2_hit

sub = wide[
    wide['overlaps_tf_locus']
    & (wide[LOOP_CLASS_COL_A] == LOOP_CLASS)
    & (wide[LOOP_CLASS_COL_B] == LOOP_CLASS)
].copy()

able_a, able_b = f'AbLE_score_{STAGE_A}', f'AbLE_score_{STAGE_B}'
sub = sub.dropna(subset=[able_a, able_b])
sub = sub[(sub[able_a] != 0) & (sub[able_b] != 0)]

print(f"{LOOP_CLASS} loops overlapping a TF-family locus, present+scored in both {STAGE_A} and {STAGE_B}: {len(sub)}")
print(f"\n{STAGE_A} AbLE: mean={sub[able_a].mean():.4f}  median={sub[able_a].median():.4f}")
print(f"{STAGE_B} AbLE: mean={sub[able_b].mean():.4f}  median={sub[able_b].median():.4f}")

stat, p = wilcoxon(sub[able_a], sub[able_b])
print(f"\nWilcoxon signed-rank ({STAGE_A} vs {STAGE_B}): stat={stat:.2f}, p={p:.4g}")
