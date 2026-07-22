import pandas as pd
import numpy as np

BEDPATH    = '/mnt/florenceshares/Masahiro/Saitoulab_data/saitoulab_chip/blacklist/mm10_chip_blacklist.bed'
DIR = 'loops/merged_calls'
in_loops = pd.read_csv(f'{DIR}/merged_consensus_q0.05.loop_filt.bedpe', sep='\t', header=None,
                       names=['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2',
                              'FDR', 'left', 'right', 'res', 'size'])

def load_bed(path: str) -> pd.DataFrame:
    print(f"Loading BED filter: {path}")
    bed = pd.read_csv(path, sep='\t', header=None,
                       usecols=[0, 1, 2],
                       names=['chrom', 'start', 'end'],
                       comment='#')
    bed = bed[~bed['chrom'].astype(str).str.startswith('track')]
    bed = bed[~bed['chrom'].astype(str).str.startswith('browser')]
    bed['start'] = pd.to_numeric(bed['start'], errors='coerce')
    bed['end']   = pd.to_numeric(bed['end'],   errors='coerce')
    bed = bed.dropna().reset_index(drop=True)
    bed['start'] = bed['start'].astype(int)
    bed['end']   = bed['end'].astype(int)
    print(f"  {len(bed):,} BED regions loaded")
    return bed


def build_interval_index(bed: pd.DataFrame) -> dict:
    index = {}
    for chrom, grp in bed.groupby('chrom'):
        grp_s = grp.sort_values('start')
        index[str(chrom)] = (
            grp_s['start'].values.astype(int),
            grp_s['end'].values.astype(int),
        )
    return index


def overlaps_any(chrom: np.ndarray, starts: np.ndarray, ends: np.ndarray, index: dict) -> np.ndarray:
    result = np.zeros(len(chrom), dtype=bool)
    for c in np.unique(chrom):
        mask = chrom == c
        if c not in index:
            continue
        bed_starts, bed_ends = index[c]
        q_starts = starts[mask]
        q_ends   = ends[mask]
        hit = np.zeros(mask.sum(), dtype=bool)
        for i, (qs, qe) in enumerate(zip(q_starts, q_ends)):
            lo = np.searchsorted(bed_starts, qe, side='left')
            if lo > 0:
                hit[i] = np.any(bed_ends[:lo] > qs)
        result[mask] = hit
    return result


def filter_by_bed(df: pd.DataFrame, bed: pd.DataFrame) -> pd.DataFrame:
    n0    = len(df)
    index = build_interval_index(bed)
    hit1 = overlaps_any(df['chrom1'].values.astype(str), df['start1'].values.astype(int),
                         df['end1'].values.astype(int), index)
    hit2 = overlaps_any(df['chrom2'].values.astype(str), df['start2'].values.astype(int),
                         df['end2'].values.astype(int), index)
    overlaps = hit1 | hit2
    df = df[~overlaps].reset_index(drop=True)
    print(f"  BED filter: {n0:,} -> {len(df):,} loops (removed {n0 - len(df):,})")
    print(f"    anchor1 hits: {hit1.sum():,}  anchor2 hits: {hit2.sum():,}")
    return df

q_vals = [0.01, 0.02, 0.05]
bed = load_bed(BEDPATH)
filt_df = filter_by_bed(in_loops, bed)
for q_val in q_vals:
    mm10_filt_df = filt_df[filt_df['FDR'] <= q_val]
    mm10_filt_df.to_csv(f'{DIR}/merged_consensus_q{q_val}.mm10_loop_filt.bedpe', sep='\t', index=False, header=False)
