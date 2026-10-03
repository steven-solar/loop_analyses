# conda activate bigwig

import os
import sys
import numpy as np
import pandas as pd
import pyBigWig
import glob

def get_anchor_signal(row, anchor, signal_bw, pad=2500):
    chrom = row[f'chrom{anchor}']
    start = row[f'start{anchor}']
    end = row[f'end{anchor}']
    middle = (start + end) // 2
    signal_arr = signal_bw.values(chrom, middle-pad, middle+pad)
    return np.nansum(signal_arr) / (2*pad)  # average signal per base pair in the anchor region

def norm(signal, input_signal, epsilon=1e-6):
    return np.log2((signal + epsilon) / (input_signal + epsilon))

loop_tsv=sys.argv[1]
bigwig_pattern=sys.argv[2] #/mnt/florenceshares/Masahiro/Saitoulab_data/saitoulab_chip/bws/${cell_line}_${prefix}*pool*bw
atac_bigwig=sys.argv[3] #/mnt/florenceshares/Masahiro/Saitoulab_data/saitoulab_atac/bws/${cell_line}_${prefix}_ATAC_pool.bw
input_bigwig=sys.argv[4] #/mnt/florenceshares/Masahiro/Saitoulab_data/saitoulab_chip/bws/${cell_line}_${prefix}_Input_pool.bw
window=int(sys.argv[5])
OUT_DIR = sys.argv[6] if len(sys.argv) > 6 else '.'

print('Window size:', window)
loop_df = pd.read_csv(loop_tsv, sep='\t')
print(len(loop_df), "loops loaded from", loop_tsv)
cell_line=loop_tsv.split('/')[-1].split('_')[0]
print('Cell line:', cell_line)
print('Looking for bigwig files with pattern:', bigwig_pattern)

pad = window//2

big_wig_files = glob.glob(bigwig_pattern)
if not big_wig_files:
    raise FileNotFoundError(f"No BigWig files matched: {bigwig_pattern}")

signal_files = [f for f in big_wig_files          # exclude Input from loop
                if 'Input' not in os.path.basename(f) and 'ATAC' not in os.path.basename(f)]
print(f'Found {len(signal_files)} signal bigwig files:')

print(f'[{cell_line}] Input file: {input_bigwig}')
input_bw = pyBigWig.open(input_bigwig)
print(f'[{cell_line}] Input header: {input_bw.header()}')
for anchor in [1, 2]:
    loop_df[f'anchor{anchor}_Input'] = loop_df.apply(lambda row: get_anchor_signal(row, anchor, input_bw, pad=pad), axis=1)

print(f'[{cell_line}] ATAC file: {atac_bigwig}')
atac_bw = pyBigWig.open(atac_bigwig)
print(f'[{cell_line}] ATAC header: {atac_bw.header()}')
for anchor in [1, 2]:
    loop_df[f'anchor{anchor}_ATAC'] = loop_df.apply(lambda row: get_anchor_signal(row, anchor, atac_bw, pad=pad), axis=1)

for bigwig_file in big_wig_files:
    print('Processing bigwig file:', bigwig_file)
    fname       = os.path.basename(bigwig_file)
    signal=fname.split('_')[2]
    print('signal:', signal)
    signal_bw = pyBigWig.open(bigwig_file)
    for anchor in [1, 2]:
        loop_df[f'anchor{anchor}_{signal}'] = loop_df.apply(lambda row: get_anchor_signal(row, anchor, signal_bw, pad=pad), axis=1)
        loop_df[f'anchor{anchor}_{signal}_norm'] = norm(loop_df[f'anchor{anchor}_{signal}'], loop_df[f'anchor{anchor}_Input'])

window_name = f"{window//1000}kb"
q_vals = [0.01, 0.02, 0.05]
for q in q_vals:
    output_file = f"{OUT_DIR}/{cell_line}_q{q}_AbLE{window_name}.tsv"
    filt_df = loop_df[loop_df['FDR'] <= q]
    filt_df.to_csv(output_file, sep='\t', index=False, float_format='%.4e')

print(len(filt_df), "loops after processing")
