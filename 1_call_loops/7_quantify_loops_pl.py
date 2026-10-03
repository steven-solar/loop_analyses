# conda activate loop_quant_env

import numpy as np
import pandas as pd
import cooler
import time
from multiprocess import Pool
import sys
from functools import partial
import matplotlib.pyplot as plt

sys.path.insert(1, '/mnt/md0/jjusuf/absloopquant/AbsLoopQuant_analysis_code')
import looptools

IN_DIR = 'loops/merged_calls'
OUT_DIR = 'loops/quantified_calls'

# define fn to quantify loops
def quantify_loop_and_save_result(i, cell_line, loop_quant_df):
    chrom1, start1, end1, chrom2, start2, end2, FDR, mustache_scale, chrom, left, right, res, size, AbLE_score = loop_quant_df.loc[i]
    local_region_size = int(np.round(np.sqrt(size/1000/(32/25**2)))) * 1000
    try:
        score = lq.quantify_loop(chrom1, left, right, local_region_size=local_region_size, quant_region_size=QUANT_SIZE, k_min=2, clr_for_outlier_detection=coolers[cell_line], P_s_values_for_outlier_detection=P_s_curves[cell_line])
        if score is None:
            return np.nan
        if score <= 0:
            return np.nan
        return score
    except:
        return np.nan


cell_line = sys.argv[1]
if len(sys.argv) > 3 and sys.argv[3] is not None:
    QUANT_SIZE=int(sys.argv[3])
else:
    QUANT_SIZE=10000

window = f'{QUANT_SIZE//1000}kb'
RES=1000
coolers = {}
P_s_curves = {}
coolers={f"{cell_line}": cooler.Cooler(f'/mnt/coldstorage/shares/Masahiro/microc/mcs/{cell_line}_WT_10B.mcool::/resolutions/{RES}'), 
            f"merged": cooler.Cooler(f'../data/merged.mcool::/resolutions/{RES}')}

P_s_curves={f"{cell_line}": np.loadtxt(f'../data/P_s_curves/P_s_{cell_line}_1000bp.txt'),
            f"merged": np.loadtxt(f'../data/P_s_curves/P_s_merged_1000bp.txt')}

# set up multiprocessing
nproc=40
chunk_size=40

loop_file = f'{IN_DIR}/merged_consensus_q0.05.mm10_loop_filt.bedpe'
loop_df = pd.read_csv(loop_file, sep='\t')
print(f'{loop_file} : {len(loop_df)} loops')
print(f'Calling loops on {cell_line}')

# 3. Calc chunks
num_loops=len(loop_df)
num_chunks = int(np.ceil(len(loop_df)/chunk_size))
chunk_starts = np.arange(num_chunks)*chunk_size
chunk_ends = (np.arange(num_chunks)+1)*chunk_size
chunk_ends[-1] = len(loop_df)

# 4. Set up LoopQuant for this cell line
clr = coolers[cell_line]
P_s_values = P_s_curves[cell_line]
lq = looptools.LoopQuantifier(clr, P_s_values)

# 5. Output df
loop_quant_df = loop_df.copy()
loop_quant_df['AbLE_score'] = np.nan

# 6. process in chunks
print(f'Processing {cell_line} in {num_chunks} chunks')
for chunk_index in np.arange(num_chunks):
    start=time.time()
    with Pool(nproc) as p:
        indices_in_chunk = loop_quant_df.index[np.arange(chunk_starts[chunk_index], chunk_ends[chunk_index])]
        partial_quantify = partial(quantify_loop_and_save_result, cell_line=cell_line, loop_quant_df=loop_quant_df)
        scores_in_chunk = p.map(partial_quantify, indices_in_chunk)
        loop_quant_df.loc[indices_in_chunk, 'AbLE_score'] = np.array(scores_in_chunk)
    end=time.time()
    print(f'Processed {cell_line} chunk {chunk_index+1}/{num_chunks} in {end-start:.1f}s')
    print(f'{cell_line} chunk {chunk_index+1} : {np.sum(np.isnan(np.array(scores_in_chunk)))}/{len(scores_in_chunk)} are nan')

# 7. Output final
q_vals = [0.01, 0.02, 0.05]
for q_val in q_vals:
    loop_quant_df_q = loop_quant_df.loc[loop_quant_df['FDR']<=q_val]
    loop_quant_df_q.to_csv(f'{OUT_DIR}/{cell_line}_q{q_val}_AbLE{window}.tsv', sep='\t', index=False, float_format='%.6f')
