# conda activate loop_quant_env

import os
import numpy as np
import pandas as pd
import cooler
import time
from multiprocess import Pool
import sys
from functools import partial
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import cooltools
from cooltools.lib import plotting

sys.path.insert(1, '/mnt/md0/jjusuf/absloopquant/AbsLoopQuant_analysis_code')
import looptools

RES=1000
LOOP_PATH='/mnt/md0/sjsolar/loop_pred/loops/merge_steven'
OUT_DIR = '../data/P_s_curves'
os.makedirs(OUT_DIR, exist_ok=True)

def calculate_and_save_avg_Ps_curve(clr, nproc=1, max_sep=3000000, output_Ps_filename=None, output_der_filename=None):
    """Calculate and save the P(s) curve (average across chromosomes, weighted by chromosome size) and save to file. max_sep is the maximum genomic separation in bp to which to calculate the P(s) curve."""

    res = clr.binsize
    
    # calculate the P(s) curve
    cvd = cooltools.expected_cis(clr=clr, smooth=True, aggregate_smoothed=True, smooth_sigma=0.1, nproc=nproc)
    cvd['balanced.avg.smoothed'].loc[cvd['dist'] < 2] = np.nan

    cvd['s_bp'] = cvd['dist'] * RES
    # cvd = cvd.drop_duplicates(subset=['dist']) #[['dist_bp', 'balanced.avg.smoothed.agg']]
    cvd['der'] = np.gradient(np.log(cvd['balanced.avg.smoothed.agg']), np.log(cvd['dist_bp']))

    
    chr_names = [chrom_name for chrom_name in clr.chromnames if len(chrom_name)>3 and (chrom_name[3:].isnumeric() or chrom_name[3:]=='X')]  # only take numbered choromosomes and chrX

    # average across chromosomes
    P_s_data_all_chrs = np.zeros((1+max_sep//res, len(chr_names)))
    der_data_all_chrs = np.zeros((1+max_sep//res, len(chr_names)))
    for i, chr_name in enumerate(chr_names):
        print(i, chr_name)
        P_s_data_chr = cvd.loc[cvd['region1']==chr_name,np.array(['s_bp','balanced.avg'])]
        P_s_data_chr = P_s_data_chr.loc[P_s_data_chr['s_bp']<=max_sep]
        P_s_data_chr = P_s_data_chr.sort_values('s_bp')
        der_data_chr = cvd.loc[cvd['region1']==chr_name, np.array(['s_bp','der'])]
        der_data_chr = der_data_chr.loc[der_data_chr['s_bp']<=max_sep]
        der_data_chr = der_data_chr.sort_values('s_bp')
        assert(np.all(P_s_data_chr['s_bp']==np.arange(0,max_sep+1,res)))
        P_s_data_all_chrs[:,i] = P_s_data_chr['balanced.avg']
        der_data_all_chrs[:,i] = der_data_chr['der']

    chrom_weights = clr.chromsizes[chr_names].values
    chrom_weights = chrom_weights/np.sum(chrom_weights)
    
    P_s_data_averaged_chrs = np.average(P_s_data_all_chrs, axis=1, weights=chrom_weights)
    der_data_averaged_chrs = np.average(der_data_all_chrs, axis=1, weights=chrom_weights)

    # save as txt file
    if output_Ps_filename is None:
        output_Ps_filename = f"P_s_{clr.filename.split('/')[-1].split('.')[0]}_{RES}bp.txt"
    np.savetxt(output_Ps_filename, P_s_data_averaged_chrs)

    if output_der_filename is None:
        output_der_filename = f"P_s_der_{clr.filename.split('/')[-1].split('.')[0]}_{RES}bp.txt"
    np.savetxt(output_der_filename, der_data_averaged_chrs)

def plot_Ps_curves_together(cell_lines, colors, clr_path, tag):
    fig_ps, ax_ps = plt.subplots(figsize=(10, 7))
    fig_der, ax_der = plt.subplots(figsize=(10, 7))

    # To store peak info if you also want to use it later
    peak_positions = {}   # cell_line -> (peak_x, peak_y)

    for cell_line in cell_lines:
        # clr = cooler.Cooler(clr_path.format(cell_line=cell_line, RES=RES))
        # out_path = out_path.format(cell_line=cell_line, RES=RES, OUT_DIR=OUT_DIR)
        # tag = out_path.split('.')[-1]

        # calculate_and_save_avg_Ps_curve(
        #     clr, nproc=50, max_sep=3000000,
        #     output_Ps_filename=f'{OUT_DIR}/P_s_{cell_line}_{RES}bp.{tag}.txt',
        #     output_der_filename=f'{OUT_DIR}/P_s_der_{cell_line}_{RES}bp.{tag}.txt'
        # )

        # --- P(s) ---
        P_s_curve = np.loadtxt(f'{OUT_DIR}/P_s_{cell_line}_{RES}bp.txt')
        x_coords_ps = np.arange(1, len(P_s_curve) + 1) * RES
        ax_ps.loglog(
            x_coords_ps,
            P_s_curve,
            label=cell_line,
            color=colors[cell_line],
            linewidth=2.5,
            alpha=0.8
        )

        # --- P'(s) ---
        P_s_derivative = np.loadtxt(f'{OUT_DIR}/P_s_der_{cell_line}_{RES}bp.txt')
        x_coords_der = np.arange(1, len(P_s_derivative) + 1) * RES

        # keep only where derivative is negative (as in your existing code)
        mask = (P_s_derivative < 0)
        x_coords_der = x_coords_der[mask]
        P_s_derivative = P_s_derivative[mask]

        if len(P_s_derivative) == 0:
            continue  # skip if nothing passes the mask

        # peak in this masked region
        peak_idx = np.argmax(P_s_derivative)
        peak_x = x_coords_der[peak_idx]
        peak_y = P_s_derivative[peak_idx]
        peak_positions[cell_line] = (peak_x, peak_y)

        ax_der.semilogx(
            x_coords_der,
            P_s_derivative,
            color=colors[cell_line],
            linewidth=2.5,
            alpha=0.8,
            label=cell_line  # this will be used for the legend
        )

        # vertical line at the peak
        ax_der.axvline(x=peak_x, color=colors[cell_line], linestyle='--', alpha=0.6)

        # text marking the peak of each cell line
        # slight offset in y to avoid overlapping the line
        ax_der.text(
            peak_x,
            peak_y * 1.05,  # small vertical offset
            f'{cell_line}\n{int(peak_x):,} bp',
            color=colors[cell_line],
            fontsize=9,
            ha='center',
            va='bottom'
        )

    # --- Finalize P(s) figure (color legend is already correct via label/color) ---
    ax_ps.set_xlabel('Distance (bp)', fontsize=12)
    ax_ps.set_ylabel('P(s) HiC', fontsize=12)
    ax_ps.set_title('P(s) HiC', fontsize=14)
    ax_ps.legend(title='Cell line')
    ax_ps.grid(True, which="both", ls="-", alpha=0.2)
    fig_ps.tight_layout()
    fig_ps.savefig(f'{OUT_DIR}/Combined_Ps.{tag}.svg', format='svg')

    # --- Finalize P'(s) figure ---
    ax_der.set_xlabel('Distance (bp)', fontsize=12)
    ax_der.set_ylabel("P'(s) HiC [d(logP)/d(logS)]", fontsize=12)
    ax_der.set_title("P'(s) HiC", fontsize=14)
    ax_der.legend(title='Cell line', loc='best', frameon=True)
    ax_der.grid(True, which="both", ls="-", alpha=0.2)
    fig_der.tight_layout()
    fig_der.savefig(f'{OUT_DIR}/Combined_Derivative.{tag}.svg', format='svg')
    
def read_colors(tsv):
    """Read colors from a TSV file with columns 'cell_line' and 'color'."""
    df = pd.read_csv(tsv, sep='\t')
    return dict(zip(df['cell_line'], df['color']))

cell_lines = ['ESC', 'EpiLC', 'd4c7PGCLC', 'GSC']
colors = read_colors('../data/cl_colors.tsv')
coolers = {}
P_s_curves = {}
# plot_Ps_curves_together(cell_lines, colors, clr_path='/mnt/coldstorage/shares/Masahiro/microc/mcs/{cell_line}_WT_HiC_900M.mcool::/resolutions/{RES}', tag='HiC_900M')

cell_lines += ['merged']
colors['merged'] = '#000000'
plot_Ps_curves_together(cell_lines, colors, clr_path='/mnt/coldstorage/shares/Masahiro/microc/mcs/{cell_line}_WT_10B.mcool::/resolutions/{RES}', tag='MicroC_10B')
