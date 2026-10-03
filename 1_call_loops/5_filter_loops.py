# conda activate loop_quant_env

import numpy as np
import pandas as pd
import subprocess as sp
import time
from multiprocessing import Pool
import cv2
import cooler
from functools import partial
import sys


# load the coolers & P(s) curves
RES = 1000
LOOP_PATH='loops/merged_calls'
OUT_PATH=LOOP_PATH
MIN_READ_COUNTS_PER_PIXEL = 0.4
MAX_DIST_TO_CENTER_PX = 2.5
LOCAL_REGION_SIZE = 10000
CELL_LINE = 'merged'
CLR_PATH = f'../data/{CELL_LINE}.mcool::/resolutions/1000'
COOLER = cooler.Cooler(CLR_PATH)
P_S_CURVE = np.loadtxt(f'../data/P_s_curves/P_s_{CELL_LINE}_{RES}bp.txt')


def get_window(left, right):
    snap_left = (left // RES) * RES
    snap_right = (right // RES) * RES
    start1 = snap_left - LOCAL_REGION_SIZE
    end1   = snap_left + LOCAL_REGION_SIZE + RES
    start2 = snap_right - LOCAL_REGION_SIZE
    end2   = snap_right + LOCAL_REGION_SIZE + RES
    return start1, end1, start2, end2

def acceptable_size_and_location(clr, chrom, left, right, min_size=32000):
    '''
    Check that loop is an appropriate size (>=min_size) and is not too close to end of chromosome.
    Requires global variable LOCAL_REGION_SIZE.
    Returns True if passed, False if failed.
    '''    
    if right-left < min_size:
        return False
    if left-LOCAL_REGION_SIZE < 0 or right+LOCAL_REGION_SIZE > clr.chromsizes[chrom]:
        return False
    return True

def no_NaNs_near_center(clr, chrom, left, right, na_stripe_dist_to_center_px_cutoff=5):
    '''
    Check that there are no NaN stripes too close to the center (<=na_stripe_dist_to_center_px_cutoff away).
    Requires global variable LOCAL_REGION_SIZE.
    Returns True if passed, False if failed.
    '''
    # get the image
    start1, end1, start2, end2 = get_window(left, right)
    img = clr.matrix().fetch(f'{chrom}:{start1}-{end1}',f'{chrom}:{start2}-{end2}').astype('float')
    
    # find NA stripes, if any
    length_of_img = img.shape[0]  # height/width of square image
    ver_na_stripe_indices = np.where(np.sum(np.isnan(img),0)==length_of_img)[0]  # get indices of NA stripes
    hor_na_stripe_indices = np.where(np.sum(np.isnan(img),1)==length_of_img)[0]
    any_na_stripes = len(ver_na_stripe_indices)>0 or len(hor_na_stripe_indices)>0

    if any_na_stripes:
        middle_index = length_of_img//2
        ver_na_stripe_indices_from_middle = np.abs(ver_na_stripe_indices - middle_index)
        hor_na_stripe_indices_from_middle = np.abs(hor_na_stripe_indices - middle_index)
        if np.any(ver_na_stripe_indices_from_middle<=na_stripe_dist_to_center_px_cutoff) or np.any(hor_na_stripe_indices_from_middle<=na_stripe_dist_to_center_px_cutoff):
            return False  # NA values are too close to center; can't be resolved   
    return True

def global_maximum_dist_to_center(clr, chrom, left, right, P_s_data, s_px_matrix, gaussian_blur_sigma_px=2.5, ignore_diag_cutoff_px=5):
    '''
    Calculate the Euclidean distance (in pixels) of the global maximum to the center of the image (the location of the loop). The global maximum is calculated on the observed/expected matrix.
    Requires global variable LOCAL_REGION_SIZE.
    '''
    snap_left = (left // RES) * RES
    snap_right = (right // RES) * RES
    start1 = snap_left - LOCAL_REGION_SIZE
    end1   = snap_left + LOCAL_REGION_SIZE + RES
    start2 = snap_right - LOCAL_REGION_SIZE
    end2   = snap_right + LOCAL_REGION_SIZE + RES

    # get the image
    img = clr.matrix().fetch(f'{chrom}:{start1}-{end1}',f'{chrom}:{start2}-{end2}').astype('float')

    # get the expected global background image
    bg_img = P_s_data[s_px_matrix]

    # in the image and background image, make all pixels near diagonal NA
    img[s_px_matrix<=ignore_diag_cutoff_px] = np.nan
    bg_img[s_px_matrix<=ignore_diag_cutoff_px] = np.nan

    # divide the image by the expected global background
    img_over_bg = img/bg_img

    # resolve any NA values (only do this if no NaNs near center)
    img_over_bg_NAs_removed = np.nan_to_num(img_over_bg, nan=np.nanmedian(img))  # replace NA values with median value in the image
    
    # blur image
    ksize = int(np.ceil(3*gaussian_blur_sigma_px)//2*2+1)  # round up to next odd integer >= 3 sigma
    img_over_bg_blurred = cv2.GaussianBlur(img_over_bg_NAs_removed,ksize=(ksize,ksize),sigmaX=gaussian_blur_sigma_px)
    
    # find global maximum
    center_pixel_indices = np.array([i[0] for i in np.where(np.logical_and(x_px==0,y_px==0))])
    brightest_pixel_indices = np.array(np.unravel_index(np.nanargmax(img_over_bg_blurred), img_over_bg_blurred.shape))
    dist_to_brightest_pixel = np.linalg.norm(brightest_pixel_indices-center_pixel_indices)
    return dist_to_brightest_pixel

def read_counts_per_pixel(clr, chrom, left, right):
    '''
    Calculate the number of reads divided by the number of pixels in the local region.
    Requires global variable LOCAL_REGION_SIZE.
    '''
    start1, end1, start2, end2 = get_window(left, right)
    img_unbalanced = clr.matrix(balance=False).fetch(f'{chrom}:{start1}-{end1}',f'{chrom}:{start2}-{end2}').astype('float')
    read_count = np.sum(img_unbalanced)
    num_pixels = np.size(img_unbalanced)
    return read_count/num_pixels

def run_global_maximum_dist_to_center(chrom, left, right):
    loop_size_px = right//RES-left//RES
    s_px_matrix = loop_size_px+y_px-x_px  # genomic separation in units of res
    s_px_matrix[s_px_matrix<0] = 0  # don't allow negative values of s
    return global_maximum_dist_to_center(COOLER, chrom, left, right, P_S_CURVE, s_px_matrix)

def run_filter_loop(i, loops_bedpe):
    row = loops_bedpe.iloc[i]
    chrom = row['chrom']
    left = row['left']
    right = row['right']
    loop_size_px = right//RES-left//RES
    s_px_matrix = loop_size_px+y_px-x_px  # genomic separation in units of res
    s_px_matrix[s_px_matrix<0] = 0  # don't allow negative values of s

    size_and_loc_pass = acceptable_size_and_location(COOLER, chrom, left, right, min_size=32000)
    NaN_stripe_pass = no_NaNs_near_center(COOLER, chrom, left, right, na_stripe_dist_to_center_px_cutoff=5)
    global_max_dist = run_global_maximum_dist_to_center(chrom, left, right)
    global_max_pass = global_maximum_dist_to_center(COOLER, chrom, left, right, P_S_CURVE, s_px_matrix) <= MAX_DIST_TO_CENTER_PX
    read_count_pass = read_counts_per_pixel(COOLER, chrom, left, right) >= MIN_READ_COUNTS_PER_PIXEL
    return size_and_loc_pass, NaN_stripe_pass, global_max_dist, global_max_pass, read_count_pass

# calculate matrices that depend on LOCAL_REGION_SIZE
a = LOCAL_REGION_SIZE//RES
y_px, x_px = np.meshgrid(np.arange(-a, a+1),np.arange(-a, a+1))  # matrices of pixel coordinates relative to center pixel; units are pixels; shape is (2a,2a)
chunk_size = 40
nproc = 40
q_vals = [0.01, 0.02, 0.05]

# load in most permissive FDR filtering, then just refilter at the end for the other q values
loops_bedpe = pd.read_csv(f'{LOOP_PATH}/{CELL_LINE}_consensus_q0.05.sorted.bedpe', sep='\t', header=None, names=['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2', 'FDR', 'mustache_scale'])
print(f'loaded {len(loops_bedpe)} loops for {CELL_LINE} @ q=0.05')

loops_bedpe['chrom'] = loops_bedpe['chrom1']  # for convenience, since chrom1 and chrom2 are always the same in our data
loops_bedpe['left'] = (loops_bedpe['start1']+loops_bedpe['end1'])//2
loops_bedpe['right'] = (loops_bedpe['start2']+loops_bedpe['end2'])//2
loops_bedpe['res'] = loops_bedpe['end1'] - loops_bedpe['start1']
loops_bedpe['size'] = loops_bedpe['right']-loops_bedpe['left']
loops_bedpe['size_and_loc_pass'] = False  # initialize with False; will be updated to True if passes size and location filter
loops_bedpe['NaN_stripe_pass'] = False  # initialize with False; will be updated to True if passes NaN stripe filter
loops_bedpe['global_max_dist'] = np.inf  # initialize with infinity; will be updated with actual max distance after filtering
loops_bedpe['global_max_pass'] = False  # initialize with False; will be updated to True if passes global max distance filter
loops_bedpe['read_count_pass'] = False  # initialize with False; will be updated to True if passes read count filter

# multiprocessing
num_chunks = int(np.ceil(len(loops_bedpe)/chunk_size))
chunk_starts = np.arange(num_chunks)*chunk_size
chunk_ends = (np.arange(num_chunks)+1)*chunk_size
chunk_ends[-1] = len(loops_bedpe)

for chunk_index in np.arange(num_chunks):
    start = time.time()
    with Pool(nproc) as p:
        indices_in_chunk = np.arange(chunk_starts[chunk_index],chunk_ends[chunk_index])
        partial_loop_filt = partial(run_filter_loop, loops_bedpe=loops_bedpe)
        loop_results_in_chunk = p.map(partial_loop_filt, indices_in_chunk)
        loops_bedpe.loc[indices_in_chunk, ['size_and_loc_pass', 'NaN_stripe_pass', 'global_max_dist', 'global_max_pass', 'read_count_pass']] = loop_results_in_chunk
    if chunk_index % 10 == 0:
        # to avoid losing progress if the script is interrupted, save the results after every 10 chunks
        loops_bedpe.to_csv(f'{OUT_PATH}/{CELL_LINE}_consensus_q0.05.sorted.marked.bedpe', sep='\t', index=False, header=False)  # save after applying FDR filter
    end = time.time()
    # print progress
    print(f'Processed {CELL_LINE} chunk {chunk_index+1}/{num_chunks} in {end-start:.1f}s')

for q_val in q_vals:
    loops_bedpe_q = loops_bedpe.loc[loops_bedpe['FDR']<=q_val]
    loops_bedpe_q['pass'] = loops_bedpe_q['global_max_pass'] & loops_bedpe_q['NaN_stripe_pass'] & loops_bedpe_q['size_and_loc_pass'] & loops_bedpe_q['read_count_pass']
    loops_bedpe_q.to_csv(f'{OUT_PATH}/{CELL_LINE}_consensus_q{q_val}.sorted.marked.tsv', sep='\t', index=False, header=False)  # save after applying FDR filter and marking which loops pass filters
    filtered_loops_df = loops_bedpe_q.loc[loops_bedpe_q['pass']]  # only keep loops that pass all filters
    filtered_loops_df = filtered_loops_df[['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2', 'FDR', 'left', 'right', 'res', 'size']]  # only keep the original columns
    filtered_loops_df.to_csv(f'{OUT_PATH}/{CELL_LINE}_consensus_q{q_val}.loop_filt.bedpe', sep='\t', index=False, header=False)  # save after applying FDR filter
