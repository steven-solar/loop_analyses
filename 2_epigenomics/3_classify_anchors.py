'''
For each anchor, apply rules:
	- Is CTCF 
		- CTCF: CTCF (narrow) **and** Rad21 (narrow)
	- Is E
		- K4me1 (narrow) **or** K27ac (broad)
	- Is P
		- TSS
	- CRE = E **or** P
	- Is PRC
		- H2Aub **and** Ring1b
	- save the underlying results of each thing (ie. is_k4me1 and is_k27ac) so we can see at the end too
    - experimented w narrow vs broad peaks for PRC, felt narrow were more reliable and better overlapped expected loci, gave more reasonable #s, so went with that
'''

# conda activate pybedtools_env
import pandas as pd
import pybedtools #v0.12.0
import sys

NARROW_FOLDER='/mnt/florenceshares/Masahiro/Saitoulab_data/saitoulab_chip/peaks/macs2/narrow'
BROAD_FOLDER='/mnt/florenceshares/Masahiro/Saitoulab_data/saitoulab_chip/peaks/macs2/broad'

def get_centered_bt(df, window_name):
    mid = df['mid']
    if window_name == '3kb':
        pad = 1500
    elif window_name == '10kb':
        pad = 5000
    else:
        raise ValueError(f'Unknown window: {window_name}')
    
    # Create the padded coordinates
    temp_df = pd.DataFrame({
        'chrom': df['chrom'],
        'start': (mid - pad),
        'end': mid + pad,
        'index': df.index, 
        'anchor_id': df['anchor_id'],
    })
    
    # Convert to BedTool
    return pybedtools.BedTool.from_dataframe(temp_df)

def annotate_anchors(df, signals, window_name):
    anchor_bt = get_centered_bt(df, window_name)
    results = pd.DataFrame.from_dict({
        int(f.name): {
            f'CTCF_peak': False,
            f'RAD21_peak': False,
            f'is_CTCF': False,
            f'K4me1_peak': False,
            f'K27ac_peak': False,
            f'is_E': False,
            'is_P': False,
            f'Ring1b_peak': False,
            f'H2Aub_peak_narrow': False,
            f'is_PRC_narrow': False,
            f'is_PRC_broad': False
        } for f in anchor_bt},
        orient='index'
    )

    ctcf_hits = anchor_bt.intersect(signals['CTCF'], u=True)
    ctcf_idx = [int(hit.name) for hit in ctcf_hits]
    results.loc[ctcf_idx, f'CTCF_peak'] = True
    rad21_hits = anchor_bt.intersect(signals['RAD21'], u=True)
    rad21_idx = [int(hit.name) for hit in rad21_hits]
    results.loc[rad21_idx, f'RAD21_peak'] = True

    is_ctcf_idx = list(set(ctcf_idx) & set(rad21_idx))
    results.loc[is_ctcf_idx, f'is_CTCF'] = True
    
    k27_hits = anchor_bt.intersect(signals['H3K27ac'], u=True)
    k27_idx = [int(hit.name) for hit in k27_hits]
    results.loc[k27_idx, f'K27ac_peak'] = True
    k4_hits = anchor_bt.intersect(signals['H3K4me1'], u=True)
    k4_idx = [int(hit.name) for hit in k4_hits]
    results.loc[k4_idx, f'K4me1_peak'] = True

    is_e_idx    = list(set(k27_idx) & set(k4_idx))
    results.loc[is_e_idx, f'is_E'] = True
    
    p_hits = anchor_bt.intersect(signals['TSS'], u=True)
    p_idx = [int(hit.name) for hit in p_hits]
    results.loc[p_idx, 'is_P'] = True
    
    h2_narrow_hits = anchor_bt.intersect(signals['H2Aub_narrow'], u=True)
    ring1b_narrow_hits = anchor_bt.intersect(signals['Ring1b_narrow'], u=True)
    ring1b_broad_hits = anchor_bt.intersect(signals['Ring1b_broad'], u=True)
    h2_narrow_idx = [int(hit.name) for hit in h2_narrow_hits]
    ring1b_narrow_idx    = [int(hit.name) for hit in ring1b_narrow_hits]
    ring1b_broad_idx = [int(hit.name) for hit in ring1b_broad_hits]

    is_prc_narrow_idx = list(set(h2_narrow_idx) & set(ring1b_narrow_idx))
    is_prc_broad_idx = list(set(h2_narrow_idx) & set(ring1b_broad_idx))
    results.loc[h2_narrow_idx, f'H2Aub_peak_narrow'] = True
    results.loc[ring1b_narrow_idx,    f'Ring1b_peak']        = True
    results.loc[is_prc_narrow_idx, f'is_PRC_narrow'] = True
    results.loc[is_prc_broad_idx, f'is_PRC_broad'] = True
    
    annotated_df = df.join(results)
    return annotated_df

windows = ['3kb'] # windows = ['3kb', '10kb']
q_vals = [0.01] # q_vals = [0.01, 0.02, 0.05]
cutoffs = [10] # cutoffs = [10, 20]
cell_lines = ['ESC', 'EpiLC', 'd4c7PGCLC', 'GSC']
prefixes = ['BDF121', 'BDF121', 'BDF121', 'AAG']
IN_DIR = 'anchor_epigenomics'
# cutoff = int(sys.argv[1]) if len(sys.argv) > 1 else 20

for cutoff in cutoffs:
    print(f'Filtering peaks with -log(p) >= {cutoff}...')
    signals_by_cl = {}
    for i, cell_line in enumerate(cell_lines):
        print(f'Loading signals for {cell_line}...')
        prefix = prefixes[i]
        bed_paths = {
            'CTCF': f'{NARROW_FOLDER}/{cell_line}_{prefix}_CTCF_pool_peaks.narrowPeak',
            'RAD21': f'{NARROW_FOLDER}/{cell_line}_{prefix}_RAD21_pool_peaks.narrowPeak',
            'H3K4me1': f'{NARROW_FOLDER}/{cell_line}_{prefix}_K4me1_pool_peaks.narrowPeak',
            'H3K27ac': f'{BROAD_FOLDER}/{cell_line}_{prefix}_K27ac_pool_peaks.broadPeak',
            'TSS': '/mnt/md1/Masahiro/germ_microc/loops/annotation/tss.1kb.norm.mass.promoter.bed',
            'Ring1b_narrow': f'{NARROW_FOLDER}/{cell_line}_{prefix}_Ring1b_pool_peaks.narrowPeak',
            'Ring1b_broad': f'{BROAD_FOLDER}/{cell_line}_{prefix}_Ring1b_pool_peaks.broadPeak',
            'H2Aub_narrow': f'{NARROW_FOLDER}/{cell_line}_{prefix}_H2Aub_pool_peaks.narrowPeak',
        }
        signals = {k: pybedtools.BedTool(v) for k, v in bed_paths.items()}
        for key in signals:
            if key == 'TSS': continue # Skip filtering for TSS since it's already a curated set
            print(f'Original {key} peaks: {len(signals[key])}')
            signals[key] = signals[key].filter(lambda row: float(row[7]) >= cutoff).saveas()
            print(f'Filtered {key} peaks (-log(p) >= {cutoff}): {len(signals[key])}')
        signals_by_cl[cell_line] = signals

    for q_val in q_vals:
        for window in windows:
            print(f'Annotating anchors for window {window} and q-value {q_val}...')
            all_anchors = None
            anchor_df = pd.read_csv(f'{IN_DIR}/anchors_q{q_val}_{window}.tsv', sep='\t')
            for _, cell_line in enumerate(cell_lines):
                cl_df = anchor_df[(anchor_df['cell_line'] == cell_line)]
                cl_annotated_df = annotate_anchors(cl_df, signals_by_cl[cell_line], window)
                all_anchors = cl_annotated_df if all_anchors is None else pd.concat(
                    [all_anchors, cl_annotated_df], ignore_index=True
                )
            all_anchors.to_csv(f'{IN_DIR}/anchors_q{q_val}_{window}_annotated.p{cutoff}.tsv', sep='\t', index=False)
            print(f"CTCF: {all_anchors[f'is_CTCF'].sum()}, "
                  f"E: {all_anchors[f'is_E'].sum()}, P: {all_anchors[f'is_P'].sum()}, "
                  f"PRC narrow: {all_anchors[f'is_PRC_narrow'].sum()}, PRC broad: {all_anchors[f'is_PRC_broad'].sum()}")
    print('Finished processing cell line:', cell_line)
    pybedtools.cleanup(verbose=False)
    print(f'Finished processing cutoff: {cutoff}')
