#!/usr/bin/env bash

# conda activate bigwig
# pyBigWig Library Version: 0.3.22

cell_lines=("ESC" "EpiLC" "d4c7PGCLC" "GSC")
prefixes=("BDF121" "BDF121" "BDF121" "AAG")
BW_DIR="/mnt/florenceshares/Masahiro/Saitoulab_data/saitoulab_chip/bws"
ATAC_DIR="/mnt/coldstorage/shares/Masahiro/Saitoulab_data/saitoulab_atac/bws"

LOOP_DIR="../1_call_loops/loops/quantified_calls"
OUT_DIR="loop_epigenomics"

mkdir -p $OUT_DIR

for ((i=0; i<${#cell_lines[@]}; i++)); do
	cell_line=${cell_lines[$i]}
	prefix=${prefixes[$i]}
	echo "Starting ${cell_line} ${prefix}"
	python 1_quantify_epigenomics.py \
		"${LOOP_DIR}/${cell_line}_q0.05_AbLE3kb.tsv" \
		"${BW_DIR}/${cell_line}_${prefix}*pool*bw" \
		"${ATAC_DIR}/${cell_line}_${prefix}_ATAC_pool.bw" \
		"${BW_DIR}/${cell_line}_${prefix}_Input_pool.bw" \
		3000 \
		"${OUT_DIR}" &

	python 1_quantify_epigenomics.py \
		"${LOOP_DIR}/${cell_line}_q0.05_AbLE10kb.tsv" \
		"${BW_DIR}/${cell_line}_${prefix}*pool*bw" \
		"${ATAC_DIR}/${cell_line}_${prefix}_ATAC_pool.bw" \
		"${BW_DIR}/${cell_line}_${prefix}_Input_pool.bw" \
		10000 \
		"${OUT_DIR}" &
done

wait `jobs -p`
echo "All done"
