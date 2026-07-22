#!/usr/bin/env bash

DIR="loops"
IN_DIR="${DIR}/mustache_calls"
OUT_DIR="${DIR}/merged_calls"
mkdir -p "${OUT_DIR}"

q_vals=(0.01 0.02 0.05)
onekb_ST=0.95
two_five_kb_ST=0.88
SLOP=5000

for q in ${q_vals[@]}; do
	echo "merging merged.mcool loops for q=${q}"

	consensus_bedpe="${OUT_DIR}/merged_consensus_q${q}.bedpe"
    consensus_bedpe_sorted="${OUT_DIR}/merged_consensus_q${q}.sorted.bedpe"
		
	# Start w 1kb loops
	cp "${IN_DIR}/merged_1kb_q${q}.tsv" "${consensus_bedpe}"

	# Filter out 2kb explained by 1kb, append
	pairToPair -a "${IN_DIR}/merged_2kb_q${q}.tsv" -b "${consensus_bedpe}" -slop $SLOP -type notboth >> "${consensus_bedpe}"
	
	# Filter out 5kb explained by 1 or 2kb, append
	pairToPair -a "${IN_DIR}/merged_5kb_q${q}.tsv" -b "${consensus_bedpe}" -slop $SLOP -type notboth >> "${consensus_bedpe}"

	# Sort by position
	sort -k1,1 -k2,2n "${consensus_bedpe}" > "${consensus_bedpe_sorted}"
	echo "$(wc -l ${consensus_bedpe_sorted}) loops merged for q=${q}"
done
