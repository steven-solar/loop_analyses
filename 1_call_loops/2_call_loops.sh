#!/usr/bin/env bash
# conda activate mustache

DATA_DIR="../data"

call_loops () {
	cool_file_path=$1
	res=$2
	pt=$3
	st=$4
	sz=$5
	d=$6
	cell_line=$7
	out=$8
	mustache \
		-p 16 \
		-f "${cool_file_path}" \
		-r "${res}" \
		-pt "${pt}" \
		-st "${st}" \
		-sz "${sz}" \
		-d "${d}" \
		-o "${out}"
	echo "Done calling ${cool_file_path} at ${res} resolution"
	q_vals=(0.01 0.02 0.5)
	for Q_VAL in ${q_vals[@]}; do
		echo "Filtering ${res} loops to FDR <= ${Q_VAL}"
		awk -v q=${Q_VAL} -F'\t' '$7 < q {print}' "${OUT_DIR}/${cell_line}_${res}.tsv" > "${OUT_DIR}/${cell_line}_${res}_q${Q_VAL}.tsv"
	done
    echo "Done filtering ${res} loops" 
}

MUSTACHE_PY=~/mustache/mustache/mustache.py
N_PROC=16
OUT_DIR="loops/mustache_calls"
mkdir -p "${OUT_DIR}" "${OUT_DIR}"/logs

# Mustache parameters
P_THRESHOLD=0.1   # q-value threshold (-pt)
SIGMA_ZERO=1.6    # sigma zero (-sz)
DISTANCE=2mb      # maximum distance (-d)

resolutions=("1kb" "2kb" "5kb")
cool_file_path="${DATA_DIR}/merged.mcool"
for res in ${resolutions[@]}; do
	if [[ "${res}" == "1kb" ]]; then
		ST=0.95
	else
		ST=0.88
	fi
	echo "Calling loops for merged.mcool at ${res} resolution"
	call_loops \
        "${cool_file_path}" \
        "${res}" \
        "${P_THRESHOLD}" \
        "${ST}" \
        "${SIGMA_ZERO}" \
        "${DISTANCE}" \
        "merged" \
        "${OUT_DIR}/merged_${res}.tsv" \
    &> "${OUT_DIR}/logs/2_call_${res}.out" &
done

wait $(jobs -p)
echo "done calling loops for merged.mcool"
