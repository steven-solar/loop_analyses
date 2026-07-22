#!/usr/bin/env bash
# conda activate higlass_env

chromsizes="../data/mm10.chromsizes"
project_name="SJSOLAR_SPERM_LOOP_PREDICTION_v2_07032026"

for tsv_file in loops/quantified_calls/*_q0.01_AbLE3kb.tsv; do
    echo "Processing $tsv_file"
    name=$(basename "$tsv_file" .tsv)

    # ── existing BEDPE pipeline ───────────────────────────────────────────────
    bedpe_file="${tsv_file%.tsv}.bedpe"
    awk -F'\t' -v OFS='\t' 'NR>1 {print $1, $2, $3, $4, $5, $6, $7, $13}' "$tsv_file" > "${bedpe_file}"
    output_db="${bedpe_file%.*}.bedpedb"
    higlass_name="/mnt/md1/DataRepository/HiGlass/${name}.bedpedb"
    higlass_bn=$(basename "${higlass_name}")
    clodius aggregate bedpe \
        --chromsizes-filename "${chromsizes}" \
        --output-file "${output_db}" \
        "${bedpe_file}"
    cp "${output_db}" "${higlass_name}"

    # ── NEW: anchors as 1D BED ────────────────────────────────────────────────
    bed_file="${tsv_file%.tsv}_anchors.bed"

    # left anchors + right anchors → single BED, sorted
    awk -F'\t' -v OFS='\t' 'NR>1 {
        print $2, $3, $4, "left_anchor";
        print $5, $6, $7, "right_anchor"
    }' "$tsv_file" \
    | sort -k1,1 -k2,2n \
    > "${bed_file}"

    # aggregate for HiGlass
    bed_db="${bed_file%.bed}.beddb"
    clodius aggregate bedfile \
        --chromsizes-filename "${chromsizes}" \
        --output-file "${bed_db}" \
        "${bed_file}"

    higlass_bed_name="/mnt/md1/DataRepository/HiGlass/${name}_anchors.beddb"
    higlass_bed_bn=$(basename "${higlass_bed_name}")
    cp "${bed_db}" "${higlass_bed_name}"

    # ingest as 1D track
    cd /mnt/md1/DataRepository/HiGlass/
    higlass-manage ingest \
        --no-upload \
        --filetype beddb \
        --datatype bedlike \
        --project-name "${project_name}" \
        "${higlass_bed_bn}"

    # ingest BEDPE
    higlass-manage ingest \
        --no-upload \
        --filetype bed2ddb \
        --datatype 2d-rectangle-domains \
        --project-name "${project_name}" \
        "${higlass_bn}"

    echo "Done processing $tsv_file"
    cd /mnt/md0/sjsolar/loop_pred/loop_analyses/1_call_loops
done
