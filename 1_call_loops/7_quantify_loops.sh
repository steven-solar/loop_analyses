#!/usr/bin/env bash

quantify_loops_pl() {
    cell_line=$1
    name=$2
    quant=$3
    echo "Processing ${name} loops for ${cell_line} at quant size ${quant}"
    python 7_quantify_loops_pl.py ${cell_line} ${name} ${quant} &> ../logs/7_quantify_loops_pl.${cell_line}.${name}.${quant}.out
}

cell_lines=("ESC" "EpiLC" "d4c7PGCLC" "GSC")
loop_names=("merged") #all_cell_lines

for cell_line in "${cell_lines[@]}"; do
    for name in "${loop_names[@]}"; do
        echo "Quantifying loops for ${cell_line} - ${name}"
        quantify_loops_pl ${cell_line} ${name} 3000 &
        quantify_loops_pl ${cell_line} ${name} 10000 &
    done
done

wait `jobs -p`

echo "All done!"
