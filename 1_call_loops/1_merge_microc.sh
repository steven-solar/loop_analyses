#!/bin/bash

# use cooler to merge the 4 cell lines mcool files into a merge
# in_path is the path to the mcool files
# resolution is the resolution to merge at
# out_path is the path to the output merged mcool file
# example usage: ./merge_microc.sh /mnt/coldstorage/shares/Masahiro/microc/mcs/*_WT_10B.mcool data/merged.mcool
in_path="$1"
out_path="$2"
resolution="$3"
target_res="::/resolutions/${resolution}"

files=(${in_path})
files_with_res=( "${files[@]/%/$target_res}" )

out_cool="../data/merged.cool"
out_mcool="../data/merged.mcool"
echo "Merging ${#files_with_res[@]} files"
echo "${files_with_res[@]}"

cooler merge "${out_cool}" "${files_with_res[@]}"

echo "Zoomifying and balancing ${out_cool} -> ${out_mcool}"

cooler zoomify --balance -r 1000,2000,5000 -p 24 -o "${out_mcool}" "${out_cool}"
