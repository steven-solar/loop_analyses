#!/usr/bin/env python3
"""
compute_scales_from_loops.py

Derive genome-browser track y-axis scales from per-cell-line loop dataframes,
each with anchor1_<mark> / anchor2_<mark> raw signal columns (and matching
_norm columns, which are excluded here).

For each cell line: anchor1 and anchor2 values for a mark are pooled into a
single set of "all anchors", then mean/std computed over that pooled set.
max_value = mean + N_STD * std; min_value = 0.
global_scales.tsv then averages max_value for each mark across cell lines.
"""

import csv

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config -- edit these
# ---------------------------------------------------------------------------
LOOPS_PATH_PAT = "../2_epigenomics/loop_epigenomics/{cl}_q0.01_AbLE3kb.tsv"
CELL_LINES = ["ESC", "EpiLC", "d4c7PGCLC", "GSC"]
N_STD = 10.0
PER_CELL_OUT = "per_cell_line_scales.tsv"
GLOBAL_OUT = "global_scales.tsv"


def load_loops(cl):
    path = LOOPS_PATH_PAT.format(cl=cl)
    return pd.read_csv(path, sep="\t")


def detect_marks(df):
    """Find marks with both anchor1_<mark> and anchor2_<mark> raw (non-_norm) columns."""
    marks = []
    for col in df.columns:
        if not col.startswith("anchor1_") or col.endswith("_norm"):
            continue
        mark = col[len("anchor1_"):]
        a1_col, a2_col = f"anchor1_{mark}", f"anchor2_{mark}"
        if a2_col not in df.columns:
            print(f"[warn] {a1_col} has no matching {a2_col} -- skipping")
            continue
        if not pd.api.types.is_numeric_dtype(df[a1_col]) or not pd.api.types.is_numeric_dtype(df[a2_col]):
            print(f"[warn] {mark} columns not numeric -- skipping")
            continue
        marks.append(mark)
    return marks


def pooled_anchor_values(df, mark):
    """Concatenate anchor1_<mark> and anchor2_<mark> into one array, dropping non-finite."""
    a1 = df[f"anchor1_{mark}"].to_numpy(dtype=float)
    a2 = df[f"anchor2_{mark}"].to_numpy(dtype=float)
    pooled = np.concatenate([a1, a2])
    return pooled[np.isfinite(pooled)]


def compute_per_cell_line_scales(cell_lines, n_std, out_tsv):
    rows = []
    for cl in cell_lines:
        try:
            df = load_loops(cl)
        except FileNotFoundError as e:
            print(f"[warn] missing loop file for {cl}: {e}")
            continue

        marks = detect_marks(df)
        if not marks:
            print(f"[warn] no usable raw mark columns found for {cl}")
            continue

        for mark in marks:
            values = pooled_anchor_values(df, mark)
            if values.size == 0:
                print(f"[warn] no finite values for {cl}/{mark} -- skipping")
                continue
            mean = float(np.mean(values))
            std = float(np.std(values))
            max_value = mean + n_std * std
            # max_value = np.max(values)
            rows.append((cl, mark, mean, std, max_value, values.size))
            print(f"{cl}\t{mark}\tn={values.size}\tmean={mean:.3f}\tstd={std:.3f}\tmax={max_value:.3f}")

    with open(out_tsv, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["cell_line", "mark", "mean", "std", "max_value", "n_anchors"])
        w.writerows(rows)
    print(f"[done] wrote {out_tsv} ({len(rows)} rows)")
    return rows


def compute_global_scales(per_cell_line_rows, cell_lines_expected, out_tsv):
    """Average max_value across cell lines per mark -> out_tsv (min=0, max=avg_max)."""
    mark_values = {}
    for cl, mark, mean, std, max_value, n_anchors in per_cell_line_rows:
        mark_values.setdefault(mark, []).append(max_value)

    n_expected = len(cell_lines_expected)
    mark_avg = {}
    with open(out_tsv, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["mark", "min_value", "max_value", "n_cell_lines"])
        for mark, values in sorted(mark_values.items()):
            avg = float(np.mean(values))
            mark_avg[mark] = avg
            if len(values) < n_expected:
                print(f"[warn] {mark}: averaged over {len(values)}/{n_expected} cell lines")
            w.writerow([mark, 0.0, avg, len(values)])
    print(f"[done] wrote {out_tsv} ({len(mark_avg)} marks)")
    return mark_avg


if __name__ == "__main__":
    per_cell_line_rows = compute_per_cell_line_scales(CELL_LINES, N_STD, PER_CELL_OUT)
    if not per_cell_line_rows:
        raise ValueError("no scales computed -- check LOOPS_PATH_PAT and CELL_LINES")
    compute_global_scales(per_cell_line_rows, CELL_LINES, GLOBAL_OUT)