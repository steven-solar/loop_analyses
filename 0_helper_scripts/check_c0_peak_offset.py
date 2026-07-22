"""
check_c0_peak_offset.py

Question: are C0 anchors offset from true ChIP-seq peak summits (i.e. an
artifact of imprecise/off-center loop-anchor calling merging two nearby but
distinct peaks into one anchor bin), or do CTCF and CRE marks genuinely
coincide within the same anchor window?

Approach:
  1. For each anchor, compute distance from the anchor midpoint to the nearest
     peak summit for each mark (CTCF, Rad21, Stag1, Stag2 -> "CTCF-side"; K4me1,
     K4me3, K27ac -> "CRE-side"), using bedtools closest.
  2. Compare the distance distributions for C0 vs your pure CTCF-only and
     CRE-only reference clusters:
       - If C0's distance-to-CTCF-summit distribution looks like the pure CTCF
         cluster's (tight, near zero), and its distance-to-CRE-summit
         distribution looks like the pure CRE cluster's, then both features are
         genuinely centered at the anchor -> real coincidence, not noise.
       - If C0's distances are systematically larger/more variable than the
         pure clusters' distances to their respective marks, that supports an
         off-center-calling / resolution-limited explanation.
  3. Within C0 specifically, scatter distance-to-nearest-CTCF-summit vs
     distance-to-nearest-CRE-summit per anchor. If most anchors sit near the
     origin (both small), that's coincidence. If anchors spread out along one
     axis (small CTCF distance / large CRE distance, or vice versa, split
     roughly 50/50), that suggests the anchor is picking up two nearby but
     spatially separate features -> more consistent with an artifact.

Requires: pybedtools (uses your existing pybedtools/bedtools install).
Peak files should be BED/narrowPeak per mark. If narrowPeak (10-col MACS2
format), the summit (col 10, 0-based offset from peak start) is used as the
point feature; otherwise peak midpoint is used.

Usage:
    python check_c0_peak_offset.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pybedtools

plt.rcParams['svg.fonttype'] = 'none'

# ----------------------------- CONFIG ------------------------------------ #

ANCHOR_TABLE = "anchors_q0.01_3kb_annotated.tsv"
CLUSTER_COL = "cluster"
TARGET_CLUSTER = 0
CTCF_REF_CLUSTER = 1
CRE_REF_CLUSTER = 2

# anchor coordinate columns -- adjust to your table's naming
CHROM_COL = "chrom"
START_COL = "start"
END_COL = "end"
ANCHOR_ID_COL = "anchor_id"

# peak files per mark -- point these at your actual ChIP peak calls
# (narrowPeak format assumed; plain BED with midpoint fallback also works)
PEAK_FILES = {
    "CTCF":  "peaks/CTCF_peaks.narrowPeak",
    "Rad21": "peaks/Rad21_peaks.narrowPeak",
    "Stag1": "peaks/Stag1_peaks.narrowPeak",
    "Stag2": "peaks/Stag2_peaks.narrowPeak",
    "K4me1": "peaks/K4me1_peaks.narrowPeak",
    "K4me3": "peaks/K4me3_peaks.narrowPeak",
    "K27ac": "peaks/K27ac_peaks.narrowPeak",
}

CTCF_MARKS = ["CTCF", "Rad21", "Stag1", "Stag2"]
CRE_MARKS = ["K4me1", "K4me3", "K27ac"]

TMP_DIR = "/mnt/md0/tmp_bedtools"  # redirect pybedtools tmp to avoid /tmp space issues
OUT_PREFIX = "c0_peak_offset"

# --------------------------------------------------------------------------- #


def setup_bedtools_tmp():
    os.makedirs(TMP_DIR, exist_ok=True)
    pybedtools.set_tempdir(TMP_DIR)


def peaks_to_summit_bed(peak_path):
    """
    Build a BedTool of single-bp summit points from a peak file.
    Uses narrowPeak summit offset (col 10) if present and looks valid,
    otherwise falls back to the peak midpoint.
    """
    df = pd.read_csv(peak_path, sep="\t", header=None)
    is_narrowpeak = df.shape[1] >= 10

    rows = []
    for _, r in df.iterrows():
        chrom, start, end = r[0], int(r[1]), int(r[2])
        if is_narrowpeak:
            summit_offset = int(r[9])
            summit = start + summit_offset if summit_offset >= 0 else (start + end) // 2
        else:
            summit = (start + end) // 2
        rows.append((chrom, summit, summit + 1))

    summit_df = pd.DataFrame(rows, columns=["chrom", "start", "end"])
    return pybedtools.BedTool.from_dataframe(summit_df).sort()


def anchor_midpoints_bed(anchor_df):
    mids = anchor_df.copy()
    mids["mid_start"] = ((mids[START_COL] + mids[END_COL]) // 2).astype(int)
    mids["mid_end"] = mids["mid_start"] + 1
    bed_df = mids[[CHROM_COL, "mid_start", "mid_end", ANCHOR_ID_COL]]
    bed_df.columns = ["chrom", "start", "end", "name"]
    return pybedtools.BedTool.from_dataframe(bed_df).sort(), mids


def closest_distance(anchor_bed, summit_bed):
    """
    bedtools closest -d: returns, per anchor, the nearest summit and the
    distance (bp). Distance of -1 means no feature on that chromosome.
    """
    closest = anchor_bed.closest(summit_bed, d=True)
    cols = ["chrom", "start", "end", "anchor_id",
            "s_chrom", "s_start", "s_end", "distance"]
    df = closest.to_dataframe(names=cols)
    df = df[["anchor_id", "distance"]].rename(columns={"distance": "dist"})
    df.loc[df["dist"] < 0, "dist"] = np.nan  # no feature found on chrom
    return df


def compute_all_distances(anchor_df):
    """Return anchor_df with one distance column added per mark."""
    setup_bedtools_tmp()
    anchor_bed, mids = anchor_midpoints_bed(anchor_df)

    out = mids.set_index(ANCHOR_ID_COL)
    for mark, path in PEAK_FILES.items():
        if not os.path.exists(path):
            print(f"[warn] peak file missing for {mark}: {path}, skipping")
            continue
        summit_bed = peaks_to_summit_bed(path)
        dist_df = closest_distance(anchor_bed, summit_bed).set_index("anchor_id")
        out[f"dist_{mark}"] = dist_df["dist"]
        print(f"  computed distances to {mark} ({path})")

    return out.reset_index()


def composite_side_distance(df, marks, colname):
    """Min distance across a group of marks (nearest feature of that 'side')."""
    cols = [f"dist_{m}" for m in marks if f"dist_{m}" in df.columns]
    df[colname] = df[cols].min(axis=1, skipna=True)
    return df


def plot_distance_comparison(df, cluster_col):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    order = ["CTCF ref", "CRE ref", "C0"]
    palette = {"CTCF ref": "tab:blue", "CRE ref": "tab:green", "C0": "tab:brown"}

    import seaborn as sns
    sns.boxplot(data=df, x=cluster_col, y="dist_ctcf_side", order=order,
                palette=palette, ax=axes[0], showfliers=False)
    axes[0].set_title("Distance to nearest CTCF-side summit")
    axes[0].set_ylabel("distance (bp)")

    sns.boxplot(data=df, x=cluster_col, y="dist_cre_side", order=order,
                palette=palette, ax=axes[1], showfliers=False)
    axes[1].set_title("Distance to nearest CRE-side summit")
    axes[1].set_ylabel("distance (bp)")

    fig.suptitle("Anchor-to-peak-summit distance: C0 vs pure reference clusters")
    fig.tight_layout()
    fig.savefig(f"{OUT_PREFIX}_boxplot.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT_PREFIX}_boxplot.svg")


def plot_c0_scatter(c0_df):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(c0_df["dist_ctcf_side"], c0_df["dist_cre_side"],
               s=8, alpha=0.35, color="tab:brown", edgecolor="none")
    lim = np.nanpercentile(
        pd.concat([c0_df["dist_ctcf_side"], c0_df["dist_cre_side"]]), 95
    )
    ax.set_xlim(-lim * 0.05, lim)
    ax.set_ylim(-lim * 0.05, lim)
    ax.axline((0, 0), slope=1, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("distance to nearest CTCF-side summit (bp)")
    ax.set_ylabel("distance to nearest CRE-side summit (bp)")
    ax.set_title("C0 anchors: CTCF-side vs CRE-side summit distance")
    fig.savefig(f"{OUT_PREFIX}_c0_scatter.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT_PREFIX}_c0_scatter.svg")


def main():
    anchors = pd.read_csv(ANCHOR_TABLE, sep="\t")
    keep_clusters = [TARGET_CLUSTER, CTCF_REF_CLUSTER, CRE_REF_CLUSTER]
    anchors = anchors[anchors[CLUSTER_COL].isin(keep_clusters)].copy()

    dist_df = compute_all_distances(anchors)

    merged = anchors.merge(dist_df[[ANCHOR_ID_COL] +
                                    [c for c in dist_df.columns if c.startswith("dist_")]],
                            on=ANCHOR_ID_COL, how="left")

    merged = composite_side_distance(merged, CTCF_MARKS, "dist_ctcf_side")
    merged = composite_side_distance(merged, CRE_MARKS, "dist_cre_side")

    label_map = {
        TARGET_CLUSTER: "C0",
        CTCF_REF_CLUSTER: "CTCF ref",
        CRE_REF_CLUSTER: "CRE ref",
    }
    merged["cluster_label"] = merged[CLUSTER_COL].map(label_map)

    plot_distance_comparison(merged, "cluster_label")

    c0 = merged[merged["cluster_label"] == "C0"]
    plot_c0_scatter(c0)

    print("\nMedian distances (bp):")
    print(merged.groupby("cluster_label")[["dist_ctcf_side", "dist_cre_side"]]
          .median())

    print("""
How to read this:
- Compare C0's median dist_ctcf_side to the CTCF-ref cluster's median
  dist_ctcf_side: if they're similar (both near 0), C0 anchors sit right on top
  of real CTCF summits, same as bona fide CTCF anchors. Same logic for
  dist_cre_side vs the CRE-ref cluster.
- If C0's distances are systematically larger than either reference cluster's,
  that anchor's "true" feature may not actually be centered where the anchor
  was called -> supports a calling-resolution artifact.
- In the C0-only scatter, points near the origin (both distances small) support
  genuine coincidence of CTCF and CRE marks at the same anchor. Points spread
  along one axis (one distance small, the other large) suggest the anchor is
  capturing two distinct, nearby-but-separate features rather than a true
  hybrid site.
""")


if __name__ == "__main__":
    main()