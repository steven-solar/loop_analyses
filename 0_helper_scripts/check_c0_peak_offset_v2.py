"""
check_c0_peak_offset.py  (v2)

Changes from v1
───────────────
• CTCF_REF_CLUSTERS / CRE_REF_CLUSTERS are full groups; every cluster in
  each list is assigned the same reference label for plotting.
• Anchors carry a CELL_LINE_COL column; peak files are resolved per cell
  line via CELL_LINE_PREFIX + PEAK_FILE_TEMPLATES so each anchor is
  compared against the right experiment.
• Anchors expose a single MID_COL midpoint coordinate (bp); no start/end
  midpoint arithmetic is performed.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pybedtools

plt.rcParams["svg.fonttype"] = "none"

# ───────────────────────────── CONFIG ──────────────────────────────────── #

ANCHOR_TABLE  = "/mnt/md0/sjsolar/loop_pred/loop_analyses/3_anchor_clustering/clustering/anchors_q0.01_3kb_pca_clustered_umap.tsv"
CLUSTER_COL   = "cluster"
CELL_LINE_COL = "cell_line"   # column in anchor table giving the cell line

TARGET_CLUSTER    = 0              # single cluster treated as "C0"
CTCF_REF_CLUSTERS = [1, 3, 7]     # all mapped to one "CTCF ref" group
CRE_REF_CLUSTERS  = [2, 4, 6]     # all mapped to one "CRE ref" group

# Anchor coordinate columns
CHROM_COL     = "chrom"
MID_COL       = "mid"        # single midpoint (bp) – no start/end needed
ANCHOR_ID_COL = "anchor_id"

# Cell-line → filename prefix
CELL_LINE_PREFIX = {
    "ESC":       "BDF121",
    "EpiLC":     "BDF121",
    "d4c7PGCLC": "BDF121",
    "GSC":       "AAG",
}

NARROW_FOLDER = (
    "/mnt/florenceshares/Masahiro/Saitoulab_data/"
    "saitoulab_chip/peaks/macs2/narrow"
)
BROAD_FOLDER = (
    "/mnt/florenceshares/Masahiro/Saitoulab_data/"
    "saitoulab_chip/peaks/macs2/broad"
)

# Placeholders: {narrow}, {broad}, {cell_line}, {prefix}
PEAK_FILE_TEMPLATES = {
    "CTCF":  "{narrow}/{cell_line}_{prefix}_CTCF_pool_peaks.narrowPeak",
    "RAD21": "{narrow}/{cell_line}_{prefix}_RAD21_pool_peaks.narrowPeak",
    "STAG1": "{narrow}/{cell_line}_{prefix}_STAG1_pool_peaks.narrowPeak",
    "STAG2": "{narrow}/{cell_line}_{prefix}_STAG2_pool_peaks.narrowPeak",
    "K4me1": "{narrow}/{cell_line}_{prefix}_K4me1_pool_peaks.narrowPeak",
    "K27ac": "{narrow}/{cell_line}_{prefix}_K27ac_pool_peaks.narrowPeak",
}

# Keys must match PEAK_FILE_TEMPLATES keys exactly
CTCF_MARKS = ["CTCF", "RAD21", "STAG1", "STAG2"]
CRE_MARKS  = ["K4me1", "K27ac"]

TMP_DIR    = "/mnt/md0/tmp_bedtools"
OUT_PREFIX = "c0/c0_peak_offset"

# ─────────────────────────── HELPERS ───────────────────────────────────── #


def setup_bedtools_tmp():
    os.makedirs(TMP_DIR, exist_ok=True)
    pybedtools.set_tempdir(TMP_DIR)


def resolve_peak_files(cell_line):
    """Return {mark: resolved_path} for *cell_line*."""
    prefix = CELL_LINE_PREFIX[cell_line]
    return {
        mark: tmpl.format(
            narrow=NARROW_FOLDER,
            broad=BROAD_FOLDER,
            cell_line=cell_line,
            prefix=prefix,
        )
        for mark, tmpl in PEAK_FILE_TEMPLATES.items()
    }


def peaks_to_summit_bed(peak_path):
    """
    Build a single-bp summit BedTool from a narrowPeak or plain BED file.
    narrowPeak col 10 (0-based summit offset from peak start) is used when
    available; otherwise the peak midpoint is used.
    """
    df = pd.read_csv(peak_path, sep="\t", header=None)
    is_narrowpeak = df.shape[1] >= 10

    rows = []
    for _, r in df.iterrows():
        chrom, start, end = r[0], int(r[1]), int(r[2])
        if is_narrowpeak:
            offset = int(r[9])
            summit = start + offset if offset >= 0 else (start + end) // 2
        else:
            summit = (start + end) // 2
        rows.append((chrom, summit, summit + 1))

    summit_df = pd.DataFrame(rows, columns=["chrom", "start", "end"])
    return pybedtools.BedTool.from_dataframe(summit_df).sort()


def anchor_points_bed(anchor_df):
    """
    Build a single-bp BedTool directly from the pre-computed MID_COL.
    The anchor midpoint is treated as a 1-bp interval [mid, mid+1).
    """
    mid = anchor_df[MID_COL].astype(int)
    bed_df = pd.DataFrame({
        "chrom": anchor_df[CHROM_COL],
        "start": mid,
        "end":   mid + 1,
        "name":  anchor_df[ANCHOR_ID_COL],
    })
    return pybedtools.BedTool.from_dataframe(bed_df).sort()


def closest_distance(anchor_bed, summit_bed):
    """
    Run bedtools closest -d; return DataFrame[anchor_id, dist].
    Distance < 0 (no feature on chromosome) is replaced with NaN.
    Ties are broken by keeping the first hit.
    """
    closest = anchor_bed.closest(summit_bed, d=True)
    # A(4 cols) + B(3 cols) + distance(1) = 8 cols
    col_names = [
        "chrom", "start", "end", "anchor_id",
        "s_chrom", "s_start", "s_end", "distance",
    ]
    df = closest.to_dataframe(names=col_names, dtype={"distance": float})
    df = (
        df[["anchor_id", "distance"]]
        .rename(columns={"distance": "dist"})
        .drop_duplicates(subset="anchor_id", keep="first")
    )
    df.loc[df["dist"] < 0, "dist"] = np.nan
    return df


# ─────────────────── CELL-LINE-AWARE DISTANCE COMPUTATION ──────────────── #


def compute_all_distances(anchor_df):
    """
    Group anchors by CELL_LINE_COL, load the matching peak files for each
    group, and compute the nearest-summit distance per mark.

    Returns a DataFrame with [ANCHOR_ID_COL, dist_<mark>, ...].
    """
    setup_bedtools_tmp()
    pieces = []

    for cell_line, grp in anchor_df.groupby(CELL_LINE_COL, sort=False):
        print(f"\n── cell line: {cell_line}  ({len(grp):,} anchors)")
        peak_files = resolve_peak_files(cell_line)
        anchor_bed = anchor_points_bed(grp)

        rec = {ANCHOR_ID_COL: grp[ANCHOR_ID_COL].tolist()}

        for mark, path in peak_files.items():
            if not os.path.exists(path):
                print(f"   [warn] missing peak file for {mark}: {path}")
                rec[f"dist_{mark}"] = [np.nan] * len(grp)
                continue

            summit_bed = peaks_to_summit_bed(path)
            dist_df    = closest_distance(anchor_bed, summit_bed)
            id2dist    = dist_df.set_index("anchor_id")["dist"]

            # reindex preserves order and fills missing anchor IDs with NaN
            rec[f"dist_{mark}"] = id2dist.reindex(
                grp[ANCHOR_ID_COL]
            ).values.tolist()
            print(f"   {mark}: done")

        pieces.append(pd.DataFrame(rec))

    return pd.concat(pieces, ignore_index=True)


# ─────────────────────── POST-PROCESSING ───────────────────────────────── #


def composite_side_distance(df, marks, colname):
    """
    Add *colname* = minimum distance across all marks in the group.
    Anchors missing every mark in the group get NaN.
    """
    cols = [f"dist_{m}" for m in marks if f"dist_{m}" in df.columns]
    if not cols:
        print(f"[warn] no distance columns found for marks {marks}; "
              f"{colname} will be NaN")
        df[colname] = np.nan
    else:
        df[colname] = df[cols].min(axis=1, skipna=True)
    return df


def build_label_map():
    """
    Map every cluster ID to its group label.
    All clusters in CTCF_REF_CLUSTERS → "CTCF ref"
    All clusters in CRE_REF_CLUSTERS  → "CRE ref"
    TARGET_CLUSTER                    → "C0"
    """
    lmap = {TARGET_CLUSTER: "C0"}
    for c in CTCF_REF_CLUSTERS:
        lmap[c] = "CTCF ref"
    for c in CRE_REF_CLUSTERS:
        lmap[c] = "CRE ref"
    return lmap


# ──────────────────────────── PLOTS ────────────────────────────────────── #


def plot_distance_comparison(merged):
    import seaborn as sns

    order   = ["CTCF ref", "CRE ref", "C0"]
    palette = {"CTCF ref": "tab:blue", "CRE ref": "tab:green", "C0": "tab:brown"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, ycol, title in zip(
        axes,
        ["dist_ctcf_side", "dist_cre_side"],
        ["Distance to nearest CTCF-side summit",
         "Distance to nearest CRE-side summit"],
    ):
        sns.boxplot(
            data=merged, x="cluster_label", y=ycol,
            order=order, palette=palette,
            ax=ax, showfliers=False,
        )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("distance (bp)")

    fig.suptitle("Anchor-to-peak-summit distance: C0 vs reference clusters")
    fig.tight_layout()
    out = f"{OUT_PREFIX}_boxplot.svg"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def plot_c0_scatter(c0_df):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        c0_df["dist_ctcf_side"], c0_df["dist_cre_side"],
        s=8, alpha=0.35, color="tab:brown", edgecolor="none",
    )
    combined = pd.concat(
        [c0_df["dist_ctcf_side"], c0_df["dist_cre_side"]]
    ).dropna()
    lim = float(np.nanpercentile(combined, 95)) if len(combined) else 1000
    ax.set_xlim(-lim * 0.05, lim)
    ax.set_ylim(-lim * 0.05, lim)
    ax.axline((0, 0), slope=1, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("distance to nearest CTCF-side summit (bp)")
    ax.set_ylabel("distance to nearest CRE-side summit (bp)")
    ax.set_title("C0 anchors: CTCF-side vs CRE-side summit distance")
    out = f"{OUT_PREFIX}_c0_scatter.svg"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ─────────────────────────── MAIN ──────────────────────────────────────── #


def main():
    os.makedirs(os.path.dirname(OUT_PREFIX) or ".", exist_ok=True)

    anchors = pd.read_csv(ANCHOR_TABLE, sep="\t")

    keep_clusters = [TARGET_CLUSTER] + CTCF_REF_CLUSTERS + CRE_REF_CLUSTERS
    anchors = anchors[anchors[CLUSTER_COL].isin(keep_clusters)].copy()
    print(
        f"Anchors retained: {len(anchors):,}  "
        f"clusters={keep_clusters}  "
        f"cell lines={sorted(anchors[CELL_LINE_COL].unique())}"
    )

    # ── per-cell-line distance computation ──────────────────────────────── #
    dist_df = compute_all_distances(anchors)

    dist_cols = [c for c in dist_df.columns if c.startswith("dist_")]
    merged = anchors.merge(
        dist_df[[ANCHOR_ID_COL] + dist_cols],
        on=ANCHOR_ID_COL,
        how="left",
    )

    # ── composite side distances ─────────────────────────────────────────── #
    merged = composite_side_distance(merged, CTCF_MARKS, "dist_ctcf_side")
    merged = composite_side_distance(merged, CRE_MARKS,  "dist_cre_side")

    # ── cluster → group label ────────────────────────────────────────────── #
    label_map = build_label_map()
    merged["cluster_label"] = merged[CLUSTER_COL].map(label_map)

    unknown = merged["cluster_label"].isna().sum()
    if unknown:
        print(f"[warn] {unknown} anchors could not be labelled "
              f"(unexpected cluster IDs)")

    # ── plots ────────────────────────────────────────────────────────────── #
    plot_distance_comparison(merged)
    plot_c0_scatter(merged[merged["cluster_label"] == "C0"])

    # ── summary stats ────────────────────────────────────────────────────── #
    print("\nMedian distances (bp) – by group label:")
    print(
        merged.groupby("cluster_label")[["dist_ctcf_side", "dist_cre_side"]]
        .median()
        .to_string()
    )

    print("\nMedian distances (bp) – by individual cluster:")
    print(
        merged.groupby(CLUSTER_COL)[["dist_ctcf_side", "dist_cre_side"]]
        .median()
        .to_string()
    )

    print("""
Interpretation guide
────────────────────
• dist_ctcf_side (C0) ≈ dist_ctcf_side (CTCF ref) → C0 anchors sit on
  genuine CTCF summits, same as pure CTCF anchors. Mirror logic for CRE side.

• C0 distances systematically larger than either reference → the 'true'
  peak feature is not centred at the called anchor → likely a
  resolution/merging artefact.

• C0 scatter near the origin (both distances small) → genuine co-occupancy
  of CTCF and CRE marks at the same anchor.
  Points spread along one axis (one small, one large) → the anchor is
  capturing two distinct, spatially-separate features → artefactual hybrid.
""")


if __name__ == "__main__":
    main()
