"""
check_c0_bimodality.py

Question: is cluster C0 a genuine CTCF+CRE hybrid anchor class, or is it actually
a mixture of two separable subpopulations (pure-CTCF-like anchors and pure-CRE-like
anchors) that got merged by the clustering / averaged out in meta-profiles?

Approach:
  1. Scatter PC1 (CRE) vs PC2 (CTCF) for C0 anchors only, with marginal densities.
     A genuine hybrid class -> single blob roughly centered between the pure classes.
     A disguised mixture    -> two separable lobes / bimodal marginals.
  2. Same scatter but using composite raw z-scores (mean of CRE marks vs mean of
     CTCF/cohesin marks) instead of PCs, since PCs can mix in other variance.
  3. Quantify bimodality per axis with a Gaussian Mixture Model: fit k=1 vs k=2
     components and compare BIC. Lower BIC for k=2 (by a meaningful margin) is
     evidence of a mixture; k=1 winning supports a genuine single population.
  4. If the `diptest` package is available, also run Hartigan's dip test on each
     axis (a formal unimodality test) as a cross-check on the GMM result.
  5. Repeat steps 1-3 for a couple of "pure" reference clusters (your CTCF-only
     and CRE-only clusters) as a sanity check / baseline for what "genuinely
     unimodal" looks like in this same feature space.

Usage:
    python check_c0_bimodality.py

Edit the CONFIG block below to match your actual file paths / column names.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture

plt.rcParams['svg.fonttype'] = 'none'

# ----------------------------- CONFIG ------------------------------------ #

ANCHOR_TABLE = "/mnt/md0/sjsolar/loop_pred/loop_analyses/3_anchor_clustering/clustering/anchors_q0.01_3kb_pca_clustered_umap.tsv"   # per-anchor table w/ PCs + z-scores
CLUSTER_COL  = "cluster"                            # Leiden cluster label column
TARGET_CLUSTER = 0                                  # the C0 cluster you're checking

# clusters to use as "pure" reference baselines -- adjust to your actual mapping
CTCF_REF_CLUSTERS = [1,3,7]     # e.g. "CTCF (ESC, EpiLC)"
CRE_REF_CLUSTERS  = [2,4,6]     # e.g. "CRE (ESC, EpiLC)"

PC_CRE_COL  = "PC1"
PC_CTCF_COL = "PC2"

# raw z-score columns that feed each composite score
# adjust suffixes (_ESC / _EpiLC / etc.) to whichever cell-line columns you want,
# or point these at pooled/joint z-score columns if you have them
CRE_FEATURES  = ["K4me1_clip", "K4me3_clip", "K27ac_clip"]
CTCF_FEATURES = ["CTCF_clip", "Stag1_clip", "Stag2_clip", "Rad21_clip"]

OUT_PREFIX = "c0/c0_bimodality_check"

# --------------------------------------------------------------------------- #


def composite_scores(df):
    """Mean z-score across CRE marks and across CTCF/cohesin marks, per anchor."""
    cre_cols = [c for c in CRE_FEATURES if c in df.columns]
    ctcf_cols = [c for c in CTCF_FEATURES if c in df.columns]
    missing = set(CRE_FEATURES + CTCF_FEATURES) - set(cre_cols + ctcf_cols)
    if missing:
        print(f"[warn] missing columns, skipping: {missing}")
    df = df.copy()
    df["cre_score"] = df[cre_cols].mean(axis=1)
    df["ctcf_score"] = df[ctcf_cols].mean(axis=1)
    return df


def gmm_bimodality(x, max_components=2, n_init=10, random_state=0):
    """
    Fit 1- and 2-component 1D GMMs, return BIC for each and the fitted 2-component
    model. A large positive (BIC_1 - BIC_2) favors a two-population mixture.
    """
    x = np.asarray(x).reshape(-1, 1)
    x = x[~np.isnan(x).ravel()]
    bics = {}
    models = {}
    for k in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=k, n_init=n_init, random_state=random_state)
        gmm.fit(x)
        bics[k] = gmm.bic(x)
        models[k] = gmm
    return bics, models


def try_dip_test(x):
    """Optional: Hartigan's dip test if `diptest` package is installed."""
    try:
        import diptest
    except ImportError:
        return None
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    dip, pval = diptest.diptest(x)
    return dip, pval


def report_axis(name, x):
    print(f"\n--- {name} ---")
    bics, models = gmm_bimodality(x)
    print(f"  BIC k=1: {bics[1]:.1f}   BIC k=2: {bics[2]:.1f}   "
          f"delta (k1-k2): {bics[1] - bics[2]:.1f}  "
          f"({'favors 2 components' if bics[2] < bics[1] else 'favors 1 component'})")
    dip_result = try_dip_test(x)
    if dip_result is not None:
        dip, pval = dip_result
        print(f"  Hartigan dip stat: {dip:.4f}, p={pval:.4f} "
              f"({'reject unimodality' if pval < 0.05 else 'fail to reject unimodality'})")
    else:
        print("  (install `diptest` via pip for a formal unimodality test: "
              "pip install diptest --break-system-packages)")
    return bics, models


def plot_joint(df, x_col, y_col, title, fname):
    g = sns.jointplot(
        data=df, x=x_col, y=y_col, kind="scatter",
        joint_kws=dict(s=8, alpha=0.35, edgecolor="none"),
        marginal_kws=dict(bins=60, kde=True),
        height=6,
    )
    g.fig.suptitle(title, y=1.02)
    g.savefig(fname, bbox_inches="tight")
    plt.close(g.fig)
    print(f"  saved {fname}")


def analyze_cluster(df, cluster_name, cluster_ids, label):
    sub = df[df[CLUSTER_COL].isin(cluster_ids)]
    if len(sub) < 20:
        print(f"[warn] cluster {cluster_name} ({label}) has only {len(sub)} anchors, "
              f"bimodality stats will be unreliable")

    print(f"\n=== {label} (cluster {cluster_name}, n={len(sub)}) ===")

    # PC space
    plot_joint(sub, PC_CRE_COL, PC_CTCF_COL,
               f"{label}: PC1 (CRE) vs PC2 (CTCF)",
               f"{OUT_PREFIX}_{cluster_name}_pcs.svg")
    report_axis(f"{label} PC1", sub[PC_CRE_COL])
    report_axis(f"{label} PC2", sub[PC_CTCF_COL])

    # raw composite score space
    plot_joint(sub, "cre_score", "ctcf_score",
               f"{label}: composite CRE score vs composite CTCF score",
               f"{OUT_PREFIX}_{cluster_name}_rawscores.svg")
    report_axis(f"{label} cre_score", sub["cre_score"])
    report_axis(f"{label} ctcf_score", sub["ctcf_score"])

    return sub


def main():
    df = pd.read_csv(ANCHOR_TABLE, sep="\t")
    df = composite_scores(df)

    c0 = analyze_cluster(df, "C0", [TARGET_CLUSTER], "C0 (candidate hybrid)")
    ctcf_ref = analyze_cluster(df, "CTCF_ref", CTCF_REF_CLUSTERS, "CTCF reference cluster")
    cre_ref = analyze_cluster(df, "CRE_ref", CRE_REF_CLUSTERS, "CRE reference cluster")

    # side-by-side comparison figure: C0 vs the two pure reference clusters,
    # all in the same raw-score space so you can eyeball whether C0 sits as a
    # single blob between them or overlaps/splits into two sub-lobes
    fig, ax = plt.subplots(figsize=(6, 6))
    for sub, label, color in [
        (ctcf_ref, "CTCF ref", "tab:blue"),
        (cre_ref, "CRE ref", "tab:green"),
        (c0, "C0", "tab:brown"),
    ]:
        ax.scatter(sub["cre_score"], sub["ctcf_score"], s=6, alpha=0.3,
                   label=f"{label} (n={len(sub)})", color=color, edgecolor="none")
    ax.set_xlabel("composite CRE score (mean z)")
    ax.set_ylabel("composite CTCF score (mean z)")
    ax.legend(markerscale=3)
    ax.set_title("C0 vs pure CTCF / CRE reference clusters")
    fig.savefig(f"{OUT_PREFIX}_overlay.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {OUT_PREFIX}_overlay.svg")

    print("""
How to read this:
- If C0's scatter in the overlay is one blob sitting between the CTCF-ref and
  CRE-ref blobs (and doesn't overlap much with either), that's consistent with
  a genuine intermediate/hybrid population.
- If C0's scatter instead looks like two clouds, one sitting on top of the
  CTCF-ref cloud and another on top of the CRE-ref cloud, that's evidence C0 is
  a merged mixture rather than a true hybrid class.
- The GMM BIC delta and dip test give you a number to back up the visual call:
  a 2-component GMM winning clearly (BIC drop of ~10+) plus a significant dip
  test on the same axis is reasonably strong evidence for a mixture.
""")


if __name__ == "__main__":
    main()
