#!/usr/bin/env python3
"""
Ridge regression on loop AbLE_score from epigenomic (_clip_) features and/or
loop-size features. Produces waterfall feature-importance plots (RdBu-colored,
in-sample R2 in the title) and a per-loop-class R2 bar chart segmented by
CTCF-CTCF / CRE-CRE / PRC-PRC.

Usage:
    python ridge_loop_regression.py loops.tsv --outdir figs

Assumes a tab-separated table with the columns from make_input_filt.py output:
AbLE_score, size, loop_name (values like "CTCF-CTCF & ..." or exactly
"CTCF-CTCF"/"CRE-CRE"/"PRC-PRC"), and *_clip_left / *_clip_right epigenomic
feature columns.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({
    'font.size':     6,
    'axes.linewidth': 0.25,
    'axes.labelsize': 6,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'svg.fonttype':  'none',       # keeps text as real text, not outlines, in Illustrator
})

import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "AbLE_score"
LOOP_CLASSES = ["CTCF-CTCF", "CRE-CRE", "PRC-PRC"]
ALPHAS = np.logspace(-3, 4, 50)


def log(msg: str) -> None:
    print(f"[ridge_loop_regression] {msg}", file=sys.stderr)


def load_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    log(f"loaded {df.shape[0]} rows x {df.shape[1]} cols from {path}")
    assert TARGET in df.columns, f"missing target column {TARGET!r}"
    assert "size" in df.columns, "missing 'size' column"
    assert "loop_name" in df.columns, "missing 'loop_name' column"
    return df


def clip_feature_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.endswith("_clip_left") or c.endswith("_clip_right")]
    assert cols, "no *_clip_left / *_clip_right columns found"
    log(f"found {len(cols)} clip feature columns")
    return sorted(cols)


def add_size_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    size = df["size"].astype(float)
    assert (size > 0).all(), "non-positive loop size present; log/inv undefined"
    df["log_size"] = np.log(size)
    df["inv_size"] = 1.0 / size
    return df


def assign_loop_class(df: pd.DataFrame) -> pd.Series:
    # loop_name may embed the class token (e.g. "CTCF-CTCF & ...") or equal it
    # exactly; match against the known classes and leave everything else NaN.
    cls = pd.Series(np.nan, index=df.index, dtype=object)
    for c in LOOP_CLASSES:
        cls[df["loop_name"].astype(str).str.contains(c, regex=False)] = c
    return cls


def build_matrix(df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    sub = df[feature_cols + [TARGET]].apply(pd.to_numeric, errors="coerce")
    n_before = len(sub)
    sub = sub.dropna()
    n_dropped = n_before - len(sub)
    if n_dropped:
        log(f"dropped {n_dropped}/{n_before} rows with NaNs in {len(feature_cols)} features + target")
    X = sub[feature_cols].to_numpy()
    y = sub[TARGET].to_numpy()
    y = np.log10(y)
    return X, y, sub.index.to_list()


def fit_ridge(X: np.ndarray, y: np.ndarray):
    assert len(X) > len(set(X.shape)), "not enough rows to fit"
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
    model.fit(X, y)
    r2 = model.score(X, y)
    alpha = model.named_steps["ridgecv"].alpha_
    coefs = model.named_steps["ridgecv"].coef_
    log(f"fit ridge: n={len(y)}, p={X.shape[1]}, alpha={alpha:.4g}, in-sample R2={r2:.4f}")
    return model, coefs, r2, alpha


def plot_waterfall(coefs: np.ndarray, feature_names: list[str], r2: float, title: str, outpath: Path) -> None:
    order = np.argsort(np.abs(coefs))[::-1]
    coefs_sorted = coefs[order]
    names_sorted = [feature_names[i] for i in order]

    vmax = np.abs(coefs_sorted).max() if len(coefs_sorted) else 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    colors = plt.cm.RdBu_r(norm(coefs_sorted))

    fig_h = max(3, 0.28 * len(names_sorted) + 1.2)
    fig, ax = plt.subplots(figsize=(8, fig_h))
    y_pos = np.arange(len(names_sorted))[::-1]
    ax.barh(y_pos, coefs_sorted, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_sorted, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Standardized ridge coefficient")
    ax.set_title(f"{title}\n(R\u00b2 = {r2:.3f}, in-sample, n_features={len(names_sorted)})")

    sm = ScalarMappable(norm=norm, cmap=plt.cm.RdBu_r)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Coefficient", fraction=0.046, pad=0.04)

    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    log(f"saved {outpath}")


def run_model(df: pd.DataFrame, feature_cols: list[str], title: str, outpath: Path):
    X, y, _ = build_matrix(df, feature_cols)
    model, coefs, r2, _alpha = fit_ridge(X, y)
    plot_waterfall(coefs, feature_cols, r2, title, outpath)
    return model, coefs, feature_cols, r2


def compute_r2_set(model, df: pd.DataFrame, feature_cols: list[str], loop_class: pd.Series) -> tuple[dict, dict]:
    """Score an already-fitted joint model (no refitting) on all loops together
    and on each loop-class subset."""
    r2s, ns = {}, {}

    X, y, _ = build_matrix(df, feature_cols)
    r2s["All loops"] = model.score(X, y)
    ns["All loops"] = len(y)
    log(f"joint model scored on All loops: n={len(y)}, R2={r2s['All loops']:.4f}")

    for c in LOOP_CLASSES:
        sub = df.loc[loop_class == c]
        X, y, _ = build_matrix(sub, feature_cols)
        if len(y) == 0:
            log(f"skipping {c}: no usable rows")
            continue
        r2s[c] = model.score(X, y)
        ns[c] = len(y)
        log(f"joint model scored on {c}: n={len(y)}, R2={r2s[c]:.4f}")
    return r2s, ns


def plot_class_r2_combined(
    r2_with_size: dict, n_with_size: dict,
    r2_no_size: dict, n_no_size: dict,
    title: str, outpath: Path,
) -> None:
    base_colors = {"All loops": "#666666", "CTCF-CTCF": "#B2182B", "CRE-CRE": "#4EB2E0", "PRC-PRC": "#4EB265"}
    # Hatch fills rely on an SVG <pattern> def that Illustrator's importer often
    # drops, leaving blank bars -- use flat alpha instead, which is just a
    # plain fill color and survives SVG->AI round-trips.
    ALPHA_WITH_SIZE, ALPHA_NO_SIZE = 1.0, 0.55

    rows = (
        [("All loops", r2_with_size, n_with_size, "with size", ALPHA_WITH_SIZE)]
        + [("All loops", r2_no_size, n_no_size, "without size", ALPHA_NO_SIZE)]
        + [(c, r2_no_size, n_no_size, "without size", ALPHA_NO_SIZE) for c in LOOP_CLASSES]
    )
 
    labels, vals, colors = [], [], []
    for c, r2s, ns, group_name, alpha in rows:
        if c not in r2s:
            continue
        labels.append(f"{c}  (n={ns[c]:,}) \u2014 {group_name}")
        vals.append(r2s[c])
        colors.append(base_colors.get(c, "gray"))
 
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(labels) + 1.5))
    y_pos = np.arange(len(labels))[::-1]
    ax.barh(y_pos, vals, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axhline(len(labels) - 2.5, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel(f"R\u00b2  [{TARGET}]  (in-sample, joint model)")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
 
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="black", edgecolor="black", alpha=ALPHA_WITH_SIZE, label="with size features"),
        Patch(facecolor="black", edgecolor="black", alpha=ALPHA_NO_SIZE, label="without size features"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8)
 
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    log(f"saved {outpath}")
 

def main() -> None:
    df = load_table('../loops/loop_df_reclustered_per_cl.tsv')
    df = add_size_features(df)
    clip_cols = clip_feature_cols(df)
    size_cols = ["size", "log_size", "inv_size"]

    # 1) full model: epigenomic + size features, all loops
    model_with_size, _coefs_full, _feats_full, _r2_full = run_model(
        df, clip_cols + size_cols,
        "Feature importance: epigenomic + size features [All loops]",
        Path("01_waterfall_full_with_size.svg"),
    )

    # 2) size-only model, all loops
    run_model(
        df, size_cols,
        "Feature importance: size features only [All loops]",
        Path("02_waterfall_size_only.svg"),
    )

    # 3) epigenomic-only model (no size), all loops
    model_no_size, _coefs_noclip, _feats_noclip, _r2_noclip = run_model(
        df, clip_cols,
        "Feature importance: epigenomic features only, no size [All loops]",
        Path("03_waterfall_no_size.svg"),
    )

    # 4) reassess each SAME joint (all-loops) model on all loops together and
    #    on each loop-class subset -- no refitting, just scoring predictions
    loop_class = assign_loop_class(df)
    log("loop class counts: " + ", ".join(f"{c}={int((loop_class == c).sum())}" for c in LOOP_CLASSES))
    r2_with_size, n_with_size = compute_r2_set(model_with_size, df, clip_cols + size_cols, loop_class)
    r2_no_size, n_no_size = compute_r2_set(model_no_size, df, clip_cols, loop_class)
    plot_class_r2_combined(
        r2_with_size, n_with_size, r2_no_size, n_no_size,
        f"R\u00b2 [{TARGET}] by loop class, joint model  (n = {len(df):,} total loops)",
        Path("04_loop_class_r2_combined.svg"),
    )


if __name__ == "__main__":
    main()