#!/usr/bin/env python3
"""
Ridge regression on loop AbLE_score from epigenomic (_clip_) features and/or
loop-size features. Produces waterfall feature-importance plots (RdBu-colored,
in-sample R2 in the title) and a per-loop-class R2 bar chart segmented by
CTCF-CTCF / CRE-CRE / PRC-PRC.

Usage:
    python justification.py path/to/loops.tsv --outdir figs

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
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

TARGET = "AbLE_score"
LOOP_CLASSES = ["CTCF-CTCF", "CRE-CRE", "PRC-PRC"]
ALPHAS = np.logspace(-3, 4, 50)
N_TREES = 500


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


def fit_rf(X: np.ndarray, y: np.ndarray):
    """Random forest counterpart to fit_ridge. No StandardScaler needed --
    tree splits are threshold-based, not distance-based, so scaling is a
    no-op here (unlike ridge, which needs it for the L2 penalty to treat
    features fairly). Returns feature_importances_ in place of signed
    coefficients, and None in place of ridge's alpha (no equivalent
    regularization-strength scalar for a random forest)."""
    assert len(X) > len(set(X.shape)), "not enough rows to fit"
    model = RandomForestRegressor(n_estimators=N_TREES, random_state=0, n_jobs=-1)
    model.fit(X, y)
    r2 = model.score(X, y)
    importances = model.feature_importances_
    log(f"fit rf: n={len(y)}, p={X.shape[1]}, n_trees={N_TREES}, in-sample R2={r2:.4f}")
    return model, importances, r2, None


def evaluate_held_out(
    df: pd.DataFrame,
    feature_cols: list[str],
    loop_class: pd.Series,
    model_type: str,
    feature_set_name: str,
    test_size: float = 0.2,
    random_state: int = 0,
) -> list[dict]:
    """Train/test split evaluation (held-out R2), same pattern as the
    original PLS loop_feature_regression script: fit on the train split,
    score on the untouched test split. Unlike compute_r2_set (which scores
    an in-sample-fit model and is prone to overstating a flexible model
    like RF), this gives an honest ridge-vs-RF comparison.

    Fits once on the full train split, then scores on the test split as a
    whole AND on each loop-class subset of that SAME test split (rather
    than re-splitting per class), since re-splitting per loop class would
    make the per-class test sets small and the resulting R2 noisy.
    """
    X, y, idx = build_matrix(df, feature_cols)
    idx = np.asarray(idx)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, idx, test_size=test_size, random_state=random_state
    )

    if model_type == "ridge":
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
    elif model_type == "rf":
        model = RandomForestRegressor(n_estimators=N_TREES, random_state=random_state, n_jobs=-1)
    else:
        raise ValueError(f"unknown model_type {model_type!r}, expected 'ridge' or 'rf'")

    model.fit(X_train, y_train)
    r2_all = r2_score(y_test, model.predict(X_test))
    log(f"[{model_type}] {feature_set_name}: held-out R2 (All loops) = {r2_all:.4f} "
        f"(n_train={len(y_train)}, n_test={len(y_test)})")

    records = [{
        "model_type": model_type, "feature_set": feature_set_name, "loop_class": "All loops",
        "r2_test": r2_all, "n_train": len(y_train), "n_test": len(y_test),
    }]

    test_loop_class = loop_class.loc[idx_test]
    for c in LOOP_CLASSES:
        mask = (test_loop_class == c).to_numpy()
        if mask.sum() == 0:
            log(f"[{model_type}] {feature_set_name}: skipping {c}, no test rows")
            continue
        r2_c = r2_score(y_test[mask], model.predict(X_test[mask]))
        log(f"[{model_type}] {feature_set_name}: held-out R2 ({c}) = {r2_c:.4f} (n_test={int(mask.sum())})")
        records.append({
            "model_type": model_type, "feature_set": feature_set_name, "loop_class": c,
            "r2_test": r2_c, "n_train": len(y_train), "n_test": int(mask.sum()),
        })

    return records


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


def plot_importance_bar(importances: np.ndarray, feature_names: list[str], r2: float, title: str, outpath: Path) -> None:
    """RF counterpart to plot_waterfall. Importances are unsigned (always
    >= 0, a Gini/impurity-reduction share), so this is a single-color bar
    chart sorted descending rather than a signed RdBu diverging one --
    there's no "pushes the prediction down" direction to show, just
    relative contribution."""
    order = np.argsort(importances)[::-1]
    imp_sorted = importances[order]
    names_sorted = [feature_names[i] for i in order]

    fig_h = max(3, 0.28 * len(names_sorted) + 1.2)
    fig, ax = plt.subplots(figsize=(8, fig_h))
    y_pos = np.arange(len(names_sorted))[::-1]
    ax.barh(y_pos, imp_sorted, color="#4C72B0", edgecolor="black", linewidth=0.4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_sorted, fontsize=8)
    ax.set_xlabel("Random forest feature importance (mean impurity decrease)")
    ax.set_title(f"{title}\n(R\u00b2 = {r2:.3f}, in-sample, n_features={len(names_sorted)})")

    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    log(f"saved {outpath}")


def run_model(df: pd.DataFrame, feature_cols: list[str], title: str, outpath: Path, model_type: str = "ridge"):
    """model_type: 'ridge' (default, unchanged behavior) or 'rf'.
    Returns (model, coefs_or_importances, feature_cols, r2) either way, so
    existing call sites that only used the ridge path keep working as-is."""
    X, y, _ = build_matrix(df, feature_cols)
    if model_type == "ridge":
        model, coefs, r2, _alpha = fit_ridge(X, y)
        plot_waterfall(coefs, feature_cols, r2, title, outpath)
    elif model_type == "rf":
        model, coefs, r2, _alpha = fit_rf(X, y)
        plot_importance_bar(coefs, feature_cols, r2, title, outpath)
    else:
        raise ValueError(f"unknown model_type {model_type!r}, expected 'ridge' or 'rf'")
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
 

def plot_held_out_r2(records: list[dict], title: str, outpath: Path) -> None:
    """Grouped bar chart: held-out R2 for ridge vs rf, grouped by
    (feature_set, loop_class). This is the honest counterpart to
    plot_class_r2_combined's in-sample numbers."""
    rec_df = pd.DataFrame(records)
    group_order = (
        rec_df[["feature_set", "loop_class"]]
        .drop_duplicates()
        .sort_values(["feature_set", "loop_class"])
        .apply(tuple, axis=1)
        .tolist()
    )
    model_colors = {"ridge": "#4C72B0", "rf": "#DD8452"}
    model_types = [m for m in ["ridge", "rf"] if m in rec_df["model_type"].unique()]

    n_groups = len(group_order)
    bar_h = 0.8 / max(len(model_types), 1)
    fig, ax = plt.subplots(figsize=(8, 0.5 * n_groups * len(model_types) + 1.5))
    y_base = np.arange(n_groups)[::-1]

    labels = []
    for gi, (fset, cl) in enumerate(group_order):
        row = rec_df[(rec_df.feature_set == fset) & (rec_df.loop_class == cl)]
        n_test = row["n_test"].iloc[0] if len(row) else 0
        labels.append(f"{fset} / {cl}  (n_test={n_test:,})")
        for mi, m in enumerate(model_types):
            sub = row[row.model_type == m]
            if sub.empty:
                continue
            y = y_base[gi] - bar_h * mi + bar_h * (len(model_types) - 1) / 2
            ax.barh(y, sub["r2_test"].iloc[0], height=bar_h,
                    color=model_colors.get(m, "gray"), edgecolor="black", linewidth=0.4,
                    label=m if gi == 0 else None)

    ax.set_yticks(y_base)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"Held-out R\u00b2  [{TARGET}]  (test split, test_size=0.2)")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", fontsize=8)

    fig.savefig(outpath, dpi=200, bbox_inches="tight")
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

    # 1b-3b) same three feature sets, but with a random forest in place of
    # ridge -- same return shape (model, importances, feature_cols, r2), so
    # everything downstream (compute_r2_set, plot_class_r2_combined) works
    # unchanged regardless of which model_type produced it.
    rf_model_with_size, _rfcoefs_full, _rffeats_full, _rfr2_full = run_model(
        df, clip_cols + size_cols,
        "Feature importance (RF): epigenomic + size features [All loops]",
        Path("01b_rf_importance_full_with_size.svg"),
        model_type="rf",
    )

    run_model(
        df, size_cols,
        "Feature importance (RF): size features only [All loops]",
        Path("02b_rf_importance_size_only.svg"),
        model_type="rf",
    )

    rf_model_no_size, _rfcoefs_noclip, _rffeats_noclip, _rfr2_noclip = run_model(
        df, clip_cols,
        "Feature importance (RF): epigenomic features only, no size [All loops]",
        Path("03b_rf_importance_no_size.svg"),
        model_type="rf",
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

    # 4b) same loop-class R2 breakdown for the RF joint models
    rf_r2_with_size, rf_n_with_size = compute_r2_set(rf_model_with_size, df, clip_cols + size_cols, loop_class)
    rf_r2_no_size, rf_n_no_size = compute_r2_set(rf_model_no_size, df, clip_cols, loop_class)
    plot_class_r2_combined(
        rf_r2_with_size, rf_n_with_size, rf_r2_no_size, rf_n_no_size,
        f"R\u00b2 [{TARGET}] by loop class, joint RF model  (n = {len(df):,} total loops)",
        Path("04b_rf_loop_class_r2_combined.svg"),
    )

    # 5) honest ridge-vs-RF comparison: train/test split (test_size=0.2),
    # fit on train, score on held-out test -- unlike steps 1-4b above,
    # this isn't in-sample, so it won't make RF look artificially strong.
    feature_sets = {
        "epigenomic+size": clip_cols + size_cols,
        "size-only": size_cols,
        "epigenomic-only": clip_cols,
    }
    held_out_records = []
    for fset_name, cols in feature_sets.items():
        for model_type in ("ridge", "rf"):
            held_out_records += evaluate_held_out(
                df, cols, loop_class, model_type, fset_name,
                test_size=0.2, random_state=0,
            )

    held_out_df = pd.DataFrame(held_out_records)
    held_out_df.to_csv("05_held_out_r2.tsv", sep="\t", index=False)
    log(f"saved 05_held_out_r2.tsv ({len(held_out_df)} rows)")

    plot_held_out_r2(
        held_out_records,
        f"Held-out R\u00b2 [{TARGET}]: ridge vs random forest  (test_size=0.2, n = {len(df):,} total loops)",
        Path("05_held_out_r2_comparison.svg"),
    )


if __name__ == "__main__":
    main()