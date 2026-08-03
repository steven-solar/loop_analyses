#!/usr/bin/env python

import os
import re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from patsy import dmatrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# ---------------- basic config ----------------
INPUT_FILE  = "loops/loop_df_reclustered_per_cl.tsv"
OUTPUT_DIR  = "per_cl_reclustered"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ABLE_COL     = "AbLE_score"
LOG_ABLE_COL = "AbLE_log"
SIZE_COL     = "size"
CLASS_COL    = "loop_name"
CELL_COL     = "cell_line"

LOOP_CLASSES = ["CTCF-CTCF", "CRE-CRE", "PRC-PRC"]
CLASS_COLORS = {
    "CTCF-CTCF": "#CC3412",
    "CRE-CRE":   "#33BCEE",
    "PRC-PRC":   "#4EB265",
}

SPLINE_DF  = 5
N_PLS_COMP = 4
MIN_N      = 50   # minimum points per (class, cell line) to fit

matplotlib.rcParams.update({
    'font.size': 6,
    'axes.linewidth': 0.25,
    'axes.labelsize': 6,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'svg.fonttype': 'none',
    'path.simplify': False,
    'agg.path.chunksize': 0,
})


def get_signal_columns(df):
    """*_left / *_right columns with 'clip' in the name."""
    return sorted(c for c in df.columns
                  if (c.endswith("_left") or c.endswith("_right"))
                  and "clip" in c.lower())


def make_spline(x, df=SPLINE_DF):
    return dmatrix(f"1 + cr(x, df={df})", {"x": np.asarray(x)}, return_type="dataframe")


def fit_log_spline(x, y_log, df=SPLINE_DF):
    Xs = make_spline(x, df)
    coef = np.linalg.lstsq(Xs.values, y_log, rcond=None)[0]
    fit = Xs.values @ coef
    resid = y_log - fit
    r2 = 1.0 - np.sum(resid ** 2) / np.sum((y_log - y_log.mean()) ** 2)

    def predict(x_new):
        return make_spline(x_new, df).values @ coef

    return predict, resid, r2

def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.25, length=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.25)


def plot_pooled_spline(df, group_col, groups, color_map, xlabel, ylabel, title, outname, legend_ncol=1):
    """One curve per group (cell line, or loop class), pooled over everything else."""
    gsub = df.dropna(subset=[LOG_ABLE_COL, SIZE_COL, ABLE_COL])
    s_min, s_max = gsub[SIZE_COL].min(), gsub[SIZE_COL].max()
    s_grid = np.linspace(s_min, s_max, 200)

    fig, ax = plt.subplots(figsize=(4, 3))
    y_all = []
    for g in groups:
        sub = gsub[gsub[group_col] == g]
        if len(sub) < MIN_N:
            continue
        predict, _, _ = fit_log_spline(sub[SIZE_COL].values, sub[LOG_ABLE_COL].values)
        y_fit = np.exp(predict(s_grid))
        ax.plot(s_grid, y_fit, lw=0.25, color=color_map.get(g, "grey"), label=g)
        y_all.append(y_fit)

    ax.set_xscale("log", base=10)
    ax.set_yscale("log")
    style_axes(ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if y_all:
        y_all = np.concatenate(y_all)
        ax.set_xlim(s_min, s_max)
        ax.set_ylim(y_all.min() / 1.2, y_all.max() * 1.2)
    ax.legend(frameon=False, fontsize=5, ncol=legend_ncol)
    fig.savefig(os.path.join(OUTPUT_DIR, outname), bbox_inches="tight")
    plt.close(fig)


def main():
    df = pd.read_csv(INPUT_FILE, sep="\t")
    df = df[df[ABLE_COL] > 0].copy()
    df[LOG_ABLE_COL] = np.log(df[ABLE_COL])

    if SIZE_COL not in df.columns and {"left", "right"}.issubset(df.columns):
        df[SIZE_COL] = (df["right"] - df["left"]).abs()

    feature_cols = get_signal_columns(df)
    print(f"Using {len(feature_cols)} signal features")

    cell_lines_all = sorted(df[CELL_COL].dropna().unique())

    spline_params = {}
    spline_r2_records = []
    r2_records = []
    ridge_coefs = {}

    # ---------------- per (class, cell line) modelling ----------------
    for cl in LOOP_CLASSES:
        for cell in cell_lines_all:
            sub = df[(df[CLASS_COL] == cl) & (df[CELL_COL] == cell)].copy()
            sub = sub.dropna(subset=feature_cols + [LOG_ABLE_COL, SIZE_COL])
            if len(sub) < MIN_N:
                continue

            # 1) size spline + residuals (cached for reuse in every plot below)
            predict, resid, r2_spline = fit_log_spline(
                sub[SIZE_COL].values, sub[LOG_ABLE_COL].values
            )
            sub["logAble_resid"] = resid
            spline_params[(cl, cell)] = (r2_spline, predict)
            spline_r2_records.append({"loop_class": cl, "cell_line": cell,
                                       "n": len(sub), "spline_r2_logAbLE": r2_spline})

            # 2) PLS on residual log(AbLE)
            X_train, X_test, y_train, y_test = train_test_split(
                sub[feature_cols].values, sub["logAble_resid"].values,
                test_size=0.2, random_state=0
            )
            scaler = StandardScaler()
            X_train_sc, X_test_sc = scaler.fit_transform(X_train), scaler.transform(X_test)

            pls = PLSRegression(n_components=N_PLS_COMP)
            pls.fit(X_train_sc, y_train)
            r2 = r2_score(y_test, pls.predict(X_test_sc).ravel())
            r2_records.append({"cell_line": cell, "loop_class": cl, "r2": r2, "n": len(sub)})

            # 3) ridge for coefficient / waterfall plot
            ridge = Ridge(alpha=1.0).fit(X_train_sc, y_train)
            ridge_coefs[(cl, cell)] = (feature_cols, ridge.coef_)

    spline_r2_df = pd.DataFrame(spline_r2_records)
    spline_r2_df.to_csv(os.path.join(OUTPUT_DIR, "spline_r2_per_class_cell.tsv"),
                         sep="\t", index=False)
    r2_df = pd.DataFrame(r2_records)
    r2_df.to_csv(os.path.join(OUTPUT_DIR, "pls_r2_per_class_cell.tsv"), sep="\t", index=False)

    # ---------------- per-cell-line before/after spline panels (reuses cache) ----------------
    for cell in cell_lines_all:
        df_cell = df[df[CELL_COL] == cell].dropna(subset=[LOG_ABLE_COL, SIZE_COL, ABLE_COL])
        if df_cell.empty:
            continue
        s_min, s_max = df_cell[SIZE_COL].min(), df_cell[SIZE_COL].max()
        s_grid = np.linspace(s_min, s_max, 200)

        fig, axes = plt.subplots(len(LOOP_CLASSES), 2, figsize=(6, 4.5), sharex="col")
        axes = np.atleast_2d(axes)
        right_y = []

        for row_idx, cl in enumerate(LOOP_CLASSES):
            sub = df_cell[df_cell[CLASS_COL] == cl]
            cached = spline_params.get((cl, cell))
            if len(sub) < MIN_N or cached is None:
                continue
            r2_spline, predict = cached

            y_log_fit = predict(sub[SIZE_COL].values)
            ratio = np.exp(sub[LOG_ABLE_COL].values - y_log_fit)
            y_grid = np.exp(predict(s_grid))

            ax_left = axes[row_idx, 0]
            ax_left.scatter(sub[SIZE_COL], sub[ABLE_COL], s=2, alpha=0.15,
                             color=CLASS_COLORS[cl], rasterized=True)
            ax_left.plot(s_grid, y_grid, color="black", lw=0.25,
                         label=f"Spline (R²={r2_spline:.3f})")
            ax_left.set_xscale("log", base=10)
            ax_left.set_yscale("log")
            style_axes(ax_left)
            ax_left.set_ylabel(f"{cl}\nAbLE (log scale)")
            ax_left.legend(frameon=False, fontsize=5, loc="upper right")

            ax_right = axes[row_idx, 1]
            ax_right.scatter(sub[SIZE_COL], ratio, s=2, alpha=0.15,
                             color=CLASS_COLORS[cl], rasterized=True)
            ax_right.axhline(1.0, color="grey", lw=0.25, ls="--")
            ax_right.set_xscale("log", base=10)
            ax_right.set_yscale("log")
            style_axes(ax_right)
            ax_right.set_ylabel("AbLE / spline(size)\n(log scale)")

            valid_ratio = ratio[ratio > 0]
            if len(valid_ratio):
                right_y.append(valid_ratio)

        axes[-1, 0].set_xlabel("loop size (bp, log scale)")
        axes[-1, 1].set_xlabel("loop size (bp, log scale)")

        if right_y:
            right_y = np.concatenate(right_y)
            y_min_plot, y_max_plot = right_y.min() / 1.2, right_y.max() * 1.2
            for r in range(len(LOOP_CLASSES)):
                axes[r, 1].set_xlim(s_min, s_max)
                axes[r, 1].set_ylim(y_min_plot, y_max_plot)

        fig.suptitle(f"{cell}: spline removal of size trend", fontsize=7)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(os.path.join(OUTPUT_DIR, f"{cell}_perclass_spline_before_after.svg"), dpi=300)
        plt.close(fig)

    # ---------------- spline grid: 4 cell lines x 3 classes, shared scale (reuses cache) ----------------
    cell_lines_4 = sorted(r2_df["cell_line"].unique())[:4]
    gsub = df[df[CELL_COL].isin(cell_lines_4) & df[CLASS_COL].isin(LOOP_CLASSES)]
    gsub = gsub.dropna(subset=[LOG_ABLE_COL, SIZE_COL, ABLE_COL])
    s_min, s_max = gsub[SIZE_COL].min(), gsub[SIZE_COL].max()
    s_grid = np.linspace(s_min, s_max, 200)

    def plot_splines_grid(log_x, x_label, out_name):
        fig, axes = plt.subplots(2, 2, figsize=(6, 4), sharex=True, sharey=True)
        axes = axes.ravel()
        y_all = []

        for ax, cell in zip(axes, cell_lines_4):
            for cl in LOOP_CLASSES:
                cached = spline_params.get((cl, cell))
                if cached is None:
                    continue
                _, predict = cached
                y_fit = np.exp(predict(s_grid))
                ax.plot(s_grid, y_fit, color=CLASS_COLORS[cl], lw=0.25, label=cl)
                y_all.append(y_fit)
            ax.set_title(cell)
            style_axes(ax)

        for ax in axes[2:]:
            ax.set_xlabel(x_label)
        for ax in axes[::2]:
            ax.set_ylabel("AbLE_score (log scale)")
        handles, labels_ = axes[0].get_legend_handles_labels()
        axes[1].legend(handles, labels_, frameon=False, fontsize=5)

        if y_all:
            y_all = np.concatenate(y_all)
            y_lo, y_hi = max(y_all.min(), 1e-6) / 1.2, y_all.max() * 1.2
            for ax in axes:
                if log_x:
                    ax.set_xscale("log", base=10)
                    ax.set_xlim(s_min, s_max)
                else:
                    pad = 0.02 * (s_max - s_min)
                    ax.set_xlim(s_min - pad, s_max + pad)
                ax.set_yscale("log")
                ax.set_ylim(y_lo, y_hi)

        fig.suptitle("Size splines per cell line (shared scale)", fontsize=7)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(os.path.join(OUTPUT_DIR, out_name), dpi=300)
        plt.close(fig)

    plot_splines_grid(True, "loop size (bp, log scale)", "splines_all_cells_logx_logy.svg")
    plot_splines_grid(False, "loop size (bp)", "splines_all_cells_lin_logy.svg")

    # ---------------- pooled splines: per cell line, and per loop class ----------------
    cl_palette_df = pd.read_csv("../data/cl_colors.tsv", sep="\t")
    CL_COLORS = dict(zip(cl_palette_df["cell_line"], cl_palette_df["color"]))

    plot_pooled_spline(df, CELL_COL, cell_lines_all, CL_COLORS,
                        "loop size (bp, log scale)", "AbLE_score (log scale)",
                        "Spline per cell line (all loops)",
                        "splines_per_cellline_all_loops_logx_logy.svg", legend_ncol=2)

    plot_pooled_spline(df, CLASS_COL, LOOP_CLASSES, CLASS_COLORS,
                        "loop size (bp, log scale)", "AbLE_score (log scale)",
                        "Spline per loop class (all cell lines)",
                        "splines_per_class_all_cells_logx_logy.svg")

    # ---------------- R² bar plot per cell line ----------------
    if not r2_df.empty:
        pivot = r2_df.pivot(index="cell_line", columns="loop_class", values="r2").reindex(columns=LOOP_CLASSES)
        y_pos, bar_h = np.arange(len(pivot.index)), 0.18

        fig, ax = plt.subplots(figsize=(3, 2.3))
        for i, cl in enumerate(LOOP_CLASSES):
            ax.barh(y_pos + (i - 1) * bar_h, pivot[cl].values, height=bar_h,
                    color=CLASS_COLORS[cl], label=cl.split("-")[0])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(pivot.index)
        ax.set_xlim(0, 0.5)
        ax.set_xticks(np.linspace(0, 0.5, 6))
        ax.set_xlabel("PLS R² (test)")
        ax.set_title("Loop-class PLS R²")
        style_axes(ax)
        ax.legend(frameon=False, fontsize=5)
        fig.savefig(os.path.join(OUTPUT_DIR, "r2_barplot.svg"))
        plt.close(fig)

    # ---------------- ridge waterfall plots (top 20 |beta| per class/cell) ----------------
    for (cl, cell), (feat_names, coefs) in ridge_coefs.items():
        idx = np.argsort(np.abs(coefs))[::-1][:20]
        names, vals = [feat_names[i] for i in idx], coefs[idx]
        colors = ["#d7191c" if v > 0 else "#2c7bb6" for v in vals]

        fig, ax = plt.subplots(figsize=(3, 2.3))
        y = np.arange(len(idx))
        ax.barh(y, vals, color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=4)
        ax.invert_yaxis()
        ax.set_xlabel("Ridge coefficient")
        ax.set_title(f"{cell} - {cl}")
        style_axes(ax)
        fig.tight_layout()
        fname = f"ridge_waterfall_{cell}_{re.sub('[^A-Za-z0-9]+', '_', cl)}.svg"
        fig.savefig(os.path.join(OUTPUT_DIR, fname), dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()