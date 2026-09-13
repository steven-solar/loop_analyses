"""
All-in-one loop transition analysis:
  1. Per-loop euclidean distance traveled across cell-state transitions
     (ESC->EpiLC->d4c7PGCLC->GSC), combining left+right anchor PC1-4 into
     a single 8D distance per transition (see note in compute_transition_distances).
  2. Violin plots of that distance per transition.
  3. Scatter of distance vs. change in AbLE score per transition, w/ Pearson/Spearman.
  4. Per-loop PC-pair trajectory grid (L anchor top row, R anchor bottom row,
     all 6 PC1-4 pairs) for any loop(s) you want to inspect individually.

Assumes wide df, one row per loop, columns named PC{pc}_{side}_{cell_line},
e.g. PC1_left_ESC, and AbLE score cols ABLE_COL_PATTERN.format(cl=cell_line),
default "AbLE_{cl}". EDIT THESE PATTERNS if your real column names differ --
validate_columns() will tell you exactly what's missing rather than failing
with a cryptic KeyError downstream.
"""

import os
import sys
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats

matplotlib.use('Agg')
plt.rcParams.update({
    "text.usetex": False})
plt.rcParams['svg.fonttype'] = 'none'
plt.rc('pdf', fonttype=42)
plt.rcParams["ps.useafm"] = True
matplotlib.rcParams['path.simplify'] = False        # don't simplify vector paths
matplotlib.rcParams['agg.path.chunksize'] = 0       # no chunking

matplotlib.rcParams.update({
    'font.size':     6,
    'axes.linewidth': 0.25,
    'axes.labelsize': 6,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'svg.fonttype':  'none',       # keeps text as real text, not outlines, in Illustrator
})

import matplotlib.patches as mpatches

# --- global rcParams so every text element and stroke matches your target ---
PT_TO_IN = 1 / 72.0
TARGET_WIDTH_PT = 150
STROKE_PT = 0.25
FONT_PT = 5

plt.rcParams.update({
    "font.size": FONT_PT,
    "axes.titlesize": FONT_PT,
    "axes.labelsize": FONT_PT,
    "xtick.labelsize": FONT_PT,
    "ytick.labelsize": FONT_PT,
    "legend.fontsize": FONT_PT,
    "axes.linewidth": STROKE_PT,
    "xtick.major.width": STROKE_PT,
    "ytick.major.width": STROKE_PT,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "svg.fonttype": "none",   # keep text as live/editable text in Illustrator, not outlines
})

# ----------------------------- config ------------------------------------
CELL_LINES = ["ESC", "EpiLC", "d4c7PGCLC", "GSC"]
N_PCS = 4
SIDES = {"left": "left", "right": "right"}   # row label -> column-name token
PC_COL_PATTERN = "PC{pc}_{side}_{cl}"        # e.g. PC1_left_ESC  -- CHECK/EDIT
ABLE_COL_PATTERN = "AbLE_score_{cl}"         # e.g. AbLE_score_ESC
PC_AXIS_LABELS = {1: "PC1 (CRE)", 2: "PC2 (CTCF)", 3: "PC3 (PRC)", 4: "PC4 (Lamin)"}
PC_PAIRS = list(itertools.combinations(range(1, N_PCS + 1), 2))
CL_COLORS_PATH = "../data/cl_colors.tsv"
PC_AXIS_LIM = (-2, 2)
PC_AXIS_TICK_STEP = 0.5

INPUT_PATH = "../4_loop_strength_prediction/loops/loop_df_reclustered_per_cl_wide.tsv"
OUTPUT_DIR = "."
LOOP_ID_COL = None                  # set to a column name if loops aren't uniquely indexed by df.index


def _read_df(path):
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".pkl") or path.endswith(".pickle"):
        return pd.read_pickle(path)
    if path.endswith(".tsv"):
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _pc_col(pc, side, cl, pattern=PC_COL_PATTERN):
    return pattern.format(pc=pc, side=side, cl=cl)


def _load_cl_colors(path=CL_COLORS_PATH, cell_lines=CELL_LINES):
    """
    Loads cell-line -> color mapping from a 2-col tsv. Tries common column
    name variants; falls back to first two columns with a warning if none
    match, since I don't know your exact header names.
    """
    ccdf = pd.read_csv(path, sep="\t")
    name_col_candidates = ["cell_line", "cellline", "cl", "name"]
    color_col_candidates = ["color", "colour", "hex"]

    name_col = next((c for c in name_col_candidates if c in ccdf.columns), None)
    color_col = next((c for c in color_col_candidates if c in ccdf.columns), None)
    if name_col is None or color_col is None:
        print(f"[WARN] Could not find expected column names in {path} "
              f"(got {list(ccdf.columns)}); falling back to first two columns "
              f"as (name, color). Verify this is correct.", file=sys.stderr)
        name_col, color_col = ccdf.columns[0], ccdf.columns[1]

    colors_by_cl = dict(zip(ccdf[name_col], ccdf[color_col]))
    missing = [cl for cl in cell_lines if cl not in colors_by_cl]
    if missing:
        raise ValueError(
            f"{path} is missing colors for cell line(s): {missing}. "
            f"Found entries for: {list(colors_by_cl.keys())}"
        )
    return colors_by_cl


# --------------------------- validation -----------------------------------
def validate_columns(df, sides=SIDES, cell_lines=CELL_LINES, n_pcs=N_PCS,
                      pc_pattern=PC_COL_PATTERN, able_pattern=ABLE_COL_PATTERN):
    missing_pc, missing_able = [], []
    for side_tok in sides.values():
        for cl in cell_lines:
            for pc in range(1, n_pcs + 1):
                col = _pc_col(pc, side_tok, cl, pc_pattern)
                if col not in df.columns:
                    missing_pc.append(col)
    for cl in cell_lines:
        able_col = able_pattern.format(cl=cl)
        if able_col not in df.columns:
            missing_able.append(able_col)

    if missing_pc:
        raise ValueError(
            f"Missing expected PC columns: {missing_pc}\n"
            f"Columns containing 'PC': {[c for c in df.columns if 'PC' in c]}\n"
            f"-> check PC_COL_PATTERN / SIDES against your actual naming."
        )
    has_able = True
    if missing_able:
        print(
            f"[WARN] Missing expected AbLE columns: {missing_able}\n"
            f"Columns containing 'able'/'AbLE': {[c for c in df.columns if 'able' in c.lower()]}\n"
            f"-> distance-vs-AbLE correlation plot will be skipped.",
            file=sys.stderr,
        )
        has_able = False
    return has_able


# ----------------------- transition distance calc --------------------------
def compute_transition_distances(df, cell_lines=CELL_LINES, sides=SIDES, n_pcs=N_PCS,
                                  pc_pattern=PC_COL_PATTERN, able_pattern=ABLE_COL_PATTERN,
                                  has_able=True, loop_id_col=LOOP_ID_COL):
    """
    For each consecutive cell-line pair, compute per-loop euclidean distance
    combining BOTH anchors' PC1-4 into one 8D vector (sqrt(dist_left^2 + dist_right^2)),
    which is the geometrically consistent "total loop displacement" metric --
    NOT a sum or average of separate left/right distances (summing raw distances
    double-counts, averaging isn't a proper metric). Per-anchor distances are
    also returned separately as a diagnostic for L/R asymmetry.

    Returns long df: [loop_id, transition, distance, dist_left, dist_right,
                       able_diff, able_diff_abs (if available)]
    """
    loop_ids = df[loop_id_col] if loop_id_col else df.index

    records = []
    for cl_from, cl_to in zip(cell_lines[:-1], cell_lines[1:]):
        transition = f"{cl_from}->{cl_to}"

        side_dists = {}
        for side_label, side_tok in sides.items():
            from_cols = [_pc_col(pc, side_tok, cl_from, pc_pattern) for pc in range(1, n_pcs + 1)]
            to_cols = [_pc_col(pc, side_tok, cl_to, pc_pattern) for pc in range(1, n_pcs + 1)]

            from_vals = df[from_cols].to_numpy(dtype=float)
            to_vals = df[to_cols].to_numpy(dtype=float)

            nan_rows = np.isnan(from_vals).any(axis=1) | np.isnan(to_vals).any(axis=1)
            if nan_rows.any():
                print(f"[WARN] {transition} ({side_label}): {nan_rows.sum()}/{len(df)} rows "
                      f"have NaN in PC cols.", file=sys.stderr)

            d = np.sqrt(np.nansum((to_vals - from_vals) ** 2, axis=1))
            d[nan_rows] = np.nan
            side_dists[side_label] = d

        # combined 8D distance = sqrt(sum of squared per-anchor distances)
        combined = np.sqrt(sum(d ** 2 for d in side_dists.values()))

        able_diff = able_diff_abs = None
        if has_able:
            able_from_col = able_pattern.format(cl=cl_from)
            able_to_col = able_pattern.format(cl=cl_to)
            able_diff = (df[able_to_col] - df[able_from_col]).to_numpy(dtype=float)
            able_diff_abs = np.abs(able_diff)

        chunk = pd.DataFrame({
            "loop_id": loop_ids.values if hasattr(loop_ids, "values") else loop_ids,
            "transition": transition,
            "distance": combined,
        })
        for side_label, d in side_dists.items():
            chunk[f"dist_{side_label}"] = d
        if able_diff is not None:
            chunk["able_diff"] = able_diff
            chunk["able_diff_abs"] = able_diff_abs
        records.append(chunk)

    long_df = pd.concat(records, ignore_index=True)
    n_nan = long_df["distance"].isna().sum()
    if n_nan:
        print(f"[INFO] {n_nan}/{len(long_df)} rows have NaN combined distance "
              f"(dropped from plots, kept in returned df).", file=sys.stderr)
    return long_df


# ------------------------------- plots --------------------------------------
# def plot_distance_violins(long_df, cell_lines=CELL_LINES, out_path=None):
#     order = [f"{a}->{b}" for a, b in zip(cell_lines[:-1], cell_lines[1:])]
#     plot_df = long_df.dropna(subset=["distance"])
#     fig, ax = plt.subplots(figsize=(1.6 * len(order) + 2, 5))
#     sns.violinplot(data=plot_df, x="transition", y="distance", order=order, cut=0, inner="quartile", ax=ax)
#     ax.set_xlabel("")
#     ax.set_ylabel("Euclidean distance (PC1-4, both anchors combined)")
#     ax.set_title("Loop displacement in PC space per cell-state transition")
#     plt.tight_layout()
#     if out_path:
#         fig.savefig(out_path)
#         print(f"[INFO] saved {out_path}")
#     return fig

def plot_distance_violins(long_df, cell_lines=CELL_LINES, out_path=None,
                           whisker_tick_halfwidth=0.15):
    order = [f"{a}->{b}" for a, b in zip(cell_lines[:-1], cell_lines[1:])]
    plot_df = long_df.dropna(subset=["distance"])

    fig, ax = plt.subplots(figsize=(1.6 * len(order) + 2, 5))
    sns.violinplot(data=plot_df, x="transition", y="distance", order=order,
                   cut=0, inner="quartile", ax=ax)

    # add whisker-position ticks (standard 1.5*IQR rule, clipped to data range)
    # so you have a reference for where to draw box whiskers in Illustrator
    for i, transition in enumerate(order):
        vals = plot_df.loc[plot_df["transition"] == transition, "distance"].dropna()
        if vals.empty:
            continue
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        lo_fence, hi_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        whisker_lo = vals[vals >= lo_fence].min()
        whisker_hi = vals[vals <= hi_fence].max()

        ax.hlines([whisker_lo, whisker_hi],
                  i - whisker_tick_halfwidth, i + whisker_tick_halfwidth,
                  color="black", linewidth=1, zorder=5)

    ax.set_xlabel("")
    ax.set_ylabel("Euclidean distance (PC1-4, both anchors combined)")
    ax.set_title("Loop displacement in PC space per cell-state transition")
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    if out_path:
        fig.savefig(out_path)
        print(f"[INFO] saved {out_path}")
    return fig

def plot_distance_histograms(long_df, cell_lines=CELL_LINES, out_path=None,
                              n_bins=250, colors=None):
    """
    Overlapping step-histogram of per-loop transition distances (the same
    'distance' column plot_distance_violins uses), one line per transition,
    linear-spaced shared bins on x, raw loop counts on y.
    """
    order = [f"{a}->{b}" for a, b in zip(cell_lines[:-1], cell_lines[1:])]

    vals_by_transition = {}
    for transition in order:
        v = long_df.loc[long_df["transition"] == transition, "distance"].to_numpy(dtype=float)
        n_nan = np.isnan(v).sum()
        if n_nan:
            print(f"[WARN] {transition}: dropping {n_nan}/{len(v)} NaN distances "
                  f"before binning.", file=sys.stderr)
        vals_by_transition[transition] = v[np.isfinite(v)]

    pooled = np.concatenate(list(vals_by_transition.values()))
    if pooled.size == 0:
        print("[WARN] plot_distance_histograms: no finite distances, skipping.",
              file=sys.stderr)
        return None
    bins = np.linspace(pooled.min(), pooled.max(), n_bins + 1)

    if colors is None:
        colors = {order[i]: c for i, c in enumerate(["#1f77b4", "#ff7f0e", "#d62728"])}

    fig, ax = plt.subplots(figsize=(6, 4))
    for transition in order:
        v = vals_by_transition[transition]
        if v.size == 0:
            continue
        ax.hist(v, bins=bins, density=False, histtype="step",
                linewidth=1.2, color=colors[transition], label=transition)

    ax.set_xlabel("Euclidean distance (PC1-4, both anchors combined)")
    ax.set_xlim(0, 10)
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Loop Displacement per Cell-State Transition")
    ax.legend(title="Transition", frameon=True, fontsize=6, title_fontsize=7)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    if out_path:
        fig.savefig(out_path)
        print(f"[INFO] saved {out_path}")
    return fig

def plot_distance_vs_able(long_df, cell_lines=CELL_LINES, out_path=None, use_abs=False):
    if "able_diff" not in long_df.columns:
        print("[WARN] no able_diff column present, skipping correlation plot.", file=sys.stderr)
        return None

    order = [f"{a}->{b}" for a, b in zip(cell_lines[:-1], cell_lines[1:])]
    y_col = "able_diff_abs" if use_abs else "able_diff"
    plot_df = long_df.dropna(subset=["distance", y_col])

    g = sns.lmplot(
        data=plot_df, x="distance", y=y_col, col="transition", col_order=order,
        col_wrap=min(len(order), 4), height=4, scatter_kws={"alpha": 0.2, "s": 10},
        line_kws={"color": "red"}, facet_kws={"sharex": False, "sharey": False},
    )
    for ax, transition in zip(g.axes.flat, order):
        sub = plot_df[plot_df["transition"] == transition]
        if len(sub) >= 2:
            r, p = stats.pearsonr(sub["distance"], sub[y_col])
            rho, p_s = stats.spearmanr(sub["distance"], sub[y_col])
            ax.text(0.05, 0.95, f"pearson r={r:.2f} (p={p:.1e})\nspearman ρ={rho:.2f}",
                     transform=ax.transAxes, va="top", fontsize=8)
        ax.set_title(transition)

    g.set_axis_labels("Euclidean distance (PC1-4, both anchors combined)",
                       "|ΔAbLE score|" if use_abs else "ΔAbLE score")
    plt.tight_layout()
    if out_path:
        g.savefig(out_path, dpi=300)
        print(f"[INFO] saved {out_path}")
    return g


def compute_mean_pc_trajectories(df, sides=SIDES, cell_lines=CELL_LINES, n_pcs=N_PCS,
                                  pc_col_pattern=PC_COL_PATTERN):
    """
    Mean (+ SEM) PC1-4 position per cell line, per anchor side, averaged
    across ALL loops in df. Returns {side_label: {'mean': df[cell_line x PC],
    'sem': df[cell_line x PC], 'n': int}}.
    """
    out = {}
    for side_label, side_tok in sides.items():
        means, sems = {}, {}
        for cl in cell_lines:
            cols = [_pc_col(pc, side_tok, cl, pc_col_pattern) for pc in range(1, n_pcs + 1)]
            vals = df[cols]
            n_nan = vals.isna().any(axis=1).sum()
            if n_nan:
                print(f"[WARN] {side_label}/{cl}: {n_nan}/{len(df)} loops have NaN, "
                      f"excluded from mean.", file=sys.stderr)
            means[cl] = vals.mean(axis=0).to_numpy()
            sems[cl] = vals.sem(axis=0).to_numpy()
        mean_df = pd.DataFrame(means, index=[f"PC{pc}" for pc in range(1, n_pcs + 1)]).T
        sem_df = pd.DataFrame(sems, index=[f"PC{pc}" for pc in range(1, n_pcs + 1)]).T
        out[side_label] = {"mean": mean_df, "sem": sem_df, "n": len(df)}
    return out


def plot_mean_pc_trajectories(traj, cell_lines=CELL_LINES, sides=SIDES, pc_pairs=PC_PAIRS,
                               pc_axis_labels=PC_AXIS_LABELS, output_dir=OUTPUT_DIR,
                               out_base="mean_pc_trajectories_all_pairs",
                               cl_colors_path=CL_COLORS_PATH, axis_lim=PC_AXIS_LIM,
                               axis_tick_step=PC_AXIS_TICK_STEP):
    """
    Grid of mean-across-loops PC trajectories: rows = (left, right) anchor,
    cols = each PC pair. Fixed axis_lim/axis_tick_step applied to every panel
    (not data-driven) -- so points outside axis_lim will be clipped; check
    your PC value range if trajectories look cut off. Error bars = SEM.
    """
    side_labels = list(sides.keys())
    colors_by_cl = _load_cl_colors(cl_colors_path, cell_lines)

    ticks = np.arange(axis_lim[0], axis_lim[1] + 1e-9, axis_tick_step)

    n_cols = len(pc_pairs)
    fig, axes = plt.subplots(len(side_labels), n_cols, figsize=(4.2 * n_cols, 4.0 * len(side_labels)))
    if len(side_labels) == 1:
        axes = axes.reshape(1, n_cols)

    for col_idx, (pc_a, pc_b) in enumerate(pc_pairs):
        xa, ya = f"PC{pc_a}", f"PC{pc_b}"

        for row_idx, side_label in enumerate(side_labels):
            ax = axes[row_idx, col_idx]
            mean_df, sem_df = traj[side_label]["mean"], traj[side_label]["sem"]
            xs, ys = mean_df[xa].to_numpy(), mean_df[ya].to_numpy()
            xerr, yerr = sem_df[xa].to_numpy(), sem_df[ya].to_numpy()

            out_of_range = (xs < axis_lim[0]) | (xs > axis_lim[1]) | (ys < axis_lim[0]) | (ys > axis_lim[1])
            if out_of_range.any():
                print(f"[WARN] {side_label} {xa} vs {ya}: {out_of_range.sum()} cell-line "
                      f"mean(s) fall outside axis_lim={axis_lim} and will be clipped.",
                      file=sys.stderr)

            ax.errorbar(xs, ys, xerr=xerr, yerr=yerr, fmt="none",
                        ecolor="grey", alpha=0.5, zorder=1)
            for i in range(len(cell_lines) - 1):
                ax.annotate("", xy=(xs[i + 1], ys[i + 1]), xytext=(xs[i], ys[i]),
                            arrowprops=dict(arrowstyle="->", color="black", alpha=0.6), zorder=2)
            for cl, x, y in zip(cell_lines, xs, ys):
                ax.scatter(x, y, color=colors_by_cl[cl], s=60, zorder=3,
                           edgecolor="black", linewidth=0.5, label=cl)
                ax.annotate(f"({x:.2f}, {y:.2f})", (x, y), textcoords="offset points",
                            xytext=(6, 4), fontsize=7, color=colors_by_cl[cl])

            ax.set_xlim(*axis_lim); ax.set_ylim(*axis_lim)
            ax.set_xticks(ticks); ax.set_yticks(ticks)
            ax.set_xlabel(pc_axis_labels.get(pc_a, xa))
            ax.set_ylabel(pc_axis_labels.get(pc_b, ya))
            if row_idx == 0:
                ax.set_title(f"{xa} vs {ya}", fontsize=10)
            if col_idx == 0:
                ax.annotate(f"{side_label.capitalize()} anchor", xy=(0, 0.5),
                            xycoords="axes fraction", xytext=(-55, 0),
                            textcoords="offset points", fontsize=11,
                            rotation=90, va="center", ha="center")

    axes[0, -1].legend(fontsize=8, frameon=False, loc="best")
    n = next(iter(traj.values()))["n"]
    fig.suptitle(f"Mean loop PC trajectory across cell states (n={n} loops); "
                 f"error bars = SEM", y=1.01)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{out_base}.svg")
    fig.savefig(path, bbox_inches="tight")
    print(f"[INFO] saved {path}")
    return fig, axes


# --------------------------------- main --------------------------------------
if __name__ == "__main__":
    df = _read_df(INPUT_PATH)
    has_able = validate_columns(df)

    long_df = compute_transition_distances(df, has_able=has_able)
    long_df.to_csv(os.path.join(OUTPUT_DIR, "transition_distances_long.csv"), index=False)
    print("[INFO] wrote transition_distances_long.csv")

    plot_distance_violins(long_df, out_path=os.path.join(OUTPUT_DIR, "transition_distance_violins.svg"))
    plot_distance_histograms(long_df, out_path=os.path.join(OUTPUT_DIR, "transition_distance_histograms.svg"))
    if has_able:
        plot_distance_vs_able(long_df, out_path=os.path.join(OUTPUT_DIR, "distance_vs_able_diff.svg"))

    traj = compute_mean_pc_trajectories(df)
    plot_mean_pc_trajectories(traj)