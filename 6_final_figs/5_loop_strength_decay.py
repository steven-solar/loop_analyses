import pandas as pd
import numpy as np
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

import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import math

CELL_LINES = ['ESC', 'EpiLC', 'd4c7PGCLC', 'GSC']

# LOOP_DF_PATH = '../4_loop_strength_prediction/loops/loop_df_3kb_wide.tsv'
LOOP_DF_PATH = '../4_loop_strength_prediction/loops/loop_df_reclustered_per_cl_wide.tsv'

PER_CELL_LINE_LOOP_NAME = True
LOOP_NAME_COL_TEMPLATE  = 'loop_name_{cl}'
GLOBAL_LOOP_NAME_COL    = 'loop_name'

ABLE_COL_TEMPLATE = 'AbLE_score_{cl}'

CL_COLOR_TSV   = '../data/cl_colors.tsv'
LOOP_COLOR_TSV = '../4_loop_strength_prediction/loops/loop_colors.tsv'

MANUAL_BOOL_COL_TEMPLATE = 'is_only_{feat}_{side}_{cl}'
MANUAL_NONE_COL_TEMPLATE = 'is_None_{side}_{cl}'

FEAT_MAP = {
    'CRE':        'CRE',
    'CTCF':       'CTCF',
    'PRC_narrow': 'PRC',
}
FEAT_RANK = {'CTCF': 0, 'CRE': 1, 'PRC': 2}

MANUAL_LOOP_NAME_COL_TEMPLATE = 'manual_loop_name_{cl}'

# All raw classes produced by manual annotation
MANUAL_CLASSES = ['CTCF-CTCF', 'CTCF-CRE', 'CTCF-PRC',
                  'CRE-CRE', 'CRE-PRC', 'PRC-PRC', 'Other-related']

OUT_DIR = Path('loop_strength_decay')
N_BINS  = 20
MIN_N   = 5

BIN_STRATEGY     = 'log'
GAP_BREAK_FACTOR = 4

HARMONIZE_BINS  = True
HARMONIZE_STRAT = 'all'   # 'all' | 'largest' | 'smallest'

PLOT_CATEGORIES = ['CTCF-CTCF', 'CRE-CRE', 'PRC-PRC'] #, 'Weak-related', 'Other-related']
# PLOT_CATEGORIES = None    # ← uncomment for ALL classes, no merging
PRC_RELATED_SOURCES = {'PRC-PRC', 'CRE-PRC', 'CTCF-PRC', 'PRC-CRE', 'PRC-CTCF'}

OUT_DIR.mkdir(exist_ok=True, parents=True)

def load_color_map(path):
    d = pd.read_csv(path, sep='\t')
    name_col, color_col = d.columns[:2]
    cmap = dict(zip(d[name_col], d[color_col]))
    print(f"Loaded {len(cmap)} colors from {path}: {list(cmap.keys())}")
    return cmap


CELL_COLORS  = load_color_map(CL_COLOR_TSV)
CLASS_COLORS = load_color_map(LOOP_COLOR_TSV)
CLASS_COLORS['PRC-related'] = '#4daf4a'
CLASSES      = list(CLASS_COLORS.keys())   # all raw cluster-annotation classes

df = pd.read_csv(LOOP_DF_PATH, sep='\t')
print(f"Loaded {len(df)} loops from {LOOP_DF_PATH}")
print(f"Columns (first 20): {list(df.columns)[:20]}{'...' if len(df.columns) > 20 else ''}")

def loop_name_col(cl):
    return LOOP_NAME_COL_TEMPLATE.format(cl=cl) if PER_CELL_LINE_LOOP_NAME \
        else GLOBAL_LOOP_NAME_COL


def anchor_identity_vec(df, side, cl):
    """Per-row anchor identity ('CTCF'/'CRE'/'PRC'/'None') from boolean columns."""
    ident = pd.Series('None', index=df.index, dtype=object)
    for feat, label in FEAT_MAP.items():
        col = MANUAL_BOOL_COL_TEMPLATE.format(feat=feat, side=side, cl=cl)
        if col in df.columns:
            ident = ident.where(~df[col].astype(bool), label)
        else:
            print(f"  WARNING: column '{col}' not found — '{label}' won't be called "
                  f"for {side} side of {cl}.")
    none_col = MANUAL_NONE_COL_TEMPLATE.format(side=side, cl=cl)
    if none_col in df.columns:
        ident = ident.where(~df[none_col].astype(bool), 'None')
    return ident


def derive_manual_loop_class(df):
    """Add per-cell-line manual_loop_name_{cl} columns from is_only_* boolean columns."""
    def pair_name(a, b):
        if a == 'None' or b == 'None':
            return 'Other-related'
        lo, hi = sorted([a, b], key=lambda x: FEAT_RANK[x])
        return f'{lo}-{hi}'

    out = df.copy()
    any_derived = False

    for cl in CELL_LINES:
        any_found = any(
            MANUAL_BOOL_COL_TEMPLATE.format(feat=feat, side=side, cl=cl) in df.columns
            for feat in FEAT_MAP for side in ('left', 'right')
        )
        if not any_found:
            print(f"WARNING: no is_only_* columns for {cl} — skipping manual annotation.")
            continue

        left_id  = anchor_identity_vec(out, 'left',  cl)
        right_id = anchor_identity_vec(out, 'right', cl)
        manual   = [pair_name(a, b) for a, b in zip(left_id, right_id)]

        col = MANUAL_LOOP_NAME_COL_TEMPLATE.format(cl=cl)
        out[col] = manual
        print(f"[{cl}] manual classes: {pd.Series(manual).value_counts().to_dict()}")
        any_derived = True

    if not any_derived:
        print("WARNING: no is_only_* columns found — skipping manual annotation.")
        return None
    return out

def get_class_sub(df, name_col, cls):
    """
    Sub-DataFrame for class `cls`.

    'PRC-related' → union of all rows whose name_col value is in
                    PRC_RELATED_SOURCES (PRC-PRC, CTCF-PRC, CRE-PRC …).
    All other cls → exact equality match on name_col.
    """
    if cls == 'PRC-related':
        return df[df[name_col].isin(PRC_RELATED_SOURCES)]
    return df[df[name_col] == cls]


def resolve_classes(all_classes, plot_categories, df, name_col_fn):
    """
    Return the ordered list of classes to actually draw.

    1. If plot_categories is None  → use all_classes unchanged.
    2. If plot_categories is a list → use that list.

    In both cases, silently drop any class that has < MIN_N rows in
    every cell line (so empty panels are never created).

    Note: 'Weak-related' will be dropped automatically for manual
    annotation because that label does not exist in the manual data.
    """
    candidates = all_classes if plot_categories is None else plot_categories

    with_data = []
    for cls in candidates:
        has_data = any(
            len(get_class_sub(df, name_col_fn(cl), cls)) >= MIN_N
            for cl in CELL_LINES
            if name_col_fn(cl) in df.columns
        )
        if has_data:
            with_data.append(cls)
        else:
            print(f"  [resolve_classes] '{cls}' — no data in any cell line, skipping")
    return with_data

def compute_shared_edges(df, eff_classes, name_col, size_col,
                          n_bins, bin_strategy, min_n, strategy='all'):
    """
    Compute one set of bin edges to share across all classes in a panel.

    strategy
    --------
    'all'      reference = every row in df (most inclusive x-range)
    'largest'  reference = the effective class with the most loops
    'smallest' reference = the effective class with the fewest loops
               (most conservative: no class will have empty bins at extremes)
    """
    if strategy == 'all':
        ref_vals = df[size_col].dropna().values
    else:
        counts = {cls: len(get_class_sub(df, name_col, cls)) for cls in eff_classes}
        counts = {k: v for k, v in counts.items() if v >= min_n}
        if not counts:
            return None
        ref_cls  = (max if strategy == 'largest' else min)(counts, key=counts.get)
        ref_vals = get_class_sub(df, name_col, ref_cls)[size_col].dropna().values

    if len(ref_vals) == 0:
        return None

    eff_bins = max(3, min(n_bins, len(ref_vals) // min_n))

    if bin_strategy == 'quantile':
        return np.unique(np.quantile(ref_vals, np.linspace(0, 1, eff_bins + 1)))
    elif bin_strategy == 'log':
        vmin = max(ref_vals.min(), 1)
        return np.logspace(np.log10(vmin), np.log10(ref_vals.max()), eff_bins + 1)
    else:
        return np.linspace(ref_vals.min(), ref_vals.max(), eff_bins + 1)


def bin_and_summarize(df_sub, size_col, able_col, n_bins, bin_strategy, min_n,
                      shared_edges=None):
    """
    Bin loops by size and return (centers, means, sems).

    Only loops with finite AbLE scores are used in the statistics; a bin
    is dropped if it has < min_n valid AbLE values.
    """
    vals = df_sub[size_col].values
    able = df_sub[able_col].values

    # Require at least some finite AbLE scores overall
    valid_all = np.isfinite(able)
    if valid_all.sum() < min_n:
        return np.array([]), np.array([]), np.array([])

    vals = vals[valid_all]
    able = able[valid_all]

    if shared_edges is not None:
        edges = shared_edges
    else:
        n_loops  = len(vals)
        eff_bins = max(3, min(n_bins, n_loops // min_n))
        if eff_bins < 3:
            return np.array([]), np.array([]), np.array([])
        if bin_strategy == 'quantile':
            edges = np.unique(np.quantile(vals, np.linspace(0, 1, eff_bins + 1)))
        elif bin_strategy == 'log':
            vmin  = max(vals.min(), 1)
            edges = np.logspace(np.log10(vmin), np.log10(vals.max()), eff_bins + 1)
        else:
            edges = np.linspace(vals.min(), vals.max(), eff_bins + 1)

    centers, means, sems = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (vals >= lo) & (vals < hi)
        y = able[mask]

        # Drop bins with too few valid AbLE scores
        if y.size < min_n:
            continue

        centers.append((lo + hi) / 2)
        means.append(np.mean(y))
        sems.append(stats.sem(y))

    return np.array(centers), np.array(means), np.array(sems)

def insert_gaps(centers, means, sems, gap_factor, log_x=False):
    """
    Insert NaN breaks at large data gaps so the line visually breaks
    instead of interpolating across missing size ranges.

    log_x=True  → gap detection in log10 space (correct for log-axis plots;
                   prevents every transition to a large size from being
                   flagged as a gap).
    log_x=False → gap detection in linear space (correct for linear-axis plots).
    """
    if len(centers) < 3:
        return centers, means, sems

    if log_x:
        space  = np.log10(np.clip(centers, 1, None))
        mid_fn = lambda a, b: np.sqrt(a * b)    # geometric midpoint on original scale
    else:
        space  = centers.astype(float).copy()
        mid_fn = lambda a, b: (a + b) / 2.0     # arithmetic midpoint

    diffs = np.diff(space)
    med   = np.median(diffs)
    if med == 0:
        return centers, means, sems

    new_c, new_m, new_s = [centers[0]], [means[0]], [sems[0]]
    for i in range(1, len(centers)):
        if diffs[i - 1] > gap_factor * med:
            new_c.append(mid_fn(centers[i - 1], centers[i]))
            new_m.append(np.nan)
            new_s.append(np.nan)
        new_c.append(centers[i])
        new_m.append(means[i])
        new_s.append(sems[i])

    return np.array(new_c), np.array(new_m), np.array(new_s)

def add_powerlaw_fit(ax, centers, means, color, label_prefix=''):
    valid = ~np.isnan(means)
    c, m = centers[valid], means[valid]
    if len(c) < 5:
        return
    slope, intercept, r, *_ = stats.linregress(np.log10(c),
                                                np.log10(np.clip(m, 1e-10, None)))
    x_fit = np.logspace(np.log10(c.min()), np.log10(c.max()), 200)
    y_fit = 10 ** (intercept + slope * np.log10(x_fit))
    ax.plot(x_fit, y_fit, linestyle='--', color=color, linewidth=1.2, alpha=0.7,
            label=f'{label_prefix} b={slope:.2f} r²={r**2:.2f}')


def style_ax(ax, log_x, log_y):
    if log_x: ax.set_xscale('log')
    if log_y: ax.set_yscale('log')
    ax.set_xlabel('Loop size (bp)')
    ax.set_ylabel('Mean AbLE score')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False)


def safe_fill(ax, centers, means, sems, color, alpha=0.15):
    """fill_between that propagates NaN breaks correctly."""
    lower = np.where(np.isnan(means), np.nan, np.clip(means - sems, 1e-10, None))
    upper = np.where(np.isnan(means), np.nan, means + sems)
    ax.fill_between(centers, lower, upper, color=color, alpha=alpha)


def grid_dims(n, max_cols=3):
    ncols = min(n, max_cols)
    nrows = math.ceil(n / ncols)
    return nrows, ncols


SCALE_COMBOS = [
    (True,  True,  'logx_logy'),
    (True,  False, 'logx_liny'),
    (False, True,  'linx_logy'),
    (False, False, 'linx_liny'),
]

# one panel per cell line, show the diff classes
def plot_layout1(df, all_classes, class_colors, name_col_fn,
                 out_tag, title_tag,
                 harmonize=HARMONIZE_BINS,
                 harmonize_strat=HARMONIZE_STRAT,
                 plot_cats=PLOT_CATEGORIES,
                 shared_scale=True):
    """
    One subplot per cell line; one coloured line per loop class.
    plot_cats
    ---------
    PLOT_CATEGORIES  → simplified view  (PRC-related merged, others filtered)
    None             → all raw classes, no merging

    shared_scale : bool
        If True, all 4 cell-line panels use the same x/y axis limits,
        derived from the min/max across ALL panels/classes in this figure.
    """
    classes = resolve_classes(all_classes, plot_cats, df, name_col_fn)
    if not classes:
        print(f"[layout1/{out_tag}] no classes with data — skipping")
        return

    for log_x, log_y, tag in SCALE_COMBOS:
        nrows, ncols = grid_dims(len(CELL_LINES), max_cols=2)
        LETTER_SIZE = (12, 10)
        fig, axes = plt.subplots(nrows, ncols, figsize=LETTER_SIZE,
                                  sharex=False, sharey=False)
        axes = np.atleast_1d(axes).flatten()

        # Track global data range across all panels for this scale combo
        all_x, all_y_lo, all_y_hi = [], [], []

        for ax, cl in zip(axes, CELL_LINES):
            able_col = ABLE_COL_TEMPLATE.format(cl=cl)
            name_col = name_col_fn(cl)
            if name_col not in df.columns:
                ax.set_visible(False)
                continue

            shared_edges = (
                compute_shared_edges(df, classes, name_col, 'size',
                                     N_BINS, BIN_STRATEGY, MIN_N, harmonize_strat)
                if harmonize else None
            )

            for cls in classes:
                sub = get_class_sub(df, name_col, cls)
                if len(sub) < MIN_N:
                    continue

                centers, means, sems = bin_and_summarize(
                    sub, 'size', able_col, N_BINS, BIN_STRATEGY, MIN_N,
                    shared_edges=shared_edges)
                if len(centers) == 0:
                    continue

                centers, means, sems = insert_gaps(
                    centers, means, sems, GAP_BREAK_FACTOR, log_x=log_x)

                color = class_colors.get(cls, '#888888')
                ax.plot(centers, means, 'o-', color=color, linewidth=1.2,
                        markersize=3, alpha=0.85, label=cls)
                safe_fill(ax, centers, means, sems, color)
                # if log_x and log_y:
                #     add_powerlaw_fit(ax, centers, means, color, label_prefix=cls)

                if shared_scale:
                    valid = ~np.isnan(means)
                    if valid.any():
                        all_x.extend(centers[valid])
                        all_y_lo.extend((means - sems)[valid])
                        all_y_hi.extend((means + sems)[valid])

            ax.set_title(cl, fontsize=13, fontweight='bold')
            style_ax(ax, log_x, log_y)

        for ax in axes[len(CELL_LINES):]:
            ax.set_visible(False)

        # Apply one common x/y range to every visible panel
        if shared_scale and all_x:
            x_arr = np.asarray(all_x, dtype=float)
            y_lo_arr = np.asarray(all_y_lo, dtype=float)
            y_hi_arr = np.asarray(all_y_hi, dtype=float)

            if log_x:
                xmin = max(x_arr.min(), 1)
                xmax = x_arr.max()
                xlim = (xmin / 1.15, xmax * 1.15)
            else:
                pad = 0.05 * (x_arr.max() - x_arr.min() or 1)
                xlim = (x_arr.min() - pad, x_arr.max() + pad)

            if log_y:
                ymin = max(y_lo_arr[y_lo_arr > 0].min() if (y_lo_arr > 0).any() else y_hi_arr.min(), 1e-10)
                ymax = y_hi_arr.max()
                ylim = (ymin / 1.15, ymax * 1.15)
            else:
                pad = 0.05 * (y_hi_arr.max() - y_lo_arr.min() or 1)
                ylim = (y_lo_arr.min() - pad, y_hi_arr.max() + pad)

            for ax, cl in zip(axes[:len(CELL_LINES)], CELL_LINES):
                if ax.get_visible():
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)

        htag     = f'_harm_{harmonize_strat}' if harmonize else '_perclass'
        cats_tag = 'simplified' if plot_cats is not None else 'all'
        scale_tag = '_sharedscale' if shared_scale else ''
        fig.suptitle(
            f'Loop strength vs size — {title_tag}\n'
            f'one panel per cell line · {tag} · {cats_tag}',
            fontsize=14, fontweight='bold')
        plt.tight_layout()
        out_path = OUT_DIR / \
            f'loop_strength_by_cellline_{out_tag}_{tag}{htag}_{cats_tag}{scale_tag}.svg'
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_path}")

# one panel per loop class, across cls
def plot_layout2(df, all_classes, class_colors, name_col_fn,
                 out_tag, title_tag,
                 harmonize=HARMONIZE_BINS,
                 plot_cats=PLOT_CATEGORIES):
    """
    One subplot per loop class; one coloured line per cell line.
    plot_cats : same semantics as plot_layout1.
    """
    classes = resolve_classes(all_classes, plot_cats, df, name_col_fn)
    if not classes:
        print(f"[layout2/{out_tag}] no classes with data — skipping")
        return

    for log_x, log_y, tag in SCALE_COMBOS:
        nrows, ncols = grid_dims(len(classes), max_cols=3)
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows),
                                  sharex=False, sharey=False)
        axes = np.atleast_1d(axes).flatten()

        for ax, cls in zip(axes, classes):
            # Shared edges for this panel: pool `cls` sizes across ALL cell lines
            if harmonize:
                pieces = [
                    get_class_sub(df, name_col_fn(cl), cls)['size']
                    for cl in CELL_LINES
                    if name_col_fn(cl) in df.columns
                ]
                combined = pd.concat(pieces).dropna().values if pieces else np.array([])
                if len(combined) >= MIN_N:
                    eff = max(3, min(N_BINS, len(combined) // MIN_N))
                    if BIN_STRATEGY == 'quantile':
                        shared_edges = np.unique(
                            np.quantile(combined, np.linspace(0, 1, eff + 1)))
                    elif BIN_STRATEGY == 'log':
                        shared_edges = np.logspace(
                            np.log10(max(combined.min(), 1)),
                            np.log10(combined.max()), eff + 1)
                    else:
                        shared_edges = np.linspace(
                            combined.min(), combined.max(), eff + 1)
                else:
                    shared_edges = None
            else:
                shared_edges = None

            for cl in CELL_LINES:
                able_col = ABLE_COL_TEMPLATE.format(cl=cl)
                name_col = name_col_fn(cl)
                if name_col not in df.columns:
                    continue

                sub = get_class_sub(df, name_col, cls)
                if len(sub) < MIN_N:
                    continue

                centers, means, sems = bin_and_summarize(
                    sub, 'size', able_col, N_BINS, BIN_STRATEGY, MIN_N,
                    shared_edges=shared_edges)
                if len(centers) == 0:
                    continue

                centers, means, sems = insert_gaps(
                    centers, means, sems, GAP_BREAK_FACTOR, log_x=log_x)

                color = CELL_COLORS.get(cl, '#888888')
                ax.plot(centers, means, 'o-', color=color, linewidth=1.2,
                        markersize=3, alpha=0.85, label=cl)
                safe_fill(ax, centers, means, sems, color)
                if log_x and log_y:
                    add_powerlaw_fit(ax, centers, means, color, label_prefix=cl)

            ax.set_title(cls, fontsize=13, fontweight='bold')
            style_ax(ax, log_x, log_y)

        for ax in axes[len(classes):]:
            ax.set_visible(False)

        htag     = '_harm_combined' if harmonize else '_perclass'
        cats_tag = 'simplified' if plot_cats is not None else 'all'
        fig.suptitle(
            f'Loop strength vs size — by cell line\n'
            f'one panel per {title_tag} · {tag} · {cats_tag}',
            # + (' · shared edges (combined)' if harmonize else ''),
            fontsize=14, fontweight='bold')
        plt.tight_layout()
        out_path = OUT_DIR / \
            f'loop_strength_by_loopname_{out_tag}_{tag}{htag}_{cats_tag}.svg'
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_path}")


# swap plot_cats=PLOT_CATEGORIES to plot_cats=None to toggle
plot_layout1(df, CLASSES, CLASS_COLORS, loop_name_col,
             'cluster', 'loop name (cluster annotation)',
             plot_cats=PLOT_CATEGORIES)     # ← None = all raw cluster classes

plot_layout2(df, CLASSES, CLASS_COLORS, loop_name_col,
             'cluster', 'loop name (cluster annotation)',
             plot_cats=PLOT_CATEGORIES)     # ← None = all raw cluster classes

# df_manual = derive_manual_loop_class(df)

# if df_manual is not None:
#     manual_name_col_fn = lambda cl: MANUAL_LOOP_NAME_COL_TEMPLATE.format(cl=cl)

#     # swap plot_cats=PLOT_CATEGORIES to plot_cats=None to toggle
#     plot_layout1(df_manual, MANUAL_CLASSES, CLASS_COLORS, manual_name_col_fn,
#                  'manual', 'loop name (manual annotation)',
#                  plot_cats=PLOT_CATEGORIES)  # ← None = all raw manual classes

#     plot_layout2(df_manual, MANUAL_CLASSES, CLASS_COLORS, manual_name_col_fn,
#                  'manual', 'loop name (manual annotation)',
#                  plot_cats=PLOT_CATEGORIES)  # ← None = all raw manual classes

print("\nDone. SVGs written to", OUT_DIR.resolve())