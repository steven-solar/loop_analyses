# conda activate coolpuppy_env (or really most are fine)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import math

# ============================== CONFIG ==============================
CELL_LINES = ['ESC', 'EpiLC', 'd4c7PGCLC', 'GSC']

LOOP_DF_PATH = '../4_loop_strength_prediction/loops/loop_df_3kb_wide.tsv'

PER_CELL_LINE_LOOP_NAME = True
LOOP_NAME_COL_TEMPLATE  = 'loop_name_{cl}'
GLOBAL_LOOP_NAME_COL    = 'loop_name'

ABLE_COL_TEMPLATE = 'AbLE_score_{cl}'

CL_COLOR_TSV   = '../data/cl_colors.tsv'
LOOP_COLOR_TSV = '../4_loop_strength_prediction/loops/loop_colors.tsv'

MANUAL_BOOL_COL_TEMPLATE = 'is_only_{feat}_{side}_{cl}'
MANUAL_NONE_COL_TEMPLATE = 'is_None_{side}_{cl}'

FEAT_MAP  = {'CRE': 'CRE', 'CTCF': 'CTCF', 'PRC_narrow': 'PRC'}
FEAT_RANK = {'CTCF': 0, 'CRE': 1, 'PRC': 2}

MANUAL_LOOP_NAME_COL_TEMPLATE = 'manual_loop_name_{cl}'

MANUAL_CLASSES = ['CTCF-CTCF', 'CTCF-CRE', 'CTCF-PRC',
                  'CRE-CRE',   'CRE-PRC',   'PRC-PRC',  'Other-related']

OUT_DIR  = Path('loop_strength_decay')

MIN_N             = 10     # minimum points needed to draw anything
MAX_SCATTER_N     = 3000   # max points to *display* per group (all used for the fit)
SCATTER_ALPHA     = 0.20   # raw-point transparency
SCATTER_SIZE      = 3      # raw-point marker size (pt²)
FIT_LINEWIDTH     = 2.0    # regression line width
# ====================================================================

OUT_DIR.mkdir(exist_ok=True, parents=True)
plt.rcParams['svg.fonttype'] = 'none'


# ── helpers ──────────────────────────────────────────────────────────

def load_color_map(path):
    d = pd.read_csv(path, sep='\t')
    name_col, color_col = d.columns[:2]
    cmap = dict(zip(d[name_col], d[color_col]))
    print(f"Loaded {len(cmap)} colors from {path}: {list(cmap.keys())}")
    return cmap


CELL_COLORS  = load_color_map(CL_COLOR_TSV)
CLASS_COLORS = load_color_map(LOOP_COLOR_TSV)
CLASSES      = list(CLASS_COLORS.keys())

for cl in CELL_LINES:
    if cl not in CELL_COLORS:
        print(f"WARNING: no color for cell line '{cl}', will use gray")
for c in MANUAL_CLASSES:
    if c not in CLASS_COLORS:
        print(f"WARNING: no color for manual class '{c}', will use gray")

df = pd.read_csv(LOOP_DF_PATH, sep='\t')
print(f"Loaded {len(df)} loops")
print(f"Columns: {list(df.columns)[:20]}{'...' if len(df.columns) > 20 else ''}")


def loop_name_col(cl):
    return LOOP_NAME_COL_TEMPLATE.format(cl=cl) if PER_CELL_LINE_LOOP_NAME else GLOBAL_LOOP_NAME_COL


# ── manual annotation ────────────────────────────────────────────────

def anchor_identity_vec(df, side, cl):
    """Return a Series of identity labels ('CTCF'/'CRE'/'PRC'/'None')
    for one anchor side and one cell line."""
    ident = pd.Series('None', index=df.index, dtype=object)
    for feat, label in FEAT_MAP.items():
        col = MANUAL_BOOL_COL_TEMPLATE.format(feat=feat, side=side, cl=cl)
        if col in df.columns:
            ident = ident.where(~df[col].astype(bool), label)
        else:
            print(f"  WARNING: '{col}' not found — '{label}' {side} anchors "
                  f"won't be called for {cl}.")
    none_col = MANUAL_NONE_COL_TEMPLATE.format(side=side, cl=cl)
    if none_col in df.columns:
        ident = ident.where(~df[none_col].astype(bool), 'None')
    return ident


def derive_manual_loop_class(df):
    """Add a per-cell-line manual_loop_name_{cl} column derived from
    is_only_*_{cl} boolean columns. Returns augmented df or None."""

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
            print(f"WARNING: no is_only_* columns for {cl} — skipping.")
            continue

        left_id  = anchor_identity_vec(out, 'left',  cl)
        right_id = anchor_identity_vec(out, 'right', cl)
        classes  = [pair_name(a, b) for a, b in zip(left_id, right_id)]

        dest = MANUAL_LOOP_NAME_COL_TEMPLATE.format(cl=cl)
        out[dest] = classes
        print(f"Manual classes for {cl}:",
              pd.Series(classes).value_counts().to_dict())
        any_derived = True

    if not any_derived:
        print("WARNING: no is_only_* columns found anywhere — "
              "skipping manual analysis.")
        return None
    return out


# ── core plotting primitive ──────────────────────────────────────────

def scatter_with_fit(ax, x, y, color, label, log_x, log_y,
                     rng=None):
    """Scatter raw points (subsampled for display) + best-fit line.

    The regression is performed in the *transformed* space so it plots as a
    straight line on the chosen axes:
      log-log  → power-law  (y = a·x^b)
      logx-liny → log-linear
      linx-logy → semi-log
      lin-lin  → linear

    Returns (slope, r²) in the transformed space, or (nan, nan).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # keep only finite, and positive where log scale demands it
    mask = np.isfinite(x) & np.isfinite(y)
    if log_x: mask &= (x > 0)
    if log_y: mask &= (y > 0)
    x, y = x[mask], y[mask]

    if len(x) < MIN_N:
        return np.nan, np.nan

    # ---- regression (all surviving points) ----
    tx = np.log10(x) if log_x else x.copy()
    ty = np.log10(y) if log_y else y.copy()

    if not (np.isfinite(tx).all() and np.isfinite(ty).all()):
        return np.nan, np.nan

    slope, intercept, r, *_ = stats.linregress(tx, ty)

    # ---- scatter (subsampled for display) ----
    if len(x) > MAX_SCATTER_N:
        idx = rng.choice(len(x), MAX_SCATTER_N, replace=False)
        xs, ys = x[idx], y[idx]
    else:
        xs, ys = x, y

    ax.scatter(xs, ys,
               color=color, alpha=SCATTER_ALPHA, s=SCATTER_SIZE,
               linewidths=0, rasterized=True, zorder=2)

    # ---- fit line ----
    tx_fit  = np.linspace(tx.min(), tx.max(), 300)
    ty_fit  = intercept + slope * tx_fit
    x_fit   = 10**tx_fit if log_x else tx_fit
    y_fit   = 10**ty_fit if log_y else ty_fit

    ax.plot(x_fit, y_fit,
            color=color, linewidth=FIT_LINEWIDTH, alpha=0.95, zorder=3,
            label=f'{label}  b={slope:.2f}  r²={r**2:.2f}')

    return slope, r**2


def style_ax(ax, log_x, log_y):
    if log_x: ax.set_xscale('log')
    if log_y: ax.set_yscale('log')
    ax.set_xlabel('Loop size (bp)', fontsize=11)
    ax.set_ylabel('AbLE score',     fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=7, frameon=False)


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


# ── layout 1: one panel per cell line, series = loop classes ─────────

def plot_layout1(df, classes, class_colors, name_col_fn, out_tag, title_tag):
    rng = np.random.default_rng(42)
    for log_x, log_y, tag in SCALE_COMBOS:
        nrows, ncols = grid_dims(len(CELL_LINES), max_cols=2)
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(7 * ncols, 5 * nrows),
                                  sharex=False, sharey=False)
        axes = np.atleast_1d(axes).flatten()

        for ax, cl in zip(axes, CELL_LINES):
            able_col = ABLE_COL_TEMPLATE.format(cl=cl)
            name_col = name_col_fn(cl)
            if name_col not in df.columns or able_col not in df.columns:
                ax.set_visible(False)
                continue

            for cls in classes:
                sub = df[df[name_col] == cls]
                if len(sub) < MIN_N:
                    continue
                color = class_colors.get(cls, '#888888')
                scatter_with_fit(ax,
                                 sub['size'].values, sub[able_col].values,
                                 color=color, label=cls,
                                 log_x=log_x, log_y=log_y, rng=rng)

            ax.set_title(cl, fontsize=13, fontweight='bold')
            style_ax(ax, log_x, log_y)

        for ax in axes[len(CELL_LINES):]:
            ax.set_visible(False)

        fig.suptitle(
            f'Loop strength vs size — {title_tag}\n'
            f'one panel per cell line · {tag}',
            fontsize=14, fontweight='bold')
        plt.tight_layout()
        out_path = OUT_DIR / f'scatter_by_cellline_{out_tag}_{tag}.svg'
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_path}")


# ── layout 2: one panel per loop class, series = cell lines ──────────

def plot_layout2(df, classes, class_colors, name_col_fn, out_tag, title_tag):
    rng = np.random.default_rng(42)
    for log_x, log_y, tag in SCALE_COMBOS:
        nrows, ncols = grid_dims(len(classes), max_cols=3)
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(7 * ncols, 5 * nrows),
                                  sharex=False, sharey=False)
        axes = np.atleast_1d(axes).flatten()

        for ax, cls in zip(axes, classes):
            for cl in CELL_LINES:
                able_col = ABLE_COL_TEMPLATE.format(cl=cl)
                name_col = name_col_fn(cl)
                if name_col not in df.columns or able_col not in df.columns:
                    continue

                sub = df[df[name_col] == cls]
                if len(sub) < MIN_N:
                    continue

                color = CELL_COLORS.get(cl, '#888888')
                scatter_with_fit(ax,
                                 sub['size'].values, sub[able_col].values,
                                 color=color, label=cl,
                                 log_x=log_x, log_y=log_y, rng=rng)

            ax.set_title(cls, fontsize=13, fontweight='bold')
            style_ax(ax, log_x, log_y)

        for ax in axes[len(classes):]:
            ax.set_visible(False)

        fig.suptitle(
            f'Loop strength vs size — cell lines\n'
            f'one panel per {title_tag} · {tag}',
            fontsize=14, fontweight='bold')
        plt.tight_layout()
        out_path = OUT_DIR / f'scatter_by_loopname_{out_tag}_{tag}.svg'
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_path}")


# ── run ───────────────────────────────────────────────────────────────

plot_layout1(df, CLASSES, CLASS_COLORS, loop_name_col,
             'cluster', 'loop name (cluster annotation)')
plot_layout2(df, CLASSES, CLASS_COLORS, loop_name_col,
             'cluster', 'loop name (cluster annotation)')

df_manual = derive_manual_loop_class(df)

if df_manual is not None:
    manual_name_col_fn = lambda cl: MANUAL_LOOP_NAME_COL_TEMPLATE.format(cl=cl)

    plot_layout1(df_manual, MANUAL_CLASSES, CLASS_COLORS, manual_name_col_fn,
                 'manual', 'loop name (manual annotation)')
    plot_layout2(df_manual, MANUAL_CLASSES, CLASS_COLORS, manual_name_col_fn,
                 'manual', 'loop name (manual annotation)')

print("\nDone. SVGs written to", OUT_DIR.resolve())