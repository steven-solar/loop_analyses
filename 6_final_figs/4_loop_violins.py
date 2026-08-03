import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
import os

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

import cooler
from coolpuppy.lib import numutils


OUTDIR = 'loop_violins/recluster_cre_prc'
os.makedirs(OUTDIR, exist_ok=True)

# loop_df = pd.read_csv('../4_loop_strength_prediction/loops/loop_df_3kb.tsv', sep='\t')
loop_df = pd.read_csv('../4_loop_strength_prediction/loops/loop_df_cre_prc_reclustered.tsv', sep='\t')
# loop_df = pd.read_csv('../4_loop_strength_prediction/loops/loop_df_reclustered_per_cl.tsv', sep='\t')
color_tsv = '../4_loop_strength_prediction/loops/loop_colors.tsv'


# ─── Configuration ─────────────────────────────────────────────────────────────
CELL_LINES = ['ESC', 'EpiLC', 'd4c7PGCLC', 'GSC']
CELL_LINE_COLORS = {
    'ESC':       '#4C72B0',
    'EpiLC':     '#DD8452',
    'd4c7PGCLC': '#55A868',
    'GSC':       '#C44E52',
}
COORD_COLS    = ['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2']
LOOP_NAME_COL = 'loop_name'
CELL_LINE_COL = 'cell_line'
PRC_PRC_LABEL = 'PRC-PRC'

# Hi-C parameters — keep in sync with your apa_clusters constants
RES      = 1_000        # resolution (bp)
FLANK_BP = 10_000       # pileup half-width (bp)
ENRICH_N = 3            # numutils.get_enrichment inner square half-size
MINDIST  = 20_000       # minimum loop span (bp)
MAXDIST  = 2_000_000    # maximum loop span (bp)

MCOOL_PAT = "/mnt/coldstorage/shares/Masahiro/microc/mcs/{cell_line}_WT_10B.mcool"
EXPECTED_PAT = 'expected/{cell_line}_cis_expected_res500.tsv'


def load_color_map(color_tsv):
    """Load category -> color mapping from tsv (columns: loop_name, color)."""
    cmap_df = pd.read_csv(color_tsv, sep='\t')
    return dict(zip(cmap_df['loop_name'], cmap_df['color']))


def _log10_tick_formatter(x, pos):
    """Format a value that is already in log10 units as 10^x."""
    return r'$10^{%d}$' % round(x)


def violin_by_category(plot_df, category_col, column, ylabel, out_path,
                        color_map, order=None, figsize=(8, 5),
                        bw_adjust=0.6, log_scale=False, median_unit=None):
    """
    Violin plot (with boxplot drawn inside) of `column` distribution, one
    violin per category in `category_col`, colored using a color_map dict.

    plot_df       : dataframe with category_col and `column`
    category_col  : e.g. 'loop_name' or 'cell_line'
    column        : e.g. 'AbLE_score' or 'size' (raw, untransformed values)
    ylabel        : y-axis label
    out_path      : output path (.svg recommended)
    color_map     : dict mapping category -> color
    order         : optional list to fix category order (defaults to
                    color_map order)
    log_scale     : if True, the KDE/boxplot are computed on log10(column)
                    rather than on the raw values, and the resulting
                    (linear) axis is relabeled with 10^x tick labels.
                    Requires all values in `column` to be > 0.
    median_unit   : optional str, one of {'kb', None}. If 'kb', the median
                    annotation is converted from bp -> kb (i.e. divided by
                    1000) and suffixed with " kb".
    """
    if order is None:
        order = list(color_map.keys())
    present = set(plot_df[category_col].unique())
    order = [c for c in order if c in present]

    df = plot_df.copy()
    plot_column = column
    if log_scale:
        n_nonpos = (df[column] <= 0).sum()
        if n_nonpos:
            print(f'Warning: dropping {n_nonpos} rows with {column} <= 0 '
                  f'before log10 transform')
        df = df.loc[df[column] > 0].copy()
        plot_column = f'{column}_log10'
        df[plot_column] = np.log10(df[column])

    fig, ax = plt.subplots(figsize=figsize)

    sns.violinplot(
        data=df, x=category_col, y=plot_column, order=order,
        palette=color_map, cut=0, bw_adjust=bw_adjust,
        inner=None, linewidth=0.8, ax=ax
    )

    sns.boxplot(
        data=df, x=category_col, y=plot_column, order=order,
        width=0.15, showfliers=False, boxprops={'zorder': 4, 'facecolor': 'white'},
        whiskerprops={'zorder': 4}, capprops={'zorder': 4},
        medianprops={'zorder': 4, 'color': 'black'},
        ax=ax
    )

    if log_scale:
        # ticks land at integer log10 values, labeled as 10^x
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_log10_tick_formatter))

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha='right')

    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    offset = y_range * 0.02

    for i, cat in enumerate(order):
        vals = plot_df.loc[plot_df[category_col] == cat, column].dropna()
        vals = vals[vals > 0] if log_scale else vals
        if len(vals) == 0:
            continue
        median = np.median(vals)
        top = np.log10(vals.max()) if log_scale else vals.max()

        if median_unit == 'kb':
            label = f'{median / 1000:.3g} kb'
        else:
            label = f'{median:.3g}'

        y_pos = top + offset

        ax.text(
            i, y_pos,
            label,
            ha='center', va='bottom',
            fontsize=7, color='black', fontweight='bold',
            zorder=5
        )

    ax.set_ylim(top=ax.get_ylim()[1] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.06)

    ax.set_xlabel(category_col)
    ax.set_ylabel(ylabel)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig(out_path, bbox_inches='tight', format='svg')
    plt.close(fig)


def violin_by_loop_name(loop_df, column, ylabel, out_path,
                        color_tsv, loop_name_col='loop_name',
                        order=None, figsize=(8, 5), bw_adjust=0.6,
                        log_scale=False, median_unit=None):
    """Thin wrapper over violin_by_category for the loop_name x color_tsv case."""
    color_map = load_color_map(color_tsv)
    violin_by_category(
        loop_df, loop_name_col, column, ylabel, out_path, color_map,
        order=order, figsize=figsize, bw_adjust=bw_adjust,
        log_scale=log_scale, median_unit=median_unit
    )

# ── Plot: existing per-loop-category violins ────────────────────────────────
# AbLE score: plotted on a log-scaled axis (raw values), no unit conversion
# violin_by_loop_name(
#     loop_df, 'AbLE_score', 'AbLE score',
#     f'{OUTDIR}/AbLE_score_by_loop_name.svg', color_tsv=color_tsv,
#     log_scale=True
# )
# # Loop size: plotted on a log-scaled axis (raw bp values), median labeled in kb
# violin_by_loop_name(
#     loop_df, 'size', 'Loop size (bp)',
#     f'{OUTDIR}/size_by_loop_name.svg', color_tsv=color_tsv,
#     log_scale=True, median_unit='kb'
# )

# prc_prc_df = loop_df.loc[loop_df[LOOP_NAME_COL] == PRC_PRC_LABEL].copy()
# print(f'PRC-PRC loops: {len(prc_prc_df)} / {len(loop_df)} total')
# CELL_LINE_COL = 'cell_line'
# ABLE_COL = 'AbLE_score'
# violin_by_category(
#     prc_prc_df, CELL_LINE_COL, ABLE_COL, 'AbLE score',
#     f'{OUTDIR}/AbLE_score_by_cell_line_PRC_PRC.svg',
#     color_map=CELL_LINE_COLORS, order=CELL_LINES,
#     log_scale=True
# )

#  Filter to PRC-PRC 
prc_prc_df = loop_df.loc[loop_df[LOOP_NAME_COL] == PRC_PRC_LABEL].copy()
print(f'PRC-PRC loops: {len(prc_prc_df):,} / {len(loop_df):,} total')
print(prc_prc_df.groupby(CELL_LINE_COL).size().rename('n_loops'))


# ─── Step 2 ── Load expected values ───────────────────────────────────────────
def load_expected(cl: str) -> dict[str, np.ndarray]:
    """
    Read cooltools expected-value TSV; return {chrom: array[dist → avg]}.
    Column names are auto-detected to tolerate minor format variation.
    """
    path = EXPECTED_PAT.format(cell_line=cl, res=RES)
    exp  = pd.read_csv(path, sep='\t')

    # flexible column detection (mirrors fursova script)
    rc = next(c for c in exp.columns
              if c in ('region1', 'region') or 'chrom' in c.lower())
    dc = next(c for c in exp.columns if 'dist'         in c.lower())
    vc = next(c for c in exp.columns if 'balanced.avg' in c.lower())
    print(f'  [{cl}] expected → region={rc!r}, dist={dc!r}, value={vc!r}')

    arrays: dict[str, np.ndarray] = {}
    for chrom, g in exp.groupby(rc):
        g   = g.sort_values(dc)
        md  = int(g[dc].max())
        arr = np.full(md + 1, np.nan)
        arr[g[dc].astype(int).values] = g[vc].values
        arrays[chrom] = arr
    return arrays


# ─── Step 3 ── Score O/E per loop ─────────────────────────────────────────────
def score_loops_oe(
    sub_df:     pd.DataFrame,
    cl:         str,
    exp_arrays: dict[str, np.ndarray],
    clr_cache:  dict,
) -> pd.Series:
    """
    Return a pd.Series(log2_oe, index=sub_df.index) for cell line *cl*.
    Each value is the central-pixel enrichment from a ±FLANK_BP pileup.
    """
    half = FLANK_BP // RES
    offs = np.arange(-half, half + 1)

    if cl not in clr_cache:
        clr_cache[cl] = cooler.Cooler(
            f'{MCOOL_PAT.format(cell_line=cl)}::resolutions/{RES}')
    clr = clr_cache[cl]

    # sanity-check chromosome name consistency
    all_chroms  = set(sub_df['chrom1']) | set(sub_df['chrom2'])
    missing_chr = all_chroms - set(exp_arrays)
    if missing_chr:
        print(f'  [{cl}] WARNING: {len(missing_chr)} chrom(s) absent from '
              f'expected, e.g. {sorted(missing_chr)[:4]}')

    scores, kept = [], []
    sk = dict(trans=0, dist=0, edge=0, exp=0)

    for idx, row in sub_df.iterrows():

        # ── trans / distance guard ─────────────────────────────────────────────
        if row.chrom1 != row.chrom2:
            sk['trans'] += 1; continue

        ch   = row.chrom1
        m1   = (int(row.start1) + int(row.end1)) // 2
        m2   = (int(row.start2) + int(row.end2)) // 2
        dist = abs(m2 - m1)
        if not (MINDIST <= dist <= MAXDIST):
            sk['dist'] += 1; continue

        # ── bin indices ────────────────────────────────────────────────────────
        b1, b2 = m1 // RES, m2 // RES
        lo1, hi1 = b1 - half, b1 + half + 1
        lo2, hi2 = b2 - half, b2 + half + 1
        if lo1 < 0 or lo2 < 0:
            sk['edge'] += 1; continue

        # ── fetch balanced contact matrix ──────────────────────────────────────
        try:
            off = clr.offset(ch)
            mat = clr.matrix(balance='weight')[
                off + lo1 : off + hi1,
                off + lo2 : off + hi2,
            ]
        except Exception:
            sk['edge'] += 1; continue
        if mat.shape != (2 * half + 1, 2 * half + 1):
            sk['edge'] += 1; continue

        # ── build expected matrix at matching distances ────────────────────────
        if ch not in exp_arrays:
            sk['exp'] += 1; continue
        arr = exp_arrays[ch]
        dm  = np.abs((b2 - b1) + offs[None, :] - offs[:, None])
        em  = np.full_like(dm, np.nan, dtype=float)
        ok  = dm < len(arr)
        em[ok] = arr[dm[ok]]

        # ── log2(O/E) → central enrichment ────────────────────────────────────
        with np.errstate(divide='ignore', invalid='ignore'):
            l2oe = np.log2(mat / em)
        if not np.any(np.isfinite(l2oe)):
            sk['exp'] += 1; continue

        scores.append(numutils.get_enrichment(l2oe, ENRICH_N))
        kept.append(idx)

    print(f'  [{cl}] scored {len(scores):,}/{len(sub_df):,} | skipped: {sk}')
    return pd.Series(scores, index=kept, name='log2_oe')


def build_oe_long(prc_prc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Score O/E for every (loop × cell_line) row in *prc_prc_df*.
    Returns tidy DataFrame with COORD_COLS + [cell_line, log2_oe].
    """
    clr_cache: dict = {}
    exp_cache: dict = {}
    frames = []

    for cl in CELL_LINES:
        sub = prc_prc_df[prc_prc_df[CELL_LINE_COL] == cl]
        if sub.empty:
            print(f'[O/E] {cl}: no PRC-PRC rows — skipping')
            continue

        print(f'\n[O/E] {cl}: loading expected …')
        if cl not in exp_cache:
            exp_cache[cl] = load_expected(cl)

        s  = score_loops_oe(sub, cl, exp_cache[cl], clr_cache)

        df = prc_prc_df.loc[s.index, COORD_COLS].copy()
        df[CELL_LINE_COL] = cl
        df['log2_oe']     = s.values
        frames.append(df)

    if not frames:
        raise RuntimeError('No loops scored — check MCOOL_PAT / EXPECTED_PAT.')

    oe_df = pd.concat(frames, ignore_index=True)
    print(f'\n[O/E] Total scored: {len(oe_df):,} loop×cell-line pairs')
    print(oe_df.groupby(CELL_LINE_COL)['log2_oe'].describe().round(3))
    return oe_df


# ─── Step 4 ── Violin plot ─────────────────────────────────────────────────────
def violin_oe_by_cell_line(
    oe_df:     pd.DataFrame,
    out_path:  str,
    order:     list = CELL_LINES,
    color_map: dict = CELL_LINE_COLORS,
) -> None:
    """
    One violin per cell line, y = log₂(O/E).
    Directly mirrors violin_by_category() used for AbLE scores.
    """
    # keep only cell lines that actually have data, in requested order
    cls     = [cl for cl in order if cl in oe_df[CELL_LINE_COL].values]
    palette = [color_map.get(cl, '#6699CC') for cl in cls]
    counts  = oe_df.groupby(CELL_LINE_COL)['log2_oe'].count()

    fig, ax = plt.subplots(figsize=(max(4, 1.8 * len(cls)), 5))

    sns.violinplot(
        data    = oe_df,
        x       = CELL_LINE_COL,
        y       = 'log2_oe',
        order   = cls,
        palette = palette,
        cut     = 0,           # no kernel extrapolation beyond data range
        inner   = 'box',       # show IQR box inside each violin
        ax      = ax,
    )

    ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)   # O/E = 1 reference

    ax.set_xticks(range(len(cls)))
    ax.set_xticklabels(
        [f'{cl}\n(n={counts.get(cl, 0):,})' for cl in cls],
        fontsize=9,
    )
    ax.set_xlabel('')
    ax.set_ylabel('log₂(O/E)', fontsize=11)
    ax.set_title('O/E enrichment – PRC-PRC loops', fontsize=12, fontweight='bold')

    fig.tight_layout()
    fig.savefig(out_path, format='svg', bbox_inches='tight')
    print(f'[output] {out_path}')
    plt.close(fig)


# ─── Run ───────────────────────────────────────────────────────────────────────
oe_df = build_oe_long(prc_prc_df)

violin_oe_by_cell_line(
    oe_df,
    out_path  = f'{OUTDIR}/oe_score_by_cell_line_PRC_PRC.svg',
    order     = CELL_LINES,
    color_map = CELL_LINE_COLORS,
)
