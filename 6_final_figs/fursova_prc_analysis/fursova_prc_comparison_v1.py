#!/usr/bin/env python3
"""
fursova_prc_comparison.py
==========================
Part 1 – Anchor-level comparison vs Fursova PRC domains
  • pybedtools intersect: your cluster-annotated anchors vs Fursova 2096-domain BED
  • Per-cell-line Venn: my PRC (c5/c10)  vs  Fursova-overlapping
  • Bar chart + TSV: Fursova-overlapping, non-PRC anchors → cluster breakdown

Part 2 – Loop filtering + Fursova anchor-overlap annotation
  • Restrict loop_df to loops that are PRC-PRC or PRC-CRE (my def) in ANY
    cell line (MYDEF_GLOBAL, cell-line-agnostic filter — same loops are
    then reused for every cell line downstream).
  • On that restricted set, flag each loop's two anchors vs Fursova:
        fursova_anchor1 / fursova_anchor2   (bool)
  • Four (non-mutually-exclusive) groups built from those flags:
        both       - anchor1 & anchor2
        one_side   - exactly one anchor overlaps (anchor1 XOR anchor2)
        either     - at least one anchor overlaps (anchor1 OR anchor2)
        neither    - no anchor overlaps

Part 3 – Figures (fursova groups only, on the PRC-filtered loop set)
  • APA pileups   : via run_apa_pipeline
  • O/E violins   : per cell line
  • AbLE violins  : per cell line

ASSUMPTIONS
-----------
• anchor_df cols: chrom, start, end, cell_line, cluster
  (one row per anchor × cell line; no duplicates within a cell line)
• loop_df cols:  COORD_COLS, loop_name_{cl}, AbLE_score_{cl}
• All BED files share the same chromosome naming convention (chr prefix etc.)

DEPENDENCIES  (beyond existing env)
------------------------------------
  pip install matplotlib-venn pybedtools
"""

import sys, os
import numpy as np
import pandas as pd
import pybedtools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2
import cooler

plt.rcParams['svg.fonttype'] = 'none'

sys.path.append('/mnt/md0/sjsolar/loop_pred/loop_analyses/0_helper_scripts')
import importlib, apa_clusters
importlib.reload(apa_clusters)
from apa_clusters import (
    CELL_LINES, COORD_COLS, RES, FLANK_BP, MINDIST, MAXDIST, ENRICH_N,
    MCOOL_PAT, EXPECTED_PAT, run_apa_pipeline,
)
from coolpuppy.lib import numutils
from cpu_limit import limit_cpus
limit_cpus(n=12, start=80)

# ─── Paths ────────────────────────────────────────────────────────────────────
FURSOVA_BED    = 'fursova2019_prcdomains.bed'          # 2096 PRC domains
ANCHOR_DF_PATH = '../../3_anchor_clustering/clustering_recluster_per_cellline/anchors_reclustered_per_cl.tsv'
LOOP_DF_PATH = '../../4_loop_strength_prediction/loops/loop_df_reclustered_per_cl_wide.tsv'

CL_COLORS_PATH = '../../data/cl_colors.tsv'
OUT_DIR = 'recluster_per_cellline'

# ─── Constants ────────────────────────────────────────────────────────────────
PRC_CLUSTERS   = {5, 10}
MYDEF_GLOBAL   = ['PRC-PRC', 'PRC-CRE']        # any-cell-line loop filter

FURSOVA_LOOP_CATS = ['both', 'one_side', 'either', 'neither']
FUR_ORDER = [f'fursova_{c}' for c in FURSOVA_LOOP_CATS]

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, 'apa', 'fursova'), exist_ok=True)


############################################################
# Shared utilities
############################################################

def load_cl_colors():
    df = pd.read_csv(CL_COLORS_PATH, sep='\t')
    cl_col    = next(c for c in df.columns if 'cell' in c.lower())
    color_col = next(c for c in df.columns
                     if 'color' in c.lower() or 'colour' in c.lower())
    return dict(zip(df[cl_col], df[color_col]))


def fursova_overlap_set(regions_df):
    """
    Return set of (chrom, int_start, int_end) from regions_df whose
    intervals have any overlap with FURSOVA_BED. Deduplicates before
    the bedtools call for speed.
    """
    print(regions_df.head())
    unique = (regions_df[['chrom','start','end']]
              .drop_duplicates()
              .astype({'start': int, 'end': int}))
    if unique.empty:
        return set()
    bt  = pybedtools.BedTool.from_dataframe(unique)
    fbt = pybedtools.BedTool(FURSOVA_BED)
    ov  = bt.intersect(fbt, u=True)
    try:
        ov_df = ov.to_dataframe(names=['chrom','start','end'])
    except pd.errors.EmptyDataError:
        return set()
    if ov_df.empty:
        return set()
    return set(zip(
        ov_df['chrom'],
        ov_df['start'].astype(int),
        ov_df['end'].astype(int),
    ))


def in_fursova(df, fset, chrom_col, start_col, end_col):
    """Boolean Series aligned to df.index: is each row's interval in fset?"""
    return pd.Series(
        [(c, int(s), int(e)) in fset
         for c, s, e in zip(df[chrom_col], df[start_col], df[end_col])],
        index=df.index,
    )


############################################################
# PART 1 – Anchor-level comparison
############################################################

def part1_anchor_comparison(color_map):
    """
    Load anchor_df, flag Fursova overlaps, produce Venn + bar chart.
    anchor_df expected columns: chrom, start, end, cell_line, cluster
    """
    anchor_df = pd.read_csv(ANCHOR_DF_PATH, sep='\t')
    anchor_df['start'] = anchor_df['mid'] - 1500
    anchor_df['end']   = anchor_df['mid'] + 1500
    required  = ['chrom','start','end','cell_line','cluster']
    missing   = [c for c in required if c not in anchor_df.columns]
    if missing:
        raise ValueError(
            f'anchor_df missing: {missing}\n'
            f'Found: {anchor_df.columns.tolist()}'
        )
    anchor_df[['start','end']] = anchor_df[['start','end']].astype(int)

    fset = fursova_overlap_set(anchor_df)
    anchor_df['is_my_prc']  = anchor_df['cluster'].isin(PRC_CLUSTERS)
    anchor_df['is_fursova'] = in_fursova(anchor_df, fset, 'chrom','start','end')

    # ── summary ──────────────────────────────────────────────────────────────
    print('\n[Part 1] Anchor Fursova-overlap summary (unique anchors per CL):')
    for cl in CELL_LINES:
        sub = (anchor_df[anchor_df.cell_line == cl]
               .drop_duplicates(subset=['chrom','start','end']))
        print(f'  {cl}: n={len(sub)}, '
              f'my_prc={sub.is_my_prc.sum()}, '
              f'fursova={sub.is_fursova.sum()}, '
              f'both={(sub.is_my_prc & sub.is_fursova).sum()}')

    _venn_plot(anchor_df, color_map)
    _non_prc_fursova_report(anchor_df)


def _venn_plot(anchor_df, color_map):
    n = len(CELL_LINES)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, cl in zip(axes, CELL_LINES):
        sub = (anchor_df[anchor_df.cell_line == cl]
               .drop_duplicates(subset=['chrom','start','end']))
        only_mine = int(( sub.is_my_prc & ~sub.is_fursova).sum())
        only_fur  = int((~sub.is_my_prc &  sub.is_fursova).sum())
        both      = int(( sub.is_my_prc &  sub.is_fursova).sum())

        v = venn2(
            subsets=(only_mine, only_fur, both),
            set_labels=(
                f'My PRC\n(c5/c10)\nn={only_mine + both}',
                f'Fursova\nPRC\nn={only_fur + both}',
            ),
            set_colors=(color_map.get(cl, '#6699CC'), '#E07070'),
            alpha=0.65,
            ax=ax,
        )
        for lbl in (v.set_labels or []):
            if lbl:
                lbl.set_fontsize(9)
        ax.set_title(cl, fontweight='bold', fontsize=12)

    fig.suptitle(
        'Anchor overlap: my PRC (c5/c10) vs Fursova PRC domains',
        y=1.02, fontsize=12,
    )
    fig.tight_layout()
    out = os.path.join(OUT_DIR, 'venn_prc_anchors.svg')
    fig.savefig(out, format='svg', bbox_inches='tight')
    print(f'[output] {out}')
    plt.close(fig)


def _non_prc_fursova_report(anchor_df):
    """
    Among Fursova-overlapping anchors that are NOT c5/c10:
    what cluster are they? Bar chart per cell line + TSV.
    """
    sub = (anchor_df[anchor_df.is_fursova & ~anchor_df.is_my_prc]
           .drop_duplicates(subset=['chrom','start','end','cell_line']))
    report = (
        sub.groupby(['cell_line','cluster'])
           .size()
           .reset_index(name='n_anchors')
           .sort_values(['cell_line','n_anchors'], ascending=[True, False])
    )
    out_tsv = os.path.join(OUT_DIR, 'non_prc_fursova_clusters.tsv')
    report.to_csv(out_tsv, sep='\t', index=False)
    print(f'\n[Part 1] Fursova-overlapping non-PRC anchors:\n'
          f'{report.to_string(index=False)}\n[output] {out_tsv}')

    n = len(CELL_LINES)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, cl in zip(axes, CELL_LINES):
        r = report[report.cell_line == cl].sort_values('n_anchors')
        ax.barh(r['cluster'], r['n_anchors'], color='#5B87C5')
        ax.set_title(cl)
        ax.set_xlabel('n anchors')
        ax.set_ylabel('cluster')
    fig.suptitle('Fursova-overlapping, non-PRC anchors: cluster identity',
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, 'non_prc_fursova_cluster_bar.svg')
    fig.savefig(out, format='svg')
    print(f'[output] {out}')
    plt.close(fig)


############################################################
# PART 2 – Restrict to PRC loops, annotate vs Fursova anchors
############################################################

def _name_col(cl): return f'loop_name_{cl}'
def _able_col(cl): return f'AbLE_score_{cl}'


def filter_prc_loops(loop_df):
    """
    Keep loops that are PRC-PRC or PRC-CRE (MYDEF_GLOBAL) in ANY cell line.
    Cell-line-agnostic filter — the resulting loop set is reused as-is
    for every cell line downstream.
    """
    name_cols = [_name_col(cl) for cl in CELL_LINES]
    is_global = np.column_stack(
        [loop_df[c].isin(MYDEF_GLOBAL) for c in name_cols]
    ).any(axis=1)
    filtered = loop_df[is_global].copy()
    print(f'\n[Part 2] {len(filtered)}/{len(loop_df)} loops are '
          f'PRC-PRC/PRC-CRE in >=1 cell line')
    return filtered


def part2_annotate_loops(loop_df):
    """
    Restrict loop_df to PRC-PRC/PRC-CRE loops (any cell line), then add:
      fursova_anchor1  fursova_anchor2   (bool)

    Fursova overlap is computed from the loop's own anchor coordinates
    (independent of anchor_df) to avoid any coordinate-set mismatch.
    """
    loop_df = filter_prc_loops(loop_df)

    all_anchors = pd.concat([
        loop_df[['chrom1','start1','end1']].rename(
            columns={'chrom1':'chrom','start1':'start','end1':'end'}),
        loop_df[['chrom2','start2','end2']].rename(
            columns={'chrom2':'chrom','start2':'start','end2':'end'}),
    ])
    fset = fursova_overlap_set(all_anchors)
    print(f'[Part 2] {len(fset)} unique loop-anchor coords overlap Fursova')

    loop_df['fursova_anchor1'] = in_fursova(
        loop_df, fset, 'chrom1','start1','end1')
    loop_df['fursova_anchor2'] = in_fursova(
        loop_df, fset, 'chrom2','start2','end2')

    both     = (loop_df.fursova_anchor1 &  loop_df.fursova_anchor2).sum()
    one_side = (loop_df.fursova_anchor1 ^  loop_df.fursova_anchor2).sum()
    either   = (loop_df.fursova_anchor1 |  loop_df.fursova_anchor2).sum()
    neither  = (~loop_df.fursova_anchor1 & ~loop_df.fursova_anchor2).sum()
    print(f'[Part 2] group sizes (of {len(loop_df)} PRC loops): '
          f'both={both}, one_side={one_side}, either={either}, neither={neither}')
    return loop_df


############################################################
# PART 3 – Build long-format table (4 fursova groups)
############################################################

def _fursova_masks(loop_df):
    a1, a2 = loop_df.fursova_anchor1, loop_df.fursova_anchor2
    return {
        'both':     a1 & a2,
        'one_side': a1 ^ a2,
        'either':   a1 | a2,
        'neither': ~a1 & ~a2,
    }


def build_fursova_long(loop_df):
    """
    Stack PRC loops × cell_lines × 4 Fursova overlap groups.
    Groups are NOT mutually exclusive (either = both | one_side) —
    kept that way on purpose so 'either' pools statistical power.
    """
    masks  = _fursova_masks(loop_df)
    frames = []
    for cat in FURSOVA_LOOP_CATS:
        sub = loop_df[masks[cat]]
        for cl in CELL_LINES:
            row               = sub.copy()
            row['cell_line']  = cl
            row['AbLE_score'] = row[_able_col(cl)]
            row['category']   = f'fursova_{cat}'
            frames.append(row[COORD_COLS + ['cell_line','AbLE_score','category']])
    df = pd.concat(frames, ignore_index=True)
    print('\n[Fursova long] n per cell_line × category:')
    print(df.groupby(['cell_line','category']).size().unstack(fill_value=0))
    return df


############################################################
# PART 4 – APA pileups
############################################################

def run_all_apa(fursova_long):
    print('\n[APA] Fursova categories (PRC loops only) ...')
    run_apa_pipeline(
        loop_classes={cat: fursova_long[fursova_long.category == cat]
                      for cat in FUR_ORDER},
        out_dir=os.path.join(OUT_DIR, 'apa', 'fursova'),
        tag='fursova', replot=False, n_workers=12,
    )


############################################################
# PART 5 – O/E enrichment violins
############################################################

def _load_expected(cl):
    path = EXPECTED_PAT.format(cell_line=cl, res=RES)
    exp  = pd.read_csv(path, sep='\t')

    rc_candidates = [c for c in exp.columns
                     if c in ('region1', 'region') or 'chrom' in c.lower()]
    if not rc_candidates:
        raise ValueError(
            f'[{cl}] could not find a region/chrom column in expected file '
            f'{path}. Columns present: {exp.columns.tolist()}'
        )
    rc = rc_candidates[0]

    dc_candidates = [c for c in exp.columns if 'dist' in c.lower()]
    if not dc_candidates:
        raise ValueError(
            f'[{cl}] could not find a distance column in expected file '
            f'{path}. Columns present: {exp.columns.tolist()}'
        )
    dc = dc_candidates[0]

    vc_candidates = [c for c in exp.columns if 'balanced.avg' in c.lower()]
    if not vc_candidates:
        raise ValueError(
            f'[{cl}] could not find a balanced.avg value column in expected '
            f'file {path}. Columns present: {exp.columns.tolist()}'
        )
    vc = vc_candidates[0]

    print(f'  [{cl}] expected file columns: {exp.columns.tolist()} '
          f'| using region={rc!r}, dist={dc!r}, value={vc!r} '
          f'| sample region values: {exp[rc].unique()[:5].tolist()}')

    arrays = {}
    for chrom, g in exp.groupby(rc):
        g   = g.sort_values(dc)
        md  = int(g[dc].max())
        arr = np.full(md + 1, np.nan)
        arr[g[dc].astype(int).to_numpy()] = g[vc].to_numpy()
        arrays[chrom] = arr
    return arrays


def _score_loops(long_df, cl, exp_arrays, clr_cache):
    """
    Per-loop log2(O/E) for cell line cl.
    Returns pd.Series(score, index = long_df index subset).
    Mirrors the numutils.get_enrichment metric used by the aggregate APA.
    """
    half = FLANK_BP // RES
    if cl not in clr_cache:
        clr_cache[cl] = cooler.Cooler(
            f'{MCOOL_PAT.format(cell_line=cl)}::resolutions/{RES}')
    clr  = clr_cache[cl]
    offs = np.arange(-half, half + 1)
    sub  = long_df[long_df.cell_line == cl]
    scores, kept = [], []
    sk = dict(chrom=0, dist=0, edge=0, exp_missing_chrom=0, exp_all_nan=0)

    # one-time sanity check: do chrom names in the loop df actually appear
    # in the expected-value dict for this cell line?
    loop_chroms = set(sub['chrom1']).union(sub['chrom2'])
    missing     = loop_chroms - set(exp_arrays.keys())
    if missing:
        print(f'  [{cl}] WARNING: {len(missing)} chrom(s) in loops not found '
              f'in expected dict, e.g. {sorted(missing)[:5]} vs expected keys '
              f'e.g. {sorted(exp_arrays.keys())[:5]}')

    for idx, row in sub.iterrows():
        # ── trans loop ───────────────────────────────────────────────────────
        if row.chrom1 != row.chrom2:
            sk['chrom'] += 1; continue
        ch   = row.chrom1
        m1   = (row.start1 + row.end1) // 2
        m2   = (row.start2 + row.end2) // 2
        dist = abs(m2 - m1)
        if not (MINDIST <= dist <= MAXDIST):
            sk['dist'] += 1; continue

        b1, b2   = m1 // RES, m2 // RES
        lo1, hi1 = b1 - half, b1 + half + 1
        lo2, hi2 = b2 - half, b2 + half + 1
        if lo1 < 0 or lo2 < 0:
            sk['edge'] += 1; continue

        try:
            off = clr.offset(ch)
        except KeyError:
            sk['chrom'] += 1; continue

        try:
            mat = clr.matrix(balance='weight')[
                off+lo1:off+hi1, off+lo2:off+hi2]
        except Exception:
            sk['edge'] += 1; continue
        if mat.shape != (2*half+1, 2*half+1):
            sk['edge'] += 1; continue

        if ch not in exp_arrays:
            sk['exp_missing_chrom'] += 1; continue
        arr = exp_arrays[ch]
        dm  = np.abs((b2-b1) + offs[None,:] - offs[:,None])
        ok  = dm < len(arr)
        em  = np.full_like(dm, np.nan, dtype=float)
        em[ok] = arr[dm[ok]]

        with np.errstate(divide='ignore', invalid='ignore'):
            l2oe = np.log2(mat / em)
        if not np.any(np.isfinite(l2oe)):
            sk['exp_all_nan'] += 1; continue

        scores.append(numutils.get_enrichment(l2oe, ENRICH_N))
        kept.append(idx)

    print(f'  [{cl}] {len(scores)}/{len(sub)} scored | skip: {sk}')
    return pd.Series(scores, index=kept, name='log2_oe')


def _score_df(long_df, clr_cache, exp_cache):
    """
    Score all (loop × cell_line) entries in long_df.
    Returns df with columns: orig_idx, log2_oe, cell_line, category.
    """
    idx_cat = long_df['category'].to_dict()          # index → category label
    out = []
    for cl in CELL_LINES:
        if cl not in exp_cache:
            exp_cache[cl] = _load_expected(cl)
        s = _score_loops(long_df, cl, exp_cache[cl], clr_cache)
        r = s.reset_index()
        r.columns      = ['orig_idx', 'log2_oe']
        r['cell_line'] = cl
        r['category']  = r['orig_idx'].map(idx_cat)
        out.append(r)
    return pd.concat(out, ignore_index=True)


############################################################
# PART 6 – AbLE + O/E violin figures (shared plotter)
############################################################

def _violin_fig_combined(plot_df, y_col, cat_order, ylabel, title, out_path, color_map):
    """
    Single panel: x = category, hue = cell_line, all cell lines together.
    plot_df must have columns: cell_line, category, y_col
    """
    cats = [c for c in cat_order if c in plot_df['category'].values]
    if not cats:
        print(f'[skip] no data for {out_path}')
        return

    cls = [cl for cl in CELL_LINES if cl in plot_df['cell_line'].values]
    if not cls:
        print(f'[skip] no cell lines present for {out_path}')
        return

    pal = {cl: color_map.get(cl, '#6699CC') for cl in cls}

    fig, ax = plt.subplots(figsize=(max(6, 2.2 * len(cats) * len(cls) / 2), 5.5))

    sub = plot_df[plot_df['category'].isin(cats)]
    if sub.empty or sub[y_col].dropna().empty:
        print(f'[skip] all-NaN data for {out_path}')
        plt.close(fig)
        return

    sns.violinplot(
        data=sub, x='category', y=y_col, hue='cell_line',
        order=cats, hue_order=cls, palette=pal,
        cut=0, inner='box', ax=ax, dodge=True,
    )
    ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    ax.set_xticklabels(
        [c.replace('fursova_', '') for c in cats],
        rotation=20, ha='right', fontsize=9,
    )
    ax.set_xlabel('')
    ax.set_ylabel(ylabel)
    ax.legend(title='cell line', bbox_to_anchor=(1.02, 1), loc='upper left',
              fontsize=8, title_fontsize=9)

    # n per (category, cell_line) as a small table under the title, since
    # cramming counts into tick labels gets unreadable with hue-dodging
    ns = sub.groupby(['category', 'cell_line']).size().unstack(fill_value=0)
    print(f'[{out_path}] n per category x cell_line:\n{ns}')

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, format='svg', bbox_inches='tight')
    print(f'[output] {out_path}')
    plt.close(fig)

def _violin_fig(plot_df, y_col, cat_order, ylabel, title, out_path):
    """
    One panel per cell line, x = category, y = y_col.
    plot_df must have columns: cell_line, category, y_col
    """
    cats = [c for c in cat_order if c in plot_df['category'].values]
    if not cats:
        print(f'[skip] no data for {out_path}')
        return

    n       = len(CELL_LINES)
    w_panel = max(3.5, 2.0 * len(cats))
    fig, axes = plt.subplots(1, n, figsize=(w_panel * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    # Consistent palette across all panels
    palette  = dict(zip(cats, sns.color_palette('Set2', len(cats))))
    pal_list = [palette[c] for c in cats]   # list form, max compat with seaborn

    for ax, cl in zip(axes, CELL_LINES):
        sub = plot_df[plot_df.cell_line == cl]
        ns  = sub.groupby('category').size().to_dict()

        if sub.empty or sub[y_col].dropna().empty:
            ax.set_title(cl, fontweight='bold')
            ax.text(0.5, 0.5, 'no data', ha='center',
                    va='center', transform=ax.transAxes)
            continue

        sns.violinplot(
            data=sub, x='category', y=y_col,
            order=cats, palette=pal_list,
            cut=0, inner='box', ax=ax,
        )
        ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
        ax.set_title(cl, fontweight='bold')
        ax.set_xlabel('')
        # Strip prefix from tick labels for readability
        ax.set_xticklabels(
            [f'{c.replace("fursova_","")}\n(n={ns.get(c, 0)})'
             for c in cats],
            rotation=40, ha='right', fontsize=8,
        )
        ax.set_ylabel(ylabel if cl == CELL_LINES[0] else '')

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, format='svg')
    print(f'[output] {out_path}')
    plt.close(fig)


def make_all_violins(fursova_long, color_map):
    """
    Score O/E once, then produce O/E + AbLE violins for the 4 fursova
    overlap groups (both / one_side / either / neither), PRC loops only.
    Emits both the per-cell-line-panel figs and one combined
    all-cell-lines-together fig per metric.
    """
    print('\n[Violins] Scoring O/E enrichment ...')
    clr_cache, exp_cache = {}, {}
    fur_scores = _score_df(fursova_long, clr_cache, exp_cache)

    specs = [
        # (data_df,      y_col,        cat_order, ylabel,      title,                                fname_panel,                    fname_combined)
        (fur_scores,    'log2_oe',    FUR_ORDER, 'log₂(O/E)', 'O/E – Fursova groups (PRC loops)',  'oe_violin_fursova.svg',       'oe_violin_fursova_combined.svg'),
        (fursova_long,  'AbLE_score', FUR_ORDER, 'AbLE score','AbLE – Fursova groups (PRC loops)', 'able_violin_fursova.svg',     'able_violin_fursova_combined.svg'),
    ]
    for data, y_col, cat_order, ylabel, title, fname, fname_combo in specs:
        _violin_fig(data, y_col, cat_order, ylabel, title,
                    os.path.join(OUT_DIR, fname))
        _violin_fig_combined(data, y_col, cat_order, ylabel, title,
                              os.path.join(OUT_DIR, fname_combo), color_map)
############################################################
# Main
############################################################

def main():
    color_map = load_cl_colors()

    print('=' * 65)
    print('PART 1 — Anchor comparison vs Fursova PRC domains')
    print('=' * 65)
    part1_anchor_comparison(color_map)

    print('\n' + '=' * 65)
    print('PART 2 — Restrict to PRC loops, annotate vs Fursova anchors')
    print('=' * 65)
    loop_df = pd.read_csv(LOOP_DF_PATH, sep='\t')
    loop_df = part2_annotate_loops(loop_df)
    out_ann = os.path.join(OUT_DIR, 'loop_df_prc_fursova_annotated.tsv')
    loop_df.to_csv(out_ann, sep='\t', index=False)
    print(f'[output] {out_ann}')

    print('\n' + '=' * 65)
    print('PART 3 — Figures: APA + O/E + AbLE violins (fursova groups)')
    print('=' * 65)
    fursova_long = build_fursova_long(loop_df)

    # run_all_apa(fursova_long)
    make_all_violins(fursova_long, color_map)

    print(f'\n[Done] All outputs in: {OUT_DIR}/')


if __name__ == '__main__':
    main()