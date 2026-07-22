#!/usr/bin/env python3
"""
anchor_sankey.py
----------------
Alluvial / Sankey plot for cluster or loop-type transitions.

Works for:
  Anchor clusters → plot_anchor_sankey()   col_pat='cluster_{cl}'
  Loop types      → plot_loop_sankey()     col_pat='loop_name_{cl}'
  Custom          → plot_sankey()          col_pat=<your pattern>

  Bar height  ∝ count (ALL rows in that category)
  Bar width   ∝ mean AbLE score (non-NaN rows only; None → uniform)
  Link width  ∝ count (or row-fraction when normalize=True)

When able_col_pat is set, rows that have a valid classification label
but a NaN AbLE score are relabelled '_NO_ABLE_ID' ("No AbLE") AFTER the
name-merge step.  This makes them a full Sankey category with their own
bar (white/dashed) and ribbons showing exactly which other categories
they flow to/from.
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
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
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects

LETTER_SIZE = (8.5, 11)   # inches, portrait. Use (11, 8.5) for landscape.

TRAJECTORY = ['ESC', 'EpiLC', 'd4c7PGCLC', 'GSC']

_PALETTE = [
    '#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
    '#ff7f00', '#a65628', '#f781bf', '#999999',
    '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3',
    '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3',
    '#1b9e77', '#d95f02', '#7570b3', '#e7298a',
    '#66a61e', '#e6ab02', '#a6761d', '#666666',
]

# ── sentinel for rows that have a label but no AbLE score ──────────────
_NO_ABLE_ID    = 'No AbLE'
_NO_ABLE_COLOR = '#cccccc'   # ribbon colour when the source node is No AbLE


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _sort_key(x):
    """
    Numeric IDs first (cluster 0, 1, …, 13-15),
    then alphabetical strings (CRE-CRE, CTCF-CTCF, …),
    then _NO_ABLE_ID always last → top of the bar stack.
    """
    if str(x) == _NO_ABLE_ID:
        return (3, 0, '')
    try:
        return (0, int(x), '')
    except (ValueError, TypeError):
        m = re.match(r'^(\d+)', str(x))
        return (0, int(m.group(1)), str(x)) if m else (1, 0, str(x))


def _sorted_ids(ids):
    return sorted(ids, key=_sort_key)


def _legend_label(cid: str, name: str) -> str:
    """
    Numeric / range IDs  →  'C0: Weak CRE/CTCF'  |  'C13-15: Other'
    String IDs == name   →  'CTCF-CTCF'
    String IDs != name   →  'my-id: Display Name'
    """
    if cid == name:
        return name
    if re.match(r'^\d', str(cid)):
        return f"C{cid}: {name}"
    return f"{cid}: {name}"


def _assign_colors(all_ids: list, provided: dict = None) -> dict:
    colors = dict(provided or {})
    colors.setdefault(_NO_ABLE_ID, _NO_ABLE_COLOR)   # always registered
    missing = [c for c in _sorted_ids(all_ids)
               if c not in colors and c != _NO_ABLE_ID]
    if missing:
        print(f"  Auto-assigning colors to {len(missing)} ID(s): {missing}")
        for i, c in enumerate(missing):
            colors[c] = _PALETTE[i % len(_PALETTE)]
    return colors


# ─────────────────────────────────────────────────────────────
# Cluster / name merging
# ─────────────────────────────────────────────────────────────

def merge_clusters_by_name(
    wide:           pd.DataFrame,
    trajectory:     list,
    col_pat:        str,
    cluster_names:  dict,
    cluster_colors: dict,
) -> tuple:
    """
    Merge IDs sharing the same display name into one canonical ID.
    NaN values preserved throughout — never converted to the string 'nan'.

    Numeric IDs  [13, 14, 15]  →  canonical '13-15'
    String IDs already unique  →  identity remap (no-op for loop types)
    """
    name_to_ids: dict = defaultdict(list)
    for cid, name in cluster_names.items():
        name_to_ids[name].append(str(cid))

    remap:      dict = {}
    new_colors: dict = {}
    new_names:  dict = {}

    for name, ids in name_to_ids.items():
        sids = _sorted_ids(ids)
        if len(sids) == 1:
            cid = sids[0]
            remap[cid]      = cid
            new_colors[cid] = cluster_colors.get(cid, '#777777')
            new_names[cid]  = name
        else:
            canonical = f"{sids[0]}-{sids[-1]}"
            for cid in sids:
                remap[str(cid)] = canonical
            new_colors[canonical] = cluster_colors.get(sids[0], '#777777')
            new_names[canonical]  = name
            print(f"  Merged {sids} → '{canonical}'  ('{name}')")

    # preserve data IDs not listed in cluster_names
    all_raw_ids: set = set()
    for cl in trajectory:
        col = col_pat.format(cl=cl)
        if col in wide.columns:
            all_raw_ids.update(wide[col].dropna().astype(str).unique())
    for cid in all_raw_ids:
        if cid not in remap:
            remap[cid]      = cid
            new_colors[cid] = cluster_colors.get(cid, '#777777')
            new_names[cid]  = cluster_names.get(cid, cid)

    # apply remap — guard pd.notna so NaN cells stay NaN, never become 'nan'
    wide_merged = wide.copy()
    for cl in trajectory:
        col = col_pat.format(cl=cl)
        if col in wide_merged.columns:
            wide_merged[col] = wide_merged[col].apply(
                lambda x, r=remap: r.get(str(x), str(x)) if pd.notna(x) else np.nan
            )

    return wide_merged, new_names, new_colors, remap


# ─────────────────────────────────────────────────────────────
# NaN-AbLE injection  ← new
# ─────────────────────────────────────────────────────────────

def _inject_nan_able_category(
    wide:         pd.DataFrame,
    trajectory:   list,
    col_pat:      str,
    able_col_pat: str,
) -> pd.DataFrame:
    """
    For rows where col_pat is NOT NaN but able_col_pat IS NaN, replace
    the classification value with _NO_ABLE_ID so those rows appear as a
    real Sankey category (white/dashed bar + ribbons) rather than being
    silently excluded from flows.

    Rows where col_pat is already NaN (loop absent from that cell line)
    are untouched and remain excluded from all counts.
    """
    wide = wide.copy()
    for cl in trajectory:
        col      = col_pat.format(cl=cl)
        able_col = able_col_pat.format(cl=cl)
        if col not in wide.columns or able_col not in wide.columns:
            continue
        mask = wide[col].notna() & wide[able_col].isna()
        n    = int(mask.sum())
        if n:
            wide.loc[mask, col] = _NO_ABLE_ID
            print(f"    {cl:15s}: {n:,} rows → '{_NO_ABLE_ID}'  "
                  "(valid label, NaN AbLE)")
    return wide


# ─────────────────────────────────────────────────────────────
# Build node / link data
# ─────────────────────────────────────────────────────────────

def build_sankey_data(
    wide:           pd.DataFrame,
    trajectory:     list,
    col_pat:        str  = 'cluster_{cl}',
    normalize:      bool = False,
    min_flow:       int  = 10,
    cluster_colors: dict = None,
    cluster_names:  dict = None,
) -> dict:
    """
    Purely generic — no special logic for _NO_ABLE_ID needed here because
    it is already a regular string value in the classification columns.
    """
    cluster_names = cluster_names or {}

    all_ids: set = set()
    for cl in trajectory:
        col = col_pat.format(cl=cl)
        if col in wide.columns:
            all_ids.update(wide[col].dropna().astype(str).unique())

    colors = _assign_colors(list(all_ids), cluster_colors)

    # ── nodes ──────────────────────────────────────────────────
    node_list: list = []
    node_idx:  dict = {}
    for cl in trajectory:
        col = col_pat.format(cl=cl)
        if col not in wide.columns:
            print(f"  [WARN] column '{col}' not found — skipping {cl}")
            continue
        for cid in _sorted_ids(wide[col].dropna().astype(str).unique()):
            key = (cl, cid)
            node_idx[key] = len(node_list)
            node_list.append(key)

    # ── links ──────────────────────────────────────────────────
    sources, targets, values, raw_values, link_colors = [], [], [], [], []
    for from_cl, to_cl in zip(trajectory[:-1], trajectory[1:]):
        from_col = col_pat.format(cl=from_cl)
        to_col   = col_pat.format(cl=to_cl)
        if from_col not in wide.columns or to_col not in wide.columns:
            continue
        # dropna excludes rows absent from either CL;
        # _NO_ABLE_ID rows are strings and survive dropna correctly
        pairs  = wide[[from_col, to_col]].dropna().astype(str)
        counts = pd.crosstab(pairs[from_col], pairs[to_col])
        norm   = (
            counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
            if normalize else counts.astype(float)
        )
        for from_cid in counts.index:
            for to_cid in counts.columns:
                raw = int(counts.loc[from_cid, to_cid])
                if raw < min_flow:
                    continue
                from_key = (from_cl, str(from_cid))
                to_key   = (to_cl,   str(to_cid))
                if from_key not in node_idx or to_key not in node_idx:
                    continue
                sources.append(node_idx[from_key])
                targets.append(node_idx[to_key])
                values.append(float(norm.loc[from_cid, to_cid]))
                raw_values.append(raw)
                link_colors.append(colors.get(str(from_cid), '#aaaaaa'))

    return dict(
        node_list=node_list,
        node_idx=node_idx,
        sources=sources,
        targets=targets,
        values=values,
        raw_values=raw_values,
        link_colors=link_colors,
        cluster_colors=colors,
        cluster_names=cluster_names,
        trajectory=trajectory,
        wide=wide,
        col_pat=col_pat,
    )


# ─────────────────────────────────────────────────────────────
# Bézier ribbon helper
# ─────────────────────────────────────────────────────────────

def _bezier(x0, x1, y0_b, y0_t, y1_b, y1_t, n=80):
    t   = np.linspace(0, 1, n)
    mid = x0 + (x1 - x0) * 0.5

    def _b(p0, p1, p2, p3):
        return p0*(1-t)**3 + 3*p1*(1-t)**2*t + 3*p2*(1-t)*t**2 + p3*t**3

    return (
        _b(x0, mid, mid, x1),
        _b(y0_b, y0_b, y1_b, y1_b),
        _b(y0_t, y0_t, y1_t, y1_t),
    )


# ─────────────────────────────────────────────────────────────
# Drawing
# ─────────────────────────────────────────────────────────────

def _draw_alluvial(
    sankey_data:   dict,
    out_path:      Path,
    able_col_pat:  str   = None,
    # wider bars by default
    bar_width_min: float = 0.03,
    bar_width_max: float = 0.1,
    # bring columns closer together
    col_spacing:   float = 0.25,
    gap:           float = 0.012,
    link_alpha:    float = 0.35,
    min_label_h:   float = 0.025,
    title:         str   = 'Cluster transitions',
) -> None:

    trajectory = sankey_data['trajectory']
    wide       = sankey_data['wide']
    colors     = sankey_data['cluster_colors']
    names      = sankey_data['cluster_names']
    col_pat    = sankey_data['col_pat']

    x_pos = {cl: i * col_spacing for i, cl in enumerate(trajectory)}

    # ── bar widths: mean AbLE of non-NaN rows ─────────────────
    # Global v_min / v_max across ALL (cl, cid),
    # No AbLE always gets bar_width_max via fallback in _bw().
    bar_widths: dict = {}
    able_mean:  dict = {}   # key: (cl, cid) → mean AbLE

    if able_col_pat:
        all_vals = []
        for cl in trajectory:
            col      = col_pat.format(cl=cl)
            able_col = able_col_pat.format(cl=cl)
            if col not in wide.columns:
                continue

            for cid in wide[col].dropna().astype(str).unique():
                if cid == _NO_ABLE_ID:
                    # No AbLE never contributes to AbLE scaling
                    continue
                cid_mask = (wide[col] == cid)
                if able_col in wide.columns:
                    able_mask = cid_mask & wide[able_col].notna()
                    if able_mask.any():
                        val = wide.loc[able_mask, able_col].mean()
                        if np.isfinite(val):
                            val = float(val)
                            able_mean[(cl, cid)] = val
                            all_vals.append(val)
                # if able_col missing for this cl: leave it out, will use fallback width

        if all_vals:
            v_min, v_max = min(all_vals), max(all_vals)
            if v_max == v_min:
                # all the same → just give everything max width
                for key in able_mean:
                    bar_widths[key] = bar_width_max
            else:
                rng = v_max - v_min
                for (cl, cid), val in able_mean.items():
                    bar_widths[(cl, cid)] = (
                        bar_width_min
                        + (val - v_min) / rng
                        * (bar_width_max - bar_width_min)
                    )

    def _bw(cl, cid):
        # Anything without an AbLE mean (including No AbLE) gets max width.
        return bar_widths.get((cl, cid), bar_width_max)

    # ── bar y-positions (height ∝ count) ──────────────────────
    bar_y:      dict = {}
    bar_counts: dict = {}
    for cl in trajectory:
        col = col_pat.format(cl=cl)
        if col not in wide.columns:
            continue
        vc    = wide[col].dropna().astype(str).value_counts()
        total = vc.sum()
        y_cur = 0.0
        for cid in _sorted_ids(vc.index):
            n = int(vc.get(cid, 0))
            h = n / total
            bar_y[(cl, cid)]      = (y_cur, y_cur + h)
            bar_counts[(cl, cid)] = n
            y_cur += h + gap

    y_max        = max((yt for _, yt in bar_y.values()), default=1.0)
    y_min        = min((yb for yb, _ in bar_y.values()), default=0.0)
    n_cl_present = sum(1 for cl in trajectory
                       if col_pat.format(cl=cl) in wide.columns)
    max_half_bw  = (max(bar_widths.values()) / 2 if bar_widths
                    else bar_width_max / 2)

    fig, ax = plt.subplots(
        figsize=LETTER_SIZE,
        constrained_layout=True,
    )

    # ── ribbons ────────────────────────────────────────────────
    for from_cl, to_cl in zip(trajectory[:-1], trajectory[1:]):
        from_col = col_pat.format(cl=from_cl)
        to_col   = col_pat.format(cl=to_cl)
        if from_col not in wide.columns or to_col not in wide.columns:
            continue
        pairs  = wide[[from_col, to_col]].dropna().astype(str)
        counts = pd.crosstab(pairs[from_col], pairs[to_col])
        total  = pairs.shape[0]

        from_off = {cid: bar_y.get((from_cl, cid), (0, 0))[0]
                    for cid in counts.index}
        to_off   = {cid: bar_y.get((to_cl,   cid), (0, 0))[0]
                    for cid in counts.columns}

        for from_cid in _sorted_ids(counts.index):
            for to_cid in _sorted_ids(counts.columns):
                n = int(counts.loc[from_cid, to_cid])
                if n == 0:
                    continue
                h  = n / total
                x0 = x_pos[from_cl] + _bw(from_cl, from_cid) / 2
                x1 = x_pos[to_cl]   - _bw(to_cl,   to_cid)   / 2
                bx, by_b, by_t = _bezier(
                    x0, x1,
                    from_off[from_cid], from_off[from_cid] + h,
                    to_off[to_cid],     to_off[to_cid]     + h,
                )
                from_off[from_cid] += h
                to_off[to_cid]     += h
                ax.fill_between(
                    bx, by_b, by_t,
                    color=colors.get(from_cid, '#aaa'),
                    alpha=link_alpha, linewidth=0,
                )

    # ── bars ──────────────────────────────────────────────────
    for (cl, cid), (y_bot, y_top) in bar_y.items():
        x = x_pos[cl]
        w = _bw(cl, cid)
        n = bar_counts.get((cl, cid), 0)
        h = y_top - y_bot

        if cid == _NO_ABLE_ID:
            # Exactly as before: white dashed bar + "No AbLE" label only
            ax.fill_betweenx(
                [y_bot, y_top], x - w / 2, x + w / 2,
                facecolor='white', edgecolor='#555555',
                linewidth=1.2, linestyle='--', zorder=3,
            )
            if h > min_label_h * 0.4:
                ax.text(
                    x, (y_bot + y_top) / 2,
                    f'No AbLE\nn={n:,}',
                    ha='center', va='center',
                    fontsize=6, color='#333333',
                    zorder=4, linespacing=1.3,
                )
        else:
            color = colors.get(cid, '#aaa')
            ax.fill_betweenx(
                [y_bot, y_top], x - w / 2, x + w / 2,
                color=color, linewidth=0.5,
                edgecolor='white', zorder=3,
            )

            if h > min_label_h:
                mean_val = able_mean.get((cl, cid), np.nan)
                if np.isfinite(mean_val):
                    able_str = f'AbLE={mean_val:.4f}'
                else:
                    able_str = ''

                lbl = ax.text(
                    x, (y_bot + y_top) / 2,
                    f'{names.get(cid, cid)}\nn={n:,}\n{able_str}',
                    ha='center', va='center',
                    fontsize=6, fontweight='bold',
                    color='black', zorder=4, linespacing=1.3,
                )


    # ── cell-line axis labels ─────────────────────────────────
    for cl in trajectory:
        if col_pat.format(cl=cl) not in wide.columns:
            continue
        ax.text(x_pos[cl], y_min - 0.04, cl,
                ha='center', va='top',
                fontsize=11, fontweight='bold')

    # ── legend (unchanged, including No AbLE patch) ───────────
    all_cids = _sorted_ids({cid for (_, cid) in bar_y})
    handles  = []
    for c in all_cids:
        if c == _NO_ABLE_ID:
            handles.append(mpatches.Patch(
                facecolor='white', edgecolor='#555555',
                linestyle='--', linewidth=1.2,
                label='No AbLE score',
            ))
        else:
            handles.append(plt.Rectangle(
                (0, 0), 1, 1,
                color=colors.get(c, '#aaa'),
                label=_legend_label(c, names.get(c, c)),
            ))

    ax.legend(handles=handles,
              bbox_to_anchor=(1.01, 1), loc='upper left',
              fontsize=7, frameon=False,
              ncol=max(1, len(handles) // 15))

    ax.set_xlim(-max_half_bw - 0.08,
                (n_cl_present - 1) * col_spacing + max_half_bw + 0.08)
    ax.set_ylim(y_min - 0.10, y_max + 0.06)
    ax.axis('off')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=14)

    fig.savefig(f'{out_path}.png', dpi=200, bbox_inches='tight')
    fig.savefig(f'{out_path}.svg', bbox_inches='tight', format='svg')
    plt.close(fig)
    print(f"  → Saved: {out_path}.png / .svg")

# ─────────────────────────────────────────────────────────────
# Core generic API
# ─────────────────────────────────────────────────────────────

def plot_sankey(
    wide:           pd.DataFrame,
    out_dir:        str,
    col_pat:        str  = 'cluster_{cl}',
    trajectory:     list = None,
    cluster_colors: dict = None,
    cluster_names:  dict = None,
    normalize:      bool = False,
    min_flow:       int  = 10,
    able_col_pat:   str  = None,
    merge_by_name:  bool = True,
    title:          str  = 'Cluster transitions',
) -> dict:
    """
    Parameters
    ----------
    col_pat       : f-string column pattern — '{cl}' → cell-line name.
    able_col_pat  : AbLE signal column pattern.  When set:
                      • bar widths ∝ mean AbLE (non-NaN rows only)
                      • rows with a valid label but NaN AbLE are injected
                        as the 'No AbLE' category with full Sankey flows
                    None → uniform bar widths, no No-AbLE category.
    merge_by_name : merge IDs sharing the same display name before plotting.
    """
    trajectory = trajectory or TRAJECTORY
    out_dir    = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = [col_pat.format(cl=cl) for cl in trajectory
               if col_pat.format(cl=cl) not in wide.columns]
    if missing:
        raise ValueError(
            f"Wide df missing columns: {missing}\n"
            f"Available columns: {list(wide.columns)}"
        )

    # ── 1. name-based merge ────────────────────────────────────
    if merge_by_name and cluster_names:
        print(f"\n  Merging IDs by display name (col_pat='{col_pat}') …")
        wide, cluster_names, cluster_colors, remap = merge_clusters_by_name(
            wide=wide, trajectory=trajectory, col_pat=col_pat,
            cluster_names=cluster_names, cluster_colors=cluster_colors or {},
        )
        print(f"  Remap: {remap}")
    elif merge_by_name and not cluster_names:
        print("  [WARN] merge_by_name=True but no cluster_names — skipping merge")

    # ── 2. inject 'No AbLE' category (after merge) ────────────
    if able_col_pat:
        able_found   = [able_col_pat.format(cl=cl) for cl in trajectory
                        if able_col_pat.format(cl=cl) in wide.columns]
        able_missing = [able_col_pat.format(cl=cl) for cl in trajectory
                        if able_col_pat.format(cl=cl) not in wide.columns]
        if able_missing:
            print(f"  [WARN] AbLE columns not found: {able_missing} "
                  "→ uniform bar width for those cell lines")
        if able_found:
            print(f"  AbLE columns found: {able_found}")
            # register sentinel in colours/names before injection
            cluster_colors = dict(cluster_colors or {})
            cluster_names  = dict(cluster_names  or {})
            cluster_colors.setdefault(_NO_ABLE_ID, _NO_ABLE_COLOR)
            cluster_names.setdefault(_NO_ABLE_ID,  _NO_ABLE_ID)
            print(f"\n  Injecting '{_NO_ABLE_ID}' category …")
            wide = _inject_nan_able_category(
                wide, trajectory, col_pat, able_col_pat
            )

    # ── 3. per-cell-line summary ───────────────────────────────
    print(f"\n{'='*60}")
    print(f"Sankey  col_pat='{col_pat}'  normalize={normalize}  "
          f"min_flow={min_flow}")
    for cl in trajectory:
        col = col_pat.format(cl=cl)
        if col not in wide.columns:
            continue
        valid_mask = wide[col].notna()
        n          = int(valid_mask.sum())
        k          = wide.loc[valid_mask, col].nunique()

        able_str = ''
        if able_col_pat:
            ac = able_col_pat.format(cl=cl)
            if ac in wide.columns:
                has_able   = valid_mask & wide[ac].notna()
                n_nan_able = n - int(has_able.sum())
                mean_able  = wide.loc[has_able, ac].mean()
                able_str   = (f'  mean_AbLE={mean_able:.4f}'
                              f'  NaN_AbLE={n_nan_able:,}')

        print(f"  {cl:15s}: {n:,} rows  {k} unique IDs{able_str}")

    # ── 4. build ───────────────────────────────────────────────
    sankey_data = build_sankey_data(
        wide=wide, trajectory=trajectory, col_pat=col_pat,
        normalize=normalize, min_flow=min_flow,
        cluster_colors=cluster_colors, cluster_names=cluster_names,
    )
    sankey_data['cluster_names'] = cluster_names or {}

    print(f"  Nodes : {len(sankey_data['node_list'])}")
    print(f"  Links : {len(sankey_data['sources'])}  (min_flow={min_flow})")

    # ── 5. draw ────────────────────────────────────────────────
    _draw_alluvial(
        sankey_data,
        out_path=out_dir / 'sankey',
        able_col_pat=able_col_pat,
        title=title,
    )
    return sankey_data


# ─────────────────────────────────────────────────────────────
# Convenience wrappers
# ─────────────────────────────────────────────────────────────

def plot_anchor_sankey(
    wide:           pd.DataFrame,
    out_dir:        str,
    trajectory:     list = None,
    cluster_colors: dict = None,
    cluster_names:  dict = None,
    normalize:      bool = False,
    min_flow:       int  = 10,
    able_col_pat:   str  = 'AbLE_score_{cl}',
    merge_by_name:  bool = True,
    col_pat:        str  = 'cluster_{cl}',
) -> dict:
    """Anchor cluster Sankey.  col_pat defaults to 'cluster_{cl}'."""
    return plot_sankey(
        wide=wide, out_dir=out_dir, col_pat=col_pat,
        trajectory=trajectory,
        cluster_colors=cluster_colors, cluster_names=cluster_names,
        normalize=normalize, min_flow=min_flow, able_col_pat=able_col_pat,
        merge_by_name=merge_by_name,
        title='Anchor cluster transitions',
    )


def plot_loop_sankey(
    wide:          pd.DataFrame,
    out_dir:       str,
    trajectory:    list = None,
    loop_colors:   dict = None,
    loop_names:    dict = None,
    normalize:     bool = False,
    min_flow:      int  = 10,
    able_col_pat:  str  = None,   # set to show No-AbLE flows + bar widths
    merge_by_name: bool = True,
    col_pat:       str  = 'loop_name_{cl}',
) -> dict:
    """
    Loop type Sankey.  col_pat defaults to 'loop_name_{cl}'.

    Set able_col_pat='AbLE_score_{cl}' (or your pattern) to:
      • scale bar widths by mean AbLE (non-NaN rows only)
      • show a 'No AbLE' bar with full Sankey ribbons for loops that
        have a valid loop type but no AbLE score
    """
    return plot_sankey(
        wide=wide, out_dir=out_dir, col_pat=col_pat,
        trajectory=trajectory,
        cluster_colors=loop_colors, cluster_names=loop_names,
        normalize=normalize, min_flow=min_flow,
        able_col_pat=able_col_pat,
        merge_by_name=merge_by_name,
        title='Loop type transitions',
    )


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--wide_tsv',     required=True)
    p.add_argument('--col_pat',      default='cluster_{cl}')
    p.add_argument('--trajectory',   default=None,
                   help='Comma-separated cell-line names')
    p.add_argument('--able_col_pat', default=None)
    p.add_argument('--normalize',    action='store_true')
    p.add_argument('--min_flow',     type=int, default=10)
    p.add_argument('--out_dir',      required=True)
    p.add_argument('--title',        default='Cluster transitions')
    p.add_argument('--no_merge',     action='store_true')
    return p.parse_args()


def main():
    args = _parse_args()
    wide = pd.read_csv(args.wide_tsv, sep='\t')
    print(f"Loaded {len(wide):,} rows")
    trajectory = (
        [c.strip() for c in args.trajectory.split(',')]
        if args.trajectory else None
    )
    plot_sankey(
        wide=wide, out_dir=args.out_dir, col_pat=args.col_pat,
        trajectory=trajectory, normalize=args.normalize,
        min_flow=args.min_flow, able_col_pat=args.able_col_pat,
        merge_by_name=not args.no_merge, title=args.title,
    )
    print('[done]')


if __name__ == '__main__':
    main()