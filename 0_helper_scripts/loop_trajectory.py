#!/usr/bin/env python3
"""
loop_trajectory.py
------------------
Visualise a single loop's AbLE score and anchor cluster pair
across the developmental trajectory.

    AbLE bar per cell line
    cluster pair label on top  e.g. "(5, 5)"

API:
    from loop_trajectory import plot_loop_trajectory

    plot_loop_trajectory(
        loop_df=loop_df,
        loop_id='chr1:1000000-1001000_chr1:2000000-2001000',
        out_path='plots/loop_traj',
    )

    # or pass coordinates directly
    plot_loop_trajectory(
        loop_df=loop_df,
        chrom='chr1', start1=1000000, end1=1001000,
        start2=2000000, end2=2001000,
    )
"""

from pathlib import Path
import numpy as np
import pandas as pd


import matplotlib
# only set Agg if NOT in a notebook/interactive session
import sys
if 'ipykernel' not in sys.modules:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

TRAJECTORY = ['ESC', 'EpiLC', 'd4c7PGCLC', 'GSC']

CL_COLORS = {
    'ESC':       '#e41a1c',
    'EpiLC':     '#4daf4a',
    'd4c7PGCLC': '#377eb8',
    'GSC':       '#dede00',
}

_CLUSTER_PALETTE = [
    '#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
    '#ff7f00', '#a65628', '#f781bf', '#999999',
    '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3',
    '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3',
]


############################################################
# Helpers
############################################################

def _cluster_color(cid) -> str:
    try:
        return _CLUSTER_PALETTE[int(cid) % len(_CLUSTER_PALETTE)]
    except (ValueError, TypeError):
        return '#aaaaaa'


def _find_loop(
    loop_df:  pd.DataFrame,
    loop_id:  str  = None,
    chrom:    str  = None,
    start1:   int  = None,
    end1:     int  = None,
    start2:   int  = None,
    end2:     int  = None,
    id_col:   str  = 'loop_id',
    tol:      int  = 0,
) -> pd.DataFrame:
    """
    Subset loop_df to rows matching the requested loop.
    Returns all cell-line rows for that loop (one row per cell line).
    """
    if loop_id is not None:
        if id_col not in loop_df.columns:
            raise ValueError(f"id_col '{id_col}' not in loop_df")
        rows = loop_df[loop_df[id_col] == loop_id]
        if rows.empty:
            raise ValueError(f"loop_id '{loop_id}' not found in loop_df")
        return rows

    if chrom is not None:
        mask = (
            (loop_df['chrom1'] == chrom) &
            (loop_df['start1'].between(start1 - tol, start1 + tol)) &
            (loop_df['end1'].between(end1   - tol, end1   + tol)) &
            (loop_df['start2'].between(start2 - tol, start2 + tol)) &
            (loop_df['end2'].between(end2   - tol, end2   + tol))
        )
        rows = loop_df[mask]
        if rows.empty:
            raise ValueError(
                f"No loop found at {chrom}:{start1}-{end1} / {start2}-{end2} "
                f"(tol={tol}). Try increasing tol."
            )
        return rows

    raise ValueError("Pass loop_id or chrom+start1+end1+start2+end2")


############################################################
# Main plot
############################################################

def plot_loop_trajectory(
    loop_df:         pd.DataFrame,
    loop_id:         str  = None,
    chrom:           str  = None,
    start1:          int  = None,
    end1:            int  = None,
    start2:          int  = None,
    end2:            int  = None,
    out_path:        str  = None,      # stem, saves .png + .svg; None = show
    trajectory:      list = None,
    cell_line_col:   str  = 'cell_line',
    able_col:        str  = 'AbLE_score',
    cluster_a1_col:  str  = 'cluster_left',
    cluster_a2_col:  str  = 'cluster_right',
    id_col:          str  = 'loop_id',
    tol:             int  = 0,
    title:           str  = None,
    cluster_names:   dict = None,      # {cluster_id_str: display_name}
    figsize:         tuple = (8, 4),
) -> plt.Figure:
    """
    Parameters
    ----------
    loop_df         : long-format loop df — one row per (loop × cell_line).
    loop_id         : unique loop identifier string.
    chrom/start/end : coordinate-based lookup (alternative to loop_id).
    tol             : coordinate matching tolerance in bp.
    cluster_names   : optional display names for cluster ids.
    out_path        : file stem — saves .png + .svg. None → return fig only.

    Returns
    -------
    matplotlib Figure
    """
    trajectory    = trajectory or TRAJECTORY
    cluster_names = cluster_names or {}

    rows = _find_loop(loop_df, loop_id=loop_id, chrom=chrom,
                       start1=start1, end1=end1,
                       start2=start2, end2=end2,
                       id_col=id_col, tol=tol)

    # index by cell line for easy lookup
    rows = rows.set_index(cell_line_col)

    # ---- figure ----
    fig, ax = plt.subplots(figsize=figsize)

    x      = np.arange(len(trajectory))
    width  = 0.6

    for i, cl in enumerate(trajectory):
        if cl not in rows.index:
            # missing — draw empty bar with hatching
            ax.bar(i, 0, width=width,
                   color='#eeeeee', edgecolor='#aaaaaa',
                   linewidth=1, hatch='//', zorder=2)
            ax.text(i, 0.02, 'n/a',
                    ha='center', va='bottom',
                    fontsize=8, color='#aaaaaa')
            continue

        row  = rows.loc[cl]
        able = row[able_col] if able_col in row.index else np.nan

        # bar
        bar_color = CL_COLORS.get(cl, '#888888')
        ax.bar(i, able if np.isfinite(able) else 0,
               width=width,
               color=bar_color,
               alpha=0.85,
               edgecolor='white',
               linewidth=0.8,
               zorder=2)

        # AbLE value inside bar
        if np.isfinite(able) and able > 0:
            ax.text(i, able / 2,
                    f'{able:.2f}',
                    ha='center', va='center',
                    fontsize=8, fontweight='bold',
                    color='white', zorder=3,
                    path_effects=[pe.withStroke(linewidth=2,
                                                foreground='black')])

        # cluster pair label above bar
        c_a1 = str(row[cluster_a1_col]) if cluster_a1_col in row.index else '?'
        c_a2 = str(row[cluster_a2_col]) if cluster_a2_col in row.index else '?'
        n_a1 = cluster_names.get(c_a1, c_a1)
        n_a2 = cluster_names.get(c_a2, c_a2)
        pair_label = f'({n_a1}, {n_a2})'

        y_top = able if (np.isfinite(able) and able > 0) else 0
        txt   = ax.text(i, y_top + ax.get_ylim()[1] * 0.02,
                        pair_label,
                        ha='center', va='bottom',
                        fontsize=9, fontweight='bold',
                        color='black', zorder=4)

        # # small colored squares for each cluster id in the pair label
        # for j, (cid, xoff) in enumerate(zip([c_a1, c_a2], [-0.12, 0.12])):
        #     ax.add_patch(plt.Rectangle(
        #         (i + xoff - 0.05, y_top + ax.get_ylim()[1] * 0.055),
        #         0.10, 0.04 * ax.get_ylim()[1],
        #         color=_cluster_color(cid),
        #         transform=ax.transData,
        #         zorder=5, clip_on=False,
        #     ))

    # ---- axes formatting ----
    ax.set_xticks(x)
    ax.set_xticklabels(trajectory, fontsize=11, fontweight='bold')
    ax.set_ylabel(able_col.replace('_', ' '), fontsize=10)
    ax.set_xlim(-0.6, len(trajectory) - 0.4)
    ax.spines[['top', 'right']].set_visible(False)
    ax.axhline(0, color='black', linewidth=0.8)

    # color x tick labels by cell line
    for tick, cl in zip(ax.get_xticklabels(), trajectory):
        tick.set_color(CL_COLORS.get(cl, 'black'))

    # title
    if title is None:
        if loop_id is not None:
            title = loop_id
        elif chrom is not None:
            title = f'{chrom}:{start1}-{end1}  ×  {start2}-{end2}'
        else:
            title = 'Loop trajectory'
    ax.set_title(title, fontsize=11, fontweight='bold', pad=12) 

    plt.tight_layout()

    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{out}.png', dpi=200, bbox_inches='tight')
        fig.savefig(f'{out}.svg', bbox_inches='tight', format='svg')
        print(f"  → Saved: {out}.png / .svg")
        plt.close(fig)   # only close when saving to file

    return fig    


############################################################
# CLI
############################################################

def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--loop_tsv',        required=True)
    p.add_argument('--loop_id',         default=None)
    p.add_argument('--chrom',           default=None)
    p.add_argument('--start1',          type=int, default=None)
    p.add_argument('--end1',            type=int, default=None)
    p.add_argument('--start2',          type=int, default=None)
    p.add_argument('--end2',            type=int, default=None)
    p.add_argument('--tol',             type=int, default=0)
    p.add_argument('--able_col',        default='AbLE_score')
    p.add_argument('--cluster_a1_col',  default='cluster_left')
    p.add_argument('--cluster_a2_col',  default='cluster_right')
    p.add_argument('--out_path',        default=None)
    return p.parse_args()


def main():
    args    = _parse_args()
    loop_df = pd.read_csv(args.loop_tsv, sep='\t')
    print(f"Loaded {len(loop_df):,} rows")

    fig = plot_loop_trajectory(
        loop_df=loop_df,
        loop_id=args.loop_id,
        chrom=args.chrom,
        start1=args.start1, end1=args.end1,
        start2=args.start2, end2=args.end2,
        tol=args.tol,
        able_col=args.able_col,
        cluster_a1_col=args.cluster_a1_col,
        cluster_a2_col=args.cluster_a2_col,
        out_path=args.out_path,
    )
    if args.out_path is None:
        plt.show()


if __name__ == '__main__':
    main()
