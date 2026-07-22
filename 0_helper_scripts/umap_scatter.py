#!/usr/bin/env python
"""
umap_scatter.py
----------------
Generic UMAP scatter plotter.

  plot_umap_scatter(df, out_path, panels, ...)

Each panel is a dict describing one subplot:
  col    : str   column in df to colour by          (required)
  title  : str   panel title
  labels : dict  raw_value -> display label          (optional)
  colors : dict  display label -> hex color          (optional)
  smush  : bool  merge same-label groups into one    (default False)
                 legend entry + same color

df must have columns 'UMAP1' and 'UMAP2'.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# default palette — extend as needed
_PALETTE = [
    '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
    '#a65628', '#f781bf', '#999999', '#66c2a5', '#fc8d62',
    '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494',
    '#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e',
    '#e6ab02', '#a6761d', '#b3b3b3', '#666666',
]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _fill_colors(groups: list, provided: dict) -> dict:
    """Auto-assign palette colors for any group not in provided."""
    out = dict(provided)
    missing = [g for g in groups if g not in out]
    for i, g in enumerate(missing):
        out[g] = _PALETTE[i % len(_PALETTE)]
    return out


def _prepare_panel(df: pd.DataFrame, panel: dict) -> list:
    """
    Returns list of (legend_label, color, sub_df) tuples ready for ax.scatter.

    smush=False  each raw value is its own scatter group.
                 If labels given, legend reads "DisplayName (raw_value)".
    smush=True   all raw values that share a display label are merged:
                 same color, single legend entry.
    """
    col    = panel['col']
    labels = panel.get('labels')        # raw_value -> display_label
    colors = panel.get('colors') or {}  # display_label -> hex
    smush  = panel.get('smush', False)

    raw = df[col].astype(str)

    # resolve display labels
    if labels is not None:
        str_labels = {str(k): str(v) for k, v in labels.items()}
        display = raw.map(str_labels).fillna(raw)
    else:
        display = raw

    # decide plot-groups
    if smush and labels is not None:
        group_series = display                          # merge same-named
    elif labels is not None:
        group_series = display + ' (' + raw + ')'      # distinct but named
    else:
        group_series = raw

    unique_groups = sorted(group_series.unique(), key=str)
    colors = _fill_colors(unique_groups, colors)

    return [
        (g, colors[g], df[group_series == g])
        for g in unique_groups
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_umap_scatter(
    df: pd.DataFrame,
    out_path: str,
    panels: list,
    title: str = '',
    figsize: tuple = None,
    s: float = 1.5,
    alpha: float = 0.3,
    legend_ncol: int = 2,
    legend_fontsize: int = 7,
    dpi: int = 300,
) -> None:
    """
    Parameters
    ----------
    df        : DataFrame with 'UMAP1', 'UMAP2', and any panel columns.
    out_path  : file stem (no extension); saves .png and .svg.
    panels    : list of panel dicts, one per subplot:
                  col    : str   column to colour by              (required)
                  title  : str   subplot title
                  labels : dict  raw_value -> display label
                  colors : dict  label -> hex color
                  smush  : bool  merge same-label clusters (default False)
    """
    for col in ('UMAP1', 'UMAP2'):
        if col not in df.columns:
            raise ValueError(f"df is missing required column '{col}'")

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=figsize or (8 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, panel in zip(axes, panels):
        groups = _prepare_panel(df, panel)

        for label, color, sub in groups:
            ax.scatter(
                sub['UMAP1'], sub['UMAP2'],
                c=color, s=s, alpha=alpha,
                linewidths=0, rasterized=True,
                label=f'{label}  (n={len(sub):,})',
            )

        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.set_title(panel.get('title', panel['col']), fontweight='bold')
        ax.legend(
            markerscale=5, fontsize=legend_fontsize, frameon=False,
            ncol=legend_ncol, bbox_to_anchor=(1.0, 1.0), loc='upper left',
        )
        ax.spines[['top', 'right']].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold')

    plt.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(f'{out_path}.png', dpi=dpi, bbox_inches='tight')
    plt.savefig(f'{out_path}.svg', bbox_inches='tight', format='svg')
    plt.close()
    print(f"  Saved: {out_path}.png / .svg")
