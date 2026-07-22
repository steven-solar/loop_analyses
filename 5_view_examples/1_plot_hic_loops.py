"""
plot_hic_loops.py

Per-cell-line Hi-C / Micro-C heatmaps with loops, BigWig, and BED tracks.
BigWig y-scales are shared across cell lines. Supports off-diagonal matrix
sections. Hi-C matrix exports are pixel-perfect squares.
"""

import os
import tempfile
import warnings
import logging

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter, MultipleLocator
from matplotlib.transforms import Bbox
import cooler
from coolbox.api import *
from coolbox.utilities import GenomeRange
from pygenometracks.tracks import BedTrack

plt.rcParams.update({
    "font.family":  "Arial",
    "font.size":    12,
    "text.usetex":  False,
    "pdf.fonttype": 42,
})
warnings.filterwarnings("ignore")
logging.getLogger("matplotlib.font_manager").disabled = True

# =============================================================================
# CONFIG
# =============================================================================

MCOOL_PAT  = "/mnt/coldstorage/shares/Masahiro/microc/mcs/{cell_line}_WT_10B.mcool"
ATAC_PAT   = "/mnt/coldstorage/shares/Masahiro/Saitoulab_data/saitoulab_atac/bws/{cell_line}_{prefix}_ATAC_pool.bw"
BIGWIG_PAT = "/mnt/florenceshares/Masahiro/Saitoulab_data/saitoulab_chip/bws/{cell_line}_{prefix}_{mark}_pool.bw"

CELL_LINES = ["ESC", "EpiLC", "d4c7PGCLC", "GSC"]
PREFIXES   = ["BDF121", "BDF121", "BDF121", "AAG"]
SIGNALS    = ["K27ac", "Rad21", "Ring1b"]

REGION     = "chr10:44,400,000-44,600,000"

RESOLUTION = 1000

MICROC_PARAMS = {
    "style":       "matrix",
    "transform":   "log10",
    "depth_ratio": "full",
    "balance":     True,
    "cmap":        "YlOrRd",
    "max_value":   -1,
    "min_value":   -4,
    "fontsize":    18,
}

LOOP_TSV   = "../1_call_loops/loops/merged_calls/merged_consensus_q0.01.mm10_loop_filt.bedpe"
OUTPUT_DIR = "1_plots"
OUT_PREFIX = "prdm1"

LOOP_PARAMS    = {"fill": True, "fill_color": "gray", "fill_alpha": 0.5,
                  "color": "black", "line_width": 0.075, "side": "lower", "pos": "mid"}
LOOP_PAD_BY_KB = {1000: 2000, 2000: 4000, 10000: 5000}

FRAME_WIDTH_CM = 12 * 2.54   # coolbox Frame.width is in cm
BIGWIG_BINS    = 1600
BIGWIG_HEIGHT  = 2
BIGWIG_SPACER  = 0.4
BIGWIG_CMAP    = "tab10"

BED_FILE         = None
BED_HEIGHT       = 3
BED_COLOR        = "bed_rgb"
BED_BORDER_COLOR = "none"
BED_DISPLAY      = "stacked"
BED_MAX_LABELS   = 0
BED_GENE_ROWS    = 1

EXPORT_FULL   = True
EXPORT_SQUARE = True


def _mb_fmt(x, _):
    mb = x / 1e6
    return f"{int(round(mb))} Mb" if abs(mb - round(mb)) < 0.05 else f"{mb:.1f} Mb"


def _set_mb_ticks(ax, n=6):
    """Apply evenly-spaced, Mb-labelled x-ticks to an axes."""
    ax.set_xticks(np.linspace(*ax.get_xlim(), n))
    ax.xaxis.set_major_formatter(FuncFormatter(_mb_fmt))
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True)


def _format_log10_cbar(fig):
    """Relabel all colorbars with 10^n notation."""
    for ax in fig.axes:
        if ax.get_label() == "<colorbar>":
            ax.yaxis.set_major_locator(MultipleLocator(1))
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda x, _: f"$10^{{{x:.0f}}}$")
            )
    return fig


def _save_square(fig, ax, path):
    """
    Crop to a pixel-perfect square centred on `ax` and save.

    Forces equal width and height by taking min(tight_width, tight_height),
    so the saved image is exactly square regardless of tick-label asymmetry.
    """
    ax.set_box_aspect(1)
    fig.canvas.draw()
    tight = ax.get_tightbbox(fig.canvas.get_renderer())   # display-coord (px) bbox
    side  = min(tight.width, tight.height)
    cx    = (tight.x0 + tight.x1) / 2
    cy    = (tight.y0 + tight.y1) / 2
    sq_px = Bbox([[cx - side / 2, cy - side / 2],
                  [cx + side / 2, cy + side / 2]])
    fig.savefig(path, bbox_inches=sq_px.transformed(fig.dpi_scale_trans.inverted()))
    print(f"saved square matrix: {path}")

class BedCoverageTrack(Track):
    """Thin coolbox Track wrapper around a pygenometracks BedTrack."""
    def __init__(self, cb_props, pgt_props):
        super().__init__(cb_props)
        self._pgt = BedTrack(pgt_props)

    def fetch_data(self, gr, **kwargs): pass

    def plot(self, ax, gr, **kwargs):
        self._pgt.plot(ax, gr.chrom, gr.start, gr.end)
        ax.set_xlim(gr.start, gr.end)

def _pad_loop_anchors(df):
    """Symmetrically pad loop anchors per LOOP_PAD_BY_KB."""
    df = df.copy()
    s1, e1, s2, e2 = df.columns[1], df.columns[2], df.columns[4], df.columns[5]
    widths = df[e1] - df[s1]
    for width, pad in LOOP_PAD_BY_KB.items():
        m = widths == width
        for col, delta in [(s1, -pad), (e1, pad), (s2, -pad), (e2, pad)]:
            df.loc[m, col] += delta
    return df


def _write_bedpe(df):
    tmp = tempfile.NamedTemporaryFile(suffix=".bedpe", delete=False, mode="w")
    df.to_csv(tmp.name, sep="\t", header=False, index=False)
    tmp.close()
    return tmp.name


def _add_loops(heatmap, loop_tsv, fill_col=None, border_col=None, extra_params=None):
    """Overlay loop arcs on a coolbox heatmap track. Returns (heatmap, [tmp_paths])."""
    params     = {**LOOP_PARAMS, **(extra_params or {})}
    df         = _pad_loop_anchors(pd.read_csv(loop_tsv, sep="\t"))
    group_cols = [c for c in (fill_col, border_col) if c]
    tmp_paths  = []

    if not group_cols:
        path = _write_bedpe(df)
        tmp_paths.append(path)
        return heatmap + HiCPeaksCoverage(path, style="hicpeaks", **params), tmp_paths

    for keys, grp in df.groupby(group_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        p    = {**params}
        if fill_col:   p["fill_color"] = keys[0]
        if border_col: p["color"]      = keys[-1]
        path = _write_bedpe(grp.drop(columns=group_cols))
        tmp_paths.append(path)
        heatmap = heatmap + HiCPeaksCoverage(path, style="hicpeaks", **p)

    return heatmap, tmp_paths


def compute_global_bigwig_scales(bigwig_files_per_cell, region):
    """Return dict[label] -> (0, global_max) across all cell lines in `region`."""
    label_max = {}
    for flist in bigwig_files_per_cell.values():
        for path, label in flist:
            data = BigWig(path, number_of_bins=BIGWIG_BINS).fetch_plot_data(
                GenomeRange(region)
            )
            if data is not None and len(data):
                m = float(np.nanmax(data))
                if np.isfinite(m):
                    label_max[label] = max(label_max.get(label, 0.0), m)
    return {lbl: (0.0, mx) for lbl, mx in label_max.items() if mx > 0}


def _build_bigwig_tracks(bigwig_files, region, global_scales=None):
    """Build coolbox BigWig tracks with fixed y-limits."""
    if not bigwig_files:
        return []
    colors = sns.color_palette(BIGWIG_CMAP, n_colors=len(bigwig_files))
    tracks = []
    for i, (path, label) in enumerate(bigwig_files):
        if global_scales and label in global_scales:
            y_min, y_max = global_scales[label]
        else:
            data  = BigWig(path, number_of_bins=BIGWIG_BINS).fetch_plot_data(
                GenomeRange(region)
            )
            y_min = 0.0
            y_max = float(np.nanmax(data)) if data is not None and len(data) else 0.0
        tracks.append(
            BigWig(path, number_of_bins=BIGWIG_BINS)
            + Title(label) + TrackHeight(BIGWIG_HEIGHT)
            + MinValue(y_min) + MaxValue(y_max) + Color(colors[i])
        )
    return tracks


def build_bigwig_files(cell_line, prefix):
    """Return [(path, label)] for ATAC + ChIP bigwigs that exist on disk."""
    candidates = [
        (ATAC_PAT.format(cell_line=cell_line, prefix=prefix), "ATAC"),
        *[(BIGWIG_PAT.format(cell_line=cell_line, prefix=prefix, mark=sig), sig)
          for sig in SIGNALS],
    ]
    return [(p, lbl) for p, lbl in candidates if os.path.exists(p)]

def plot_offdiag(microc_file, region_pair, resolution=RESOLUTION,
                 mc_params=None, output_dir=OUTPUT_DIR, out_base="offdiag"):
    """Matrix-only off-diagonal plot using raw cooler."""
    p      = {**MICROC_PARAMS, **(mc_params or {})}
    r1, r2 = region_pair
    mat    = (cooler.Cooler(f"{microc_file}::/resolutions/{resolution}")
              .matrix(balance=p["balance"]).fetch(r1, r2))

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(mat, origin="lower", cmap=p["cmap"],
                   norm=LogNorm(vmin=max(10 ** p["min_value"], 1e-6),
                                vmax=max(10 ** p["max_value"], 1e-5)))
    ax.set_title(f"{os.path.basename(microc_file)} | {r1} vs {r2}",
                 fontsize=p["fontsize"])
    fig.colorbar(im, ax=ax).set_label("contacts (log10 scale approx.)")

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{out_base}_offdiag.svg")
    fig.savefig(path, bbox_inches="tight")
    print(f"saved off-diagonal matrix: {path}")
    return fig, ax


def plot_hic(
    microc_file,
    region=REGION,
    resolution=RESOLUTION,
    mc_params=None,
    loop_tsv=LOOP_TSV,
    loop_fill_col=None,
    loop_border_col=None,
    loop_params=None,
    bigwig_files=None,
    global_scales=None,
    bed_file=BED_FILE,
    output_dir=OUTPUT_DIR,
    out_base="plot",
):
    """
    Diagonal Hi-C heatmap with optional loops, bigwigs, and BED track.
    Pass `region` as a 2-tuple for an off-diagonal plot (matrix only).
    """
    if isinstance(region, (tuple, list)):
        return plot_offdiag(microc_file, region, resolution,
                            mc_params, output_dir, out_base)

    params = {**MICROC_PARAMS, **(mc_params or {})}

    # Heatmap + loops
    heatmap, tmp_paths = Cool(microc_file, resolution=resolution, **params), []
    if loop_tsv:
        heatmap, tmp_paths = _add_loops(
            heatmap, loop_tsv, loop_fill_col, loop_border_col, loop_params
        )

    # Assemble coolbox Frame
    frame = Frame() + heatmap + Title(os.path.basename(microc_file))
    for bw in _build_bigwig_tracks(bigwig_files or [], region, global_scales):
        frame = frame + Spacer(BIGWIG_SPACER) + bw

    if bed_file:
        frame = frame + Spacer(BIGWIG_SPACER) + BedCoverageTrack(
            {"height": BED_HEIGHT},
            {"file": bed_file, "section_name": "anchors", "display": BED_DISPLAY,
             "max_labels": BED_MAX_LABELS, "gene_rows": BED_GENE_ROWS,
             "color": BED_COLOR, "border_color": BED_BORDER_COLOR, "border_only": False},
        )

    frame.properties["width"] = FRAME_WIDTH_CM
    fig    = _format_log10_cbar(frame.plot(region))
    tracks = list(frame.tracks.values())

    try:
        _set_mb_ticks(tracks[-1].ax)
    except Exception as e:
        print(f"Warning: Mb ticks failed: {e}")

    os.makedirs(output_dir, exist_ok=True)
    matrix_ax = tracks[0].ax

    if EXPORT_FULL:
        path = os.path.join(output_dir, f"{out_base}_full.svg")
        fig.savefig(path)
        print(f"saved full figure: {path}")

    if EXPORT_SQUARE:
        _save_square(fig, matrix_ax, os.path.join(output_dir, f"{out_base}_square.svg"))

    for f in tmp_paths:
        os.remove(f)

    return fig, matrix_ax


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    bw_files = {cl: build_bigwig_files(cl, pref)
                for cl, pref in zip(CELL_LINES, PREFIXES)}

    global_scales = (
        compute_global_bigwig_scales(bw_files, REGION)
        if isinstance(REGION, str) else None
    )
    if global_scales:
        print("Global bigwig scales:", global_scales)

    for cl in CELL_LINES:
        plot_hic(
            microc_file   = MCOOL_PAT.format(cell_line=cl),
            bigwig_files  = bw_files[cl],
            global_scales = global_scales,
            out_base      = f"{OUT_PREFIX}_{cl}",
        )

    # Merged heatmap — matrix only, no bigwigs
    plot_hic(
        microc_file = "../data/merged.mcool",
        out_base    = f"{OUT_PREFIX}_merge",
    )