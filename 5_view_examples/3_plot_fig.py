# conda activate coolbox

"""
plot_hic_loops.py

Per-cell-line Hi-C / Micro-C heatmaps with BigWig and BED tracks.
Each cell line uses its own bigwig y-scale, loaded from a precomputed
per_cell_line_scales.tsv (cell_line, mark, mean, std, max_value columns --
see compute_scales_from_loops.py). Supports off-diagonal matrix sections
and split-diagonal (top/bottom triangle) comparisons between two cell
lines, the latter styled identically to the plain (bigwig-free) matrix
plots such as the merged heatmap.
Hi-C matrix exports are pixel-perfect squares.

Also supports a loop-centric summary panel (see LOOP-CENTRIC PANEL
section near the bottom): given a loop_id, produces the split-diagonal
map with the loop location boxed, a row of per-cell-line zoomed loop
heatmaps, left/right anchor ChIP-track grids, and two shared-axis PC
trajectory plots (one per anchor).
"""

import os
import copy
import warnings
import logging

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MultipleLocator
from matplotlib.transforms import Bbox
import cooler
from coolbox.api import *
from cooltools.lib import plotting
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

PRC_SIGNALS    = ["K27ac", "K4me1", "K4me3", "Rad21", "Ring1b", "H2Aub", "K27me3"]
ALL_SIGNALS = [
    'ATAC',
    'K4me3', 'K27ac', 'K4me1', 'K36me3', 
    'CTCF', 'Rad21', 'Stag1', 'Stag2',
    'Ring1b', 'H2Aub', 'K27me3',
    'K9me2', 'K9me3', 'K36me2',
    'Laminb1',
]

CTCF_SIGNALS = ["Rad21", "CTCF", "Stag1", "Stag2"]
CRE_CTCF_SIGNALS    = ["K27ac", "K4me1", "CTCF", "Rad21"]
CRE_SIGNALS = ['ATAC', 'Rad21',  'K27ac', 'K4me1', 'K4me3'] #, 'K9me2', 'K9me3']

SIGNALS = ['ATAC', 'K27ac', 'K4me1', 'K4me3', 'Ring1b', 'K27me3']
PER_CELL_SCALES_TSV = "per_cell_line_scales_manual.tsv"  # cell_line, mark, max_value (see compute_scales_from_loops.py)

MICROC_PARAMS = {
    "style":       "matrix",
    "transform":   "log10",
    "depth_ratio": "full",
    "balance":     True,
    "cmap":        "fall",
    "max_value":   -1,
    "min_value":   -4,
    "fontsize":    18,
}

# REGION='chrX:57,880,000-58,180,000'
# OUT_PREFIX = 'zic3_chrX_57.88-58.18'
# MICROC_PARAMS['max_value'] = -1.75
# MICROC_PARAMS['min_value'] = -3.75
# SIGNALS = ['ATAC', 'K27ac', 'Ring1b', 'K27me3']
# loop_id         = "chrX:57920000-57921000 & chrX:58016000-58017000",

# REGION='chr10:79,850,000-80,000,000'
# OUT_PREFIX='med16'
# SIGNALS = ['ATAC', 'K27ac', 'K4me3', 'Rad21', 'CTCF', 'Stag1', 'Stag2']
# MICROC_PARAMS['max_value'] = -1.75
# MICROC_PARAMS['min_value'] = -3.75
# loop_id = 'chr10:79908000-79909000 & chr10:79959000-79960000'

# REGION='chr9:121,800,000-122,000,000'
# OUT_PREFIX='Higd1a'
# SIGNALS = ['ATAC', 'K27ac', 'K4me1', 'K4me3']
# MICROC_PARAMS['max_value'] = -1.99
# MICROC_PARAMS['min_value'] = -3.5
# loop_id='chr9:121858000-121859000 & chr9:121937000-121938000'

# REGION = 'chr1:8,500,000-8,975,000 '
# OUT_PREFIX = 'Sntg1'
# SIGNALS = ['ATAC', 'K27ac', 'K4me1', 'K4me3', 'K9me2', 'K9me3', 'K36me2', 'K36me3']
# MICROC_PARAMS['max_value'] = -2.5
# MICROC_PARAMS['min_value'] = -3.75
# loop_id=

REGION='chr13:5,650,000-5,900,000'
OUT_PREFIX='Klf6'
SIGNALS = ['ATAC', 'K27ac', 'K4me1', 'K4me3', 'K9me3', 'K36me3']
MICROC_PARAMS['max_value'] = -1.75
MICROC_PARAMS['min_value'] = -3.75
loop_id='chr13:5695000-5696000 & chr13:5861000-5862000'

# REGION = 'chr1:188,980,000-189,250,000'
# OUT_PREFIX = "loss_ctcf"
# MICROC_PARAMS['max_value'] = -1.99
# MICROC_PARAMS['min_value'] = -3.5
# SIGNALS = ['ATAC', 'Rad21', 'CTCF']
# loop_id='chr1:189027000-189028000 & chr1:189227000-189228000'

# REGION='chr14:122,206,358-122,523,296'
# OUT_PREFIX='defne'
# SIGNALS=['ATAC', 'Rad21', 'Stag1', 'Stag2', 'CTCF', 'K4me1', 'K4me3', 'K27me3']

EPIG_MAX = {
    "ATAC":   6,
    "K27ac":  2,
    "K4me1":  1,
    "K4me3":  1,
    "Rad21":  2,
    "Ring1b": 10,
    "H2Aub":  2,
    "K27me3": 0.5,
}

REGION_SCALES = {
    "zic3_chrX_57.88-58.18": {
        "ATAC": 2,
        "K27ac":2,
        "Ring1b":5,
        "K27me3":0.5,
    },
    "med16": {
        'ATAC':5,
        'K27ac': 10,
        'K4me1': 1,
        'K4me3': 20,
        'Rad21': 2,
        'Stag1': 5,
        'Stag2': 10,
        'CTCF': 5,
        'Ring1b': 5
    },
    "Higd1a": {
        "K4me3": 10,
        "K27ac": 2.5,
        "Ring1b": 1,
        "K27me3": 0.25,
        "H2Aub": 1
    },
    "loss_ctcf": {
        "ATAC": 3,
        "Rad21": 1,
        "CTCF": 3,
    },
    "Sntg1": {
        "ATAC": 4,
        "K27ac":5,
        "K4me3": 5,
        "K9me2":0.5,
        "K9me3":0.5,
        "K36me2":0.5,
        "K36me3":0.5
    },
    "Klf6": {
        "ATAC": 6,
        "K27ac":7.5,
        "K4me3": 10,
        "K9me3":1,
        "K36me3":1
    }

}

override = REGION_SCALES.get(OUT_PREFIX, EPIG_MAX)

RESOLUTION = 1000

OUTPUT_DIR = "plots"

FRAME_WIDTH_CM = 12 * 2.54   # coolbox Frame.width is in cm
BIGWIG_BINS    = 1600
BIGWIG_HEIGHT  = 2
BIGWIG_SPACER  = 0.4
BIGWIG_CMAP    = "tab10"  # fallback palette for any mark not in CHIP_TRACK_COLORS

# Fixed per-mark colors for the bigwig tracks in plot_hic (the per-cell-line
# heatmap+tracks plots at the top of the script). Any mark not listed here
# falls back to BIGWIG_CMAP's palette, indexed by track order, as before.
# Fill in as many/few as you want, e.g.:
CHIP_TRACK_COLORS = {
    "ATAC": "#3f77b1", 
    "K27ac": "#f58020", 
    "K4me1": "#2fa148", 
    "K4me3": "#d62a28", 
    "Rad21": "#9467bd", 
    "CTCF": "#000000", 
    "Stag1": "#e377c2",
    "Stag2": "#bcbd22",
    "Ring1b": "#865a4f", 
    "K27me3": "#808181"}
# CHIP_TRACK_COLORS = {}


BED_FILE         = "../data/mm10_genes.bed"
BED_HEIGHT       = 3
BED_COLOR        = "bed_rgb"
BED_BORDER_COLOR = "black"
BED_DISPLAY      = "stacked"
BED_MAX_LABELS   = 10
BED_GENE_ROWS    = 3

EXPORT_FULL   = True
EXPORT_SQUARE = True

os.makedirs(f'{OUTPUT_DIR}/{OUT_PREFIX}', exist_ok=True)

# def _mb_fmt(x, _):
#     mb = x / 1e6
#     return f"{int(round(mb))} Mb" if abs(mb - round(mb)) < 0.05 else f"{mb:.2f} Mb"


# def _set_mb_ticks(ax, n=6):
#     """Apply evenly-spaced, Mb-labelled x-ticks to an axes."""
#     ax.set_xticks(np.linspace(*ax.get_xlim(), n))
#     ax.xaxis.set_major_formatter(FuncFormatter(_mb_fmt))
#     ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True)


KB = 1_000
MB = 1_000_000

def _mb_fmt(x, _):
    mb = x / MB
    # Strip trailing zeros: 57.90 -> 57.9, 58.00 -> 58
    return f"{mb:.2f}".rstrip("0").rstrip(".") + " Mb"

def _set_mb_ticks(ax, step_kb=50):
    """Place x-ticks every `step_kb` kb and label in Mb."""
    ax.xaxis.set_major_locator(MultipleLocator(step_kb * KB))
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
    """
    fig.canvas.draw()
    tight = ax.get_tightbbox(fig.canvas.get_renderer())
    side  = min(tight.width, tight.height)
    cx    = (tight.x0 + tight.x1) / 2
    cy    = (tight.y0 + tight.y1) / 2
    sq_px = Bbox([[cx - side / 2, cy - side / 2],
                  [cx + side / 2, cy + side / 2]])

    sq_in = sq_px.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(path, bbox_inches=sq_in)
    print(f"saved square matrix: {path}")


def _finalize_and_export(frame, region, output_dir, out_base, highlight_loop=None):
    """
    Shared tail end of every matrix plot: render the Frame, apply the
    log10 colorbar relabeling + Mb ticks + equal aspect to the matrix
    axes, optionally box a loop location, then export _full/_square svgs.
    Used by plot_hic (including the bigwig-free merge case) and
    plot_split_diag so all matrix plots come out identically formatted.
 
    highlight_loop: optional loop row (as returned by _load_loop_row) --
    if given, draws an unfilled black box around that loop's anchor pair
    on the matrix, sized by ANCHOR_WINDOW_PAD, mirrored on both sides of
    the diagonal (since a split map shows a different cell line on each
    side, the loop location is boxed in both triangles).
    """
    frame.properties["width"] = FRAME_WIDTH_CM
    fig    = _format_log10_cbar(frame.plot(region))
    tracks = list(frame.tracks.values())
 
    matrix_ax = tracks[0].ax
    matrix_ax.get_xaxis().set_visible(True)
    matrix_ax.tick_params(
        axis="x",
        which="both",
        bottom=True,
        top=False,
        labelbottom=True,
        length=4,
    )
    if "bottom" in matrix_ax.spines:
        matrix_ax.spines["bottom"].set_visible(True)
 
    matrix_ax.set_aspect("equal", adjustable="box")
 
    try:
        _set_mb_ticks(matrix_ax)
    except Exception as e:
        print(f"Warning: Mb ticks on matrix failed: {e}")
 
    if highlight_loop is not None:
        try:
            lc, rc = _anchor_centers(highlight_loop)
            box = ANCHOR_WINDOW_PAD
            for x, y in [(lc, rc), (rc, lc)]:
                matrix_ax.add_patch(Rectangle(
                    (x - box, y - box), 2 * box, 2 * box,
                    fill=False, edgecolor="black", linewidth=1.2, zorder=5,
                ))
        except Exception as e:
            print(f"Warning: loop highlight box failed: {e}")
 
    os.makedirs(output_dir, exist_ok=True)
 
    if EXPORT_FULL:
        path = os.path.join(output_dir, f"{out_base}_full.svg")
        fig.savefig(path)
        print(f"saved full figure: {path}")
 
    if EXPORT_SQUARE:
        _save_square(fig, matrix_ax, os.path.join(output_dir, f"{out_base}_square.svg"))
 
    return fig, matrix_ax


class BedCoverageTrack(Track):
    """Thin coolbox Track wrapper around a pygenometracks BedTrack."""
    def __init__(self, cb_props, pgt_props):
        super().__init__(cb_props)
        self._pgt = BedTrack(pgt_props)

    def fetch_data(self, gr, **kwargs): pass

    def plot(self, ax, gr, **kwargs):
        self._pgt.plot(ax, gr.chrom, gr.start, gr.end)
        ax.set_xlim(gr.start, gr.end)


class SplitCoolTrack(Track):
    """
    coolbox Track that draws one Hi-C matrix split across the diagonal:
    upper triangle from `top_file`, lower triangle from `bottom_file`.
    Styled the same as a normal Cool 'matrix' track (log10 transform,
    same cmap/min/max from MICROC_PARAMS) so it drops into a Frame next
    to a Title/BED track exactly like the single-sample heatmaps.
    """
    def __init__(self, top_file, bottom_file, resolution, params, cb_props=None):
        super().__init__({"height": FRAME_WIDTH_CM, **(cb_props or {})})
        self.top_file     = top_file
        self.bottom_file  = bottom_file
        self.resolution   = resolution
        self.params       = params

    def fetch_data(self, gr, **kwargs): pass

    def plot(self, ax, gr, **kwargs):
        self.ax = ax
        region = f"{gr.chrom}:{gr.start}-{gr.end}"
        top_mat = (cooler.Cooler(f"{self.top_file}::/resolutions/{self.resolution}")
                   .matrix(balance=self.params["balance"]).fetch(region))
        bot_mat = (cooler.Cooler(f"{self.bottom_file}::/resolutions/{self.resolution}")
                   .matrix(balance=self.params["balance"]).fetch(region))

        if top_mat.shape != bot_mat.shape:
            raise ValueError(
                f"shape mismatch: top {top_mat.shape} vs bottom {bot_mat.shape} "
                f"for {region} -- do the two mcools share a bin table?"
            )

        combined = np.full(top_mat.shape, np.nan)
        iu = np.triu_indices_from(combined, k=1)
        il = np.tril_indices_from(combined, k=-1)
        combined[iu] = top_mat[iu]
        combined[il] = bot_mat[il]
        diag = np.diag_indices_from(combined)
        combined[diag] = np.nanmean(np.stack([top_mat[diag], bot_mat[diag]]), axis=0)

        small = 1e-12
        combined[combined == 0] = small
        combined[np.isnan(combined)] = small
        log_combined = np.log10(combined)

        cmap = copy.copy(plt.get_cmap(self.params["cmap"]))
        floor = cmap(0)
        cmap.set_bad(floor)
        cmap.set_under(floor)

        im = ax.imshow(
            log_combined, origin="upper", cmap=cmap,
            vmin=self.params["min_value"], vmax=self.params["max_value"],
            extent=(gr.start, gr.end, gr.end, gr.start),
        )
        ax.plot([gr.start, gr.end], [gr.start, gr.end], color="black", linewidth=0.5, alpha=0.6)
        ax.set_xlim(gr.start, gr.end)
        ax.set_ylim(gr.end, gr.start)

        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.set_label("<colorbar>")


def load_per_cell_line_scales(path, expected_cell_lines=None, expected_marks=None):
    """Load {cell_line: {mark: (0.0, max_value)}} from a per_cell_line_scales.tsv."""
    if not os.path.exists(path):
        print(f"[warn] per-cell-line scales file not found: {path} -- falling back to per-track auto-scaling")
        return {}

    df = pd.read_csv(path, sep="\t")
    required = {"cell_line", "mark", "max_value"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"{path} missing required column(s): {missing_cols}")

    scales = {}
    for _, row in df.iterrows():
        cl, mark = row["cell_line"], row["mark"]
        try:
            scales.setdefault(cl, {})[mark] = (0.0, float(row["max_value"]))
        except (TypeError, ValueError):
            print(f"[warn] non-numeric max_value for {cl}/{mark} in {path} -- skipping")

    if expected_cell_lines and expected_marks:
        for cl in expected_cell_lines:
            missing_marks = [m for m in expected_marks if m not in scales.get(cl, {})]
            if missing_marks:
                print(f"[warn] {cl}: no precomputed scale for {missing_marks} -- those tracks will auto-scale per-region")

    return scales


def _build_bigwig_tracks(bigwig_files, region, global_scales=None, override_scales=None):
    """Build coolbox BigWig tracks with fixed y-limits.

    Precedence:
      1) override_scales[label]
      2) global_scales[label]
      3) per-region auto-scale
    """
    if not bigwig_files:
        return []

    colors = sns.color_palette(BIGWIG_CMAP, n_colors=len(bigwig_files))
    tracks = []

    for i, (path, label) in enumerate(bigwig_files):
        source = "region auto-scale"
        y_min, y_max = 0.0, None

        if override_scales and label in override_scales:
            val = override_scales[label]
            if isinstance(val, (list, tuple)) and len(val) == 2:
                y_min, y_max = float(val[0]), float(val[1])
            else:
                y_min, y_max = 0.0, float(val)
            source = "override_scales"

        elif global_scales and label in global_scales:
            ymin, ymax = global_scales[label]
            y_min, y_max = float(ymin), float(ymax)
            source = "global_scales"

        else:
            data = BigWig(path, number_of_bins=BIGWIG_BINS).fetch_plot_data(
                GenomeRange(region)
            )
            y_min = 0.0
            y_max = float(np.nanmax(data)) if data is not None and len(data) else 0.0
            source = "region auto-scale"

        print(f"[track] {label}: min_value={y_min}, max_value={y_max} (source: {source})")

        color = CHIP_TRACK_COLORS.get(label, colors[i])
        tracks.append(
            BigWig(path, number_of_bins=BIGWIG_BINS, min_value=y_min, max_value=y_max)
            + Title(label)
            + TrackHeight(BIGWIG_HEIGHT)
            + Color(color)
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
    # p['max_value'] = p.get('max_value') - 0.25 
    # p['min_value'] = p.get('min_value') + 0.25
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


def plot_split_diag(
    top_cell_line,
    bottom_cell_line,
    region=REGION,
    resolution=RESOLUTION,
    mc_params=None,
    bed_file=BED_FILE,
    output_dir=OUTPUT_DIR,
    out_base=None,
    highlight_loop=None,
):
    """
    Single diagonal heatmap split across the main diagonal: upper triangle
    from `top_cell_line`, lower triangle from `bottom_cell_line` (diagonal
    bins are the mean of both, since neither side owns them). Formatted
    identically to the plain matrix plots (e.g. the merged heatmap in
    plot_hic) -- same Frame width, Title, BED track, log10 colorbar
    relabeling, Mb ticks, and _full/_square export.

    Pass `highlight_loop` (a loop row from _load_loop_row) to box that
    loop's anchor pair on the matrix.
    """
    params = {**MICROC_PARAMS, **(mc_params or {})}

    top_file = MCOOL_PAT.format(cell_line=top_cell_line)
    bot_file = MCOOL_PAT.format(cell_line=bottom_cell_line)

    heatmap = SplitCoolTrack(top_file, bot_file, resolution, params)

    frame = Frame() + heatmap + Title(f"{top_cell_line} (upper) / {bottom_cell_line} (lower)")
    if bed_file:
        frame = frame + Spacer(BIGWIG_SPACER) + BedCoverageTrack(
            {"height": BED_HEIGHT},
            {"file": bed_file, "section_name": "genes", "display": BED_DISPLAY,
             "max_labels": BED_MAX_LABELS, "gene_rows": BED_GENE_ROWS,
             "color": BED_COLOR, "border_color": BED_BORDER_COLOR, "border_only": False},
        )

    out_base = out_base or f"{top_cell_line}_top_{bottom_cell_line}_bottom"
    return _finalize_and_export(frame, region, output_dir, out_base, highlight_loop=highlight_loop)


def plot_hic(
    microc_file,
    region=REGION,
    resolution=RESOLUTION,
    mc_params=None,
    bigwig_files=None,
    global_scales=None,
    bed_file=BED_FILE,
    output_dir=OUTPUT_DIR,
    out_base="plot",
    override_scales=None,  
):
    """
    Diagonal Hi-C heatmap with optional bigwigs and BED track.
    Pass `region` as a 2-tuple for an off-diagonal plot (matrix only).
    """
    if isinstance(region, (tuple, list)):
        return plot_offdiag(microc_file, region, resolution,
                            mc_params, output_dir, out_base)

    params = {**MICROC_PARAMS, **(mc_params or {})}

    heatmap = Cool(microc_file, resolution=resolution, **params)

    frame = Frame() + heatmap + Title(os.path.basename(microc_file))
    for bw in _build_bigwig_tracks(bigwig_files or [], region, global_scales, override_scales=override_scales):
        frame = frame + Spacer(BIGWIG_SPACER) + bw

    if bed_file:
        frame = frame + Spacer(BIGWIG_SPACER) + BedCoverageTrack(
            {"height": BED_HEIGHT},
            {"file": bed_file, "section_name": "genes", "display": BED_DISPLAY,
             "max_labels": BED_MAX_LABELS, "gene_rows": BED_GENE_ROWS,
             "color": BED_COLOR, "border_color": BED_BORDER_COLOR, "border_only": False},
        )

    return _finalize_and_export(frame, region, output_dir, out_base)


# =============================================================================
# LOOP-CENTRIC PANEL
# =============================================================================
# Reproduces the figure layout: split-diagonal map with the loop boxed,
# a row of per-cell-line zoomed loop heatmaps (+/- ANCHOR_WINDOW_PAD around
# the whole anchor-to-anchor span), left/right anchor ChIP-track grids
# (+/- ANCHOR_WINDOW_PAD around each anchor's own center), and two PC
# trajectory scatter plots (one per anchor) on shared, padded axis limits.
#
# *** ADAPT THESE TO YOUR ACTUAL LOOP/PC DATAFRAME SCHEMA ***
# This assumes a single wide-format table (one row per loop, as produced
# by make_input_filt.py) with anchor coordinates, one AbLE column per cell
# line, and one PC1/PC3 column per (side, cell_line). If your PC
# coordinates live in a separate table, set PC_DF_PATH and the PC lookup
# below will read from there instead, joined on LOOP_ID_COL.

LOOP_DF_PATH = "../4_loop_strength_prediction/loops/loop_df_reclustered_per_cl_wide.tsv"   # ADAPT: your wide-format loop table
PC_DF_PATH   = None                              # ADAPT: set only if PC coords live in a separate table

LOOP_ID_COL     = "loop_id"
CHROM_COL       = "chrom1"
LEFT_START_COL  = "start1"
LEFT_END_COL    = "end1"
RIGHT_START_COL = "start2"
RIGHT_END_COL   = "end2"

ABLE_COL_PAT = "AbLE_{cell_line}"                              # e.g. AbLE_ESC
PC_COL_PAT   = {"CRE": "PC1_{side}_{cell_line}",  
                "CTCF": "PC2_{side}_{cell_line}",
                "PRC": "PC3_{side}_{cell_line}",
                "PC5-K9me3": "PC5_{side}_{cell_line}"}                # e.g. PC3_right_GSC

LOOP_ZOOM_MARKS   = ["ATAC", "K27ac", "Ring1b", "K27me3"]  # rows in the anchor ChIP grid
ANCHOR_WINDOW_PAD = 10000                                    # +/- bp: loop-zoom crop pad AND each anchor's ChIP window half-width

CELL_LINE_COLORS = {
    "ESC":        "#1f77b4",  # blue
    "EpiLC":      "#ff7f0e",  # orange
    "d4c7PGCLC":  "#d62728",  # red
    "GSC":        "#7d3c98",  # purple
}


def _read_table(path):
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    sep = "\t" if path.endswith((".tsv", ".txt")) else ","
    return pd.read_csv(path, sep=sep)


def _load_loop_row(loop_id, table_path):
    df = _read_table(table_path)
    match = df.loc[df[LOOP_ID_COL] == loop_id]
    if match.empty:
        raise ValueError(f"loop_id {loop_id!r} not found in {table_path}")
    return match.iloc[0]


def _anchor_centers(row):
    left_c  = (row[LEFT_START_COL] + row[LEFT_END_COL]) // 2
    right_c = (row[RIGHT_START_COL] + row[RIGHT_END_COL]) // 2
    return int(left_c), int(right_c)


def _loop_zoom_region(row, pad=ANCHOR_WINDOW_PAD):
    """Full anchor-to-anchor span, padded by `pad` on each side."""
    chrom = row[CHROM_COL]
    start = int(row[LEFT_START_COL]) - pad
    end   = int(row[RIGHT_END_COL]) + pad
    return f"{chrom}:{max(start, 0)}-{end}"


def _anchor_window_region(row, side, pad=ANCHOR_WINDOW_PAD):
    """A pad*2-wide window centered on one anchor's midpoint."""
    chrom = row[CHROM_COL]
    left_c, right_c = _anchor_centers(row)
    center = left_c if side == "left" else right_c
    return f"{chrom}:{max(center - pad, 0)}-{center + pad}"


def plot_loop_zoom_row(loop_id, cell_lines=CELL_LINES, resolution=RESOLUTION,
                        mc_params=None, loop_df_path=LOOP_DF_PATH,
                        output_dir=OUTPUT_DIR, out_base=None):
    """
    Row of small zoomed-in Hi-C heatmaps (one square per cell line), cropped
    to the loop's anchor-to-anchor span +/- ANCHOR_WINDOW_PAD, each titled
    with its cell line and AbLE score -- matches the small squares above
    the ChIP tracks in the figure.
    """
    p = {**MICROC_PARAMS, **(mc_params or {})}
    row = _load_loop_row(loop_id, loop_df_path)
    region = _loop_zoom_region(row)

    fig, axes = plt.subplots(1, len(cell_lines), figsize=(2.2 * len(cell_lines), 2.2))
    if len(cell_lines) == 1:
        axes = [axes]

    for ax, cl in zip(axes, cell_lines):
        mcool = MCOOL_PAT.format(cell_line=cl)
        mat = (cooler.Cooler(f"{mcool}::/resolutions/{resolution}")
               .matrix(balance=p["balance"]).fetch(region))
        mat = np.where(mat <= 0, np.nan, mat)
        ax.imshow(np.log10(mat), origin="upper", cmap=p["cmap"],
                  vmin=p["min_value"], vmax=p["max_value"])
        ax.set_xticks([]); ax.set_yticks([])

        able_col = ABLE_COL_PAT.format(cell_line=cl)
        able_txt = f"\nAbLE={row[able_col]:.2g}" if able_col in row.index else ""
        ax.set_title(f"{cl}{able_txt}", fontsize=9)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_base = out_base or f"{loop_id}_zoom"
    path = os.path.join(output_dir, f"{out_base}.svg")
    fig.savefig(path, bbox_inches="tight")
    print(f"saved loop zoom row: {path}")
    return fig, axes

def plot_anchor_chip_grid(loop_id, side, cell_lines=CELL_LINES, prefixes=PREFIXES,
                           marks=SIGNALS, loop_df_path=LOOP_DF_PATH,
                           output_dir=OUTPUT_DIR, out_base=None):
    """
    Grid of ChIP/ATAC bigwig pileups for one anchor side ("left" or
    "right"): rows = marks, columns = cell lines, each cell a small track
    over that anchor's +/- ANCHOR_WINDOW_PAD window. Y-limits per row come
    from `override` (region-specific EPIG_MAX override, same dict used by
    plot_hic) so rows are comparable across cell lines. Track color is by
    mark/signal (from CHIP_TRACK_COLORS, same constant used in the
    top-section plot_hic tracks) -- not by cell line -- so every column in
    a row is the same color and only the mark's row color changes.
    """
    assert side in ("left", "right")
    row = _load_loop_row(loop_id, loop_df_path)
    region = _anchor_window_region(row, side)
    gr = GenomeRange(region)
 
    fallback_colors = sns.color_palette(BIGWIG_CMAP, n_colors=len(marks))
 
    fig, axes = plt.subplots(len(marks), len(cell_lines),
                              figsize=(1.6 * len(cell_lines), 1.0 * len(marks)),
                              sharex=True, squeeze=False)
 
    for r, mark in enumerate(marks):
        y_max = override.get(mark, EPIG_MAX.get(mark))
        color = CHIP_TRACK_COLORS.get(mark, fallback_colors[r])
        for c, (cl, pref) in enumerate(zip(cell_lines, prefixes)):
            ax = axes[r][c]
            bw_path = (ATAC_PAT.format(cell_line=cl, prefix=pref) if mark == "ATAC"
                       else BIGWIG_PAT.format(cell_line=cl, prefix=pref, mark=mark))
            if os.path.exists(bw_path):
                data = BigWig(bw_path, number_of_bins=BIGWIG_BINS).fetch_plot_data(gr)
                x = np.linspace(gr.start, gr.end, len(data))
                ax.fill_between(x, data, color=color, linewidth=0)
            else:
                print(f"[warn] missing bigwig for {cl}/{mark}: {bw_path}")
            if y_max is None:
                print(f"[warn] y_max is None for {cl}/{mark}, setting to 1")
                y_max=1
            ax.set_ylim(0, y_max)
            ax.set_yticks([0, y_max])
            ax.set_xticks([])
            ax.set_frame_on(False)
            if c == 0:
                ax.set_ylabel(mark, fontsize=8, rotation=0, ha="right", va="center")
            else:
                ax.set_yticklabels([])
            if r == 0:
                ax.set_title(cl, fontsize=8)
 
    fig.suptitle(f"{side.capitalize()} Anchor", fontsize=10)
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_base = out_base or f"{loop_id}_{side}_anchor_chip"
    path = os.path.join(output_dir, f"{out_base}.svg")
    fig.savefig(path, bbox_inches="tight")
    print(f"saved {side} anchor ChIP grid: {path}")
    return fig, axes

def plot_pc_trajectories(loop_id, cell_lines=CELL_LINES, loop_df_path=LOOP_DF_PATH,
                          pc_df_path=PC_DF_PATH, output_dir=OUTPUT_DIR, out_base=None):
    """
    Two PC scatter plots (left anchor, right anchor) of PC1 (CRE) vs PC3
    (PRC) coordinates across cell lines, connected by an arrow trajectory
    in `cell_lines` order. Both panels share the same axis limits (global
    min/max across *both* anchors, padded by +/-1), so they're directly
    comparable -- this is the two-panel version of the single PC plot in
    the figure.
    """
    row = _load_loop_row(loop_id, pc_df_path or loop_df_path)

    coords = {"left": [], "right": []}
    for side in ("left", "right"):
        for cl in cell_lines:
            x_col = PC_COL_PAT["CRE"].format(side=side, cell_line=cl)
            y_col = PC_COL_PAT["PC5-K9me3"].format(side=side, cell_line=cl)
            coords[side].append((float(row[x_col]), float(row[y_col])))

    all_x = [xy[0] for side in coords for xy in coords[side]]
    all_y = [xy[1] for side in coords for xy in coords[side]]
    x_val = max(abs(min(all_x)), abs(max(all_x)))
    y_val = max(abs(min(all_y)), abs(max(all_y)))
    xlim = (-x_val - 1, x_val + 1)
    ylim = (-y_val - 1, y_val + 1)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, side in zip(axes, ("left", "right")):
        xy = coords[side]
        for i in range(len(xy) - 1):
            ax.annotate("", xy=xy[i + 1], xytext=xy[i],
                        arrowprops=dict(arrowstyle="->", color="black", alpha=0.6, lw=1))
        for cl, (x, y) in zip(cell_lines, xy):
            color = CELL_LINE_COLORS.get(cl, "gray")
            ax.scatter(x, y, color=color, s=60, zorder=3, label=cl)
            ax.annotate(f"({x:.2f}, {y:.2f})", (x, y), textcoords="offset points",
                        xytext=(6, 4), fontsize=8, color=color)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xlabel("PC1 (CRE)"); ax.set_ylabel("PC5 (K9me3)")
        ax.set_title(f"{side.capitalize()} Anchor PC Coordinates", fontsize=10)

    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_base = out_base or f"{loop_id}_pc_trajectories"
    path = os.path.join(output_dir, f"{out_base}.svg")
    fig.savefig(path, bbox_inches="tight")
    print(f"saved PC trajectories: {path}")
    return fig, axes


def plot_loop_summary(loop_id, top_cell_line, bottom_cell_line,
                       region=REGION, cell_lines=CELL_LINES, resolution=RESOLUTION,
                       mc_params=None, loop_df_path=LOOP_DF_PATH,
                       pc_df_path=PC_DF_PATH, output_dir=OUTPUT_DIR):
    """
    Full loop-centric summary, mirroring the figure: split-diagonal Hi-C
    map (top_cell_line upper / bottom_cell_line lower) over `region` with
    the loop boxed, a row of per-cell-line zoomed loop heatmaps, ChIP/ATAC
    track grids for the left and right anchors, and two PC-coordinate
    trajectory plots (one per anchor) on shared axes.

    Writes each component as a separate SVG under output_dir/loop_id/
    (they're separate files/axes rather than one combined figure, so each
    stays easy to re-lay-out or drop into a multi-panel assembly later).
    """
    # loop_dir = os.path.join(output_dir, loop_id)
    # os.makedirs(loop_dir, exist_ok=True)

    row = _load_loop_row(loop_id, loop_df_path)
    loop_prefix = loop_id.replace(" & ", "_")
    os.makedirs(f"plots/{OUT_PREFIX}/{loop_prefix}", exist_ok=True)

    plot_split_diag(top_cell_line, bottom_cell_line, region=region,
                     resolution=resolution, mc_params=mc_params,
                     output_dir=f"plots/{OUT_PREFIX}/{loop_prefix}",
                     out_base=f"{OUT_PREFIX}_{top_cell_line}_top_{bottom_cell_line}_bottom",
                     highlight_loop=row)

    # plot_loop_zoom_row(loop_id, cell_lines, resolution, mc_params,
    #                     loop_df_path, loop_dir, out_base=f"{loop_id}_zoom")

    for cl in CELL_LINES:
        plot_offdiag(microc_file=MCOOL_PAT.format(cell_line=cl),
                     region_pair=(_anchor_window_region(row, "left", 10000),
                                  _anchor_window_region(row, "right", 10000)),
                     resolution=resolution, mc_params=mc_params,
                     output_dir=f"plots/{OUT_PREFIX}/{loop_prefix}", out_base=f"{OUT_PREFIX}_{cl}_anchors_offdiag")

    plot_anchor_chip_grid(loop_id, "left", cell_lines, PREFIXES,
                          SIGNALS, loop_df_path, f"plots/{OUT_PREFIX}/{loop_prefix}",
                          out_base=f"{OUT_PREFIX}_left_anchor_chip")
    plot_anchor_chip_grid(loop_id, "right", cell_lines, PREFIXES,
                          SIGNALS, loop_df_path, f"plots/{OUT_PREFIX}/{loop_prefix}",
                          out_base=f"{OUT_PREFIX}_right_anchor_chip")

    plot_pc_trajectories(loop_id, cell_lines, loop_df_path, pc_df_path,
                         f"plots/{OUT_PREFIX}/{loop_prefix}", out_base=f"{OUT_PREFIX}_pc_trajectories")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    bw_files = {cl: build_bigwig_files(cl, pref)
                for cl, pref in zip(CELL_LINES, PREFIXES)}

    expected_marks = ["ATAC"] + SIGNALS
    per_cl_scales = load_per_cell_line_scales(
        PER_CELL_SCALES_TSV, expected_cell_lines=CELL_LINES, expected_marks=expected_marks
    )
    if per_cl_scales:
        print("Loaded per-cell-line bigwig scales:", per_cl_scales)

    # for cl in CELL_LINES:
    #     plot_hic(
    #         microc_file   = MCOOL_PAT.format(cell_line=cl),
    #         bigwig_files  = bw_files[cl],
    #         global_scales = per_cl_scales.get(cl, {}),
    #         out_base      = f"{OUT_PREFIX}/{OUT_PREFIX}_{cl}",
    #         override_scales = override
    #     )

    # Example loop-centric summary panel (uncomment and set a real loop_id):
    plot_loop_summary(
        loop_id         = loop_id,
        top_cell_line   = "GSC",
        bottom_cell_line= "ESC",
        region          = REGION,
    )