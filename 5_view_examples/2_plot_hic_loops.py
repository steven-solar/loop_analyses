# conda activate coolbox

"""
here for completeness sake, but ended up just using 3_plot_fig.py
2_plot_hic_loops.py
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
from matplotlib.ticker import FuncFormatter, MultipleLocator
from matplotlib.transforms import Bbox
import cooler
from cooltools.lib import plotting
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

# REGION     = "chr10:44,400,000-44,600,000"
# OUT_PREFIX = 'prdm1'

# REGION = 'chr1:9,528,380-9,739,828'
# OUT_PREFIX = 'rrs1'

# REGION = 'chr13:112,500,000-113,000,000' #chr13:112,899,382-113,053,095
# OUT_PREFIX = "ddx4"

# REGION = 'chr1:188,987,080-189,251,543'
# OUT_PREFIX = "random_ctcf"
# MICROC_PARAMS['max_value'] = -2
# MICROC_PARAMS['min_value'] = -3.1

# REGION = 'chr19:44,663,756-46,592,873'
# OUT_PREFIX = 'pax2_chr19_44.6-46.6'

# REGION='chrX:57,850,103-58,053,314'
# OUT_PREFIX = 'zic3_chrX_57.9-58.2'

# REGION='chrX:57,892,452-58,289,190'
# OUT_PREFIX='mock'

# REGION=('chrX:57,980,000-58,060,000', 'chrX:58,110,000-58,160,000')
# OUT_PREFIX='zic3_ctcf'

# REGION='chr13:112,651,138-112,881,482'
# OUT_PREFIX = "ddx4"

# REGION='chr17:14,116,163-14,236,893'
# OUT_PREFIX='Dact2'

# REGION='chr19:23,063,344-23,172,278'
# OUT_PREFIX='Klf9'

# REGION='chr14:99,252,418-99,602,528'
# OUT_PREFIX='Klf5'

# REGION='chr4:124,400,000-124,750,000'
# OUT_PREFIX='Pou3f1'

# REGION='chrX:57,800,000-58,300,000'
# OUT_PREFIX = 'zic3_chrX_57.8-58.3'
# MICROC_PARAMS['max_value'] = -1.75
# MICROC_PARAMS['min_value'] = -3.25

REGION='chrX:57,880,000-58,180,000'
OUT_PREFIX = 'zic3_chrX_57.88-58.18'
MICROC_PARAMS['max_value'] = -1.25
MICROC_PARAMS['min_value'] = -3.75
SIGNALS = ['ATAC', 'K27ac', 'Ring1b', 'K27me3']

# REGION='chr11:116,000,000-116,820,148'
# OUT_PREFIX='Unk'

# REGION='chr9:121,800,000-122,000,000'
# OUT_PREFIX='Higd1a'
# MICROC_PARAMS['max_value'] = -1.99
# MICROC_PARAMS['min_value'] = -3.25

# REGION='chr13:21,089,347-22,101,997'
# OUT_PREFIX='Zfp184_c12'
# SIGNALS = ['ATAC', 'K27ac', 'K4me1', 'K4me3', 'Rad21', 'CTCF', 'Stag1', 'Stag2', 'Ring1b', 'K27me3']
# MICROC_PARAMS['max_value'] = -1.25
# MICROC_PARAMS['min_value'] = -4.5

# REGION='chr11:97,750,000-98,000,000'
# OUT_PREFIX='Lasp1_c12'
# SIGNALS = ['ATAC', 'K27ac', 'K4me1', 'K4me3', 'Rad21', 'CTCF', 'Stag1', 'Stag2', 'Ring1b', 'K27me3']
# MICROC_PARAMS['max_value'] = -1.5
# MICROC_PARAMS['min_value'] = -3.25

# REGION='chr13:23,420,000-23,540,000'
# OUT_PREFIX='Btn1a1_c12'
# SIGNALS = ['ATAC', 'K27ac', 'K4me1', 'K4me3', 'Rad21', 'CTCF', 'Stag1', 'Stag2', 'Ring1b', 'K27me3']
# MICROC_PARAMS['max_value'] = -1.25
# MICROC_PARAMS['min_value'] = -3

# REGION='chr10:79,902,872-79,963,675'
# OUT_PREFIX='Med16_c12'
# SIGNALS = ['ATAC', 'K27ac', 'K4me1', 'K4me3', 'Rad21', 'CTCF', 'Stag1', 'Stag2', 'Ring1b', 'K27me3']
# MICROC_PARAMS['max_value'] = -1.25
# MICROC_PARAMS['min_value'] = -3.1


# REGION = 'chr1:8,500,000-8,975,000 '
# OUT_PREFIX = 'Sntg1'
# SIGNALS = ['ATAC', 'K27ac', 'K4me1', 'K4me3', 'K9me2', 'K9me3', 'K36me2', 'K36me3']
# MICROC_PARAMS['max_value'] = -2.5
# MICROC_PARAMS['min_value'] = -3.75

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
    "zic3_big": {
        "ATAC": 2,
        "K27ac": 2,
        "K4me3": 5,
        "Rad21": 0.5,
        "Ring1b": 5,
    },
    "mock": {
        "ATAC": 2,
        "K27ac": 2,
        "K4me1": 0.5,
        "K4me3": 5,
        "Rad21": 0.5,
        "Ring1b": 5,
        "CTCF":0.5,
        "H2Aub": 1,
        "Stag1": 1,
        "Stag2": 1,
    },
    "random_ctcf": {
        "CTCF": 1,
        "Rad21": 1,
        "Stag1": 1,
        "Stag2": 1
    },
    "ddx4": {
        "CTCF":2,
        "Rad21": 1,
    },
    "Dact2": {
        'ATAC': 3,
        'K4me3': 10,
        'K9me2': 0.5,
        'K9me3': 0.5,
    },
    "Klf9": {
        "ATAC": 3,
        "Rad21": 1,
        "K27ac": 5,
        "K4me1": 10,
        'K9me2': 0.5,
        'K9me3': 0.5,
    },
    "Pou3f1": {
        "ATAC": 3,
        "K27ac": 5,
        "K4me1": 2,
        "K4me3": 10,
        "Rad21": 2,
    },
    "Higd1a": {
        "K4me3": 10,
        "K27ac": 2.5,
    },
    "zic3_chrX_57.8-58.3": {
        "ATAC": 2,
        "K27ac": 2,
        "K4me3": 5,
        "Rad21": 0.5,
        "Ring1b": 5,
    },
    'Zfp184_c12': {
        'ATAC':2,
        'K27ac': 5,
        'K4me1': 1,
        'K4me3': 10,
        'Rad21': 2,
        'Stag2': 10,
    },
    'Btn1a1_c12': {
        'ATAC':2,
        'K27ac': 5,
        'K4me1': 1,
        'K4me3': 10,
        'Rad21': 2,
        'Stag1': 10,
        'Stag2': 10,
        'CTCF':5,
    },
    "Med16_c12": {
        'ATAC':5,
        'K27ac': 10,
        'K4me1': 1,
        'K4me3': 10,
        'Rad21': 2,
        'Stag1': 5,
        'Stag2': 10,
        'CTCF':5,
        'Ring1b': 5
    },
    "Sntg1": {
        "ATAC": 4,
        "K27ac":5,
        "K9me2":0.5,
        "K9me3":0.5,
        "K36me2":0.5,
        "K36me3":0.5
    }
}

override = REGION_SCALES.get(OUT_PREFIX, EPIG_MAX)

RESOLUTION = 1000

OUTPUT_DIR = "plots"

FRAME_WIDTH_CM = 12 * 2.54   # coolbox Frame.width is in cm
BIGWIG_BINS    = 1600
BIGWIG_HEIGHT  = 2
BIGWIG_SPACER  = 0.4
BIGWIG_CMAP    = "tab10"


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
    """
    # assume aspect is already equal
    fig.canvas.draw()
    tight = ax.get_tightbbox(fig.canvas.get_renderer())  # in display units (px)
    side  = min(tight.width, tight.height)
    cx    = (tight.x0 + tight.x1) / 2
    cy    = (tight.y0 + tight.y1) / 2
    sq_px = Bbox([[cx - side / 2, cy - side / 2],
                  [cx + side / 2, cy + side / 2]])

    sq_in = sq_px.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(path, bbox_inches=sq_in)
    print(f"saved square matrix: {path}")


def _finalize_and_export(frame, region, output_dir, out_base):
    """
    Shared tail end of every matrix plot: render the Frame, apply the
    log10 colorbar relabeling + Mb ticks + equal aspect to the matrix
    axes, then export _full/_square svgs. Used by plot_hic (including the
    bigwig-free merge case) and plot_split_diag so all matrix plots come
    out identically formatted.
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
    # make sure the bottom spine is drawn
    if "bottom" in matrix_ax.spines:
        matrix_ax.spines["bottom"].set_visible(True)

    matrix_ax.set_aspect("equal", adjustable="box")

    try:
        _set_mb_ticks(matrix_ax)
    except Exception as e:
        print(f"Warning: Mb ticks on matrix failed: {e}")

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
        self.ax = ax  # coolbox tracks are expected to record their own ax; the
                      # base class only initializes it to None (see coolbox's
                      # Track.__init__) -- each concrete track sets it in plot().
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

        # Match coolbox's Cool track exactly (see coolbox's ProcessHicMat.
        # fill_zero_nan): replace true-zero and masked/NaN bins with a tiny
        # pseudocount *before* log-transforming, instead of letting them
        # come out as -inf/NaN. Otherwise NaN bins fall through matplotlib's
        # default fully-transparent "bad" color (pure white) rather than the
        # colormap's own pale floor color, which is what made masked/
        # low-coverage bins (more common on a single cell line's matrix than
        # on a pooled merge) look noticeably whiter/higher-contrast here.
        small = 1e-12
        combined[combined == 0] = small
        combined[np.isnan(combined)] = small
        log_combined = np.log10(combined)

        # coolbox also copies the cmap and pins both set_bad and set_under
        # to the colormap's own lowest color, so anything at or below the
        # floor (including the pseudocount bins above) renders identically
        # to the rest of the low end instead of matplotlib's default
        # transparent "bad" color.
        cmap = copy.copy(plt.get_cmap(self.params["cmap"]))
        floor = cmap(0)
        cmap.set_bad(floor)
        cmap.set_under(floor)

        # extent=(left, right, bottom, top): putting gr.start at "top" and
        # gr.end at "bottom" flips the y-axis so the region's start sits at
        # the top-left corner and the diagonal runs top-left -> bottom-right,
        # matching the other (non-split) heatmaps.
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
    """Load {cell_line: {mark: (0.0, max_value)}} from a per_cell_line_scales.tsv.

    Expects columns: cell_line, mark, max_value (mean/std/n_anchors columns,
    if present, are ignored here -- max_value already has n_std baked in).
    Warns on any (cell_line, mark) combo missing from the file; those
    tracks fall back to per-region auto-scaling in _build_bigwig_tracks.
    """
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

        # 1) user override_scales
        if override_scales and label in override_scales:
            val = override_scales[label]
            if isinstance(val, (list, tuple)) and len(val) == 2:
                y_min, y_max = float(val[0]), float(val[1])
            else:
                # treat scalar as max only
                y_min, y_max = 0.0, float(val)
            source = "override_scales"

        # 2) per-cell-line global_scales
        elif global_scales and label in global_scales:
            ymin, ymax = global_scales[label]
            y_min, y_max = float(ymin), float(ymax)
            source = "global_scales"

        # 3) fall back to auto-scale from data in this region
        else:
            data = BigWig(path, number_of_bins=BIGWIG_BINS).fetch_plot_data(
                GenomeRange(region)
            )
            y_min = 0.0
            y_max = float(np.nanmax(data)) if data is not None and len(data) else 0.0
            source = "region auto-scale"

        print(f"[track] {label}: min_value={y_min}, max_value={y_max} (source: {source})")

        tracks.append(
            BigWig(path, number_of_bins=BIGWIG_BINS, min_value=y_min, max_value=y_max)
            + Title(label)
            + TrackHeight(BIGWIG_HEIGHT)
            + Color(colors[i])
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


def plot_split_diag(
    top_cell_line,
    bottom_cell_line,
    region=REGION,
    resolution=RESOLUTION,
    mc_params=None,
    bed_file=BED_FILE,
    output_dir=OUTPUT_DIR,
    out_base=None,
):
    """
    Single diagonal heatmap split across the main diagonal: upper triangle
    from `top_cell_line`, lower triangle from `bottom_cell_line` (diagonal
    bins are the mean of both, since neither side owns them). Formatted
    identically to the plain matrix plots (e.g. the merged heatmap in
    plot_hic) -- same Frame width, Title, BED track, log10 colorbar
    relabeling, Mb ticks, and _full/_square export.
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
    return _finalize_and_export(frame, region, output_dir, out_base)


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

    # Assemble coolbox Frame
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

    for cl in CELL_LINES:
        plot_hic(
            microc_file   = MCOOL_PAT.format(cell_line=cl),
            bigwig_files  = bw_files[cl],
            global_scales = per_cl_scales.get(cl, {}),
            out_base      = f"{OUT_PREFIX}/{OUT_PREFIX}_{cl}",
            override_scales = override
        )

    # Merged heatmap — matrix only, no bigwigs
    # plot_hic(
    #     microc_file = "../data/merged.mcool",
    #     out_base    = f"{OUT_PREFIX}/{OUT_PREFIX}_merge",
    # )

    # Example split-diagonal comparison, e.g. ESC on top vs GSC on bottom,
    # formatted the same as the merge plot above:
    plot_split_diag("GSC", "ESC", out_base=f"{OUT_PREFIX}/{OUT_PREFIX}_ESC_vs_GSC")