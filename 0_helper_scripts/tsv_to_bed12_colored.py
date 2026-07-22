#!/usr/bin/env python3
'''
tsv_to_bed12_colored.py

Convert a wide-format anchor TSV (chrom, mid, cluster, ...) into a
cluster-colored BED12 file for genome browser visualization.

Each row is treated as a single-block ('point') anchor: the interval is
built as [mid - width/2, mid + width/2). If your TSV already carries an
explicit per-row span (e.g. a `window` column in bp), pass --width-col
window to use that instead of a fixed --width.

Usage:
    python tsv_to_bed12_colored.py \
        --input anchors.tsv \
        --output anchors_colored.bed12 \
        --width 3000

    # or, using an existing per-row width column instead of a fixed width:
    python tsv_to_bed12_colored.py \
        --input anchors.tsv \
        --output anchors_colored.bed12 \
        --width-col window
'''

import argparse
import sys
import pandas as pd

# ---------------------------------------------------------------------------
# EDIT THIS: map your cluster labels -> RGB color (as 'R,G,B' strings, 0-255)
# Keys should match the values in your `cluster` column exactly (str or int
# both fine, they'll be coerced to str for lookup).
# ---------------------------------------------------------------------------
CLUSTER_COLORS = {
    'CTCF':        '214,39,40',    # reddish
    'CRE':         '158,202,225',  # light blue
    'PRC':         '0,90,50',      # dark green
    'CTCF-CRE':    '129,55,140',   # purple
    'Cohesin-CRE': '8,48,107',     # dark blue
}

CLUSTER_NAMES = {
    0: 'CTCF-CRE',
    1: 'CTCF',
    2: 'CRE',
    3: 'CTCF',
    4: 'CRE',
    5: 'PRC',
    6: 'CRE',
    7: 'CTCF',
    8: 'CTCF',
    9: 'CTCF',
    10: 'PRC',
    11: 'CRE',
    12: 'Cohesin-CRE',
}
DEFAULT_COLOR = '128,128,128'  # gray fallback for unmapped clusters


def build_bed12(df, width=None, width_col=None, name_col='anchor_id',
                 chrom_col='chrom', mid_col='mid', cluster_col='cluster',
                 strand_default='.'):
    required = [chrom_col, mid_col, cluster_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f'ERROR: input TSV is missing required column(s): {missing}')

    if width_col is not None:
        if width_col not in df.columns:
            sys.exit(f"ERROR: --width-col {width_col} not found in TSV columns")
        half_width = df[width_col].astype(float) / 2.0
    elif width is not None:
        half_width = pd.Series(width / 2.0, index=df.index)
    else:
        sys.exit('ERROR: must supply either --width or --width-col')

    chrom_start = (df[mid_col].astype(float) - half_width).round().astype(int).clip(lower=0)
    chrom_end = (df[mid_col].astype(float) + half_width).round().astype(int)

    if name_col in df.columns:
        name = df[name_col].astype(str)
    else:
        name = df[chrom_col].astype(str) + '_' + df[mid_col].astype(str)

    cluster = df[cluster_col].astype(int)
    item_rgb = cluster.map(CLUSTER_NAMES).map(CLUSTER_COLORS).fillna(DEFAULT_COLOR)

    unmapped = sorted(set(cluster[item_rgb == DEFAULT_COLOR]))
    if unmapped:
        print(f'WARNING: {len(unmapped)} cluster label(s) had no color mapping, '
              f'used default gray: {unmapped}', file=sys.stderr)

    bed = pd.DataFrame({
        'chrom': df[chrom_col],
        'chromStart': chrom_start,
        'chromEnd': chrom_end,
        'name': name,
        'score': 0,
        'strand': strand_default,
        'thickStart': chrom_start,   # no CDS/thick region -> thickStart == chromStart
        'thickEnd': chrom_start,     # -> renders as a thin, fully-colored feature
        'itemRgb': item_rgb,
        'blockCount': 1,
        'blockSizes': (chrom_end - chrom_start).astype(str) + ',',
        'blockStarts': '0,',
    })

    # BED convention: sort by chrom then start
    bed = bed.sort_values(['chrom', 'chromStart']).reset_index(drop=True)
    return bed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', required=True, help='Input TSV path')
    ap.add_argument('--output', required=True, help='Output BED12 path')
    ap.add_argument('--width', type=float, default=None,
                     help='Fixed anchor width in bp, centered on `mid` (e.g. 3000)')
    ap.add_argument('--width-col', default=None,
                     help="Use this column's value (bp) as per-row anchor width instead of --width")
    ap.add_argument('--name-col', default='anchor_id')
    ap.add_argument('--chrom-col', default='chrom')
    ap.add_argument('--mid-col', default='mid')
    ap.add_argument('--cluster-col', default='cluster')
    ap.add_argument('--track-name', default='anchor_clusters',
                     help="UCSC track line 'name' field")
    ap.add_argument('--no-track-line', action='store_true',
                     help="Don't write a UCSC track header line")
    args = ap.parse_args()

    df = pd.read_csv(args.input, sep='\t')

    bed = build_bed12(
        df,
        width=args.width,
        width_col=args.width_col,
        name_col=args.name_col,
        chrom_col=args.chrom_col,
        mid_col=args.mid_col,
        cluster_col=args.cluster_col,
    )

    with open(args.output, 'w') as f:
        if not args.no_track_line:
            f.write(
                f"track name='{args.track_name}' description='anchors colored by cluster' "
                f"itemRgb='On'\n"
            )
        bed.to_csv(f, sep='\t', header=False, index=False)

    print(f'Wrote {len(bed)} anchors to {args.output}', file=sys.stderr)


if __name__ == '__main__':
    main()
