#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download GENCODE mouse GTF for mm10 (GRCm38) and convert to BED12 + 1 column (transcript_id).

Output columns (13 columns):
chrom, start, end, gene_name, score, strand, thickStart, thickEnd, itemRgb,
blockCount, blockSizes, blockStarts, transcript_id

- GTF is 1-based inclusive; BED is 0-based half-open.
- thickStart/thickEnd are derived from CDS; if no CDS, thickStart=chromStart and thickEnd=chromStart.
- By default, one transcript per gene is kept (best = longest CDS, then longest exon sum).
"""

import argparse
import gzip
import os
import re
import sys
import urllib.request
from collections import defaultdict

CANONICAL_CHROMS = [str(i) for i in range(1, 20)] + ["X", "Y", "MT", "M"]
CHROM_ORDER = {c: i for i, c in enumerate(CANONICAL_CHROMS, start=1)}

def parse_gtf_attributes(attr_str: str) -> dict:
    """
    Parse GTF attributes field into a dict.
    Typical format: key "value"; key2 "value2";
    """
    attrs = {}
    # split by ';' then parse key "value"
    for part in attr_str.strip().strip(";").split(";"):
        part = part.strip()
        if not part:
            continue
        m = re.match(r'^(\S+)\s+"([^"]+)"$', part)
        if m:
            attrs[m.group(1)] = m.group(2)
        else:
            # fallback: key value
            pieces = part.split(None, 1)
            if len(pieces) == 2:
                attrs[pieces[0]] = pieces[1].strip().strip('"')
    return attrs

def add_chr_prefix(chrom: str) -> str:
    if chrom.startswith("chr"):
        return chrom
    if chrom in ("MT", "M"):
        return "chrM"
    return "chr" + chrom

def download(url: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    print(f"[download] {url}\n          -> {out_path}", file=sys.stderr)
    urllib.request.urlretrieve(url, out_path)

def segments_length(segments):
    # segments are 1-based inclusive tuples (start,end)
    return sum((e - s + 1) for s, e in segments)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default="M25", help="GENCODE mouse release for mm10/GRCm38 (default: M25)")
    ap.add_argument("--gtf", default=None, help="Path to input GTF(.gz). If omitted, download from GENCODE.")
    ap.add_argument("--out", required=True, help="Output BED path")
    ap.add_argument("--download-dir", default=".", help="Where to save downloaded GTF if --gtf is omitted")
    ap.add_argument("--no-add-chr", action="store_true", help="Do NOT add 'chr' prefix (default adds chr)")
    ap.add_argument("--keep-all-transcripts", action="store_true",
                    help="Keep all transcripts (default keeps one representative transcript per gene)")
    ap.add_argument("--no-canonical-filter", action="store_true",
                    help="Do NOT restrict to canonical chromosomes (default keeps canonical only)")
    args = ap.parse_args()

    # Decide GTF path (download if needed)
    if args.gtf is None:
        # mm10 uses GRCm38; for GENCODE mouse M25 the main CHR GTF is:
        # gencode.vM25.annotation.gtf.gz (reference chromosomes only)
        gtf_name = f"gencode.v{args.release}.annotation.gtf.gz"
        url = (
            "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/"
            f"release_{args.release}/{gtf_name}"
        )
        gtf_path = os.path.join(args.download_dir, gtf_name)
        if not os.path.exists(gtf_path):
            download(url, gtf_path)
        args.gtf = gtf_path

    # Data structures
    # tx_id -> dict with gene_name, gene_id, chrom, strand, exons(set), cds(set)
    tx = {}

    def get_tx(tid, chrom, strand, gene_id, gene_name):
        if tid not in tx:
            tx[tid] = {
                "gene_id": gene_id,
                "gene_name": gene_name,
                "chrom": chrom,
                "strand": strand,
                "exons": set(),
                "cds": set(),
            }
        else:
            # fill missing if needed
            if tx[tid]["gene_name"] in (None, "", "NA") and gene_name:
                tx[tid]["gene_name"] = gene_name
        return tx[tid]

    # Parse GTF
    opener = gzip.open if args.gtf.endswith(".gz") else open
    with opener(args.gtf, "rt") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue

            chrom, source, feature, start, end, score, strand, frame, attrs = fields
            if not args.no_canonical_filter and chrom not in CANONICAL_CHROMS and not chrom.startswith("chr"):
                # When using CHR file this is usually already clean, but keep this for safety
                continue

            start_i = int(start)
            end_i = int(end)
            a = parse_gtf_attributes(attrs)

            gene_id = a.get("gene_id", "")
            tx_id = a.get("transcript_id", "")
            gene_name = a.get("gene_name") or a.get("gene") or gene_id or tx_id

            if not tx_id:
                # gene features don't have transcript_id; we only need transcript-level models
                continue

            rec = get_tx(tx_id, chrom, strand, gene_id, gene_name)

            if feature == "exon":
                rec["exons"].add((start_i, end_i))
            elif feature == "CDS":
                rec["cds"].add((start_i, end_i))

    # Convert each transcript to BED12+1
    bed_rows = []
    for tid, rec in tx.items():
        exons = sorted(rec["exons"])
        if not exons:
            continue

        chrom = rec["chrom"]
        if not args.no_add_chr:
            chrom = add_chr_prefix(chrom)

        strand = rec["strand"]
        gene_name = rec["gene_name"] if rec["gene_name"] else tid

        # BED: 0-based half-open
        exon_bed = [(s - 1, e) for s, e in exons]  # (start0, end)
        tx_start = min(s for s, e in exon_bed)
        tx_end = max(e for s, e in exon_bed)

        # blocks
        exon_bed_sorted = sorted(exon_bed, key=lambda x: x[0])
        block_sizes = [(e - s) for s, e in exon_bed_sorted]
        block_starts = [(s - tx_start) for s, e in exon_bed_sorted]
        block_count = len(exon_bed_sorted)

        # thickStart/End from CDS if available
        cds = sorted(rec["cds"])
        if cds:
            cds_bed = [(s - 1, e) for s, e in cds]
            thick_start = min(s for s, e in cds_bed)
            thick_end = max(e for s, e in cds_bed)
        else:
            thick_start = tx_start
            thick_end = tx_start

        # format lists like your example (trailing comma)
        block_sizes_str = ",".join(map(str, block_sizes)) + ","
        block_starts_str = ",".join(map(str, block_starts)) + ","

        bed_rows.append({
            "chrom": chrom,
            "start": tx_start,
            "end": tx_end,
            "name": gene_name,
            "score": 0,
            "strand": strand,
            "thickStart": thick_start,
            "thickEnd": thick_end,
            "itemRgb": 0,
            "blockCount": block_count,
            "blockSizes": block_sizes_str,
            "blockStarts": block_starts_str,
            "transcript_id": tid,
            "gene_name": gene_name,
            "cds_len": segments_length(cds) if cds else 0,
            "exon_len": segments_length(exons),
        })

    # Option: keep only one transcript per gene (like *_names_unique.bed)
    if not args.keep_all_transcripts:
        best_by_gene = {}
        for r in bed_rows:
            g = r["gene_name"]
            # rank: longest CDS, then longest exon length, then longest span
            key = (r["cds_len"], r["exon_len"], r["end"] - r["start"])
            if (g not in best_by_gene) or (key > best_by_gene[g][0]) or (key == best_by_gene[g][0] and r["transcript_id"] < best_by_gene[g][1]["transcript_id"]):
                best_by_gene[g] = (key, r)
        bed_rows = [v[1] for v in best_by_gene.values()]

    # Sort output (canonical chrom order if possible)
    def sort_key(r):
        c = r["chrom"].replace("chr", "")
        # chrM -> MT-like order
        if c == "M":
            c = "MT"
        return (CHROM_ORDER.get(c, 10**9), r["start"], r["end"], r["name"])

    bed_rows.sort(key=sort_key)

    # Write output
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wt") as out:
        for r in bed_rows:
            out.write(
                "\t".join(map(str, [
                    r["chrom"], r["start"], r["end"], r["name"], r["score"], r["strand"],
                    r["thickStart"], r["thickEnd"], r["itemRgb"],
                    r["blockCount"], r["blockSizes"], r["blockStarts"], r["transcript_id"]
                ])) + "\n"
            )

    print(f"[done] wrote {len(bed_rows)} rows -> {args.out}", file=sys.stderr)

if __name__ == "__main__":
    main()
