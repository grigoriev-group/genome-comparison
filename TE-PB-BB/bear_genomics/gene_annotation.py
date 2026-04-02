"""Gene and repeat element annotation for structural variant datasets."""

import os
import re
import logging
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np


def _rm_parse_bed(path: str) -> pd.DataFrame:
    import pandas as pd, os

    # Empty skeleton on failure
    empty = pd.DataFrame(columns=[
        "chrom","start","end","repeat_name","score",
        "strand","class_family","repeat_class","repeat_family"
    ])

    if not path or not os.path.exists(path):
        return empty

    # Read raw (no dtype enforcement yet)
    try:
        df = pd.read_csv(path, sep="\t", header=None, comment="#", engine="python")
    except Exception:
        return empty

    # 🔹 DROP BAD HEADER LINES (those with non-numeric positions)
    df = df[df[1].apply(lambda x: str(x).isdigit())]

    # Assign standard RM columns
    cols = ['chrom','start','end','repeat_name','score','strand','class_family']
    while len(cols) < df.shape[1]:
        cols.append(f"extra_{len(cols)}")
    df.columns = cols[:df.shape[1]]

    # Convert numeric columns safely
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"]   = pd.to_numeric(df["end"], errors="coerce")
    df = df.dropna(subset=["start","end"])
    df = df[df["end"] > df["start"]]

    # 🔹 Extract class & family
    if "class_family" in df.columns:
        split = df["class_family"].astype(str).str.split("/", n=1, expand=True)
        df["repeat_class"]  = split[0].str.strip()
        df["repeat_family"] = split[1].str.strip() if split.shape[1] > 1 else ""

    # Return normalized
    return df[[
        "chrom","start","end","repeat_name","score",
        "strand","class_family","repeat_class","repeat_family"
    ]].copy()


def _rm_annotate_intervals(intervals_df: pd.DataFrame, repeat_bed_path: str,
                           chrom='chrom', start='start', end='end') -> pd.DataFrame:
    """
    Vectorized, light-weight overlap annotator.
    Input intervals_df must have: chrom,start,end (ints); may carry 'join_id' for merge-back.
    Returns a DataFrame with the same index as intervals_df (or join_id if present),
    and these columns: REPEAT_OVERLAP, REPEAT_OVERLAP_BP, REPEAT_OVERLAP_PERCENT,
    REPEAT_TYPES, REPEAT_NAMES, REPEAT_FAMILIES, REPEAT_CLASSES,
    PRIMARY_REPEAT_TYPE, PRIMARY_REPEAT_NAME, PRIMARY_REPEAT_OVERLAP_BP,
    NUM_REPEAT_ELEMENTS, REPEAT_DENSITY, DISTANCE_TO_NEAREST_REPEAT.
    """
    if intervals_df is None or intervals_df.empty:
        return pd.DataFrame(index=intervals_df.index if intervals_df is not None else None)

    df = intervals_df.copy()
    df[chrom]  = df[chrom].astype(str)
    df[start]  = pd.to_numeric(df[start], errors='coerce').fillna(0).astype(int)
    df[end]    = pd.to_numeric(df[end],   errors='coerce').fillna(0).astype(int)

    # Swap bad intervals
    bad = df[end] < df[start]
    if bad.any():
        s_tmp = df.loc[bad, start].values
        df.loc[bad, start] = df.loc[bad, end].values
        df.loc[bad, end]   = s_tmp

    repeats = _rm_parse_bed(repeat_bed_path)
    if repeats.empty:
        out = pd.DataFrame(index=df.index)
        out['REPEAT_OVERLAP'] = 'No'
        out['REPEAT_OVERLAP_BP'] = 0
        out['REPEAT_OVERLAP_PERCENT'] = 0.0
        out['REPEAT_TYPES'] = ''
        out['REPEAT_NAMES'] = ''
        out['REPEAT_FAMILIES'] = ''
        out['REPEAT_CLASSES'] = ''
        out['PRIMARY_REPEAT_TYPE'] = ''
        out['PRIMARY_REPEAT_NAME'] = ''
        out['PRIMARY_REPEAT_OVERLAP_BP'] = 0
        out['NUM_REPEAT_ELEMENTS'] = 0
        out['REPEAT_DENSITY'] = 0.0
        out['DISTANCE_TO_NEAREST_REPEAT'] = -1
        return out

    # Per-chrom slice + python loop is simplest & reliable for big tables.
    # (Keeps memory bounded; faster than cartesian joins.)
    results = []
    for i, row in df.iterrows():
        c = row[chrom]; s = row[start]; e = row[end]
        del_size = max(0, e - s)
        rep_c = repeats[repeats['chrom'] == c]
        ovl   = rep_c[(rep_c['end'] > s) & (rep_c['start'] < e)]
        if ovl.empty:
            # nearest distance
            before = (rep_c['start'] >= e)
            after  = (rep_c['end']   <= s)
            dists = []
            if before.any(): dists.append(int(rep_c.loc[before, 'start'].min() - e))
            if after.any():  dists.append(int(s - rep_c.loc[after, 'end'].max()))
            nearest = (min(dists) if dists else -1)
            results.append({
                'idx': i,
                'REPEAT_OVERLAP': 'No',
                'REPEAT_OVERLAP_BP': 0,
                'REPEAT_OVERLAP_PERCENT': 0.0,
                'REPEAT_TYPES': '',
                'REPEAT_NAMES': '',
                'REPEAT_FAMILIES': '',
                'REPEAT_CLASSES': '',
                'PRIMARY_REPEAT_TYPE': '',
                'PRIMARY_REPEAT_NAME': '',
                'PRIMARY_REPEAT_OVERLAP_BP': 0,
                'NUM_REPEAT_ELEMENTS': 0,
                'REPEAT_DENSITY': 0.0,
                'DISTANCE_TO_NEAREST_REPEAT': nearest
            })
            continue

        # overlaps
        ovl_bp = (ovl[['start','end']]
                  .clip(lower=s, upper=e).eval('end - start').clip(lower=0)).sum()
        # roll up labels
        names   = set(ovl.get('repeat_name',   pd.Series([], dtype=object)).dropna().astype(str))
        classes = set(ovl.get('repeat_class',  pd.Series([], dtype=object)).dropna().astype(str))
        fams    = set(ovl.get('repeat_family', pd.Series([], dtype=object)).dropna().astype(str))
        # Use class as "type" fallback
        types   = classes.copy()

        # primary = element with max overlap
        seg = ovl.copy()
        seg['ov'] = seg[['start','end']].clip(lower=s, upper=e).eval('end - start').clip(lower=0)
        primary = seg.loc[seg['ov'].idxmax()] if not seg.empty else None

        results.append({
            'idx': i,
            'REPEAT_OVERLAP': 'Yes',
            'REPEAT_OVERLAP_BP': int(ovl_bp),
            'REPEAT_OVERLAP_PERCENT': float((ovl_bp / del_size * 100) if del_size else 0.0),
            'REPEAT_TYPES': ';'.join(sorted(t for t in types if t)),
            'REPEAT_NAMES': ';'.join(sorted(n for n in names if n)),
            'REPEAT_FAMILIES': ';'.join(sorted(f for f in fams if f)),
            'REPEAT_CLASSES': ';'.join(sorted(c for c in classes if c)),
            'PRIMARY_REPEAT_TYPE': (str(primary.get('repeat_class')) if primary is not None else ''),
            'PRIMARY_REPEAT_NAME': (str(primary.get('repeat_name'))  if primary is not None else ''),
            'PRIMARY_REPEAT_OVERLAP_BP': int(primary['ov'] if primary is not None else 0),
            'NUM_REPEAT_ELEMENTS': int(len(ovl)),
            'REPEAT_DENSITY': float((ovl_bp / del_size) if del_size else 0.0),
            'DISTANCE_TO_NEAREST_REPEAT': 0
        })
    out = pd.DataFrame(results).set_index('idx').sort_index()
    return out


def _ensure_repeat_signature_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize columns the pipeline expects.
    Derive REPEAT_SIGNATURE if missing (Class|Family|Name or fallback to PRIMARY_REPEAT_TYPE).
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    # Guarantee presence
    for col in ['REPEAT_CLASS','REPEAT_FAMILY','REPEAT_NAME','REPEAT_TYPES','REPEAT_CLASSES',
                'PRIMARY_REPEAT_TYPE','PRIMARY_REPEAT_NAME']:
        if col not in out.columns:
            out[col] = out.get(col, 'None')
    if 'REPEAT_SIGNATURE' not in out.columns:
        # Prefer single primary labels; fallback to combined label
        out['REPEAT_SIGNATURE'] = out.apply(
            lambda r: (f"{r.get('PRIMARY_REPEAT_TYPE','')}"
                       if r.get('PRIMARY_REPEAT_TYPE') else
                       (r.get('REPEAT_CLASSES') or r.get('REPEAT_TYPES') or 'None')),
            axis=1
        )
    return out


def _parse_repeatmasker_bed(path: str) -> pd.DataFrame:
    """
    Parse a RepeatMasker BED-like file into a normalized DataFrame.

    Returns columns:
      chrom, start, end, repeat_name, score, strand, class_family, repeat_class, repeat_family
    Notes:
      - start/end are coerced to int with non-numeric → 0 then filtered (end > start, start >= 0)
      - class_family is split on '/' → repeat_class, repeat_family
    """
    import os
    import pandas as pd

    # Empty skeleton for early returns
    empty = pd.DataFrame(columns=[
        "chrom","start","end","repeat_name","score","strand",
        "class_family","repeat_class","repeat_family"
    ])

    if not path or not os.path.exists(path):
        return empty

    # Read as strings; we'll coerce numerics ourselves
    try:
        df = pd.read_csv(path, sep="\t", header=None, comment="#", dtype=str, engine="python")
    except Exception:
        try:
            df = pd.read_csv(path, sep=r"\s+", header=None, comment="#", dtype=str, engine="python")
        except Exception as e:
            raise RuntimeError(f"Unable to parse RepeatMasker file {path}: {e}")

    if df.shape[1] < 3:
        raise RuntimeError(f"RepeatMasker file has <3 columns: {path}")

    df = df.fillna("")

    # Build normalized output
    out = pd.DataFrame(index=df.index)
    out["chrom"] = df.iloc[:, 0].astype(str).str.strip()

    # robust integer coercion
    out["start"] = pd.to_numeric(df.iloc[:, 1], errors="coerce").fillna(0).astype(int)
    out["end"]   = pd.to_numeric(df.iloc[:, 2], errors="coerce").fillna(0).astype(int)

    # Optional columns
    out["repeat_name"] = (df.iloc[:, 3].astype(str).str.strip()) if df.shape[1] > 3 else ""
    out["score"]       = (df.iloc[:, 4].astype(str).str.strip()) if df.shape[1] > 4 else ""
    out["strand"]      = (df.iloc[:, 5].astype(str).str.strip()) if df.shape[1] > 5 else ""

    # Heuristically locate class/family column (look right-to-left for a "class/family" pattern)
    class_family_series = None
    # Search from the last column down to column 6 (inclusive)
    for c in range(df.shape[1] - 1, 5, -1):
        vals = df.iloc[:, c].astype(str)
        # Only sample first ~200 to keep it quick
        if vals.head(200).str.contains("/", regex=False).any():
            class_family_series = vals
            break

    if class_family_series is None:
        # Fallback to column 6 if it exists, else empty
        class_family_series = (df.iloc[:, 6].astype(str) if df.shape[1] > 6
                               else pd.Series([""] * len(df), index=df.index))

    out["class_family"] = class_family_series.astype(str).str.replace("?", "", regex=False).str.strip()

    # Split class/family
    split = out["class_family"].str.split("/", n=1, expand=True)
    out["repeat_class"]  = split[0].fillna("").str.strip()
    out["repeat_family"] = (split[1] if split.shape[1] > 1 else "").fillna("").astype(str).str.strip()

    # Keep only sane intervals
    out = out[(out["end"] > out["start"]) & (out["start"] >= 0)].copy()

    # (Optional) also expose the legacy 'name' field if other code references it
    if "name" not in out.columns:
        out["name"] = out["repeat_name"]

    return out


def _safe_int(x, default=0):
    try: return int(x)
    except Exception: return default


def _overlap_len(a_start, a_end, b_start, b_end):
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def _build_gene_model_from_gff(gff_path: str, gene_name_field: str = "gene", logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    cols = ["chrom","source","feature","start","end","score","strand","phase","attrs"]
    if not gff_path or (not os.path.exists(gff_path)): raise FileNotFoundError(f"GFF not found: {gff_path}")
    rows = []
    with open(gff_path, "r") as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"): continue
            parts = line.rstrip("\\n").split("\\t")
            if len(parts) < 9: continue
            chrom, source, feature, start, end, score, strand, phase, attrs = parts
            try:
                start = int(start); end = int(end)
                if end <= start: continue
            except Exception: continue
            rows.append((chrom, source, feature, start, end, score, strand, phase, attrs))
    if not rows: raise RuntimeError("No features parsed from GFF")
    df = pd.DataFrame(rows, columns=cols)
    def extract_name(a: str) -> str:
        for key in [gene_name_field,"gene_name","gene","Name","ID","gene_id","transcript_id"]:
            m = re.search(rf"{re.escape(key)}=([^;]+)", a)
            if m: return m.group(1)
        for pat in [r'gene_name "([^"]+)"', r'gene_id "([^"]+)"', r'Name "([^"]+)"']:
            m = re.search(pat, a)
            if m: return m.group(1)
        return ""
    df["gene_name"] = df["attrs"].apply(extract_name)
    keep = df["feature"].isin(["gene","mRNA","transcript","exon","CDS"])
    df = df[keep].copy()
    gene_groups = []
    for gname, gdf in df.groupby("gene_name"):
        if not gname: continue
        chrom = gdf["chrom"].mode().iat[0]
        s_candidates = gdf[gdf["feature"].isin(["gene","mRNA","transcript"])]["strand"]
        strand = s_candidates.mode().iat[0] if not s_candidates.empty else gdf["strand"].mode().iat[0]
        bounds = gdf[gdf["feature"].isin(["gene","mRNA","transcript"])]
        if bounds.empty: bounds = gdf
        gstart = bounds["start"].min(); gend = bounds["end"].max()
        exons = gdf[gdf["feature"]=="exon"][["start","end"]].values.tolist()
        cds   = gdf[gdf["feature"]=="CDS"][["start","end"]].values.tolist()
        mrnas = gdf[gdf["feature"].isin(["mRNA","transcript"])][["start","end"]].values.tolist()
        if strand == "-": tss, tes = gend, gstart
        else:             tss, tes = gstart, gend
        gene_groups.append({
            "gene_name": gname, "chrom": chrom, "strand": strand,
            "gene_start": int(gstart), "gene_end": int(gend),
            "tss": int(tss), "tes": int(tes),
            "exons": exons, "cds": cds, "mrnas": mrnas
        })
    return pd.DataFrame(gene_groups)


def _classify_region_for_row(row: pd.Series, primary_gene: Dict[str, Any], promoter_bp: int = 2000,
                             terminator_bp: int = 2000) -> Dict[str, Any]:
    """
    Classify genomic region relative to primary gene with complete logic.

    Args:
        row: DataFrame row containing variant/region information
        primary_gene: Dictionary with gene annotation data
        promoter_bp: Promoter region size in base pairs
        terminator_bp: Terminator region size in base pairs

    Returns:
        Dictionary with region classification and metadata
    """
    # Extract coordinates with fallback options
    start = _safe_int(row.get("START", row.get("MIN_START", row.get("start", 0))))
    end = _safe_int(row.get("END", row.get("MAX_END", row.get("end", 0))))
    chrom = str(row.get("CHROM", row.get("chrom", "")))

    # Handle intergenic case (no primary gene or different chromosome)
    if not primary_gene or chrom != str(primary_gene.get("chrom", "")):
        return {
            "REGION_CLASS": "intergenic",
            "SIDE_RELATIVE_TO_PRIMARY": "intergenic",
            "DIST_TO_PRIMARY_5P": np.nan,
            "DIST_TO_PRIMARY_3P": np.nan,
            "PROMOTER_OVERLAP_BP": 0,
            "TERMINATOR_OVERLAP_BP": 0,
            "OVERLAPS_EXON": False,
            "OVERLAPS_CDS": False,
            "NEARBY_EXON": False,
            "NEARBY_CDS": False,
            "OVERLAPS_MRNA": False,
            "NEARBY_MRNA": False,
            "MRNA_OVERLAP_BP": 0
        }

    # Extract gene features
    gstart = int(primary_gene["gene_start"])
    gend = int(primary_gene["gene_end"])
    strand = primary_gene["strand"]
    tss = int(primary_gene["tss"])
    tes = int(primary_gene["tes"])
    exons = [(int(a), int(b)) for (a, b) in primary_gene.get("exons", [])]
    cds = [(int(a), int(b)) for (a, b) in primary_gene.get("cds", [])]
    mrnas = [(int(a), int(b)) for (a, b) in primary_gene.get("mrnas", [])]

    # Helper function to calculate overlap with interval lists
    def _ovl(ints):
        return max((_overlap_len(start, end, s, e) for s, e in ints), default=0) if ints else 0

    # Calculate overlaps
    gene_ovl = _overlap_len(start, end, gstart, gend)
    exon_ovl = _ovl(exons)
    cds_ovl = _ovl(cds)
    mrna_ovl = _ovl(mrnas)

    # Define promoter and terminator regions based on strand
    if strand == "-":
        promoter_start, promoter_end = tes, tes + promoter_bp
        terminator_start, terminator_end = tss - terminator_bp, tss
    else:
        promoter_start, promoter_end = tss - promoter_bp, tss
        terminator_start, terminator_end = tes, tes + terminator_bp

    # Calculate regulatory region overlaps
    prom_ovl = _overlap_len(start, end, promoter_start, promoter_end)
    term_ovl = _overlap_len(start, end, terminator_start, terminator_end)

    # Classify region hierarchically (FIXED: complete if-elif chain)
    if exon_ovl > 0 or cds_ovl > 0:
        region_class = "exonic"
    elif gene_ovl > 0:
        region_class = "intronic"
    elif prom_ovl > 0:
        region_class = "promoter"
    elif term_ovl > 0:
        region_class = "terminator"
    else:
        region_class = "intergenic"

    # Calculate distances to gene anchors
    five_anchor = tss if strand != "-" else tes
    three_anchor = tes if strand != "-" else tss

    def anchor_dist(a):
        """Calculate distance from region to anchor point."""
        if end < a:
            return a - end
        if start > a:
            return start - a
        return 0

    dist_5p = anchor_dist(five_anchor)
    dist_3p = anchor_dist(three_anchor)

    # Calculate minimum gap to feature intervals
    def min_gap(ints):
        """Calculate minimum gap between region and intervals."""
        if not ints:
            return np.inf
        gaps = []
        for s, e in ints:
            if end < s:
                gaps.append(s - end)
            elif start > e:
                gaps.append(start - e)
            else:
                gaps.append(0)
        return min(gaps) if gaps else np.inf

    # Determine "nearby" features (within threshold distance)
    near_thresh = max(int(promoter_bp), int(terminator_bp))
    near_exon = (exon_ovl == 0) and (min_gap(exons) <= near_thresh)
    near_cds = (cds_ovl == 0) and (min_gap(cds) <= near_thresh)
    near_mrna = (mrna_ovl == 0) and (min_gap(mrnas) <= near_thresh)

    # Determine side relative to gene
    if (end < five_anchor and strand != "-") or (start > five_anchor and strand == "-"):
        side = "5_prime"
    elif (start > five_anchor and strand != "-") or (end < five_anchor and strand == "-"):
        side = "3_prime"
    else:
        side = "within_gene"

    # Return complete classification dictionary
    return {
        "REGION_CLASS": region_class,
        "SIDE_RELATIVE_TO_PRIMARY": side,
        "DIST_TO_PRIMARY_5P": int(dist_5p),
        "DIST_TO_PRIMARY_3P": int(dist_3p),
        "PROMOTER_OVERLAP_BP": int(prom_ovl),
        "TERMINATOR_OVERLAP_BP": int(term_ovl),
        "OVERLAPS_EXON": bool(exon_ovl > 0),
        "OVERLAPS_CDS": bool(cds_ovl > 0),
        "NEARBY_EXON": bool(near_exon),
        "NEARBY_CDS": bool(near_cds),
        "OVERLAPS_MRNA": bool(mrna_ovl > 0),
        "NEARBY_MRNA": bool(near_mrna),
        "MRNA_OVERLAP_BP": int(mrna_ovl)
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Annotate variants with gene and repeat information")
    parser.add_argument("--input", required=True, help="Species-specific variants CSV")
    parser.add_argument("--gff", required=True, help="GFF/GFF3 annotation file")
    parser.add_argument("--repeatmasker-bed", default=None, help="RepeatMasker BED file (optional)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import pandas as pd
    df = pd.read_csv(args.input)
    gene_model = _build_gene_model_from_gff(args.gff, gene_name_field="Name", logger=logging.getLogger())
    annotated = df.apply(lambda row: _classify_region_for_row(row, gene_model, promoter_bp=2000, logger=logging.getLogger()), axis=1, result_type="expand")
    result = pd.concat([df, annotated], axis=1)
    if args.repeatmasker_bed:
        repeat_annot = _rm_annotate_intervals(df, args.repeatmasker_bed)
        result = pd.concat([result, repeat_annot], axis=1)
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result)} annotated variants to {args.output}")
