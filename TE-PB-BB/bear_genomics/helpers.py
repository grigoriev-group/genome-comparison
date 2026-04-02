"""
bear_genomics.helpers
=====================
Low-level utility functions used throughout the bear genomics pipeline.
These helpers have no dependencies on other bear_genomics submodules and
may be imported freely by any part of the pipeline.

Contains:
  - fix_reference_genome_path()   — repair hard-coded test reference paths
  - _select_iv_cols()             — pick the right coordinate columns from a DataFrame
  - _swap_bad_intervals_inplace() — fix inverted start/end coordinates in-place
  - _safe_int()                   — coerce a value to int with a fallback default
  - _overlap_len()                — compute the overlap length of two intervals
  - _looks_like_rm_signature()    — detect RepeatMasker class/family strings
  - _is_legacy_clip_signature()   — detect legacy soft-clip annotation strings
"""

import os
import logging
import pandas as pd


def fix_reference_genome_path(config, logger=None):
    """
    Fix hardcoded test reference genome paths
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if hasattr(config, 'reference_genome_path'):
        ref_path = str(config.reference_genome_path)

        # Check for problematic test paths
        if '/tmp/test_reference.fasta' in ref_path or 'test_reference' in ref_path:
            logger.warning("⚠️ Detected test reference path - attempting to fix...")

            # Try to find a real reference genome
            possible_refs = []

            if hasattr(config, 'polar_reference_genome') and config.polar_reference_genome:
                if os.path.exists(config.polar_reference_genome):
                    possible_refs.append(('polar', config.polar_reference_genome))

            if hasattr(config, 'brown_reference_genome') and config.brown_reference_genome:
                if os.path.exists(config.brown_reference_genome):
                    possible_refs.append(('brown', config.brown_reference_genome))

            if possible_refs:
                # Use the first available reference
                ref_type, ref_path = possible_refs[0]
                config.reference_genome_path = ref_path
                logger.info(f"✅ Fixed reference path to {ref_type}: {ref_path}")
                return True
            else:
                logger.warning("⚠️ No valid reference genome found - disabling assembly validation")
                config.reference_genome_path = None
                if hasattr(config, 'enable_assembly_validation'):
                    config.enable_assembly_validation = False
                return False

    return True

def _select_iv_cols(df: pd.DataFrame):
    """
    FIXED: Selects coordinate columns with robust fallbacks for all
    dataframe types (refined, clustered, overlapping).
    """
    chrom_col = "CHROM"

    # Priority: START/END > Average > Min/Max
    if "START" in df.columns:
        start_col = "START"
    elif "AVG_START" in df.columns:
        start_col = "AVG_START"
    elif "MIN_START" in df.columns:
        start_col = "MIN_START"
    else:
        start_col = "start"  # Default fallback

    if "END" in df.columns:
        end_col = "END"
    elif "AVG_END" in df.columns:
        end_col = "AVG_END"
    elif "MAX_END" in df.columns:
        end_col = "MAX_END"
    else:
        end_col = "end"  # Default fallback

    return chrom_col, start_col, end_col

def _swap_bad_intervals_inplace(df: pd.DataFrame, start_col: str, end_col: str) -> int:
    bad = df[end_col] < df[start_col]
    n_bad = int(bad.sum())
    if n_bad:
        # robust vectorized swap
        s_tmp = df.loc[bad, start_col].values
        df.loc[bad, start_col] = df.loc[bad, end_col].values
        df.loc[bad, end_col]   = s_tmp
    return n_bad

def _safe_int(x, default=0):
    try: return int(x)
    except Exception: return default

def _overlap_len(a_start, a_end, b_start, b_end):
    return max(0, min(a_end, b_end) - max(a_start, b_start))

def _looks_like_rm_signature(s: str) -> bool:
    if not isinstance(s, str) or not s: return False
    if '/' in s: return True
    for hint in ["LINE","SINE","LTR","DNA","Simple_repeat","Low_complexity",
                 "Satellite","rRNA","tRNA","snRNA","scRNA","srpRNA","Unknown"]:
        if hint.lower() in s.lower(): return True
    return False

def _is_legacy_clip_signature(s: str) -> bool:
    if not isinstance(s, str) or not s: return False
    s_low = s.lower()
    return any(tok in s_low for tok in ["clip","mh=","sc=","ins="])
