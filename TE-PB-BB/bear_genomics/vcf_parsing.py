"""
vcf_parsing.py
--------------
VCF file parsing utilities for bear genomics structural variant analysis.

Provides:
- FixedVCFParser: Enhanced VCF parser with comprehensive error handling,
  genotype (GT) extraction, and deletion-focused QC filtering.
- variants_to_dataframe: Convert parsed variant dicts to a tidy pandas DataFrame.
- process_vcf_files_sequential: Single-process fallback for batch VCF parsing.
- process_vcf_files_parallel: Multiprocessing-accelerated batch VCF parsing.
- process_single_vcf_file_standalone: Worker entry-point for parallel processing.
"""

import os
import logging
import multiprocessing
from typing import Dict, List, Optional
from collections import defaultdict

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class FixedVCFParser:
    """Enhanced VCF parser with comprehensive error handling."""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.parsing_statistics = {}
        self.qc_metrics = defaultdict(int)
        self.error_log = []
        self.sample_cache = {}
        self.parsing_stats = {'total_lines': 0, 'header_lines': 0, 'data_lines': 0, 'variants_parsed': 0, 'variants_passed_qc': 0, 'parsing_errors': 0, 'variants_extracted': 0}

    def _standardize_chrom(self, chrom_name):
        """Standardize chromosome names"""
        if isinstance(chrom_name, str):
            return chrom_name.replace('chr', '').replace('Chr', '').replace('CHR', '')
        return str(chrom_name)

    def get_parsing_statistics(self) -> Dict:
        """Get parsing statistics with success rate calculation - ONLY VERSION NEEDED"""
        stats = dict(self.parsing_stats)
        if stats.get('data_lines', 0) > 0:
            stats['success_rate'] = stats.get('variants_extracted', 0) / stats['data_lines']
        else:
            stats['success_rate'] = 0.0
        return stats

    def parse_vcf_file(self, file_path: str, enable_multi_allelic: bool = True) -> List[Dict]:
        """Parse VCF file and extract GT (Genotype) for Zygosity analysis."""
        if not os.path.exists(file_path):
            self.logger.error(f'VCF file not found: {file_path}')
            return []

        self.parsing_stats = {'total_lines': 0, 'header_lines': 0, 'data_lines': 0, 'variants_parsed': 0,
                              'variants_passed_qc': 0, 'parsing_errors': 0, 'variants_extracted': 0}
        variants = []
        file_name = os.path.basename(file_path)
        sample_name = os.path.splitext(file_name)[0]

        # Config defaults
        min_qual = getattr(self.config, 'min_qual', 20)
        min_size = getattr(self.config, 'min_size', 50)
        max_size = getattr(self.config, 'max_size', 100000)

        try:
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    self.parsing_stats['total_lines'] += 1
                    if line.startswith('#'):
                        self.parsing_stats['header_lines'] += 1
                        continue

                    self.parsing_stats['data_lines'] += 1
                    fields = line.strip().split('\t')

                    # === DEBUG: Print the first valid data line to log ===
                    if self.parsing_stats['data_lines'] == 1:
                        self.logger.info(f"DEBUG VCF LINE 1 cols: {len(fields)}")
                        if len(fields) >= 10:
                            self.logger.info(f"DEBUG FORMAT: {fields[8]}")
                            self.logger.info(f"DEBUG SAMPLE: {fields[9]}")

                    # Check minimum columns (8 for standard VCF, 10 for Genotype)
                    if len(fields) < 8:
                        continue

                    chrom, pos, _, ref, alt, qual, _, info = fields[:8]

                    # === 1. EXTRACT GENOTYPE (GT) ===
                    genotype = './.'  # Default

                    # Ensure we have Sample Data (Col 10)
                    if len(fields) >= 10:
                        fmt_field = fields[8].strip()
                        sample_values = fields[9].strip()

                        # Robust extraction
                        if 'GT' in fmt_field:
                            try:
                                fmt_parts = fmt_field.split(':')
                                val_parts = sample_values.split(':')

                                if 'GT' in fmt_parts:
                                    gt_idx = fmt_parts.index('GT')
                                    # Safety check: ensure sample has enough parts
                                    if gt_idx < len(val_parts):
                                        genotype = val_parts[gt_idx]
                            except Exception:
                                pass  # Keep default ./ .
                    # ================================

                    # === 2. FILTER FOR DELETIONS ===
                    try:
                        info_dict = self._parse_info_field_safe(info)
                        is_deletion = False
                        sv_type = info_dict.get('SVTYPE', '')

                        # Strict DEL check
                        if sv_type == 'DEL' or 'DEL' in str(sv_type):
                            is_deletion = True
                        elif len(ref) > len(alt) and len(ref) > 1:
                            is_deletion = True

                        if not is_deletion:
                            continue

                        self.parsing_stats['variants_parsed'] += 1
                        chrom = self._standardize_chrom(chrom)
                        start = int(pos)

                        # Pass genotype explicitly
                        if enable_multi_allelic and ',' in alt:
                            alt_alleles = alt.split(',')
                            for alt_idx, alt_allele in enumerate(alt_alleles):
                                variant = self._process_single_allele(
                                    chrom, start, ref, alt_allele, qual, info_dict,
                                    file_name, sample_name, alt_idx,
                                    min_qual, min_size, max_size,
                                    genotype=genotype
                                )
                                if variant:
                                    variants.append(variant)
                                    self.parsing_stats['variants_passed_qc'] += 1
                        else:
                            variant = self._process_single_allele(
                                chrom, start, ref, alt, qual, info_dict,
                                file_name, sample_name, 0,
                                min_qual, min_size, max_size,
                                genotype=genotype
                            )
                            if variant:
                                variants.append(variant)
                                self.parsing_stats['variants_passed_qc'] += 1

                    except Exception as e:
                        self.logger.debug(f'Error parsing variant details at line {line_num}: {e}')
                        continue

        except Exception as e:
            self.logger.error(f'Error parsing {file_path}: {e}')
            return []

        self.parsing_stats['variants_extracted'] = len(variants)
        self.logger.info(f'Parsed {len(variants)} deletions from {file_name}')
        return variants

    def _process_single_allele(self, chrom: str, start: int, ref: str, alt: str, qual: str, info_dict: Dict,
                               file_name: str, sample_name: str, alt_idx: int, min_qual: float, min_size: int,
                               max_size: int, genotype: str = './.') -> Optional[Dict]:
        """Process a single allele from potentially multi-allelic site"""
        try:
            end = None
            if 'END' in info_dict:
                try:
                    end = int(info_dict['END'])
                except (ValueError, TypeError):
                    pass
            if end is None and 'SVLEN' in info_dict:
                try:
                    svlen = int(info_dict['SVLEN'])
                    end = start + abs(svlen)
                except (ValueError, TypeError):
                    pass
            if end is None:
                if len(ref) > len(alt):
                    end = start + len(ref)
                else:
                    end = start + min_size
            if end <= start:
                if 'SVLEN' in info_dict:
                    try:
                        svlen = abs(int(info_dict['SVLEN']))
                        if svlen > 0:
                            end = start + svlen
                        else:
                            return None
                    except (ValueError, TypeError):
                        return None
                else:
                    end = start + 1
            if start >= end:
                return None
            size = end - start
            if size < min_size or size > max_size:
                return None
            try:
                if qual.lower() in ['inf', 'infinity', '.', '999']:
                    quality = float('inf')
                else:
                    quality = float(qual)
                    if quality < min_qual:
                        return None
            except (ValueError, AttributeError):
                quality = 0.0
                if quality < min_qual:
                    return None

            # Construct variant dictionary with Genotype
            variant = {
                'chrom': chrom,
                'start': start,
                'end': end,
                'size': size,
                'quality': quality,
                'sample_name': sample_name,
                'file_name': file_name,
                'svtype': 'DEL',
                'svmethod': 'VCF_PARSED',
                'alt_index': alt_idx,
                'support_reads': 1,
                'total_reads': 1,
                'precision_score': 0.5,
                'qc_flags': [],
                'info_dict': info_dict,
                'genotype': genotype
            }
            return variant
        except Exception as e:
            self.logger.debug(f'Error processing allele {alt_idx} at {chrom}:{start}: {e}')
            return None

    def _parse_vcf_line_safe(self, line: str, file_name: str, sample_name: str, line_number: int,
                             enable_multi_allelic: bool, file_stats: Dict) -> List[Dict]:
        """Parse VCF line with safe error handling AND genotype extraction."""
        fields = line.split('\t')
        if len(fields) < 8:
            file_stats['parsing_errors'] += 1
            return []

        chrom, pos, var_id, ref, alt, qual, filt, info = fields[:8]

        # === NEW: Extract Genotype ===
        genotype = './.'
        if len(fields) > 9:
            fmt_field = fields[8]
            sample_field = fields[9]
            if 'GT' in fmt_field:
                try:
                    fmt_parts = fmt_field.split(':')
                    sample_parts = sample_field.split(':')
                    gt_idx = fmt_parts.index('GT')
                    if gt_idx < len(sample_parts):
                        genotype = sample_parts[gt_idx]
                except ValueError:
                    pass
        # =============================

        try:
            info_dict = self._parse_info_field_safe(info)
        except Exception:
            file_stats['parsing_errors'] += 1
            return []

        if info_dict.get('SVTYPE') != 'DEL':
            return []

        file_stats['variants_parsed'] += 1

        if enable_multi_allelic and ',' in alt:
            # Note: You might need to update _handle_multiallelic_variant_safe to accept genotype too
            # For now, we inject it into the single variant processor
            variants = self._handle_multiallelic_variant_safe(chrom, pos, ref, alt, qual, info_dict, file_name,
                                                              sample_name, line_number)
            # Patch genotypes if returned
            for v in variants:
                if v: v['genotype'] = genotype
        else:
            # Pass genotype to the safe processor
            variant = self._process_single_variant_safe(chrom, pos, ref, alt, qual, info_dict, file_name, sample_name,
                                                        0, genotype=genotype)
            variants = [variant] if variant else []

        valid_variants = [v for v in variants if v is not None]
        file_stats['variants_passed_qc'] += len(valid_variants)
        return valid_variants

    def _parse_info_field_safe(self, info: str) -> Dict[str, str]:
        """Safely parse VCF INFO field."""
        info_dict = {}
        if not info or info == '.':
            return info_dict
        try:
            for item in info.split(';'):
                if not item:
                    continue
                if '=' in item:
                    key, value = item.split('=', 1)
                    info_dict[key] = value
                else:
                    info_dict[item] = True
        except Exception as e:
            self.logger.debug(f"Error parsing INFO field '{info}': {e}")
        return info_dict

    def _process_single_variant_safe(self, chrom: str, pos: str, ref: str, alt: str, qual: str, info_dict: Dict,
                                     file_name: str, sample_name: str, alt_index: int, genotype: str = './.') -> \
    Optional[Dict]:
        """Process a single variant with safe error handling and Genotype support."""
        try:
            start = int(pos)
            if 'END' in info_dict:
                end = int(info_dict['END'])
            elif 'SVLEN' in info_dict:
                svlen = int(info_dict['SVLEN'])
                end = start + abs(svlen)
            elif ref and alt and (alt != '<DEL>'):
                end = start + max(len(ref), len(alt))
            else:
                end = start + getattr(self.config, 'min_size', 50)

            if start >= end:
                return None

            size = end - start
            if size < getattr(self.config, 'min_size', 50) or size > getattr(self.config, 'max_size', 1000000):
                return None

            try:
                if qual.lower() in ['inf', 'infinity', '.']:
                    quality = float('inf')
                else:
                    quality = float(qual)
            except (ValueError, AttributeError):
                quality = 0.0

            min_qual = getattr(self.config, 'min_qual', 10.0)
            if quality != float('inf') and quality < min_qual:
                return None

            variant = {
                'chrom': chrom,
                'start': start,
                'end': end,
                'size': size,
                'quality': quality,
                'sample_name': sample_name,
                'file_name': file_name,
                'svtype': info_dict.get('SVTYPE', 'DEL'),
                'svmethod': 'VCF_PARSED',
                'alt_index': alt_index,
                'support_reads': 1,
                'total_reads': 1,
                'qc_flags': [],
                'precision_score': 0.5,
                'genotype': genotype  # <--- ADDED FIELD
            }
            return variant
        except Exception as e:
            self.logger.debug(f'Error processing variant at {chrom}:{pos}: {e}')
            return None

    def _handle_multiallelic_variant_safe(self, chrom, pos, ref, alt, qual, info_dict, file_name, sample_name,
                                          line_number, genotype='./.'):
        """
        Handle multi-allelic variants safely.
        FIXED: Now propagates genotype to split variants.
        """
        variants = []
        alt_alleles = alt.split(',')
        for i, alt_allele in enumerate(alt_alleles):
            variant = self._process_single_variant_safe(
                chrom, pos, ref, alt_allele, qual, info_dict,
                file_name, sample_name, i, genotype=genotype
            )
            if variant:
                variants.append(variant)
        return variants

    def _calculate_variant_size(self, info_dict, pos, ref, alt):
        """Calculate variant size from various sources"""
        if 'SVLEN' in info_dict:
            return abs(int(info_dict['SVLEN']))
        elif 'END' in info_dict:
            return int(info_dict['END']) - int(pos)
        else:
            return abs(len(ref) - len(alt))

    def _log_processing_summary_fixed(self, file_path: str, variants: List[Dict], file_stats: Dict):
        """Log comprehensive processing summary with meaningful success rates."""
        file_name = os.path.basename(file_path)
        required_keys = ['lines_processed', 'header_lines', 'data_lines', 'variants_parsed', 'variants_passed_qc', 'parsing_errors']
        for key in required_keys:
            if key not in file_stats:
                file_stats[key] = 0
        data_lines = file_stats['data_lines']
        variants_extracted = len(variants)
        if data_lines > 0:
            parsing_rate = file_stats['variants_parsed'] / data_lines
            overall_rate = variants_extracted / data_lines
        else:
            parsing_rate = 0.0
            overall_rate = 0.0
        if file_stats['variants_parsed'] > 0:
            qc_rate = file_stats['variants_passed_qc'] / file_stats['variants_parsed']
        else:
            qc_rate = 0.0
        self.logger.info(f'Completed processing {file_path}: {variants_extracted} variants extracted')
        self.logger.info(f"File {file_name}: parsing rate {parsing_rate:.1%} ({file_stats['variants_parsed']}/{data_lines} data lines)")
        self.logger.info(f"File {file_name}: QC pass rate {qc_rate:.1%} ({file_stats['variants_passed_qc']}/{file_stats['variants_parsed']} parsed variants)")
        self.logger.info(f'File {file_name}: overall success rate {overall_rate:.1%} ({variants_extracted}/{data_lines} final variants)')
        if file_stats['parsing_errors'] > 0:
            self.logger.debug(f"  Parsing errors: {file_stats['parsing_errors']}")

    def _update_global_stats(self, file_stats: Dict):
        """Update global parsing statistics."""
        for key, value in file_stats.items():
            if key in self.parsing_stats:
                self.parsing_stats[key] += value
        self.parsing_stats['total_files_processed'] += 1


def process_vcf_files_sequential(vcf_files, config):
    """
    Sequential fallback used when multiprocessing is unavailable.
    Mirrors the behavior of process_vcf_files_parallel: returns a flat list of variant dicts.
    """
    all_variants = []
    for vcf in vcf_files:
        try:
            variants = process_single_vcf_file_standalone(vcf, config)
            # process_single_vcf_file_standalone() returns a list of variant dicts
            all_variants.extend(variants)
        except Exception as e:
            logger.error(f"Failed to parse {vcf}: {e}")
    return all_variants


def process_vcf_files_parallel(vcf_files, config, max_workers=4):

    try:
        with multiprocessing.Pool(processes=max_workers) as pool:
            worker_args = [(vcf_file, config) for vcf_file in vcf_files]
            results = pool.starmap(process_single_vcf_file_standalone, worker_args)
        all_variants = []
        for file_variants in results:
            all_variants.extend(file_variants)
        return all_variants
    except Exception as e:
        logger.error(f'Parallel processing failed: {e}')
        return process_vcf_files_sequential(vcf_files, config)


def process_single_vcf_file_standalone(vcf_file, config):
    """
    FIXED: Standalone VCF processor for multiprocessing
    """
    parser = FixedVCFParser(config)
    flag = getattr(config, 'enable_multiallelic_parsing', True)
    return parser.parse_vcf_file(vcf_file, enable_multi_allelic=flag)


def variants_to_dataframe(variants: List[Dict]) -> pd.DataFrame:
    """Convert variants to pandas DataFrame with all metadata preserved."""
    if not variants:
        return pd.DataFrame()
    core_data = []
    for variant in variants:
        core_data.append({
            'CHROM': variant.get('chrom', variant.get('CHROM', 'unknown')),
            'START': variant.get('start', variant.get('START', 0)),
            'END': variant.get('end', variant.get('END', 0)),
            'SIZE': variant.get('size', variant.get('SIZE', 0)),
            'QUAL': variant.get('quality', variant.get('QUAL', 20.0)),
            'FILE': variant.get('file_name', variant.get('FILE', 'unknown')),
            'SAMPLE': variant.get('sample_name', variant.get('SAMPLE', 'unknown')),
            'PRECISION_SCORE': variant.get('precision_score', 0.5),
            'SUPPORT_READS': variant.get('support_reads', 1),
            'TOTAL_READS': variant.get('total_reads', 1),
            'SVTYPE': variant.get('svtype', variant.get('SVTYPE', 'DEL')),
            'SVMETHOD': variant.get('svmethod', 'VCF_PARSED'),
            'QC_FLAGS': ';'.join(variant.get('qc_flags', [])),
            # === CRITICAL FIX: ADD THIS LINE ===
            'genotype': variant.get('genotype', './.')
        })

    df = pd.DataFrame(core_data)
    # Avoid division by zero errors
    total_reads = df['TOTAL_READS'].replace(0, 1)
    df['SUPPORT_FRACTION'] = df['SUPPORT_READS'] / total_reads

    # Safe Log10 calculation
    sizes = df['SIZE'].replace(0, 1)
    df['SIZE_LOG10'] = np.log10(sizes)

    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Parse VCF files to CSV")
    parser.add_argument("--vcf-folder", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from bear_genomics.config import load_fixed_config
    config = load_fixed_config(args.config)
    results = process_vcf_files_parallel([args.vcf_folder], config)
    df = variants_to_dataframe(results)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} variants to {args.output}")
