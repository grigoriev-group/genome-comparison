"""Cross-species structural variant comparison engine for brown vs polar bear analysis."""

import os
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd

from pathlib import Path
from multiprocessing import Pool, cpu_count


@dataclass
class ComparisonResults:
    """Results container for species comparison analysis."""
    overlapping_deletions: pd.DataFrame = field(default_factory=pd.DataFrame)
    brown_specific_deletions: pd.DataFrame = field(default_factory=pd.DataFrame)
    polar_specific_deletions: pd.DataFrame = field(default_factory=pd.DataFrame)
    comparison_stats: Dict[str, Any] = field(default_factory=dict)
    brown_full: pd.DataFrame = field(default_factory=pd.DataFrame)
    polar_full: pd.DataFrame = field(default_factory=pd.DataFrame)

    def as_dict(self) -> Dict[str, pd.DataFrame]:
        return {
            "overlapping":    self.overlapping_deletions,
            "brown_specific": self.brown_specific_deletions,
            "polar_specific": self.polar_specific_deletions,
        }


    def get(self, key: str, default=None):
        return self.as_dict().get(key, default)


class SpeciesComparisonEngine:
    """
    Cross-species deletion comparison engine.

    Compares clustered deletions between brown bears and polar bears to identify
    species-specific variants and common deletions using overlap-based analysis.
    """

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.overlap_threshold = getattr(config, 'overlap_threshold', 0.3)
        self.threads = getattr(config, 'threads', cpu_count())
        self.process_brown_specific = getattr(config, 'process_brown_specific', True)
        self.process_polar_specific = getattr(config, 'process_polar_specific', True)
        self.process_overlapping = getattr(config, 'process_overlapping', False)
        if hasattr(config, 'focus_species') and config.focus_species is not None:
            focus = config.focus_species.lower()
            if focus == 'brown':
                self.process_brown_specific = True
                self.process_polar_specific = False
                self.process_overlapping = False
            elif focus == 'polar':
                self.process_brown_specific = False
                self.process_polar_specific = True
                self.process_overlapping = False
            elif focus == 'common':
                self.process_brown_specific = False
                self.process_polar_specific = False
                self.process_overlapping = True
            elif focus == 'all':
                self.process_brown_specific = True
                self.process_polar_specific = True
                self.process_overlapping = True
        self.logger.info(f'Species comparison configured: brown={self.process_brown_specific}, polar={self.process_polar_specific}, overlapping={self.process_overlapping}')

    def compare_species_deletions(self, datasets: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], ComparisonResults]:
        """
        Compare brown and polar bear deletions to identify species-specific variants.

        Args:
            datasets: Dictionary containing clustered deletion dataframes
                     Expected keys: 'brown', 'polar', 'panda_brown', 'panda_polar'

        Returns:
            Tuple of (filtered_datasets, comparison_results)
            - filtered_datasets: Only datasets selected for downstream processing
            - comparison_results: Complete comparison results for analysis
        """
        self.logger.info('Starting cross-species deletion comparison')
        brown_df = datasets.get('brown', pd.DataFrame())
        polar_df = datasets.get('polar', pd.DataFrame())
        if brown_df.empty and polar_df.empty:
            self.logger.warning('No brown or polar deletions available for comparison')
            return (datasets, ComparisonResults(overlapping_deletions=pd.DataFrame(), brown_specific_deletions=pd.DataFrame(), polar_specific_deletions=pd.DataFrame(), comparison_stats={'error': 'no_data'}))
        comparison_results = self._perform_overlap_analysis(brown_df, polar_df)
        filtered_datasets = self._filter_datasets_for_downstream_processing(datasets, comparison_results)
        self._log_comparison_summary(comparison_results)
        self._save_comparison_artifacts(comparison_results)
        return (filtered_datasets, comparison_results)

    def _save_comparison_artifacts(self, results: ComparisonResults):
        """Saves detailed species comparison files with debug probing."""

        # 1. Robust output folder retrieval
        if hasattr(self.config, 'output_folder'):
            base_out = self.config.output_folder
        elif isinstance(self.config, dict):
            base_out = self.config.get('output_folder', '.')
        else:
            base_out = '.'

        # 2. Create Directory (Force Absolute Path)
        out_dir = Path(base_out).resolve() / "species_comparison"
        out_dir.mkdir(parents=True, exist_ok=True)

        # 3. Map DataFrames
        outputs = {
            "brown_specific":        results.brown_specific_deletions,
            "polar_specific":        results.polar_specific_deletions,
            "overlapping_deletions": results.overlapping_deletions,
        }

        # 4. Save
        saved_count = 0
        for name, df in outputs.items():
            if df is not None and not df.empty:
                df.to_csv(out_dir / f"{name}.csv", index=False)

                if all(col in df.columns for col in ['CHROM', 'START', 'END']):
                    bed_df = df[['CHROM', 'START', 'END']].rename(
                        columns={'CHROM': 'chrom', 'START': 'start', 'END': 'end'}
                    )
                    bed_df.to_csv(out_dir / f"{name}.bed", sep='\t', header=False, index=False)

                saved_count += 1

        self.logger.info(f"✅ Saved {saved_count} species comparison files.")

    def _perform_overlap_analysis(self, brown_df: pd.DataFrame, polar_df: pd.DataFrame) -> ComparisonResults:
        self.logger.info("Starting overlap analysis (using _find_overlaps_parallel)...")

        # 1. Run existing spatial overlap (Uses your existing parallel logic)
        # This avoids using 'intervaltree' directly
        overlapping, brown_specific, polar_specific = self._find_overlaps_parallel(
            brown_df, polar_df, self.overlap_threshold, self.threads
        )

        self.logger.info(
            f"Overlap analysis complete: {len(overlapping)} overlapping, "
            f"{len(brown_specific)} brown-specific, {len(polar_specific)} polar-specific"
        )

        return ComparisonResults(
            overlapping_deletions=overlapping,
            brown_specific_deletions=brown_specific,
            polar_specific_deletions=polar_specific,
            comparison_stats={'overlapping': len(overlapping)}
        )

    def _find_overlaps_parallel(self, brown_df: pd.DataFrame, polar_df: pd.DataFrame, threshold: float, threads: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Parallelized overlap detection between brown and polar deletions.
        Adapted from run_bear_analysis_05242025.py find_overlaps function.
        """
        optimal_threads = min(threads, len(brown_df), cpu_count())
        polar_records = list(polar_df.to_dict('records'))
        brown_records = list(brown_df.to_dict('records'))
        with Pool(processes=optimal_threads, initializer=self._init_worker, initargs=(polar_records,)) as pool:
            results = pool.map(self._find_best_match_for_brown, brown_records)
        overlapping_pairs = []
        brown_specific_records = []
        used_polar_indices = set()
        for brown_record, polar_index, overlap_score in results:
            if polar_index is not None and overlap_score >= threshold:
                overlapping_pairs.append((brown_record, polar_records[polar_index], overlap_score))
                used_polar_indices.add(polar_index)
            else:
                brown_specific_records.append(brown_record)
        overlapping_rows = []
        for brown_rec, polar_rec, overlap_score in overlapping_pairs:
            overlapping_rows.append(self._create_overlapping_record(brown_rec, polar_rec, overlap_score))
        polar_specific_indices = set(range(len(polar_records))) - used_polar_indices
        polar_specific_records = [polar_records[i] for i in polar_specific_indices]
        overlapping_df = pd.DataFrame(overlapping_rows)
        brown_specific_df = pd.DataFrame(brown_specific_records)
        polar_specific_df = pd.DataFrame(polar_specific_records)
        self.logger.info(f'Overlap analysis: {len(overlapping_df)} overlapping, {len(brown_specific_df)} brown-specific, {len(polar_specific_df)} polar-specific')
        return (overlapping_df, brown_specific_df, polar_specific_df)

    def _init_worker(self, polar_records: List[Dict]):
        """Initialize worker process with polar records for parallel processing."""
        global _worker_polar_records
        _worker_polar_records = {i: rec for i, rec in enumerate(polar_records)}

    def _find_best_match_for_brown(self, brown_record: Dict) -> Tuple[Dict, Optional[int], float]:
        """Find the best matching polar deletion for a single brown deletion."""
        global _worker_polar_records
        best_score = 0.0
        best_index = None
        for polar_index, polar_record in _worker_polar_records.items():
            if polar_record.get('CHROM') != brown_record.get('CHROM'):
                continue
            overlap_score = self._calculate_overlap_score(brown_record, polar_record)
            if overlap_score > best_score:
                best_score = overlap_score
                best_index = polar_index
        return (brown_record, best_index, best_score)

    def _get_coordinate(self, record: Dict, possible_keys: List[str]) -> Optional[int]:
        """Extract coordinate value trying multiple possible column names."""
        for key in possible_keys:
            if key in record and record[key] is not None:
                try:
                    return int(record[key])
                except (ValueError, TypeError):
                    continue
        return None

    def _get_coordinate_with_fallback(self, record: Dict, possible_keys: List[str]) -> Optional[int]:
        """Extract coordinate value prioritizing AVG coordinates over MIN/MAX."""
        # Prioritize AVG coordinates for breakpoint analysis
        avg_keys = [key for key in possible_keys if 'AVG' in key]
        other_keys = [key for key in possible_keys if 'AVG' not in key]

        # Try AVG coordinates first
        for key in avg_keys + other_keys:
            if key in record and record[key] is not None:
                try:
                    return int(record[key])
                except (ValueError, TypeError):
                    continue
        return None

    def _calculate_overlap_score(self, record1: Dict, record2: Dict) -> float:
        """Calculate overlap score between two deletion records."""
        start1 = self._get_coordinate(record1, ['MIN_START', 'START', 'POS'])
        end1 = self._get_coordinate(record1, ['MAX_END', 'END', 'POS'])
        start2 = self._get_coordinate(record2, ['MIN_START', 'START', 'POS'])
        end2 = self._get_coordinate(record2, ['MAX_END', 'END', 'POS'])
        if None in [start1, end1, start2, end2]:
            return 0.0
        if end1 < start1:
            end1 = start1 + 1
        if end2 < start2:
            end2 = start2 + 1
        overlap = max(0, min(end1, end2) - max(start1, start2))
        union = max(end1 - start1, end2 - start2)
        if union > 0:
            return overlap / union
        else:
            return 0.0

    def _create_overlapping_record(self, brown_rec: Dict, polar_rec: Dict, overlap_score: float) -> Dict:
        """Create a combined record for overlapping deletions with standardized columns."""

        # Use AVG coordinates as primary, fallback to MIN/MAX
        brown_avg_start = brown_rec.get("AVG_START", brown_rec.get("MIN_START"))
        brown_avg_end = brown_rec.get("AVG_END", brown_rec.get("MAX_END"))
        polar_avg_start = polar_rec.get("AVG_START", polar_rec.get("MIN_START"))
        polar_avg_end = polar_rec.get("AVG_END", polar_rec.get("MAX_END"))

        return {
            "CHROM": brown_rec.get("CHROM"),

            # Standardized coordinate columns (for breakpoint analysis)
            "AVG_START": brown_avg_start,  # PRIMARY coordinates
            "AVG_END": brown_avg_end,
            "MIN_START": min(brown_rec.get("MIN_START", brown_avg_start),
                             polar_rec.get("MIN_START", polar_avg_start)),
            "MAX_END": max(brown_rec.get("MAX_END", brown_avg_end),
                           polar_rec.get("MAX_END", polar_avg_end)),

            # Brown bear specific data
            "BROWN_MIN_START": brown_rec.get("MIN_START"),
            "BROWN_MAX_END": brown_rec.get("MAX_END"),
            "BROWN_AVG_START": brown_avg_start,
            "BROWN_AVG_END": brown_avg_end,
            "BROWN_AVG_SIZE": brown_rec.get("AVG_SIZE", brown_avg_end - brown_avg_start),
            "BROWN_AVG_QUAL": brown_rec.get("AVG_QUAL", 0),
            "BROWN_SAMPLES_COUNT": brown_rec.get("SAMPLES_COUNT", 0),
            "BROWN_PRECISION": brown_rec.get("PRECISION_SCORE", 0),

            # Polar bear specific data
            "POLAR_MIN_START": polar_rec.get("MIN_START"),
            "POLAR_MAX_END": polar_rec.get("MAX_END"),
            "POLAR_AVG_START": polar_avg_start,
            "POLAR_AVG_END": polar_avg_end,
            "POLAR_AVG_SIZE": polar_rec.get("AVG_SIZE", polar_avg_end - polar_avg_start),
            "POLAR_AVG_QUAL": polar_rec.get("AVG_QUAL", 0),
            "POLAR_SAMPLES_COUNT": polar_rec.get("SAMPLES_COUNT", 0),
            "POLAR_PRECISION": polar_rec.get("PRECISION_SCORE", 0),

            # Comparison metrics
            "OVERLAP_RATIO": overlap_score,
            "SIZE_RATIO": self._calculate_size_ratio(brown_rec, polar_rec),
            "FLAG": "OVERLAPPING"
        }

    def _calculate_size_ratio(self, rec1: Dict, rec2: Dict) -> float:
        """Calculate size ratio between two deletions."""
        size1 = rec1.get('AVG_SIZE', rec1.get('MAX_END', 0) - rec1.get('MIN_START', 0))
        size2 = rec2.get('AVG_SIZE', rec2.get('MAX_END', 0) - rec2.get('MIN_START', 0))
        if size1 > 0 and size2 > 0:
            return min(size1, size2) / max(size1, size2)
        else:
            return 0.0

    def _filter_datasets_for_downstream_processing(self, original_datasets: Dict[str, pd.DataFrame], comparison_results: ComparisonResults) -> Dict[str, pd.DataFrame]:
        """Filter datasets based on processing configuration."""
        filtered_datasets = {}
        process_brown = getattr(self.config, 'process_brown_specific', True)
        process_polar = getattr(self.config, 'process_polar_specific', False)
        process_overlapping = getattr(self.config, 'process_overlapping', False)

        # Normalize: attribute-first, then dict fallback
        brown_df = getattr(comparison_results, 'brown_specific_deletions', None)
        if brown_df is None and isinstance(comparison_results, dict):
            brown_df = comparison_results.get('brown_specific')

        polar_df = getattr(comparison_results, 'polar_specific_deletions', None)
        if polar_df is None and isinstance(comparison_results, dict):
            polar_df = comparison_results.get('polar_specific')

        overlap_df = getattr(comparison_results, 'overlapping_deletions', None)
        if overlap_df is None and isinstance(comparison_results, dict):
            overlap_df = comparison_results.get('overlapping')

        if process_brown and isinstance(brown_df, pd.DataFrame) and not brown_df.empty:
            filtered_datasets['brown'] = brown_df
            self.logger.info("Including %d brown-specific deletions", len(brown_df))

        if process_polar and isinstance(polar_df, pd.DataFrame) and not polar_df.empty:
            filtered_datasets['polar'] = polar_df
            self.logger.info("Including %d polar-specific deletions", len(polar_df))

        if process_overlapping and isinstance(overlap_df, pd.DataFrame) and not overlap_df.empty:
            filtered_datasets['overlapping'] = overlap_df
            self.logger.info("Including %d overlapping deletions", len(overlap_df))

        return filtered_datasets

    def _calculate_comparison_statistics(self, brown_df: pd.DataFrame, polar_df: pd.DataFrame, overlapping_df: pd.DataFrame, brown_specific_df: pd.DataFrame, polar_specific_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive comparison statistics."""
        total_brown = len(brown_df)
        total_polar = len(polar_df)
        total_overlapping = len(overlapping_df)
        total_brown_specific = len(brown_specific_df)
        total_polar_specific = len(polar_specific_df)
        stats = {'input_counts': {'brown_deletions': total_brown, 'polar_deletions': total_polar}, 'output_counts': {'overlapping_deletions': total_overlapping, 'brown_specific_deletions': total_brown_specific, 'polar_specific_deletions': total_polar_specific}, 'percentages': {'brown_specific_percent': total_brown_specific / total_brown * 100 if total_brown > 0 else 0, 'polar_specific_percent': total_polar_specific / total_polar * 100 if total_polar > 0 else 0, 'brown_overlapping_percent': total_overlapping / total_brown * 100 if total_brown > 0 else 0, 'polar_overlapping_percent': total_overlapping / total_polar * 100 if total_polar > 0 else 0}, 'configuration': {'overlap_threshold': self.overlap_threshold, 'process_brown_specific': self.process_brown_specific, 'process_polar_specific': self.process_polar_specific, 'process_overlapping': self.process_overlapping}}
        return stats

    def _get_required_columns(self) -> List[str]:
        """Get list of required columns for comparison analysis."""
        return ['CHROM', 'MIN_START', 'MAX_END']

    def _validate_dataframe_columns(self, df: pd.DataFrame, required_cols: List[str], name: str) -> bool:
        """Validate that dataframe has required columns."""
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            self.logger.warning(f'{name} dataframe missing columns: {missing_cols}')
            return False
        return True

    def _standardize_dataframe_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attempt to standardize dataframe column names for compatibility."""
        df_copy = df.copy()
        column_mapping = {'POS': 'MIN_START', 'START': 'MIN_START', 'END': 'MAX_END', 'CHROMOSOME': 'CHROM', 'CHR': 'CHROM'}
        for old_col, new_col in column_mapping.items():
            if old_col in df_copy.columns and new_col not in df_copy.columns:
                df_copy[new_col] = df_copy[old_col]
        if 'MAX_END' not in df_copy.columns and 'MIN_START' in df_copy.columns:
            if 'SIZE' in df_copy.columns:
                df_copy['MAX_END'] = df_copy['MIN_START'] + df_copy['SIZE']
            elif 'LENGTH' in df_copy.columns:
                df_copy['MAX_END'] = df_copy['MIN_START'] + df_copy['LENGTH']
            else:
                df_copy['MAX_END'] = df_copy['MIN_START'] + 1
        return df_copy

    def _log_comparison_summary(self, comparison_results: ComparisonResults):
        """Log summary of comparison results."""
        stats = comparison_results.comparison_stats
        self.logger.info('=== Species Comparison Summary ===')
        if 'input_counts' in stats:
            input_counts = stats['input_counts']
            self.logger.info(f"Input: {input_counts['brown_deletions']} brown, {input_counts['polar_deletions']} polar deletions")
        if 'output_counts' in stats:
            output_counts = stats['output_counts']
            self.logger.info(f"Results: {output_counts['overlapping_deletions']} overlapping, {output_counts['brown_specific_deletions']} brown-specific, {output_counts['polar_specific_deletions']} polar-specific")
        if 'percentages' in stats:
            percentages = stats['percentages']
            self.logger.info(f"Brown species-specificity: {percentages['brown_specific_percent']:.1f}%")
            self.logger.info(f"Polar species-specificity: {percentages['polar_specific_percent']:.1f}%")
        self.logger.info('=== End Species Comparison Summary ===')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare brown vs polar bear structural variants")
    parser.add_argument("--brown", required=True, help="Brown bear clustered variants CSV")
    parser.add_argument("--polar", required=True, help="Polar bear clustered variants CSV")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    import os
    from bear_genomics.config import load_fixed_config
    config = load_fixed_config(args.config)
    brown_df = pd.read_csv(args.brown)
    polar_df = pd.read_csv(args.polar)
    engine = SpeciesComparisonEngine(config)
    results = engine.compare_species_deletions(brown_df, polar_df)
    os.makedirs(args.output_dir, exist_ok=True)
    results.brown_specific_deletions.to_csv(os.path.join(args.output_dir, "brown_specific.csv"), index=False)
    results.polar_specific_deletions.to_csv(os.path.join(args.output_dir, "polar_specific.csv"), index=False)
    results.overlapping_deletions.to_csv(os.path.join(args.output_dir, "overlapping.csv"), index=False)
    print(f"Brown-specific: {len(results.brown_specific_deletions)}")
    print(f"Polar-specific: {len(results.polar_specific_deletions)}")
    print(f"Overlapping: {len(results.overlapping_deletions)}")
