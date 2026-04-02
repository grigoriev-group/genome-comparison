"""
clustering.py
-------------
Variant clustering utilities for bear genomics structural variant analysis.

Provides:
- ClusteringMetrics: Dataclass holding comprehensive clustering quality metrics.
- OptimizedClusteringEngine: Full-featured clustering engine with sliding-window
  grouping, DBSCAN support, adaptive tolerance calculation, quality filtering,
  and cluster merging.  Handles both small and large (chunked) datasets.
"""

import logging
import time
import warnings
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


@dataclass
class ClusteringMetrics:
    """Comprehensive clustering quality metrics."""
    total_variants: int
    num_clusters: int
    clustered_variants: int
    noise_variants: int
    clustering_efficiency: float
    silhouette_score: float
    calinski_harabasz_score: float
    avg_cluster_size: float
    cluster_size_std: float
    cluster_density: float
    eps_parameter: float
    min_samples_parameter: int

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary."""
        return {'total_variants': self.total_variants, 'num_clusters': self.num_clusters, 'clustered_variants': self.clustered_variants, 'noise_variants': self.noise_variants, 'clustering_efficiency': self.clustering_efficiency, 'silhouette_score': self.silhouette_score, 'calinski_harabasz_score': self.calinski_harabasz_score, 'avg_cluster_size': self.avg_cluster_size, 'cluster_size_std': self.cluster_size_std, 'cluster_density': self.cluster_density, 'eps_parameter': self.eps_parameter, 'min_samples_parameter': self.min_samples_parameter}


class OptimizedClusteringEngine:
    """
    Optimized clustering engine with:
    - Fixed O(n²) complexity issues
    - Proper adaptive tolerance calculation
    - DBSCAN parameter optimization
    - Cluster quality assessment
    - Memory-efficient processing for large datasets
    """

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.base_tolerance = getattr(config, 'tolerance_bp', 50)
        self.clustering_strategy = getattr(config, 'clustering_strategy', 'adaptive')
        self.min_samples = getattr(config, 'brown_min_samples', 5)
        self.max_variants_for_optimization = 10000
        self.chunk_size = 5000
        self.parameter_cache = {}
        self.clustering_metrics = {}

    def cluster_variants(self, df: pd.DataFrame, dataset_name: str = 'unknown') -> pd.DataFrame:
        """Alias for cluster_variants_comprehensive for convenience."""
        return self.cluster_variants_comprehensive(df, dataset_name=dataset_name)

    def _merge_overlapping_clusters_from_records(self, cluster_records: List[Dict]) -> List[Dict]:
        """Merge overlapping clusters from records."""
        if len(cluster_records) <= 1:
            return cluster_records
        sorted_clusters = sorted(cluster_records, key=lambda x: (x['CHROM'], x['MIN_START']))
        merged = []
        current = sorted_clusters[0]
        for next_cluster in sorted_clusters[1:]:
            if current['CHROM'] == next_cluster['CHROM'] and current['MAX_END'] + self.base_tolerance >= next_cluster['MIN_START']:
                current = self._merge_two_clusters(current, next_cluster)
            else:
                merged.append(current)
                current = next_cluster
        merged.append(current)
        return merged

    def _merge_two_clusters(self, cluster1: Dict, cluster2: Dict) -> Dict:
        """
        Merge two overlapping clusters.
        FIXED: Merges genotypes and handles unique sample counting correctly.
        """
        # 1. Merge Genotypes (CRITICAL for Zygosity Analysis)
        gt1 = cluster1.get('genotypes', {})
        gt2 = cluster2.get('genotypes', {})

        # Update gt1 with gt2. If a sample is in both, gt2's call takes precedence.
        merged_genotypes = gt1.copy()
        merged_genotypes.update(gt2)

        # 2. Recalculate Unique Sample Counts
        # Use keys from merged genotypes to ensure uniqueness
        unique_samples = list(merged_genotypes.keys())

        # Fallback for legacy data without genotypes
        if not unique_samples:
            s1 = set(cluster1.get('SAMPLE_NAMES', '').split(';'))
            s2 = set(cluster2.get('SAMPLE_NAMES', '').split(';'))
            # Remove empty strings
            s1.discard('')
            s2.discard('')
            unique_samples = list(s1 | s2)

        new_sample_count = len(unique_samples)

        # 3. Fix Statistics Weighting (Use original counts for weighting)
        count1 = cluster1['SAMPLES_COUNT']
        count2 = cluster2['SAMPLES_COUNT']
        total_weight_count = count1 + count2

        # Prevent division by zero
        if total_weight_count == 0:
            weight1, weight2 = 0.5, 0.5
        else:
            weight1 = count1 / total_weight_count
            weight2 = count2 / total_weight_count

        # 4. Recalculate Percent Correctly
        # We back-calculate the Total Dataset Size from Cluster 1
        # Formula: Total = Count / (Percent/100)
        p1 = cluster1.get('SAMPLES_PERCENT', 0)
        if p1 > 0 and count1 > 0:
            estimated_total_samples = count1 / (p1 / 100.0)
            new_percent = (new_sample_count / estimated_total_samples) * 100.0
        else:
            # Fallback if we can't reverse engineer the total size
            new_percent = min(100.0, p1 + cluster2.get('SAMPLES_PERCENT', 0))

        merged_cluster = {
            'CHROM': cluster1['CHROM'],
            'MIN_START': min(cluster1['MIN_START'], cluster2['MIN_START']),
            'MAX_END': max(cluster1['MAX_END'], cluster2['MAX_END']),
            'AVG_START': round(cluster1['AVG_START'] * weight1 + cluster2['AVG_START'] * weight2, 1),
            'AVG_END': round(cluster1['AVG_END'] * weight1 + cluster2['AVG_END'] * weight2, 1),
            'AVG_SIZE': round(cluster1['AVG_SIZE'] * weight1 + cluster2['AVG_SIZE'] * weight2, 1),
            'AVG_QUAL': round(cluster1['AVG_QUAL'] * weight1 + cluster2['AVG_QUAL'] * weight2, 2),

            'SAMPLES_COUNT': new_sample_count,
            'SAMPLES_PERCENT': round(new_percent, 2),
            'SAMPLE_NAMES': ';'.join(sorted(unique_samples)[:10]),  # Sort and limit to keep string short

            'PRECISION_SCORE': round((cluster1['PRECISION_SCORE'] + cluster2['PRECISION_SCORE']) / 2, 3),
            'SIZE_STDEV': round(np.sqrt(
                (cluster1['SIZE_STDEV'] ** 2 * count1 + cluster2['SIZE_STDEV'] ** 2 * count2) / total_weight_count), 2),
            'genotypes': merged_genotypes
        }
        return merged_cluster

    def cluster_variants_comprehensive(self, variants_df: pd.DataFrame, dataset_name: str = 'unknown',
                                       min_samples_override: Optional[int] = None) -> pd.DataFrame:
        """
        Perform comprehensive variant clustering with optimization and quality assessment.
        """
        if variants_df.empty:
            self.logger.warning(f'Empty variants DataFrame for clustering: {dataset_name}')
            return variants_df

        self.logger.info(f'Starting clustering for {dataset_name}: {len(variants_df)} variants')
        start_time = time.time()

        variants_df = self._standardize_column_names(variants_df)
        unified_variants = self._create_unified_variant_dataset(variants_df)

        if not unified_variants:
            self.logger.error(f'No valid variants for clustering in {dataset_name}')
            return pd.DataFrame()

        effective_min_samples = min_samples_override or self.min_samples

        # Route to appropriate handler
        if len(unified_variants) > self.chunk_size:
            clustered_df = self._cluster_large_dataset_fixed(unified_variants, dataset_name, effective_min_samples)
        else:
            clustered_df = self._cluster_single_dataset_fixed(unified_variants, dataset_name, effective_min_samples)

        if not clustered_df.empty:
            clustered_df = self._ensure_clustering_output_format(clustered_df, dataset_name)

        clustering_time = time.time() - start_time
        self.logger.info(
            f'Clustering completed for {dataset_name} in {clustering_time:.2f}s: {len(clustered_df)} clusters generated')
        return clustered_df

    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """FIXED: Standardize column names to prevent KeyErrors"""
        if df.empty:
            return df
        df = df.copy()
        column_mapping = {
            'chrom': 'CHROM', 'chromosome': 'CHROM', 'Chromosome': 'CHROM',
            'start': 'START', 'Start': 'START', 'MIN_START': 'START',
            'end': 'END', 'End': 'END', 'MAX_END': 'END',
            'size': 'SIZE', 'Size': 'SIZE', 'AVG_SIZE': 'SIZE',
            'quality': 'QUAL', 'Quality': 'QUAL', 'AVG_QUAL': 'QUAL',
            'sample_name': 'SAMPLE', 'SAMPLE_NAMES': 'SAMPLE',
            'file_name': 'FILE', 'FILE_NAME': 'FILE'
        }
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and new_name not in df.columns:
                df[new_name] = df[old_name]
        required_columns = {
            'CHROM': 'unknown', 'START': 0, 'END': 1,
            'SIZE': 1, 'QUAL': 20.0, 'SAMPLE': 'unknown', 'FILE': 'unknown'
        }
        for col, default_val in required_columns.items():
            if col not in df.columns:
                df[col] = default_val
        return df

    def _create_unified_variant_dataset(self, variants_df: pd.DataFrame) -> List[Dict]:
        """FIXED: Create unified variant dataset preserving Genotype info."""
        unified = []
        for idx, row in variants_df.iterrows():
            try:
                variant = {
                    'variant_id': f'var_{idx}',
                    'chrom': str(row.get('CHROM', '')),
                    'start': int(row.get('START', 0)),
                    'end': int(row.get('END', 0)),
                    'size': int(row.get('SIZE', 0)),
                    'quality': float(row.get('QUAL', 20.0)),
                    'sample_name': str(row.get('SAMPLE', f'sample_{idx}')),
                    'sample_count': int(row.get('SAMPLES_COUNT', 1)),
                    'precision_score': float(row.get('PRECISION_SCORE', 0.5)),
                    # === FIX 2: Pass Genotype to Clustering ===
                    'genotype': str(row.get('genotype', './.'))
                }

                # Sanity checks
                if variant['start'] >= variant['end']:
                    variant['end'] = variant['start'] + max(variant['size'], 1)
                if variant['size'] <= 0:
                    variant['size'] = variant['end'] - variant['start']

                unified.append(variant)
            except Exception as e:
                self.logger.debug(f'Error processing variant {idx}: {e}')
                continue
        return unified

    def _cluster_single_dataset_fixed(self, variants: List[Dict], dataset_name: str, min_samples: int) -> pd.DataFrame:
        """
        FIXED: Single dataset clustering.
        (This is the correct version that calculates total_dataset_samples)
        """
        all_sample_names = set((v.get('sample_name', 'unknown') for v in variants))
        total_dataset_samples = len(all_sample_names)

        self.logger.info(
            f'Dataset {dataset_name}: {total_dataset_samples} unique samples, {len(variants)} total variants')

        all_clusters = []
        chrom_groups = defaultdict(list)

        for variant in variants:
            chrom_groups[variant['chrom']].append(variant)

        for chrom, chrom_variants in chrom_groups.items():
            if len(chrom_variants) < 2:
                continue

            chrom_variants.sort(key=lambda x: x['start'])
            groups = self._sliding_window_grouping(chrom_variants)

            for group in groups:
                if len(group) >= min_samples:
                    # Pass total_dataset_samples to the stats calculator
                    cluster_record = self._calculate_cluster_statistics(group, chrom, total_dataset_samples)
                    all_clusters.append(cluster_record)

        if all_clusters:
            clustered_df = pd.DataFrame(all_clusters)
            clustered_df = clustered_df.drop(columns=['genotypes'], errors='ignore')
            clustered_df = self._assign_global_cluster_ids(clustered_df, dataset_name)
        else:
            clustered_df = pd.DataFrame()

        return clustered_df

    def _sliding_window_grouping(self, variants: List[Dict]) -> List[List[Dict]]:
        """FIXED: Sliding window grouping with proper tolerance handling"""
        if len(variants) < 2:
            return [variants] if variants else []
        groups = []
        current_group = [variants[0]]
        for i in range(1, len(variants)):
            current_var = variants[i]
            prev_var = current_group[-1]
            distance = current_var['start'] - prev_var['end']
            if distance <= self.base_tolerance:
                current_group.append(current_var)
            else:
                if len(current_group) >= 2:
                    groups.append(current_group)
                current_group = [current_var]
        if len(current_group) >= 2:
            groups.append(current_group)
        return groups

    def _calculate_cluster_statistics(self, group: List[Dict], chrom: str, total_dataset_samples: int = None) -> Dict:
        """
        Calculate cluster statistics with RSSD and bounded breakpoint refinement.

        UPDATED: Now includes:
        - RSSD (Relative Size Standard Deviation) for characterizing cluster tightness
        - BREAKPOINT_WINDOW: adaptive search window based on SIZE_STDEV
        - START/END: rounded average coordinates (int(round(avg)))

        START/END are set to round(AVG_START) / round(AVG_END).
        RSSD is retained as a cluster tightness metric.
        """

        try:
            # 1. Sample Logic
            sample_names = list(set((v.get('sample_name', 'unknown') for v in group)))
            unique_sample_count = len(sample_names)
            total_variant_count = len(group)

            if total_dataset_samples and total_dataset_samples > 0:
                samples_percent = unique_sample_count / total_dataset_samples * 100
            else:
                samples_percent = 100.0

            # 2. Aggregate Genotypes
            genotypes_map = {}
            for v in group:
                s_name = v.get('sample_name')
                gt = v.get('genotype', './.')
                if s_name:
                    genotypes_map[s_name] = gt

            # 3. Basic Statistics
            starts = [v['start'] for v in group]
            ends = [v['end'] for v in group]
            sizes = [v['size'] for v in group]
            qualities = [v['quality'] for v in group if np.isfinite(v.get('quality', 0))]

            min_start = min(starts)
            max_end = max(ends)
            avg_start = sum(starts) / total_variant_count
            avg_end = sum(ends) / total_variant_count
            avg_size = sum(sizes) / total_variant_count
            avg_qual = sum(qualities) / len(qualities) if qualities else 20.0

            # 4. Size Statistics with RSSD
            if len(sizes) > 1:
                size_stdev = float(np.std(sizes))
                size_consistency = max(0.0, 1.0 - size_stdev / avg_size) if avg_size > 0 else 0.0
                position_span = max_end - min_start
                position_consistency = max(0.0, 1.0 - position_span / max(avg_size * 2, 1))
                precision_score = (size_consistency + position_consistency) / 2
            else:
                size_stdev = 0.0
                precision_score = 1.0

            precision_score = max(0.0, min(1.0, precision_score))

            # 5. Calculate RSSD (Relative Size Standard Deviation)
            # RSSD = SIZE_STDEV / AVG_SIZE - dimensionless measure of cluster tightness
            # Used for information and window sizing, NOT filtering (filtering disabled)
            rssd = size_stdev / avg_size if avg_size > 0 else 0.0

            # 6. START/END = rounded average coordinates
            start = int(round(avg_start))
            end = int(round(avg_end))
            if end <= start:
                end = start + max(int(avg_size), 1)

            # 7. Construct record
            cluster_record = {
                'CHROM': chrom,
                'MIN_START': min_start,
                'MAX_END': max_end,
                'AVG_START': round(avg_start, 1),
                'AVG_END': round(avg_end, 1),
                'AVG_SIZE': round(avg_size, 1),
                'AVG_QUAL': round(avg_qual, 2),
                'SAMPLES_COUNT': unique_sample_count,
                'SAMPLES_PERCENT': round(samples_percent, 1),
                'SAMPLE_NAMES': ';'.join(sample_names[:10]),
                'PRECISION_SCORE': round(precision_score, 3),
                'SIZE_STDEV': round(size_stdev, 2),
                'RSSD': round(rssd, 4),
                'START': start,
                'END': end,
                'SIZE': round(avg_size, 1),
                'QUAL': round(avg_qual, 2),
                'genotypes': genotypes_map,
            }
            return cluster_record

        except Exception as e:
            self.logger.error(f'Error calculating cluster statistics: {e}')
            return {
                'CHROM': chrom,
                'MIN_START': group[0]['start'],
                'MAX_END': group[0]['end'],
                'AVG_SIZE': group[0]['size'],
                'SAMPLES_COUNT': 1,
                'SAMPLES_PERCENT': 0.0,
                'PRECISION_SCORE': 0.5,
                'RSSD': 0.0,
                'START': group[0]['start'],
                'END': group[0]['end'],
                'genotypes': {},
            }

    def _assign_global_cluster_ids(self, clustered_df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """FIXED: Assign global cluster IDs"""
        if clustered_df.empty:
            return clustered_df
        clustered_df = clustered_df.reset_index(drop=True)
        clustered_df['CLUSTER_ID'] = [f'{dataset_name}_cluster_{i}' for i in range(len(clustered_df))]
        return clustered_df

    def _ensure_clustering_output_format(self, clustered_df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """FIXED: Ensure clustering output has expected format"""
        if clustered_df.empty:
            return clustered_df
        expected_columns = {'CHROM': 'unknown', 'MIN_START': 0, 'MAX_END': 1, 'AVG_SIZE': 1, 'AVG_QUAL': 20.0, 'SAMPLES_COUNT': 1, 'PRECISION_SCORE': 0.5, 'CLUSTER_ID': f'{dataset_name}_cluster_0'}
        for col, default_val in expected_columns.items():
            if col not in clustered_df.columns:
                clustered_df[col] = default_val
        return clustered_df

    def _cluster_large_dataset_fixed(self, variants: List[Dict], dataset_name: str, min_samples: int) -> pd.DataFrame:
        """
        FIXED: Handle clustering of large datasets with configurable quality validation.
        """
        # Get species-specific quality parameters
        params = self._get_quality_params(dataset_name)
        max_size_stdev = params['max_size_stdev']
        min_quality_fraction = params['min_quality_fraction']
        min_quality_value = params['min_quality_value']

        all_sample_names = set((v.get('sample_name', 'unknown') for v in variants))
        total_dataset_samples = len(all_sample_names)
        self.logger.info(f'Large dataset {dataset_name}: {total_dataset_samples} unique samples')
        self.logger.info(
            f'Quality filters for {dataset_name}: max_stdev={max_size_stdev}, '
            f'min_qual_frac={min_quality_fraction}, min_qual_val={min_quality_value}'
        )
        self.logger.info(f'Large dataset detected ({len(variants)} variants), using chunked processing')

        # --- Prepare log message for stdev check ---
        stdev_log_msg = f"rejected_stdev>{max_size_stdev}"
        stdev_check_enabled = max_size_stdev is not None and max_size_stdev > 0
        if not stdev_check_enabled:
            stdev_log_msg = "rejected_stdev(disabled)"
        # -------------------------------------------

        chrom_groups = defaultdict(list)
        for variant in variants:
            chrom_groups[variant['chrom']].append(variant)

        all_clusters = []

        # Counters for tracking rejections across all chromosomes
        total_groups_all_chroms = 0
        rejected_quality_all_chroms = 0
        rejected_stdev_all_chroms = 0
        accepted_all_chroms = 0

        for chrom, chrom_variants in chrom_groups.items():
            if len(chrom_variants) < min_samples:
                continue

            self.logger.info(f'Processing chromosome {chrom}: {len(chrom_variants)} variants')
            chrom_variants.sort(key=lambda x: x['start'])

            # Per-chromosome counters
            total_groups_chrom = 0
            rejected_quality_chrom = 0
            rejected_stdev_chrom = 0
            accepted_chrom = 0

            for i in range(0, len(chrom_variants), self.chunk_size):
                chunk = chrom_variants[i:i + self.chunk_size]
                if len(chunk) < min_samples:
                    continue

                chunk_groups = self._sliding_window_grouping(chunk)

                for group in chunk_groups:
                    if len(group) < min_samples:
                        continue

                    total_groups_chrom += 1
                    total_groups_all_chroms += 1

                    # QUALITY VALIDATION: Check if >= min_quality_fraction have QUAL >= min_quality_value
                    qualities = [v['quality'] for v in group if np.isfinite(v.get('quality', 0))]
                    if qualities:
                        high_quality_count = sum(1 for q in qualities if q >= min_quality_value)
                        if high_quality_count < len(qualities) * min_quality_fraction:
                            rejected_quality_chrom += 1
                            rejected_quality_all_chroms += 1
                            continue

                    # Calculate cluster statistics
                    cluster_record = self._calculate_cluster_statistics(group, chrom, total_dataset_samples)

                    # SIZE_STDEV VALIDATION: Must be <= max_size_stdev
                    stdev = cluster_record.get('SIZE_STDEV', None)

                    # FIX: Check if validation is enabled before comparing
                    if stdev_check_enabled and (stdev is None or pd.isna(stdev) or float(stdev) > max_size_stdev):
                        rejected_stdev_chrom += 1
                        rejected_stdev_all_chroms += 1
                        continue
                    # If check is disabled, skip the rejection logic and proceed

                    # Passed all validations
                    all_clusters.append(cluster_record)
                    accepted_chrom += 1
                    accepted_all_chroms += 1

            # Log per-chromosome summary
            self.logger.info(
                f"[Clustering:{chrom}:{dataset_name}] groups considered={total_groups_chrom}, "
                f"accepted={accepted_chrom}, rejected_quality={rejected_quality_chrom} "
                f"(min_frac={min_quality_fraction}, min_val={min_quality_value}), "
                f"{stdev_log_msg}={rejected_stdev_chrom}"
            )

        # Log overall summary
        self.logger.info(
            f"[Clustering:{dataset_name} TOTAL] groups considered={total_groups_all_chroms}, "
            f"accepted={accepted_all_chroms}, rejected_quality={rejected_quality_all_chroms}, "
            f"{stdev_log_msg}={rejected_stdev_all_chroms}"
        )

        if all_clusters:
            clustered_df = pd.DataFrame(all_clusters)
            clustered_df = clustered_df.drop(columns=['genotypes'], errors='ignore')
            clustered_df = self._assign_global_cluster_ids(clustered_df, dataset_name)
        else:
            clustered_df = pd.DataFrame()

        return clustered_df

    def _get_quality_params(self, dataset_name: str) -> dict:
        """
        Get clustering quality parameters for a specific dataset/species.

        Args:
            dataset_name: Name of the dataset (e.g., 'brown', 'polar', 'panda')

        Returns:
            Dict with quality parameters
        """
        # Default parameters
        defaults = {
            'max_size_stdev': 20.0,
            'min_quality_fraction': 0.5,
            'min_quality_value': 999,
            # RSSD settings (0 = disabled)
            'max_rssd': 0,  # Disabled by default - RSSD still calculated for breakpoint window
            'max_breakpoint_window': 500,
            'min_breakpoint_window': 20,
        }

        # Get global settings
        quality_config = getattr(self.config, 'clustering_quality_filters', {})

        if quality_config:
            # Update with global settings
            defaults['max_size_stdev'] = quality_config.get('max_size_stdev', defaults['max_size_stdev'])
            defaults['max_rssd'] = quality_config.get('max_rssd', defaults['max_rssd'])
            defaults['max_breakpoint_window'] = quality_config.get('max_breakpoint_window', defaults['max_breakpoint_window'])
            defaults['min_breakpoint_window'] = quality_config.get('min_breakpoint_window', defaults['min_breakpoint_window'])
            defaults['min_quality_fraction'] = quality_config.get('min_quality_fraction',
                                                                  defaults['min_quality_fraction'])
            defaults['min_quality_value'] = quality_config.get('min_quality_value', defaults['min_quality_value'])

            # Check for species-specific overrides
            species_overrides = quality_config.get('species_overrides', {})

            # Extract species name from dataset_name (e.g., 'brown' from 'brown_specific')
            species = dataset_name.replace('_specific', '').replace('_brown', '').replace('_polar', '')

            if species in species_overrides:
                override = species_overrides[species]
                defaults['max_size_stdev'] = override.get('max_size_stdev', defaults['max_size_stdev'])
                defaults['max_rssd'] = override.get('max_rssd', defaults['max_rssd'])
                defaults['min_quality_fraction'] = override.get('min_quality_fraction',
                                                                defaults['min_quality_fraction'])
                defaults['min_quality_value'] = override.get('min_quality_value', defaults['min_quality_value'])

                self.logger.debug(
                    f"Using species-specific quality params for {dataset_name}: "
                    f"max_stdev={defaults['max_size_stdev']}, max_rssd={defaults['max_rssd']}, "
                    f"min_qual_frac={defaults['min_quality_fraction']}, "
                    f"min_qual_val={defaults['min_quality_value']}"
                )

        return defaults

    def _validate_cluster_quality_composition(self, cluster_variants: pd.DataFrame,
                                              dataset_name: str = 'unknown') -> bool:
        """
        Validate that required fraction of variants in cluster meet quality threshold.

        Args:
            cluster_variants: DataFrame of variants in a cluster
            dataset_name: Name of the dataset for species-specific parameters

        Returns:
            True if cluster meets quality threshold, False otherwise
        """
        if cluster_variants.empty:
            return False

        # Get species-specific quality parameters
        params = self._get_quality_params(dataset_name)
        min_quality_fraction = params['min_quality_fraction']
        min_quality_value = params['min_quality_value']

        # Accept common QUAL column aliases
        qual_col = None
        for c in ("QUAL", "quality", "AVG_QUAL", "SCORE"):
            if c in cluster_variants.columns:
                qual_col = c
                break

        if qual_col is None:
            return False

        n = len(cluster_variants)
        if n == 0:
            return False

        # Convert to numeric and check for >= min_quality_value
        qual_values = pd.to_numeric(cluster_variants[qual_col], errors='coerce')
        n_high_quality = (qual_values >= min_quality_value).sum()

        return (n_high_quality >= min_quality_fraction * n)

    def _cluster_single_chromosome(self, chrom_variants: pd.DataFrame, chrom: str,
                                   min_samples: int, dataset_name: str = 'unknown') -> List[Dict]:
        """Perform clustering on a single chromosome."""
        if len(chrom_variants) < 2:
            return []
        sorted_variants = chrom_variants.sort_values('START').reset_index(drop=True)
        eps = self._calculate_adaptive_tolerance(sorted_variants, chrom)
        features = self._prepare_clustering_features(sorted_variants)
        cluster_labels = self._perform_dbscan_clustering(features, eps, min_samples, chrom)
        clusters = self._convert_clusters_to_records(sorted_variants, cluster_labels, chrom,
                                                     min_samples, dataset_name)
        return clusters

    def _validate_eps_for_data(self, eps: float, starts: np.ndarray, sizes: np.ndarray) -> float:
        """Validate that eps makes sense for the actual data distribution."""
        if len(starts) < 3:
            return eps
        try:
            from sklearn.neighbors import NearestNeighbors
            sample_size = min(len(starts), 100)
            sample_indices = np.random.choice(len(starts), sample_size, replace=False)
            sample_coords = starts[sample_indices].reshape(-1, 1)
            k = min(3, len(sample_coords) - 1)
            if k > 0:
                nbrs = NearestNeighbors(n_neighbors=k + 1, metric='manhattan').fit(sample_coords)
                distances, _ = nbrs.kneighbors(sample_coords)
                k_distances = distances[:, k]
                median_k_distance = np.median(k_distances)
                q75_k_distance = np.percentile(k_distances, 75)
                if eps > q75_k_distance * 3:
                    eps = q75_k_distance * 2
                    self.logger.debug(f'Adjusted eps to {eps:.1f} based on k-distance analysis')
                elif eps < median_k_distance * 0.5:
                    eps = median_k_distance
                    self.logger.debug(f'Increased eps to {eps:.1f} based on k-distance analysis')
        except Exception as e:
            self.logger.debug(f'K-distance validation failed: {e}')
        return eps

    def _prepare_clustering_features(self, variants_df: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for clustering with proper scaling.
        Includes SIZE and read depth support fraction when available.
        """
        starts = variants_df['START'].values.reshape(-1, 1)
        ends = variants_df['END'].values.reshape(-1, 1)
        features = np.column_stack([starts, ends])

        if 'SIZE' in variants_df.columns and len(variants_df) > 100:
            sizes = variants_df['SIZE'].values.reshape(-1, 1)
            size_scaler = StandardScaler()
            scaled_sizes = size_scaler.fit_transform(sizes)
            features = np.column_stack([features, scaled_sizes * 100])

            # Add read depth support fraction
            rd_weight = getattr(self, 'read_depth_cluster_weight', 50)
            if rd_weight > 0:
                if 'SUPPORT_FRACTION' in variants_df.columns:
                    sf = variants_df['SUPPORT_FRACTION'].fillna(0.5).values.reshape(-1, 1)
                    features = np.column_stack([features, sf * rd_weight])
                    self.logger.info(f'DBSCAN features: pos + SUPPORT_FRACTION (weight={rd_weight})')
                elif 'SUPPORT_READS' in variants_df.columns and 'TOTAL_READS' in variants_df.columns:
                    sr = variants_df['SUPPORT_READS'].fillna(0).values
                    tr = np.where(variants_df['TOTAL_READS'].fillna(1).values == 0, 1,
                                  variants_df['TOTAL_READS'].fillna(1).values)
                    sf = (sr / tr).reshape(-1, 1)
                    features = np.column_stack([features, sf * rd_weight])
                    self.logger.info(f'DBSCAN features: pos + computed SUPPORT_FRACTION (weight={rd_weight})')
                else:
                    self.logger.debug(f'DBSCAN features: pos only (no read depth columns found)')

            self.logger.info(
                f'Clustering feature matrix: {features.shape[1]} features, {features.shape[0]} variants')

        return features

    def _perform_dbscan_clustering(self, features: np.ndarray, eps: float, min_samples: int, chrom: str) -> np.ndarray:
        """
        Perform DBSCAN clustering with improved error handling and validation.
        FIXED: Better parameter validation and fallback strategies.
        """
        if len(features) == 0:
            return np.array([])
        original_min_samples = min_samples
        if len(features) < min_samples:
            min_samples = max(1, len(features) // 3)
            self.logger.warning(f'Chromosome {chrom}: adjusted min_samples from {original_min_samples} to {min_samples} (insufficient variants: {len(features)})')
        if eps <= 0:
            eps = self.base_tolerance
            self.logger.warning(f'Invalid eps for {chrom}, using base tolerance: {eps}')
        try:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='manhattan', n_jobs=1)
            cluster_labels = dbscan.fit_predict(features)
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = list(cluster_labels).count(-1)
            noise_rate = n_noise / len(cluster_labels) if len(cluster_labels) > 0 else 0
            self.logger.debug(f'Chromosome {chrom}: {n_clusters} clusters, {n_noise} noise points ({noise_rate:.1%} noise rate, eps={eps:.1f}, min_samples={min_samples})')
            if noise_rate > 0.8 and eps > self.base_tolerance and (len(features) > 10):
                self.logger.warning(f'High noise rate ({noise_rate:.1%}) for {chrom}, trying smaller eps')
                smaller_eps = max(self.base_tolerance, eps * 0.7)
                return self._perform_dbscan_clustering(features, smaller_eps, min_samples, chrom)
            if n_clusters == 0 and len(features) > 1:
                self.logger.warning(f'No clusters formed for {chrom} - all variants classified as noise')
            elif n_clusters == 1 and len(features) > 20:
                self.logger.info(f'Single large cluster for {chrom} - consider increasing eps or reducing min_samples')
            return cluster_labels
        except Exception as e:
            self.logger.error(f'DBSCAN clustering failed for chromosome {chrom}: {e}')
            self.logger.info(f'Using fallback clustering for {chrom}')
            return np.arange(len(features))

    def _convert_clusters_to_records(
            self,
            sorted_variants: pd.DataFrame,
            cluster_labels: np.ndarray,
            chrom: str,
            min_samples: int,
            dataset_name: str = 'unknown'
    ) -> List[Dict]:
        """Convert cluster labels to cluster records with quality & dispersion validation."""

        # Get species-specific quality parameters
        params = self._get_quality_params(dataset_name)
        max_size_stdev = params['max_size_stdev']
        min_quality_fraction = params['min_quality_fraction']
        min_quality_value = params['min_quality_value']

        clusters: List[Dict] = []
        unique_labels = [l for l in np.unique(cluster_labels) if l != -1]

        # --- Prepare log message for stdev check ---
        stdev_log_msg = f"rejected_stdev>{max_size_stdev}"
        stdev_check_enabled = max_size_stdev is not None and max_size_stdev > 0
        if not stdev_check_enabled:
            stdev_log_msg = "rejected_stdev(disabled)"
        # -------------------------------------------

        # counters for logging
        total_considered = 0
        rejected_small = 0
        rejected_quality = 0
        rejected_dispersion = 0
        accepted = 0

        for label in unique_labels:
            cluster_mask = (cluster_labels == label)
            cluster_variants = sorted_variants.loc[cluster_mask]

            total_considered += 1

            # Enforce minimum cluster size
            if len(cluster_variants) < min_samples:
                rejected_small += 1
                self.logger.debug(
                    f"Cluster {label} on {chrom} rejected: size {len(cluster_variants)} < min_samples {min_samples}"
                )
                continue

            # Quality validation with configurable parameters
            if not self._validate_cluster_quality_composition(cluster_variants, dataset_name):
                rejected_quality += 1
                self.logger.debug(
                    f"Cluster {label} on {chrom} rejected: <{min_quality_fraction * 100}% variants have QUAL>={min_quality_value}"
                )
                continue

            # Compute stats
            cluster_record = self._calculate_cluster_statistics(
                cluster_variants.to_dict("records"),
                chrom,
                len(sorted_variants)
            )
            cluster_record.setdefault("CLUSTER_LABEL", int(label))

            # SIZE_STDEV validation with configurable threshold
            stdev = cluster_record.get("SIZE_STDEV", None)

            # FIX: Check if validation is enabled before comparing
            if stdev_check_enabled and (stdev is None or pd.isna(stdev) or float(stdev) > max_size_stdev):
                rejected_dispersion += 1
                self.logger.debug(
                    f"Cluster {label} on {chrom} rejected: SIZE_STDEV ({stdev}) > {max_size_stdev}"
                )
                continue
            # If check is disabled, skip the rejection logic and proceed

            clusters.append(cluster_record)
            accepted += 1

        # one-line summary per chromosome
        self.logger.info(
            f"[Clustering:{chrom}:{dataset_name}] clusters considered={total_considered}, "
            f"accepted={accepted}, rejected_small={rejected_small}, "
            f"rejected_quality={rejected_quality} (min_frac={min_quality_fraction}, min_val={min_quality_value}), "
            f"{stdev_log_msg}={rejected_dispersion}"
        )

        return clusters

    def _merge_overlapping_clusters(self, clusters: List[Dict], chrom: str) -> List[Dict]:
        """
        Merge overlapping clusters from different chunks.
        FIXED: Efficient O(n log n) algorithm instead of O(n²).
        """
        if len(clusters) <= 1:
            return clusters
        sorted_clusters = sorted(clusters, key=lambda x: x['MIN_START'])
        merged = []
        current_cluster = sorted_clusters[0].copy()
        for next_cluster in sorted_clusters[1:]:
            if self._clusters_overlap(current_cluster, next_cluster):
                current_cluster = self._merge_two_clusters(current_cluster, next_cluster)
            else:
                merged.append(current_cluster)
                current_cluster = next_cluster.copy()
        merged.append(current_cluster)
        return merged

    def _clusters_overlap(self, cluster1: Dict, cluster2: Dict) -> bool:
        """Check if two clusters overlap based on their coordinates."""
        tolerance = self.base_tolerance * 2
        overlap_start = max(cluster1['MIN_START'], cluster2['MIN_START'])
        overlap_end = min(cluster1['MAX_END'], cluster2['MAX_END'])
        return overlap_end - overlap_start > -tolerance

    def _calculate_clustering_metrics(self, original_df: pd.DataFrame, clustered_df: pd.DataFrame, dataset_name: str, clustering_time: float):
        """
        Calculate comprehensive clustering quality metrics.
        FIXED: Proper metrics calculation with meaningful success rates.
        """
        if clustered_df.empty:
            metrics = ClusteringMetrics(total_variants=len(original_df), num_clusters=0, clustered_variants=0, noise_variants=len(original_df), clustering_efficiency=0.0, silhouette_score=0.0, calinski_harabasz_score=0.0, avg_cluster_size=0.0, cluster_size_std=0.0, cluster_density=0.0, eps_parameter=self.base_tolerance, min_samples_parameter=self.min_samples)
            clustering_efficiency = 0.0
        else:
            clustered_variants = clustered_df['SAMPLES_COUNT'].sum() if 'SAMPLES_COUNT' in clustered_df.columns else len(clustered_df)
            cluster_sizes = clustered_df['SAMPLES_COUNT'].values if 'SAMPLES_COUNT' in clustered_df.columns else np.ones(len(clustered_df))
            clustering_efficiency = len(clustered_df) / len(original_df) if len(original_df) > 0 else 0.0
            try:
                if len(clustered_df) > 1:
                    centers = clustered_df[['AVG_START', 'AVG_END']].values if 'AVG_START' in clustered_df.columns else clustered_df[['MIN_START', 'MAX_END']].values
                    labels = np.arange(len(clustered_df))
                    if len(centers) > 1:
                        from sklearn.metrics import silhouette_score, calinski_harabasz_score
                        silhouette = silhouette_score(centers, labels)
                        calinski_harabasz = calinski_harabasz_score(centers, labels)
                    else:
                        silhouette = 1.0
                        calinski_harabasz = 0.0
                else:
                    silhouette = 1.0
                    calinski_harabasz = 0.0
            except Exception as e:
                self.logger.debug(f'Error calculating clustering quality metrics: {e}')
                silhouette = 0.0
                calinski_harabasz = 0.0
            metrics = ClusteringMetrics(total_variants=len(original_df), num_clusters=len(clustered_df), clustered_variants=int(clustered_variants), noise_variants=max(0, len(original_df) - int(clustered_variants)), clustering_efficiency=clustering_efficiency, silhouette_score=silhouette, calinski_harabasz_score=calinski_harabasz, avg_cluster_size=float(cluster_sizes.mean()), cluster_size_std=float(cluster_sizes.std()), cluster_density=len(clustered_df) / len(original_df) if len(original_df) > 0 else 0.0, eps_parameter=self.base_tolerance, min_samples_parameter=self.min_samples)
        self.clustering_metrics[dataset_name] = metrics
        reduction_rate = 1.0 - clustering_efficiency
        self.logger.info(f'Clustering metrics for {dataset_name}:')
        self.logger.info(f'  Input variants: {metrics.total_variants}')
        self.logger.info(f'  Output clusters: {metrics.num_clusters}')
        self.logger.info(f'  Reduction rate: {reduction_rate:.1%} ({metrics.total_variants} -> {metrics.num_clusters})')
        self.logger.info(f'  Avg cluster size: {metrics.avg_cluster_size:.1f}')
        self.logger.info(f'  Processing time: {clustering_time:.2f}s')
        if metrics.silhouette_score > 0:
            self.logger.info(f'  Clustering quality (silhouette): {metrics.silhouette_score:.3f}')

    def optimize_clustering_parameters(self, variants_df: pd.DataFrame, chrom: str) -> Tuple[float, int]:
        """
        Optimize DBSCAN parameters using elbow method and silhouette analysis.
        FIXED: Efficient parameter optimization for better clustering quality.
        """
        if len(variants_df) > self.max_variants_for_optimization:
            return (self._calculate_adaptive_tolerance(variants_df, chrom), self.min_samples)
        self.logger.info(f'Optimizing clustering parameters for chromosome {chrom}')
        features = self._prepare_clustering_features(variants_df)
        base_eps = self._calculate_adaptive_tolerance(variants_df, chrom)
        eps_range = np.linspace(base_eps * 0.5, base_eps * 2.0, 10)
        min_samples_range = [max(2, self.min_samples // 2), self.min_samples, self.min_samples * 2]
        best_score = -1
        best_eps = base_eps
        best_min_samples = self.min_samples
        for eps in eps_range:
            for min_samples in min_samples_range:
                try:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='manhattan')
                    labels = dbscan.fit_predict(features)
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    if n_clusters <= 1 or n_clusters >= len(features) // 2:
                        continue
                    score = silhouette_score(features, labels)
                    if score > best_score:
                        best_score = score
                        best_eps = eps
                        best_min_samples = min_samples
                except:
                    continue
        self.logger.info(f'Optimized parameters for {chrom}: eps={best_eps:.1f}, min_samples={best_min_samples}, score={best_score:.3f}')
        return (best_eps, best_min_samples)

    def get_clustering_report(self) -> Dict:
        """Generate comprehensive clustering analysis report."""
        if not self.clustering_metrics:
            return {'error': 'No clustering metrics available'}
        report = {'summary': {'datasets_processed': len(self.clustering_metrics), 'total_variants': sum((m.total_variants for m in self.clustering_metrics.values())), 'total_clusters': sum((m.num_clusters for m in self.clustering_metrics.values())), 'overall_efficiency': sum((m.clustered_variants for m in self.clustering_metrics.values())) / sum((m.total_variants for m in self.clustering_metrics.values()))}, 'dataset_metrics': {name: metrics.to_dict() for name, metrics in self.clustering_metrics.items()}, 'parameter_cache': self.parameter_cache, 'recommendations': self._generate_clustering_recommendations()}
        return report

    def _calculate_adaptive_tolerance(self, variants_df: pd.DataFrame, chrom: str) -> float:
        """
        Calculate adaptive tolerance (eps) for DBSCAN clustering.
        FIXED: Implementation of missing method.
        """
        if len(variants_df) < 2:
            return self.base_tolerance
        starts = variants_df['START'].values
        sizes = variants_df['SIZE'].values
        position_diffs = np.diff(sorted(starts))
        valid_diffs = position_diffs[(position_diffs > 0) & (position_diffs < 100000)]
        if len(valid_diffs) == 0:
            return self.base_tolerance
        if self.clustering_strategy == 'conservative':
            eps = np.percentile(valid_diffs, 25)
        elif self.clustering_strategy == 'generous':
            eps = np.percentile(valid_diffs, 75)
        else:
            eps = np.median(valid_diffs)
        if len(sizes) > 0:
            median_size = np.median(sizes)
            size_factor = min(median_size / 1000.0, 2.0)
            eps *= 1.0 + size_factor * 0.1
        eps = max(eps, self.base_tolerance * 0.5)
        eps = min(eps, self.base_tolerance * 3.0)
        eps = self._validate_eps_for_data(eps, starts, sizes)
        self.logger.debug(f'Adaptive tolerance for {chrom}: {eps:.1f} bp')
        return float(eps)

    def _generate_clustering_recommendations(self) -> List[str]:
        """Generate recommendations based on clustering performance."""
        recommendations = []
        if not self.clustering_metrics:
            return ['No clustering metrics available for recommendations']
        efficiencies = [m.clustering_efficiency for m in self.clustering_metrics.values()]
        avg_efficiency = np.mean(efficiencies)
        if avg_efficiency < 0.7:
            recommendations.append('Low clustering efficiency - consider adjusting tolerance parameters')
        avg_cluster_sizes = [m.avg_cluster_size for m in self.clustering_metrics.values()]
        overall_avg_size = np.mean(avg_cluster_sizes)
        if overall_avg_size < 2:
            recommendations.append('Very small average cluster size - consider reducing min_samples parameter')
        elif overall_avg_size > 20:
            recommendations.append('Very large average cluster size - consider reducing tolerance or increasing min_samples')
        silhouette_scores = [m.silhouette_score for m in self.clustering_metrics.values() if m.silhouette_score > 0]
        if silhouette_scores:
            avg_silhouette = np.mean(silhouette_scores)
            if avg_silhouette < 0.3:
                recommendations.append('Low silhouette scores indicate poor cluster separation')
        return recommendations

    def cluster_deletions_optimized(self, variants: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Enhanced clustering but NO support filtering here."""
        if variants.empty:
            return variants
        clustered_variants = self._perform_clustering(variants, dataset_name)
        # DO NOT filter here; leave thresholds to visualization/export stages.
        self.logger.info(
            f'{dataset_name}: clustering produced {len(clustered_variants)} clusters (no support filter applied)')
        return clustered_variants

    def _apply_sample_support_filter(self, variants: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Filter clusters requiring >70% sample support"""
        if variants.empty or 'SAMPLES_COUNT' not in variants.columns:
            return variants
        if 'SAMPLE_NAMES' in variants.columns:
            all_samples = set()
            for sample_names in variants['SAMPLE_NAMES'].dropna():
                all_samples.update(sample_names.split(';'))
            total_samples = len(all_samples)
        else:
            total_samples = variants['SAMPLES_COUNT'].max()
        support_threshold = max(1, int(0.7 * total_samples))
        high_support_mask = variants['SAMPLES_COUNT'] >= support_threshold
        filtered_variants = variants[high_support_mask].copy()
        self.logger.info(f'{dataset_name} sample support filter:')
        self.logger.info(f'  Total samples: {total_samples}')
        self.logger.info(f'  70% threshold: {support_threshold}')
        self.logger.info(f'  Clusters passing filter: {len(filtered_variants)}/{len(variants)}')
        return filtered_variants


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Cluster variants from CSV")
    parser.add_argument("--input", required=True, help="Input variants CSV")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import pandas as pd
    from bear_genomics.config import load_fixed_config
    config = load_fixed_config(args.config)
    df = pd.read_csv(args.input)
    engine = OptimizedClusteringEngine(config)
    result = engine.cluster_variants(df, dataset_name="variants")
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result)} clustered variants to {args.output}")
