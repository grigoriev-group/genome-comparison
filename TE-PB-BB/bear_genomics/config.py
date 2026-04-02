"""
bear_genomics.config
====================
Configuration dataclasses, loader, and validation utilities for the bear
genomics pipeline.

Contains:
  - FixedBearAnalysisConfig   — master pipeline configuration dataclass
  - load_fixed_config()       — parse a YAML file into FixedBearAnalysisConfig
  - validate_config_file()    — standalone YAML validation helper
"""

import os
import logging
import dataclasses
import yaml
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from multiprocessing import cpu_count
import shutil


# ---------------------------------------------------------------------------
# Master configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class FixedBearAnalysisConfig:
    """Master pipeline configuration."""

    # Required paths
    brown_vcf_folder: str
    polar_vcf_folder: str
    gff_file: str
    reference_genome_path: str
    output_folder: str

    # Reference genome selection
    reference_genome: str = 'polar'

    # Project
    project_name: str = 'bear_genomics'

    # Core analysis parameters
    tolerance_bp: int = 50
    overlap_threshold: float = 0.3
    clustering_strategy: str = 'adaptive'
    brown_min_samples: int = 5
    polar_min_samples: int = 5
    min_qual: float = 20.0
    min_size: int = 40
    max_size: int = 1000000
    chunk_size: int = 10000
    nearby_threshold: int = 5000
    threads: int = 8
    max_workers: int = 8
    read_depth_cluster_weight: float = 0.0  # 0 = disabled

    # Optional input files
    repeatmasker_bed: Optional[str] = None

    # VCF parsing
    enable_multiallelic_parsing: bool = True

    # Species processing flags (set dynamically via focus_species)
    process_brown_specific: bool = True
    process_polar_specific: bool = True
    process_overlapping: bool = True
    focus_species: Optional[str] = None

    # Gene annotation
    gene_annotation_settings: Dict[str, Any] = field(default_factory=lambda: {
        'include_overlapping': True,
        'include_nearby': True,
        'gene_name_field': 'gene',
        'feature_types': ['gene', 'mRNA', 'transcript', 'CDS', 'exon',
                          'five_prime_UTR', 'three_prime_UTR'],
        'strand_specific': False,
        'max_genes_per_variant': 15,
    })
    annotation_output: Dict[str, Any] = field(default_factory=lambda: {
        'include_gene_details': True,
        'include_distance_info': True,
        'include_strand_info': True,
        'separate_nearby_overlapping': True,
    })
    reference_aware_annotation: Dict[str, Any] = field(default_factory=dict)

    # Annotation dataset overrides (optional)
    manual_annotation_override: Dict[str, Any] = field(default_factory=dict)
    annotate_datasets: List[str] = field(default_factory=list)
    skip_datasets: List[str] = field(default_factory=list)

    # Clustering quality filters
    clustering_quality_filters: Dict[str, Any] = field(default_factory=lambda: {
        'max_rssd': 0,
        'max_size_stdev': 0,
        'min_quality_fraction': 0.0,
    })

    # Assembly validation
    enable_assembly_validation: bool = False
    min_assembly_identity: float = 0.85
    max_gap_size: int = 1000000

    # Repeat filtering
    enable_repeat_filtering: bool = False

    def __post_init__(self):
        self.logger = logging.getLogger(__name__)
        validation_issues = self.validate_configuration()
        if validation_issues:
            raise ValueError(
                'Configuration validation failed:\n' +
                '\n'.join(f'  - {issue}' for issue in validation_issues)
            )

    def validate_configuration(self) -> List[str]:
        issues = []

        # Required paths
        required_paths = {
            'brown_vcf_folder': self.brown_vcf_folder,
            'polar_vcf_folder': self.polar_vcf_folder,
            'gff_file': self.gff_file,
            'reference_genome_path': self.reference_genome_path,
        }
        for path_name, path_value in required_paths.items():
            if not path_value:
                issues.append(f'Required path is empty: {path_name}')
                continue
            if not os.path.exists(path_value):
                issues.append(f"Required path does not exist: {path_name} = '{path_value}'")
                continue
            if path_name.endswith('_folder'):
                if not os.path.isdir(path_value):
                    issues.append(f"Path is not a directory: {path_name} = '{path_value}'")
                elif 'vcf' in path_name:
                    if not list(Path(path_value).glob('*.vcf')):
                        issues.append(f"No VCF files found in: {path_name} = '{path_value}'")
            elif path_name == 'gff_file':
                if not path_value.lower().endswith(('.gff', '.gff3', '.gtf')):
                    issues.append(f"GFF file should have .gff, .gff3, or .gtf extension: '{path_value}'")
            elif path_name == 'reference_genome_path':
                if not path_value.lower().endswith(('.fasta', '.fa', '.fna')):
                    issues.append(f"Reference genome should have .fasta, .fa, or .fna extension: '{path_value}'")

        # Output folder writable
        try:
            os.makedirs(self.output_folder, exist_ok=True)
            test_file = Path(self.output_folder) / '.write_test'
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            issues.append(f"Cannot create or write to output directory '{self.output_folder}': {e}")

        # Optional paths
        if self.repeatmasker_bed and not os.path.exists(self.repeatmasker_bed):
            issues.append(f"Optional path does not exist: repeatmasker_bed = '{self.repeatmasker_bed}'")

        # Numeric ranges
        numeric_validations = [
            ('tolerance_bp',          self.tolerance_bp,          1,    10000,           'Tolerance BP'),
            ('overlap_threshold',     self.overlap_threshold,     0.0,  1.0,             'Overlap threshold'),
            ('brown_min_samples',     self.brown_min_samples,     1,    1000,            'Brown minimum samples'),
            ('polar_min_samples',     self.polar_min_samples,     1,    1000,            'Polar minimum samples'),
            ('min_qual',              self.min_qual,              0.0,  10000.0,         'Minimum quality'),
            ('min_size',              self.min_size,              1,    1000000,         'Minimum size'),
            ('max_size',              self.max_size,              100,  100000000,       'Maximum size'),
            ('chunk_size',            self.chunk_size,            100,  100000,          'Chunk size'),
            ('nearby_threshold',      self.nearby_threshold,      10,   1000000,         'Nearby threshold'),
            ('threads',               self.threads,               1,    cpu_count() * 2, 'Thread count'),
            ('min_assembly_identity', self.min_assembly_identity, 0.0,  1.0,            'Assembly identity'),
            ('max_gap_size',          self.max_gap_size,          1000, 10000000,        'Maximum gap size'),
        ]
        for param_name, value, min_val, max_val, description in numeric_validations:
            if not isinstance(value, (int, float)):
                issues.append(f'{description} must be numeric: {param_name} = {value}')
            elif value < min_val or value > max_val:
                issues.append(f'{description} out of range [{min_val}, {max_val}]: {param_name} = {value}')

        if self.min_size >= self.max_size:
            issues.append(f'min_size ({self.min_size}) must be less than max_size ({self.max_size})')

        valid_strategies = ['conservative', 'adaptive', 'generous']
        if self.clustering_strategy not in valid_strategies:
            issues.append(f"Invalid clustering_strategy: '{self.clustering_strategy}'. Must be one of {valid_strategies}")

        if self.enable_repeat_filtering and not self.repeatmasker_bed:
            issues.append('enable_repeat_filtering is True but no repeatmasker_bed file provided')

        if self.enable_assembly_validation and not hasattr(self, 'brown_reference_genome'):
            issues.append('enable_assembly_validation is True but no alternate reference genome provided')

        return issues

    def get_effective_config(self, dependency_status: Dict[str, bool]) -> 'FixedBearAnalysisConfig':
        effective_config = self.__class__(**{
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
        })
        if not dependency_status.get('minimap2', False):
            effective_config.enable_assembly_validation = False
            logging.warning('Assembly validation disabled - minimap2 not available')
        return effective_config

    def create_output_structure(self) -> Dict[str, Path]:
        base_path = Path(self.output_folder)
        subdirs = {
            'logs':       base_path / 'logs',
            'results':    base_path / 'results',
            'comparison': base_path / 'comparison',
        }
        for path in subdirs.values():
            path.mkdir(parents=True, exist_ok=True)
        for stale in ('visualizations', 'reports'):
            stale_path = base_path / stale
            if stale_path.exists():
                shutil.rmtree(stale_path)
                logging.info(f'Removed stale directory: {stale_path}')
        return subdirs

    def generate_summary_report(self) -> str:
        return f"""
Bear Genomics Pipeline Configuration
======================================
Project:       {self.project_name}
Output:        {self.output_folder}
Reference:     {self.reference_genome}

Input Data:
  Brown VCFs:  {self.brown_vcf_folder}
  Polar VCFs:  {self.polar_vcf_folder}
  GFF:         {self.gff_file}
  Reference:   {self.reference_genome_path}

Parameters:
  Clustering tolerance:  {self.tolerance_bp} bp
  Overlap threshold:     {self.overlap_threshold}
  Nearby threshold:      {self.nearby_threshold} bp
  Quality threshold:     {self.min_qual}
  Size range:            {self.min_size} - {self.max_size} bp
  Threads:               {self.threads}
  Brown min samples:     {self.brown_min_samples}
  Polar min samples:     {self.polar_min_samples}
"""


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _validate_min_samples_for_reference(config) -> None:
    ref = getattr(config, 'reference_genome', 'polar').lower()
    logging.info(f"Reference genome: {ref}")
    logging.info(f"  brown_min_samples: {getattr(config, 'brown_min_samples', 5)}")
    logging.info(f"  polar_min_samples: {getattr(config, 'polar_min_samples', 5)}")
    if ref == 'polar':
        logging.info("Annotation target: brown-specific variants")
    elif ref == 'brown':
        logging.info("Annotation target: polar-specific variants")
    else:
        logging.warning(f"Unknown reference_genome: '{ref}'. Expected 'polar' or 'brown'.")


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def load_fixed_config(config_file: str, allow_partial: bool = False) -> FixedBearAnalysisConfig:
    """Load YAML → FixedBearAnalysisConfig."""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f'Configuration file not found: {config_file}')

    try:
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)

        if not config_dict or not isinstance(config_dict, dict):
            raise ValueError(f'Configuration file must contain a non-empty dictionary: {config_file}')

        valid_fields = {f.name for f in dataclasses.fields(FixedBearAnalysisConfig)}
        valid_dict, extra_dict = {}, {}

        for key, value in config_dict.items():
            if key in valid_fields:
                valid_dict[key] = value
            else:
                extra_dict[key] = value

        config = FixedBearAnalysisConfig(**valid_dict)

        if extra_dict:
            logging.debug(f'Attaching {len(extra_dict)} extra config fields as dynamic attributes')
            for key, value in extra_dict.items():
                setattr(config, key, value)

        _validate_min_samples_for_reference(config)
        logging.info(f'Configuration loaded: {config_file}')
        return config

    except yaml.YAMLError as e:
        raise ValueError(f'Invalid YAML in {config_file}: {e}')
    except TypeError as e:
        if 'unexpected keyword argument' in str(e):
            import re
            match = re.search(r"'(\w+)'", str(e))
            problem_field = match.group(1) if match else 'unknown'
            logging.error(f'Unexpected config field: "{problem_field}"')
            if allow_partial:
                pruned = {k: v for k, v in config_dict.items()
                          if k in {f.name for f in dataclasses.fields(FixedBearAnalysisConfig)}
                          and k != problem_field}
                cfg = FixedBearAnalysisConfig(**pruned)
                return cfg
            raise ValueError(f'Configuration error in {config_file}: {e}')
        raise
    except Exception as e:
        raise ValueError(f'Error loading configuration from {config_file}: {e}')


def validate_config_file(config_file: str, verbose: bool = True) -> Tuple[bool, List[str]]:
    """Validate a configuration file without running the pipeline."""
    try:
        config = load_fixed_config(config_file)
        if verbose:
            print(config.generate_summary_report())
        return (True, [])
    except Exception as e:
        if verbose:
            print(f'Configuration validation failed: {e}')
        return (False, [str(e)])
