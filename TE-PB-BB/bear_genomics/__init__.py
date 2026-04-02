"""
bear_genomics
=============
Modular pipeline for structural variant (deletion) analysis in bear genomes.

Modules
-------
config              Configuration dataclasses and YAML loader
helpers             Shared interval/coordinate utility functions
vcf_parsing         VCF ingestion and normalisation
clustering          DBSCAN-based SV clustering
gene_annotation     GFF-based gene and repeat annotation
assembly_validation Assembly alignment validation (minimap2)
species_comparison  Cross-species overlap analysis
analysis_pipeline   End-to-end pipeline orchestration and CLI entry point

Quickstart
----------
    python -m bear_genomics.analysis_pipeline --config my_config.yaml
"""

from bear_genomics.config import (
    FixedBearAnalysisConfig,
    load_fixed_config,
    validate_config_file,
)
from bear_genomics.analysis_pipeline import CompleteBearGenomicsAnalysis

__all__ = [
    "FixedBearAnalysisConfig",
    "load_fixed_config",
    "validate_config_file",
    "CompleteBearGenomicsAnalysis",
]
