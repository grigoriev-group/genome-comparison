"""Assembly-based structural variant validation using minimap2 whole-genome alignment."""

import os
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
from collections import defaultdict


@dataclass
class AssemblyAlignment:
    """Data structure for assembly alignment information."""
    query_name: str
    query_length: int
    query_start: int
    query_end: int
    strand: str
    target_name: str
    target_length: int
    target_start: int
    target_end: int
    residue_matches: int
    alignment_block_length: int
    mapping_quality: int

    def __post_init__(self):
        """Calculate derived metrics."""
        self.identity = self.residue_matches / self.alignment_block_length if self.alignment_block_length > 0 else 0.0
        self.query_coverage = (self.query_end - self.query_start) / self.query_length if self.query_length > 0 else 0.0
        self.target_coverage = (self.target_end - self.target_start) / self.target_length if self.target_length > 0 else 0.0


@dataclass
class ValidationResult:
    """Results of assembly validation for a structural variant."""
    variant_id: str
    validated: bool
    confidence_score: float
    supporting_alignments: int
    alignment_identity: float
    alignment_coverage: float
    gap_analysis: Dict[str, float]
    validation_method: str
    error_message: Optional[str] = None


class FixedAssemblyValidator:
    """
    Complete assembly validation implementation with:
    - External tool dependency validation
    - Robust minimap2 integration
    - PAF file parsing with error handling
    - Alignment quality assessment
    - Memory-efficient processing
    """

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.min_assembly_identity = getattr(config, 'min_assembly_identity', 0.9)
        self.max_gap_size = getattr(config, 'max_gap_size', 500000)
        self.min_alignment_length = 1000
        self.minimap2_path = 'minimap2'
        self.samtools_path = 'samtools'
        self.alignment_cache = {}
        self.validation_stats = defaultdict(int)
        self._validate_dependencies()

    def _validate_dependencies(self):
        """
        Validate external tool dependencies.
        FIXED: Comprehensive dependency checking with helpful error messages.
        """
        self.logger.info('Validating assembly validation dependencies...')
        try:
            result = subprocess.run([self.minimap2_path, '--version'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version = result.stdout.strip()
                self.logger.info(f'✅ minimap2 available: {version}')
                self.minimap2_available = True
            else:
                self.logger.error(f'❌ minimap2 not working: {result.stderr}')
                self.minimap2_available = False
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            self.logger.error(f'❌ minimap2 not found: {e}')
            self.minimap2_available = False
        try:
            result = subprocess.run([self.samtools_path, '--version'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                self.samtools_available = True
                self.logger.info('✅ samtools available')
            else:
                self.samtools_available = False
                self.logger.warning('⚠️ samtools not available - some features disabled')
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.samtools_available = False
            self.logger.warning('⚠️ samtools not found - some features disabled')
        if not self.minimap2_available:
            self.logger.error('❌ Assembly validation requires minimap2 to be installed')
            self.logger.error('   Install with: conda install -c bioconda minimap2')
            self.logger.error('   Or download from: https://github.com/lh3/minimap2')

    def validate_variants_with_assembly(self, variants_df: pd.DataFrame, reference_genomes: Dict[str, str], output_dir: Optional[str]=None) -> pd.DataFrame:
        """
        Validate structural variants using assembly-to-assembly alignments.
        FIXED: Complete implementation with proper error handling.
        """
        if not self.minimap2_available:
            self.logger.error('Cannot perform assembly validation - minimap2 not available')
            return self._add_validation_columns_placeholder(variants_df)
        if variants_df.empty:
            self.logger.warning('Empty variants DataFrame provided for assembly validation')
            return variants_df
        self.logger.info(f'Starting assembly validation for {len(variants_df)} variants')
        if output_dir is None:
            output_dir = Path(self.config.output_folder) / 'assembly_validation'
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        alignment_files = self._ensure_assembly_alignments(reference_genomes, output_dir)
        if not alignment_files:
            self.logger.error('No assembly alignments available - cannot perform validation')
            return self._add_validation_columns_placeholder(variants_df)
        validated_df = variants_df.copy()
        validation_results = []
        for idx, variant in variants_df.iterrows():
            try:
                result = self._validate_single_variant(variant, alignment_files, output_dir)
                validation_results.append(result)
                if (idx + 1) % 100 == 0:
                    self.logger.info(f'Validated {idx + 1}/{len(variants_df)} variants')
            except Exception as e:
                self.logger.warning(f'Error validating variant {idx}: {e}')
                validation_results.append(ValidationResult(variant_id=f'variant_{idx}', validated=False, confidence_score=0.0, supporting_alignments=0, alignment_identity=0.0, alignment_coverage=0.0, gap_analysis={}, validation_method='error', error_message=str(e)))
                self.validation_stats['validation_errors'] += 1
        validated_df = self._integrate_validation_results(validated_df, validation_results)
        self._log_validation_summary(validation_results)
        return validated_df

    def _ensure_assembly_alignments(self, reference_genomes: Dict[str, str], output_dir: Path) -> Dict[str, str]:
        """
        Ensure assembly alignments exist, create if necessary.
        FIXED: Robust alignment generation with proper error handling.
        """
        alignment_files = {}
        primary_reference = self.config.reference_genome_path
        if not os.path.exists(primary_reference):
            self.logger.error(f'Primary reference genome not found: {primary_reference}')
            return alignment_files
        for genome_name, genome_path in reference_genomes.items():
            if not os.path.exists(genome_path):
                self.logger.warning(f'Reference genome not found: {genome_name} = {genome_path}')
                continue
            alignment_name = f'primary_vs_{genome_name}'
            alignment_file = self._get_or_create_alignment(alignment_name, primary_reference, genome_path, output_dir)
            if alignment_file:
                alignment_files[alignment_name] = alignment_file
                self.logger.info(f'Assembly alignment ready: {alignment_name}')
        return alignment_files

    def _get_or_create_alignment(self, alignment_name: str, query_genome: str, target_genome: str, output_dir: Path) -> Optional[str]:
        """
        Get existing alignment or create new one using minimap2.
        FIXED: Comprehensive error handling and file validation.
        """
        alignment_file = output_dir / f'{alignment_name}.paf'
        if alignment_file.exists():
            if self._validate_paf_file(alignment_file):
                self.logger.info(f'Using existing alignment: {alignment_file}')
                return str(alignment_file)
            else:
                self.logger.warning(f'Invalid existing alignment, recreating: {alignment_file}')
                alignment_file.unlink()
        if not self._validate_genome_file(query_genome):
            self.logger.error(f'Invalid query genome: {query_genome}')
            return None
        if not self._validate_genome_file(target_genome):
            self.logger.error(f'Invalid target genome: {target_genome}')
            return None
        return self._create_minimap2_alignment(query_genome, target_genome, alignment_file)

    def _validate_genome_file(self, genome_path: str) -> bool:
        """Validate genome file exists and is readable."""
        try:
            if not os.path.exists(genome_path):
                return False
            file_size = os.path.getsize(genome_path)
            if file_size < 1000000:
                self.logger.warning(f'Genome file suspiciously small: {genome_path} ({file_size} bytes)')
                return False
            with open(genome_path, 'r') as f:
                first_line = f.readline().strip()
                if not first_line.startswith('>'):
                    self.logger.error(f"Genome file doesn't appear to be FASTA format: {genome_path}")
                    return False
            return True
        except Exception as e:
            self.logger.error(f'Error validating genome file {genome_path}: {e}')
            return False

    def _validate_paf_file(self, paf_file: Path) -> bool:
        """Validate PAF file format and content."""
        try:
            if not paf_file.exists():
                return False
            if os.path.getsize(paf_file) == 0:
                return False
            with open(paf_file, 'r') as f:
                for i, line in enumerate(f):
                    if i >= 5:
                        break
                    fields = line.strip().split('\t')
                    if len(fields) < 12:
                        return False
                    try:
                        int(fields[1])
                        int(fields[2])
                        int(fields[3])
                        int(fields[6])
                        int(fields[7])
                        int(fields[8])
                        int(fields[9])
                        int(fields[10])
                        int(fields[11])
                    except ValueError:
                        return False
            return True
        except Exception as e:
            self.logger.error(f'Error validating PAF file {paf_file}: {e}')
            return False

    def _create_minimap2_alignment(self, query_genome: str, target_genome: str, alignment_file: Path) -> Optional[str]:
        """
        Create alignment using minimap2 with robust error handling.
        FIXED: Proper command construction and error handling.
        """
        self.logger.info(f'Creating alignment: {query_genome} vs {target_genome}')
        try:
            cmd = [self.minimap2_path, '-x', 'asm5', '-t', str(self.config.threads), '--secondary=no', '-N', '50', target_genome, query_genome]
            self.logger.info(f"Running minimap2 command: {' '.join(cmd)}")
            with open(alignment_file, 'w') as output_file:
                process = subprocess.Popen(cmd, stdout=output_file, stderr=subprocess.PIPE, text=True)
                try:
                    stderr_output = process.communicate(timeout=3600)[1]
                    return_code = process.returncode
                except subprocess.TimeoutExpired:
                    process.kill()
                    self.logger.error('minimap2 process timed out after 1 hour')
                    return None
            if return_code == 0:
                if self._validate_paf_file(alignment_file):
                    self.logger.info(f'Successfully created alignment: {alignment_file}')
                    return str(alignment_file)
                else:
                    self.logger.error(f'Created alignment file is invalid: {alignment_file}')
                    return None
            else:
                self.logger.error(f'minimap2 failed with return code {return_code}: {stderr_output}')
                return None
        except Exception as e:
            self.logger.error(f'Error running minimap2: {e}')
            return None

    def _parse_paf_alignments(self, paf_file: str) -> List[AssemblyAlignment]:
        """
        Parse PAF alignment file with robust error handling.
        FIXED: Complete PAF parsing with validation and error recovery.
        """
        alignments = []
        try:
            with open(paf_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        alignment = self._parse_paf_line(line.strip(), line_num)
                        if alignment:
                            alignments.append(alignment)
                    except Exception as e:
                        self.logger.debug(f'Error parsing PAF line {line_num}: {e}')
                        continue
        except Exception as e:
            self.logger.error(f'Error reading PAF file {paf_file}: {e}')
        self.logger.info(f'Parsed {len(alignments)} alignments from {paf_file}')
        return alignments

    def _parse_paf_line(self, line: str, line_num: int) -> Optional[AssemblyAlignment]:
        """Parse a single PAF line into AssemblyAlignment object."""
        if not line or line.startswith('#'):
            return None
        fields = line.split('\t')
        if len(fields) < 12:
            raise ValueError(f'Insufficient fields in PAF line: {len(fields)} < 12')
        try:
            alignment = AssemblyAlignment(query_name=fields[0], query_length=int(fields[1]), query_start=int(fields[2]), query_end=int(fields[3]), strand=fields[4], target_name=fields[5], target_length=int(fields[6]), target_start=int(fields[7]), target_end=int(fields[8]), residue_matches=int(fields[9]), alignment_block_length=int(fields[10]), mapping_quality=int(fields[11]))
            if alignment.query_start >= alignment.query_end:
                raise ValueError('Invalid query coordinates')
            if alignment.target_start >= alignment.target_end:
                raise ValueError('Invalid target coordinates')
            if alignment.residue_matches > alignment.alignment_block_length:
                raise ValueError('Residue matches exceed alignment length')
            return alignment
        except (ValueError, IndexError) as e:
            raise ValueError(f'Error parsing PAF fields: {e}')

    def _validate_single_variant(self, variant: pd.Series, alignment_files: Dict[str, str], output_dir: Path) -> ValidationResult:
        """
        Validate a single structural variant against assembly alignments.
        FIXED: Comprehensive validation logic with confidence scoring.
        """
        chrom = variant.get('CHROM', variant.get('chrom', ''))
        start = variant.get('MIN_START', variant.get('START', variant.get('start', 0)))
        end = variant.get('MAX_END', variant.get('END', variant.get('end', 0)))
        variant_id = f'{chrom}:{start}-{end}'
        all_validation_evidence = []
        for alignment_name, alignment_file in alignment_files.items():
            try:
                evidence = self._validate_against_single_alignment(chrom, start, end, alignment_file, alignment_name)
                all_validation_evidence.extend(evidence)
            except Exception as e:
                self.logger.debug(f'Error validating {variant_id} against {alignment_name}: {e}')
                continue
        return self._synthesize_validation_evidence(variant_id, all_validation_evidence)

    def _validate_against_single_alignment(self, chrom: str, start: int, end: int, alignment_file: str, alignment_name: str) -> List[Dict]:
        """Validate variant against a specific alignment file."""
        cache_key = alignment_file
        if cache_key not in self.alignment_cache:
            self.alignment_cache[cache_key] = self._parse_paf_alignments(alignment_file)
        alignments = self.alignment_cache[cache_key]
        evidence = []
        for alignment in alignments:
            if alignment.query_name != chrom:
                continue
            overlap_start = max(start, alignment.query_start)
            overlap_end = min(end, alignment.query_end)
            if overlap_end <= overlap_start:
                continue
            overlap_length = overlap_end - overlap_start
            variant_length = end - start
            overlap_fraction = overlap_length / variant_length
            if overlap_fraction < 0.5:
                continue
            validation_evidence = self._analyze_alignment_for_variant(alignment, start, end, overlap_fraction, alignment_name)
            if validation_evidence:
                evidence.append(validation_evidence)
        return evidence

    def _analyze_alignment_for_variant(self, alignment: AssemblyAlignment, variant_start: int, variant_end: int, overlap_fraction: float, alignment_name: str) -> Optional[Dict]:
        """Analyze how an alignment supports or contradicts a variant."""
        identity = alignment.identity
        coverage = overlap_fraction
        mapping_quality = alignment.mapping_quality
        variant_region_covered = alignment.query_start <= variant_start and alignment.query_end >= variant_end
        if variant_region_covered:
            if identity >= self.min_assembly_identity and mapping_quality >= 20:
                support_score = -1.0 * identity * coverage
                supports_variant = False
            else:
                support_score = 0.3 * (1 - identity) * coverage
                supports_variant = True
        else:
            support_score = 0.5 * coverage
            supports_variant = True
        return {'alignment_name': alignment_name, 'identity': identity, 'coverage': coverage, 'mapping_quality': mapping_quality, 'support_score': support_score, 'supports_variant': supports_variant, 'alignment_length': alignment.alignment_block_length, 'query_coverage': alignment.query_coverage, 'target_coverage': alignment.target_coverage}

    def _synthesize_validation_evidence(self, variant_id: str, evidence_list: List[Dict]) -> ValidationResult:
        """
        Synthesize evidence from multiple alignments into a validation result.
        FIXED: Proper evidence weighting and confidence calculation.
        """
        if not evidence_list:
            return ValidationResult(variant_id=variant_id, validated=False, confidence_score=0.0, supporting_alignments=0, alignment_identity=0.0, alignment_coverage=0.0, gap_analysis={}, validation_method='no_evidence')
        supporting_evidence = [e for e in evidence_list if e['supports_variant']]
        contradicting_evidence = [e for e in evidence_list if not e['supports_variant']]
        total_evidence = len(evidence_list)
        supporting_count = len(supporting_evidence)
        contradicting_count = len(contradicting_evidence)
        total_support_score = sum((e['support_score'] for e in evidence_list))
        avg_identity = np.mean([e['identity'] for e in evidence_list])
        avg_coverage = np.mean([e['coverage'] for e in evidence_list])
        if supporting_count > contradicting_count and total_support_score > 0:
            validated = True
            confidence = min(total_support_score / total_evidence, 1.0)
        elif contradicting_count > supporting_count and total_support_score < 0:
            validated = False
            confidence = min(abs(total_support_score) / total_evidence, 1.0)
        else:
            validated = total_support_score > 0
            confidence = 0.3
        if avg_identity < 0.8 or avg_coverage < 0.5:
            confidence *= 0.5
        return ValidationResult(variant_id=variant_id, validated=validated, confidence_score=confidence, supporting_alignments=total_evidence, alignment_identity=avg_identity, alignment_coverage=avg_coverage, gap_analysis={'supporting_alignments': supporting_count, 'contradicting_alignments': contradicting_count, 'total_support_score': total_support_score}, validation_method='assembly_alignment')

    def _add_validation_columns_placeholder(self, variants_df: pd.DataFrame) -> pd.DataFrame:
        """Add placeholder validation columns when validation cannot be performed."""
        validated_df = variants_df.copy()
        validation_columns = {'ASSEMBLY_VALIDATED': False, 'ASSEMBLY_CONFIDENCE': 0.0, 'ASSEMBLY_SUPPORT_ALIGNMENTS': 0, 'ASSEMBLY_IDENTITY': 0.0, 'ASSEMBLY_COVERAGE': 0.0, 'ASSEMBLY_METHOD': 'not_available'}
        for col, default_value in validation_columns.items():
            validated_df[col] = default_value
        return validated_df

    def _integrate_validation_results(self, variants_df: pd.DataFrame, validation_results: List[ValidationResult]) -> pd.DataFrame:
        """Integrate validation results into the variants DataFrame."""
        validated_df = variants_df.copy()
        validated_df['ASSEMBLY_VALIDATED'] = False
        validated_df['ASSEMBLY_CONFIDENCE'] = 0.0
        validated_df['ASSEMBLY_SUPPORT_ALIGNMENTS'] = 0
        validated_df['ASSEMBLY_IDENTITY'] = 0.0
        validated_df['ASSEMBLY_COVERAGE'] = 0.0
        validated_df['ASSEMBLY_METHOD'] = 'unknown'
        for i, result in enumerate(validation_results):
            if i < len(validated_df):
                validated_df.loc[i, 'ASSEMBLY_VALIDATED'] = result.validated
                validated_df.loc[i, 'ASSEMBLY_CONFIDENCE'] = result.confidence_score
                validated_df.loc[i, 'ASSEMBLY_SUPPORT_ALIGNMENTS'] = result.supporting_alignments
                validated_df.loc[i, 'ASSEMBLY_IDENTITY'] = result.alignment_identity
                validated_df.loc[i, 'ASSEMBLY_COVERAGE'] = result.alignment_coverage
                validated_df.loc[i, 'ASSEMBLY_METHOD'] = result.validation_method
        return validated_df

    def _log_validation_summary(self, validation_results: List[ValidationResult]):
        """Log comprehensive validation summary."""
        if not validation_results:
            self.logger.warning('No validation results to summarize')
            return
        total_variants = len(validation_results)
        validated_variants = sum((1 for r in validation_results if r.validated))
        high_confidence = sum((1 for r in validation_results if r.confidence_score > 0.8))
        medium_confidence = sum((1 for r in validation_results if 0.5 < r.confidence_score <= 0.8))
        low_confidence = sum((1 for r in validation_results if 0 < r.confidence_score <= 0.5))
        avg_confidence = np.mean([r.confidence_score for r in validation_results])
        avg_identity = np.mean([r.alignment_identity for r in validation_results if r.supporting_alignments > 0])
        avg_coverage = np.mean([r.alignment_coverage for r in validation_results if r.supporting_alignments > 0])
        self.logger.info('Assembly Validation Summary:')
        self.logger.info(f'  Total variants: {total_variants}')
        self.logger.info(f'  Validated variants: {validated_variants} ({validated_variants / total_variants * 100:.1f}%)')
        self.logger.info(f'  High confidence (>0.8): {high_confidence} ({high_confidence / total_variants * 100:.1f}%)')
        self.logger.info(f'  Medium confidence (0.5-0.8): {medium_confidence} ({medium_confidence / total_variants * 100:.1f}%)')
        self.logger.info(f'  Low confidence (0-0.5): {low_confidence} ({low_confidence / total_variants * 100:.1f}%)')
        self.logger.info(f'  Average confidence: {avg_confidence:.3f}')
        self.logger.info(f'  Average alignment identity: {avg_identity:.3f}')
        self.logger.info(f'  Average alignment coverage: {avg_coverage:.3f}')

    def cleanup_temporary_files(self, output_dir: Path):
        """Clean up temporary alignment files if requested."""
        try:
            temp_files = list(output_dir.glob('*.paf'))
            for temp_file in temp_files:
                if temp_file.stat().st_size > 1000000000:
                    self.logger.info(f'Removing large temporary file: {temp_file}')
                    temp_file.unlink()
        except Exception as e:
            self.logger.warning(f'Error cleaning up temporary files: {e}')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate structural variants via assembly alignment")
    parser.add_argument("--variants", required=True, help="Variants CSV to validate")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import pandas as pd
    from bear_genomics.config import load_fixed_config
    config = load_fixed_config(args.config)
    df = pd.read_csv(args.variants)
    validator = FixedAssemblyValidator(config)
    results = validator.validate_variants(df)
    results.to_csv(args.output, index=False)
    print(f"Wrote {len(results)} validated variants to {args.output}")
