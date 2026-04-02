# Bear Genomics SV Pipeline

A pipeline for structural variant (deletion) analysis in bear genomes. Designed for cross-species comparison between brown bear (*Ursus arctos*) and polar bear (*Ursus maritimus*), with support for GFF-based gene annotation and RepeatMasker-based repeat annotation.

---

## Supplementary Methods: Upstream Tools and Parameters

The pipeline takes per-sample VCF files as input. The following tools and parameters were used to generate those VCFs from raw reads.

### 1. Sequence Alignment: BWA-MEM

Whole-genome sequencing reads for all samples were aligned to both the polar bear (GCF_017311325.1, ASM1731132v1) and brown bear (GCF_023065955.2, UrsArc2.0) reference genomes using BWA-MEM. Alignments were converted to sorted, indexed BAM files using SAMtools.

| Parameter | Value |
|-----------|-------|
| Software | BWA-MEM (Li, 2013) |
| Command | `bwa mem -M -t 14 <reference> <R1.fastq> <R2.fastq>` |
| `-M` | Mark shorter split hits as secondary |
| `-t 14` | 14 threads |
| Post-processing | SAMtools (Li et al., 2009) |
| SAM → BAM | `samtools view -S -b -@ 14` |
| Sort | `samtools sort -m 4G -@ 14` |
| Index | `samtools index` |

### 2. Variant Calling: GROM

Each sample was analyzed against both reference genomes independently.

| Parameter | Value |
|-----------|-------|
| Software | GROM (Smith et al., 2017) |
| Command | `GROM -i <sorted.bam> -r <reference> -o <output.vcf> -M -e 0.1` |
| `-i` | Input: sorted BAM file |
| `-r` | Reference genome (FASTA) |
| `-o` | Output: VCF file |
| `-M` | Turn on duplicate read filtering |
| `-e 0.1` | Probability score threshold for insertions |
| References | Polar: `GCF_017311325.1_ASM1731132v1_genomic.fna` / Brown: `GCF_023065955.2_UrsArc2.0_genomic.fna` |

### 3. Variant Calling: Pindel

Pindel was run on each sample independently and output was converted to VCF format using pindel2vcf.

| Parameter | Value |
|-----------|-------|
| Software | Pindel (Ye et al., 2009) |
| Command | `pindel -f <reference> -i <config> -c ALL -o <output> -T 16` |
| `-f` | Reference genome (FASTA) |
| `-i` | Configuration file (BAM path, insert size, sample name) |
| `-c ALL` | Analyze all chromosomes |
| `-T 16` | 16 threads |
| VCF conversion | `pindel2vcf -p <pindel_D> -r <reference> -R <ref_name> -d <date> -v <output.vcf>` |
| Output types | Deletions (`*_D`) and short insertions (`*_SI`) |

**Pindel configuration file format** (tab-delimited, one sample per line):

```
<path_to_sorted.bam>    <insert_size>    <sample_name>
```

Insert size estimation:

```bash
samtools stats <sorted.bam> | grep "insert size average"
```

### 4. Repeat Annotation: RepeatMasker

RepeatMasker is run on the reference genome to identify repetitive elements. The pipeline consumes a BED file derived from the RepeatMasker output.

```bash
# Run RepeatMasker on the reference genome
RepeatMasker -species <species> -pa 16 -xsmall -gff <reference.fna>

# Convert the .out file to BED format
awk 'NR>3 {print $5"\t"$6"\t"$7"\t"$10"\t"$11"\t"$9}' <reference.fna.out> repeats.bed
```

Set `repeatmasker_bed: /path/to/repeats.bed` in your config YAML. This field is optional — if absent or the file does not exist, repeat annotation is skipped.

---

## Pipeline Overview

```
VCF files (brown + polar)
        │
        ▼
  VCF Parsing & Filtering          vcf_parsing.py
        │
        ▼
  DBSCAN Clustering                clustering.py
        │
        ▼
  Coordinate Normalization         (AVG_START/AVG_END → START/END as integers)
        │
        ▼
  Species Comparison               species_comparison.py
        │    └── brown_specific / polar_specific / overlapping
        ▼
  Repeat Annotation (non-ref set)  gene_annotation.py  ← RepeatMasker BED
        │
        ▼
  Gene Annotation (non-ref set)    gene_annotation.py  ← GFF
        │
        ▼
  CSV / TSV / JSON output          analysis_pipeline.py
```

**Reference-aware annotation:** which species-specific set gets annotated depends on `reference_genome` in the config.

| `reference_genome` | Annotated dataset  |
|--------------------|--------------------|
| `polar`            | `brown_specific`   |
| `brown`            | `polar_specific`   |

---

## Repository Structure

```
bear_genomics_pipeline/
├── bear_genomics/
│   ├── __init__.py            Public API
│   ├── config.py              Configuration dataclasses and YAML loader
│   ├── helpers.py             Shared interval/coordinate utilities
│   ├── vcf_parsing.py         VCF ingestion and normalisation
│   ├── clustering.py          DBSCAN-based SV clustering
│   ├── gene_annotation.py     GFF gene + RepeatMasker repeat annotation
│   ├── assembly_validation.py Assembly alignment validation (minimap2)
│   ├── species_comparison.py  Cross-species overlap analysis
│   └── analysis_pipeline.py  End-to-end orchestration and CLI entry point
├── configs/
│   ├── polar_ref_template.yaml   Template for polar-reference runs
│   └── brown_ref_template.yaml   Template for brown-reference runs
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-org>/bear_genomics_pipeline.git
cd bear_genomics_pipeline
```

### 2. Create a conda environment

```bash
conda create -n bear_env python=3.10
conda activate bear_env
pip install -r requirements.txt
```

Core dependencies: `numpy`, `pandas`, `scipy`, `scikit-learn`, `pyyaml`

### 3. External tools (HPC modules or PATH)

- `minimap2` — required for assembly validation

---

## Configuration

Copy a template config and edit the cluster-specific paths:

```bash
cp configs/polar_ref_template.yaml my_run.yaml
```

### Key parameters

| Parameter | Description |
|-----------|-------------|
| `reference_genome` | `"polar"` or `"brown"` — drives which set gets annotated |
| `brown_vcf_folder` | Directory containing per-sample brown bear VCF files |
| `polar_vcf_folder` | Directory containing per-sample polar bear VCF files |
| `gff_file` | GFF3 annotation file for the reference genome |
| `repeatmasker_bed` | RepeatMasker output in BED format (optional) |
| `output_folder` | Where results are written |
| `min_qual` | Minimum variant quality score (default: 20) |
| `min_size` / `max_size` | Deletion size range in bp |
| `brown_min_samples` | Min samples a deletion must appear in (brown) |
| `polar_min_samples` | Min samples a deletion must appear in (polar) |
| `overlap_threshold` | Reciprocal overlap fraction for species comparison |
| `threads` / `max_workers` | Parallelism for VCF parsing |

---

## Usage

### Local / interactive

```bash
python -m bear_genomics.analysis_pipeline --config my_run.yaml
```

### Dry-run (validates config and input files, no analysis)

```bash
python -m bear_genomics.analysis_pipeline --config my_run.yaml --dry-run
```

---

## Output Structure

```
output_folder/
├── brown/                     Clustered brown bear deletions
├── polar/                     Clustered polar bear deletions
├── species_comparison/        brown_specific, polar_specific, overlapping CSVs
├── annotation/                Gene-annotated non-reference deletions
└── logs/                      analysis.log
```

Output formats are configurable via `output_formats` in the YAML (`csv`, `tsv`, `json`).

---

## Module Descriptions

| Module | Responsibility |
|--------|----------------|
| `config.py` | Loads and validates YAML config; creates output directory structure |
| `helpers.py` | Interval arithmetic, coordinate utilities |
| `vcf_parsing.py` | Reads per-sample VCFs, applies quality/size filters, normalises fields |
| `clustering.py` | DBSCAN clustering of variants across samples; computes cluster statistics |
| `gene_annotation.py` | Annotates deletions against GFF features and RepeatMasker BED |
| `assembly_validation.py` | Validates deletion boundaries via minimap2 assembly alignment |
| `species_comparison.py` | Reciprocal-overlap comparison between species; produces species-specific sets |
| `analysis_pipeline.py` | Orchestrates all steps; CLI entry point |

---

## Notes

- VCF files are expected to be per-sample (one sample per file), in a flat directory per species.
- The `clustering_quality_filters` block controls DBSCAN cluster acceptance thresholds; set values to `0` to disable individual filters.
