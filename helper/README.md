# Helper Scripts

This directory contains utility scripts for preprocessing and manifest generation.

## Manifest Generation Scripts

### `generate_manifest.py`

**Unified manifest generation script** for image-to-image translation evaluation. Creates CSV and JSON manifests that map generated images to their original counterparts, with support for both single method and batch processing.

#### Modes

- **Single method**: `--generated <path>` - Generate manifest for one specific method
- **All methods**: `--all` - Generate manifests for all methods in GENERATED_IMAGES
- **Missing only**: `--all-missing` - Generate manifests only for methods without existing manifests

#### Features

- **Multi-structure support**: Handles various directory layouts:
  - `domain/dataset` (e.g., `foggy/ACDC/`)
  - `dataset/domain` (e.g., `ACDC/foggy/`)
  - Flat domain structure (e.g., `foggy/image.png`)
- **Domain normalization**: Maps varied naming conventions to canonical domains
- **Task type detection**: Distinguishes generation tasks (clear→adverse) from restoration tasks (adverse→clear)
- **Dataset mapping**: Automatically identifies source datasets (ACDC, BDD100k, etc.)
- **Filename normalization**: Removes generation suffixes (_fake, _translated, etc.) and dataset prefixes

#### Canonical Domains

**Generation domains** (clear → adverse weather):
- `snowy`, `rainy`, `foggy`, `night`, `cloudy`, `dawn_dusk`, `clear_day`

**Restoration domains** (adverse → clear):
- `derained` (source: rainy)
- `dehazed` (source: foggy)
- `desnowed` (source: snowy)
- `night2day` (source: night)

#### Usage

```bash
# Single method mode
python helper/generate_manifest.py \
  --generated /path/to/GENERATED_IMAGES/CUT \
  --original /path/to/original/images \
  --target /path/to/target/domain \
  -o ./manifests/CUT

# Process all methods
python helper/generate_manifest.py --all

# Process only methods without existing manifests
python helper/generate_manifest.py --all-missing

# Custom paths with all mode
python helper/generate_manifest.py --all \
  --generated-base /path/to/GENERATED_IMAGES \
  --original /path/to/original/images \
  --target /path/to/target/domain \
  --output ./manifests

# Process specific methods only
python helper/generate_manifest.py --all --methods cycleGAN stargan_v2

# Dry run (show what would be done)
python helper/generate_manifest.py --all --dry-run -v
```

#### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--generated` | - | Generated images directory (single method mode) |
| `--generated-base` | `/scratch/.../GENERATED_IMAGES` | Base directory containing method subdirectories (--all modes) |
| `--original` | `/scratch/.../FINAL_SPLITS/train/images` | Directory containing original source images |
| `--target` | `/scratch/.../AWACS/train` | Target domain images for FID reference |
| `-o, --output` | (generated directory) | Output directory for manifest files |
| `--all` | - | Process all methods in generated-base |
| `--all-missing` | - | Process only methods without existing manifests |
| `--methods` | (all) | Specific methods to process (with --all modes) |
| `-v, --verbose` | False | Show detailed progress |
| `--dry-run` | False | Preview without writing files |

#### Output

For each method, creates:
- `manifest.csv` - Tabular format with columns: `generated_path`, `original_path`, `target_path`, `domain`, `dataset`, `task_type`
- `manifest.json` - Structured format with statistics per domain and dataset

---

### `generate_manifest_report.py`

Generates comprehensive reports from manifest files, summarizing statistics across all methods.

#### Features

- **Task type separation**: Distinguishes generation vs. restoration methods
- **Multiple output formats**: Text (ASCII tables), Markdown, JSON
- **Aggregate statistics**: Totals by domain, dataset, and method
- **Visual progress bars**: Match rate visualization in text format

#### Usage

```bash
# Print text report to stdout
python helper/generate_manifest_report.py

# Save to file
python helper/generate_manifest_report.py -o manifests/REPORT.txt

# Generate all formats
python helper/generate_manifest_report.py -o manifests/REPORT --format all
# Creates: REPORT.txt, REPORT.md, REPORT.json

# Markdown format for documentation
python helper/generate_manifest_report.py -o manifests/REPORT.md --format markdown

# JSON for programmatic access
python helper/generate_manifest_report.py -o manifests/REPORT.json --format json
```

#### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--manifest-dir` | `./manifests` | Directory containing manifest subdirectories |
| `-o, --output` | (stdout) | Output file path |
| `--format` | `text` | Output format: `text`, `markdown`, `json`, or `all` |
| `--detailed` | False | Include detailed dataset breakdown per method |

#### Report Contents

- **Executive Summary**: Total counts and match rates by task type
- **Domain Summary**: Statistics per weather domain
- **Dataset Summary**: Statistics per source dataset
- **Method Details**: Per-method breakdown with visual match rate bars

#### Example Output (Markdown)

| Method | Total | Matched | Rate |
|--------|------:|--------:|-----:|
| cycleGAN | 187,398 | 187,398 | 100.0% |
| stargan_v2 | 191,400 | 191,400 | 100.0% |
| ... | ... | ... | ... |

---

## Workflow Example

```bash
# 1. Generate manifests for all methods
python helper/generate_all_manifests.py \
  --generated-base /path/to/GENERATED_IMAGES \
  --original /path/to/originals \
  --output-dir ./manifests \
  -v

# 2. Generate comprehensive report
python helper/generate_manifest_report.py \
  --manifest-dir ./manifests \
  -o ./manifests/REPORT \
  --format all

# 3. Use manifests for evaluation
python evaluate_generation.py \
  --manifest ./manifests/cycleGAN/manifest.csv \
  --metrics fid ssim lpips \
  --output results/cycleGAN_evaluation.json
```

---

### `summarize_results.py`

Collects and summarizes evaluation results from multiple image generation methods, computing a Composite Quality Score (CQS) for ranking different models.

#### Features

- **Multi-method aggregation**: Combines results from multiple evaluation JSON files
- **Composite Quality Score (CQS)**: Weighted metric combining FID, LPIPS, SSIM, PSNR, and semantic metrics
- **Ranking and comparison**: Ranks methods by CQS and provides statistical comparisons
- **Per-domain breakdown**: Shows metrics broken down by weather domains
- **Flexible input**: Accepts directory of results or individual JSON files

#### Usage

```bash
# Summarize all results in a directory
python helper/summarize_results.py --results-dir ./results

# Summarize specific result files
python helper/summarize_results.py \
  --results-files result1.json result2.json result3.json \
  --output summary.json

# Generate detailed report with rankings
python helper/summarize_results.py \
  --results-dir ./evaluation_results \
  --output ./reports/method_comparison.json \
  -v
```

#### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--results-dir` | None | Directory containing evaluation result JSON files |
| `--results-files` | None | Specific JSON result files to summarize |
| `--output` | `summary.json` | Output file for summary report |
| `-v, --verbose` | False | Show detailed progress and statistics |

#### Output

Generates a JSON report containing:
- **Method rankings**: Sorted by Composite Quality Score (CQS)
- **Per-method metrics**: FID, LPIPS, SSIM, PSNR, semantic scores
- **Statistical analysis**: Means, standard deviations, confidence intervals
- **Domain breakdowns**: Metrics per weather domain for each method

#### CQS Formula

The Composite Quality Score combines multiple metrics with weights:
- FID: 40% (lower is better)
- LPIPS: 25% (lower is better)
- SSIM: 15% (higher is better)
- PSNR: 10% (higher is better)
- Semantic mIoU: 10% (higher is better)

---

### `visualize_results.py`

Generates comprehensive diagrams and visualizations from PRISM evaluation results using seaborn and matplotlib. Creates plots for method comparison, metric distributions, correlations, and domain-wise analysis.

#### Features

- **Method comparison plots**: Bar charts comparing methods across all metrics
- **Metric distributions**: Box plots showing value distributions for each metric
- **Correlation analysis**: Heatmaps showing relationships between different metrics
- **Domain-wise analysis**: Heatmaps showing metrics across weather domains and methods
- **Radar plots**: Multi-dimensional comparison of methods across all metrics
- **Scatter plot matrices**: Pairwise relationships between metrics with histograms
- **Automated report generation**: Summary of created visualizations and statistics

#### Usage

```bash
# Visualize all results in a directory
python helper/visualize_results.py --results-dir ./results --output-dir ./plots

# Visualize specific result files
python helper/visualize_results.py \
  --results-files result1.json result2.json \
  --output-dir ./visualizations

# Generate plots with verbose output
python helper/visualize_results.py \
  --results-dir ./evaluation_results \
  --output-dir ./plots \
  -v
```

#### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--results-dir` | None | Directory containing evaluation result JSON files |
| `--results-files` | None | Specific JSON result files to visualize |
| `--output-dir` | `plots` | Output directory for generated plots |
| `-v, --verbose` | False | Show detailed progress |

#### Generated Plots

- **method_comparison.png**: Bar charts comparing methods across FID, LPIPS, SSIM, PSNR, and semantic metrics
- **metric_distributions.png**: Box plots showing distribution of each metric across methods
- **metric_correlations.png**: Heatmap showing correlations between different metrics
- **method_radar_comparison.png**: Radar plot for multi-metric method comparison (normalized)
- **metric_scatter_matrix.png**: Scatter plots and histograms showing relationships between metrics
- **{metric}_domain_comparison.png**: Domain-wise heatmaps for each metric (one per metric)
- **visualization_summary.txt**: Text summary of generated visualizations and statistics

#### Dependencies

Requires matplotlib and seaborn:
```bash
pip install matplotlib seaborn pandas numpy
```

---

## Other Helper Scripts

| Script | Description |
|--------|-------------|
| `summarize_results.py` | Aggregate and rank evaluation results with Composite Quality Score (CQS) |
| `visualize_results.py` | Generate comprehensive diagrams and visualizations from evaluation results |
| `adjust_bounding_boxes.py` | Adjust annotation bounding boxes for resized images |
| `center_crop_images.py` | Center-crop images to target dimensions |
| `count_images_per_folder.py` | Count images in directory hierarchies |
| `flatten_domain_hierarchy.py` | Reorganize nested domain/dataset structures |
| `test_lpips_extraction.py` | Test LPIPS feature extraction |

**Note**: `prepare_evaluation_manifest.py` and `generate_all_manifests.py` have been unified into `generate_manifest.py`.
