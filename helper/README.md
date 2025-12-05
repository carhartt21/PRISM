# Helper Scripts

This directory contains utility scripts for preprocessing and manifest generation.

## Manifest Generation Scripts

### `generate_all_manifests.py`

Generates manifest files for all method directories in a generated images folder. Manifests map generated images to their original source images and include metadata for evaluation.

#### Features

- **Multi-structure support**: Handles various directory layouts:
  - `domain/dataset` (e.g., `foggy/ACDC/`)
  - `dataset/domain` (e.g., `ACDC/foggy/`)
  - Flat domain structure (e.g., `foggy/image.png`)
- **Domain normalization**: Maps varied naming conventions to canonical domains
- **Task type detection**: Distinguishes generation tasks (clear→adverse) from restoration tasks (adverse→clear)
- **Dataset mapping**: Automatically identifies source datasets (ACDC, BDD100k, etc.)

#### Canonical Domains

**Generation domains** (clear → adverse weather):
- `snowy`, `rainy`, `foggy`, `night`, `cloudy`, `dawn_dusk`

**Restoration domains** (adverse → clear):
- `derained` (source: rainy)
- `dehazed` (source: foggy)
- `desnowed` (source: snowy)
- `night2day` (source: night)

#### Usage

```bash
# Process all methods with default paths
python helper/generate_all_manifests.py

# Custom paths
python helper/generate_all_manifests.py \
  --generated-base /path/to/GENERATED_IMAGES \
  --original /path/to/original/images \
  --target /path/to/target/domain \
  --output-dir ./manifests

# Process specific methods only
python helper/generate_all_manifests.py --methods cycleGAN stargan_v2

# Dry run (show what would be done)
python helper/generate_all_manifests.py --dry-run -v
```

#### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--generated-base` | `/scratch/.../GENERATED_IMAGES` | Base directory containing method subdirectories |
| `--original` | `/scratch/.../FINAL_SPLITS/train/images` | Directory containing original source images |
| `--target` | `/scratch/.../AWACS/train` | Target domain images for FID reference |
| `--output-dir` | (method directories) | Output directory for manifests |
| `--methods` | (all) | Specific methods to process |
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

## Other Helper Scripts

| Script | Description |
|--------|-------------|
| `prepare_evaluation_manifest.py` | Create evaluation manifests for specific method/domain combinations |
| `adjust_bounding_boxes.py` | Adjust annotation bounding boxes for resized images |
| `center_crop_images.py` | Center-crop images to target dimensions |
| `count_images_per_folder.py` | Count images in directory hierarchies |
| `flatten_domain_hierarchy.py` | Reorganize nested domain/dataset structures |
| `test_lpips_extraction.py` | Test LPIPS feature extraction |
