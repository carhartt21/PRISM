#!/usr/bin/env bash
set -euo pipefail

# rerun_fid_evaluations.sh
# Script to rerun evaluations for models with missing FID values.
#
# The FID is missing because:
# 1. Some models don't have manifest files (need regeneration)
# 2. Some evaluations were run with folder-based domain discovery (need CSV-based)
# 3. albumentations vs albumentations_weather naming mismatch
#
# Models to rerun:
# - albumentations_weather (was incorrectly named 'albumentations' in stats)
# - flux_kontext (has manifest, but evaluation used wrong domain discovery)
# - step1x_new (has manifest, but evaluation used wrong domain discovery)
# - Qwen-Image-Edit (has manifest, but only 'clear_day' domain - needs manifest regen)
# - VisualCloze (missing manifest)
#
# This script will:
# 1. Regenerate manifests for all models
# 2. Submit evaluation jobs via submit_evaluations.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

source venv/bin/activate

# Models that need FID re-evaluation
# Note: albumentations_weather is the correct name (albumentations folder doesn't exist)
MODELS_TO_RERUN=(
    "albumentations_weather"
    "flux_kontext"
    "step1x_new"
    "Qwen-Image-Edit"
    "VisualCloze"
)

echo "========================================="
echo "FID Re-evaluation Script"
echo "========================================="
echo "Models to process: ${MODELS_TO_RERUN[*]}"
echo ""

# Directories
GENERATED_ROOT="/scratch/aaa_exchange/AWARE/GENERATED_IMAGES"
ORIGINAL_DIR="/scratch/aaa_exchange/AWARE/FINAL_SPLITS/train/images"
TARGET_DIR="/scratch/aaa_exchange/AWARE/AWACS/train"

echo "Step 1: Regenerate manifests for all models..."
echo ""

for model in "${MODELS_TO_RERUN[@]}"; do
    GENERATED_DIR="${GENERATED_ROOT}/${model}"
    
    if [ ! -d "$GENERATED_DIR" ]; then
        echo "WARNING: Generated directory not found for $model: $GENERATED_DIR"
        echo "Skipping..."
        continue
    fi
    
    # Try to write to generated dir first, fall back to local manifests/
    OUTPUT_DIR="$GENERATED_DIR"
    if [ ! -w "$OUTPUT_DIR" ]; then
        OUTPUT_DIR="manifests/${model}"
        mkdir -p "$OUTPUT_DIR"
        echo "Note: Writing manifest to local directory: $OUTPUT_DIR"
    fi
    
    echo "Generating manifest for: $model"
    python3 helper/generate_manifest.py \
        --generated "$GENERATED_DIR" \
        --original "$ORIGINAL_DIR" \
        --target "$TARGET_DIR" \
        -o "$OUTPUT_DIR" \
        --verbose
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Manifest generation failed for $model"
        exit 1
    fi
    echo ""
done

echo "Step 2: Submit evaluation jobs..."
echo ""

# Submit all models for evaluation with manifest regeneration flag
# The regenerate-manifest flag is important to ensure consistency
./scripts/submit_evaluations.sh "${MODELS_TO_RERUN[@]}"

echo ""
echo "========================================="
echo "Jobs submitted for: ${MODELS_TO_RERUN[*]}"
echo "Monitor progress with: bjobs -w"
echo "Check logs in: logs/"
echo "========================================="
