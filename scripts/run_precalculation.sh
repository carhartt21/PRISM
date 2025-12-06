#!/bin/bash

# Script to run evaluation in precalculate mode.
# This script precalculates FID statistics, segmentation maps, and LPIPS features
# for the specified target and original directories.

# Directories
# Target: The target domain images (Reference for FID)
TARGET_DIR="/scratch/aaa_exchange/AWARE/AWACS/train"

# Original: The original source images
ORIGINAL_DIR="/scratch/aaa_exchange/AWARE/FINAL_SPLITS/train/images"

# Output directory for statistics
STATS_DIR="/scratch/aaa_exchange/AWARE/STATS"

# Cache directory for models
CACHE_DIR="/scratch/chge7185/models"

echo "Starting precalculation..."
echo "Target Dir: $TARGET_DIR"
echo "Original Dir: $ORIGINAL_DIR"
echo "Stats Dir: $STATS_DIR"
echo "Cache Dir: $CACHE_DIR"
source venv/bin/activate
# Run precalculation
# --precalculate: Enables precalculate mode (FID stats, Segmentation, LPIPS)
# --target: Target domain folder
# --original: Original domain folder
# --verbose: Show progress
python3 evaluate_generation.py \
    --precalculate \
    --target "$TARGET_DIR" \
    --original "$ORIGINAL_DIR" \
    --stats-dir "$STATS_DIR" \
    --batch-size 32 \
    --metrics fid,lpips \
    --semantic-consistency \
    --per-domain \
    --device auto \
    --cache-dir "$CACHE_DIR" \
    --batch-size 8 \
    --verbose

echo "Precalculation complete."
