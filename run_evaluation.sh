#!/bin/bash

# Script to run evaluation using precalculated statistics.
# This script evaluates generated images against original images using
# precomputed FID statistics, segmentation maps, and LPIPS features.

# Directories
# Generated: The generated/translated images to evaluate
GENERATED_DIR="/scratch/aaa_exchange/AWARE/GENERATED_IMAGES"

# Original: The original source images (for paired metrics like SSIM, LPIPS, PSNR)
ORIGINAL_DIR="/scratch/aaa_exchange/AWARE/FINAL_SPLITS/train/images"

# Target: The target domain images (only needed if stats aren't precomputed)
TARGET_DIR="/scratch/aaa_exchange/AWARE/AWACS/train"

# Statistics directory (contains precomputed FID stats, segmentation, LPIPS features)
STATS_DIR="/scratch/aaa_exchange/AWARE/STATS"

# Output file for evaluation results
OUTPUT_FILE="evaluation_results.json"

# Cache directory for models
CACHE_DIR="/scratch/chge7185/models"

echo "Starting evaluation..."
echo "Generated Dir: $GENERATED_DIR"
echo "Original Dir: $ORIGINAL_DIR"
echo "Target Dir: $TARGET_DIR"
echo "Stats Dir: $STATS_DIR"
echo "Output: $OUTPUT_FILE"
echo "Cache Dir: $CACHE_DIR"

source venv/bin/activate

# Run evaluation
# --generated: Generated/translated images to evaluate
# --original: Original domain folder (for paired metrics)
# --target: Target domain folder (used if FID stats not precomputed)
# --stats-dir: Directory with precomputed statistics
# --per-domain: Calculate metrics per domain (subfolder)
# --semantic-consistency: Compute semantic consistency via SegFormer
# --verbose: Show progress
python3 evaluate_generation.py \
    --generated "$GENERATED_DIR" \
    --original "$ORIGINAL_DIR" \
    --target "$TARGET_DIR" \
    --stats-dir "$STATS_DIR" \
    --metrics fid ssim lpips is \
    --semantic-consistency \
    --per-domain \
    --device auto \
    --cache-dir "$CACHE_DIR" \
    --batch-size 16 \
    --output "$OUTPUT_FILE" \
    --verbose

echo "Evaluation complete. Results saved to $OUTPUT_FILE"
