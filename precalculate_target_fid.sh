#!/bin/bash

# Script to precalculate FID statistics for TARGET domain images (AWACS).
# These stats are used as the reference distribution for FID computation
# when evaluating generated images.
#
# The stats will be saved as:
#   STATS_DIR/cloudy_fid.npz
#   STATS_DIR/dawn_dusk_fid.npz

# Default config: can be overridden by CLI args
TARGET_DIR="/scratch/aaa_exchange/AWARE/AWACS/train"
STATS_DIR="/scratch/aaa_exchange/AWARE/STATS"
CACHE_DIR="/scratch/chge7185/models"
IMAGE_SIZE=299
BATCH_SIZE=32
VERBOSE=1
FORCE=0
ONLY_DOMAIN=""

set -euo pipefail

echo "========================================="
echo "Precalculating Target Domain FID Stats"
echo "========================================="
echo "Target Dir: $TARGET_DIR"
echo "Stats Dir: $STATS_DIR"
echo "Cache Dir: $CACHE_DIR"
echo ""

# Activate virtual env if present
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

mkdir -p "$STATS_DIR"

# Parse optional args
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --target-dir) TARGET_DIR="$2"; shift 2;;
        --stats-dir) STATS_DIR="$2"; shift 2;;
        --batch-size) BATCH_SIZE="$2"; shift 2;;
        --force) FORCE=1; shift;
        ;;
        --domain) ONLY_DOMAIN="$2"; shift 2;;
        -h|--help) echo "Usage: $0 [--target-dir DIR] [--stats-dir DIR] [--batch-size N] [--domain name] [--force]"; exit 0;;
        *) echo "Unknown arg: $1"; shift;;
    esac
done

# Discover domains (subdirectories)
domains=()
if [ -n "$ONLY_DOMAIN" ]; then
    # Only process the specified domain
    domains=("$ONLY_DOMAIN")
else
    while IFS= read -r -d '' d; do
        domains+=("$(basename "$d")")
    done < <(find "$TARGET_DIR" -mindepth 1 -maxdepth 1 -type d -print0)
fi

if [ ${#domains[@]} -eq 0 ]; then
    echo "Error: No domain subdirectories found in $TARGET_DIR"
    exit 1
fi

echo "Found domains: ${domains[*]}"
echo ""

# Loop over domains and compute FID stats
for domain in "${domains[@]}"; do
    domain_dir="$TARGET_DIR/$domain"
    output_file="$STATS_DIR/${domain}_fid.npz"

    if [ -f "$output_file" ] && [ "$FORCE" -ne 1 ]; then
        echo "[$domain] FID stats already exist: $output_file (skipping)"
        echo "  Use --force to regenerate"
        echo ""
        continue
    fi

    echo "[$domain] Computing FID statistics..."
    echo "  Input: $domain_dir"
    echo "  Output: $output_file"

    python3 - <<PY
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, '.')
from evaluate_generation import compute_fid_statistics, save_fid_stats

_domain_dir = Path("$domain_dir")
_output_file = Path("$output_file")
_image_size = ($IMAGE_SIZE, $IMAGE_SIZE)
_batch_size = $BATCH_SIZE
_device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'

print(f'  Device: {_device}')

try:
    fid_stats = compute_fid_statistics(_domain_dir, _image_size, _batch_size, _device)
    save_fid_stats(fid_stats['mu'], fid_stats['sigma'], fid_stats['n'], _output_file)
    print(f'  Saved to: {_output_file}')
except Exception as e:
    print('  ERROR: Failed to compute FID stats:', e)
    raise
PY

    if [ $? -eq 0 ]; then
        echo "[$domain] Done."
    else
        echo "[$domain] ERROR: Failed to compute FID stats"
    fi
    echo ""
done

echo "========================================="
echo "Precalculation complete."
echo "========================================="

echo "FID stats files:"
ls -lh "$STATS_DIR"/*_fid.npz 2>/dev/null || echo "No FID stats files found"
