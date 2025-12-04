#!/bin/bash

# Script to recompute segmentation masks for the 512x512 cropped images.
# This is needed because the original masks were computed on uncropped images
# with different aspect ratios (e.g., 512x910, 512x682).

set -e

# Directories
ORIGINAL_DIR="/scratch/aaa_exchange/AWARE/FINAL_SPLITS/train/images"
STATS_DIR="/scratch/aaa_exchange/AWARE/STATS"
CACHE_DIR="/scratch/${USER}/models"

# Datasets to process
DATASETS=("ACDC" "BDD100k" "BDD10k" "IDD-AW" "MapillaryVistas" "OUTSIDE15k")

echo "========================================="
echo "Recomputing Segmentation Masks (512x512)"
echo "========================================="
echo "Original Dir: $ORIGINAL_DIR"
echo "Stats Dir: $STATS_DIR"
echo ""

# Activate virtual environment
source venv/bin/activate

# Step 1: Backup and remove old segmentation masks
echo "Step 1: Removing old segmentation masks..."
for dataset in "${DATASETS[@]}"; do
    seg_dir="${STATS_DIR}/${dataset}/original_segmentation"
    if [ -d "$seg_dir" ]; then
        count=$(find "$seg_dir" -name "*.npy" | wc -l)
        echo "  Removing $count masks from $seg_dir"
        rm -rf "$seg_dir"
    fi
done
echo ""

# Step 2: Recompute segmentation masks for each dataset
echo "Step 2: Computing new segmentation masks..."
python3 - <<'EOF'
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project to path
sys.path.insert(0, '.')

from semantic_consistency import SegFormerEvaluator
from utils.image_io import find_image_files

# Configuration
ORIGINAL_DIR = Path("/scratch/aaa_exchange/AWARE/FINAL_SPLITS/train/images")
STATS_DIR = Path("/scratch/aaa_exchange/AWARE/STATS")
CACHE_DIR = Path(f"/scratch/{__import__('os').environ.get('USER', 'unknown')}/models")
DATASETS = ["ACDC", "BDD100k", "BDD10k", "IDD-AW", "MapillaryVistas", "OUTSIDE15k"]
BATCH_SIZE = 16

# Initialize SegFormer
print("Initializing SegFormer (b5) for segmentation...")
evaluator = SegFormerEvaluator(
    model_name="nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
    device="cuda",
    cache_dir=CACHE_DIR
)

for dataset in DATASETS:
    # Find all images for this dataset (search recursively through weather subdirs)
    dataset_dir = ORIGINAL_DIR / dataset
    if not dataset_dir.exists():
        print(f"Skipping {dataset}: directory not found")
        continue
    
    image_paths = find_image_files(dataset_dir)
    if not image_paths:
        print(f"Skipping {dataset}: no images found")
        continue
    
    # Create output directory
    seg_dir = STATS_DIR / dataset / "original_segmentation"
    seg_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing {dataset}: {len(image_paths)} images -> {seg_dir}")
    
    # Process in batches
    for batch_start in tqdm(range(0, len(image_paths), BATCH_SIZE), desc=dataset):
        batch_paths = image_paths[batch_start:batch_start + BATCH_SIZE]
        
        try:
            masks = evaluator.segment_images(batch_paths, use_cache=False)
            for path, mask in zip(batch_paths, masks):
                output_path = seg_dir / f"{path.stem}.npy"
                np.save(output_path, mask.astype(np.uint8))
        except Exception as e:
            print(f"  Error processing batch: {e}")
            continue

print("\nSegmentation precomputation complete!")

# Verify results
print("\n=== Verification ===")
for dataset in DATASETS:
    seg_dir = STATS_DIR / dataset / "original_segmentation"
    if seg_dir.exists():
        masks = list(seg_dir.glob("*.npy"))
        if masks:
            sample = np.load(masks[0])
            print(f"{dataset}: {len(masks)} masks, shape={sample.shape}")
EOF

echo ""
echo "========================================="
echo "Segmentation recomputation complete!"
echo "========================================="
