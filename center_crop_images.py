#!/usr/bin/env python3
"""
Center crop all images in a directory and subdirectories to 512x512 pixels.

Usage:
    python center_crop_images.py --input /path/to/images --output /path/to/output
    python center_crop_images.py --input /path/to/images --inplace  # Overwrites originals
"""

import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def center_crop(image: Image.Image, target_size: int = 512) -> Image.Image:
    """
    Center crop image to target_size x target_size.
    
    If image is smaller than target_size, it will be padded with white.
    
    Args:
        image: PIL Image to crop
        target_size: Target size for width and height
        
    Returns:
        Center cropped image
    """
    w, h = image.size
    
    # If image is smaller than target, pad with white
    if w < target_size or h < target_size:
        new_w = max(w, target_size)
        new_h = max(h, target_size)
        new_img = Image.new('RGB', (new_w, new_h), (255, 255, 255))
        new_img.paste(image, ((new_w - w) // 2, (new_h - h) // 2))
        image = new_img
        w, h = image.size
    
    left = (w - target_size) // 2
    top = (h - target_size) // 2
    right = left + target_size
    bottom = top + target_size
    
    return image.crop((left, top, right, bottom))


def process_images(
    input_dir: str,
    output_dir: str = None,
    inplace: bool = False,
    crop_size: int = 512,
    dry_run: bool = False,
):
    """
    Process all images in input_dir and subdirectories.
    
    Args:
        input_dir: Input directory containing images
        output_dir: Output directory (preserves subdirectory structure)
        inplace: If True, overwrite original images
        crop_size: Target crop size (default 512)
        dry_run: If True, only print what would be done
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory does not exist: {input_path}")
        return
    
    if inplace:
        output_path = input_path
    elif output_dir:
        output_path = Path(output_dir)
    else:
        print("Error: Must specify --output or --inplace")
        return
    
    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.rglob(f'*{ext}'))
        image_files.extend(input_path.rglob(f'*{ext.upper()}'))
    
    image_files = sorted(set(image_files))
    
    if not image_files:
        print(f"No images found in {input_path}")
        return
    
    print(f"Found {len(image_files)} images to process")
    print(f"Crop size: {crop_size}x{crop_size}")
    print(f"Output: {'in-place' if inplace else output_path}")
    
    if dry_run:
        print("\n[DRY RUN] Would process:")
        for img_path in image_files[:10]:
            rel_path = img_path.relative_to(input_path)
            out_file = output_path / rel_path
            print(f"  {img_path} -> {out_file}")
        if len(image_files) > 10:
            print(f"  ... and {len(image_files) - 10} more")
        return
    
    processed = 0
    errors = 0
    
    for img_path in tqdm(image_files, desc="Cropping images"):
        try:
            rel_path = img_path.relative_to(input_path)
            out_file = output_path / rel_path
            
            # Load image
            img = Image.open(img_path).convert('RGB')
            
            # Center crop
            cropped = center_crop(img, crop_size)
            
            # Save
            out_file.parent.mkdir(parents=True, exist_ok=True)
            cropped.save(str(out_file), quality=95)
            
            processed += 1
            
        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            errors += 1
    
    print(f"\nDone! Processed {processed} images with {errors} errors")


def main():
    parser = argparse.ArgumentParser(
        description="Center crop all images in a directory to 512x512 pixels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Crop to new directory
    python center_crop_images.py -i /path/to/images -o /path/to/output
    
    # Overwrite originals
    python center_crop_images.py -i /path/to/images --inplace
    
    # Custom crop size
    python center_crop_images.py -i /path/to/images -o /path/to/output --size 256
    
    # Dry run (see what would be done)
    python center_crop_images.py -i /path/to/images -o /path/to/output --dry-run
"""
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input directory containing images"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory (preserves subdirectory structure)"
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite original images instead of saving to output directory"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Crop size in pixels (default: 512)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be done, don't actually process"
    )
    
    args = parser.parse_args()
    
    if not args.output and not args.inplace:
        parser.error("Must specify --output or --inplace")
    
    if args.output and args.inplace:
        parser.error("Cannot use both --output and --inplace")
    
    process_images(
        input_dir=args.input,
        output_dir=args.output,
        inplace=args.inplace,
        crop_size=args.size,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
