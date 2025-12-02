#!/usr/bin/env python3
"""
Script to adjust bounding boxes and polygon annotations after image transformations.

The images have been:
1. Resized from 1280x720 to 910x512 (maintaining aspect ratio)
2. Center cropped from 910x512 to 512x512

This script adjusts the annotations accordingly.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from tqdm import tqdm


# Original and transformed image dimensions
ORIGINAL_WIDTH = 1280
ORIGINAL_HEIGHT = 720
RESIZED_WIDTH = 910
RESIZED_HEIGHT = 512
CROP_WIDTH = 512
CROP_HEIGHT = 512


def calculate_scale_factors() -> Tuple[float, float]:
    """Calculate scale factors for the resize operation."""
    scale_x = RESIZED_WIDTH / ORIGINAL_WIDTH
    scale_y = RESIZED_HEIGHT / ORIGINAL_HEIGHT
    return scale_x, scale_y


def calculate_crop_offset() -> Tuple[float, float]:
    """Calculate the offset for center cropping."""
    offset_x = (RESIZED_WIDTH - CROP_WIDTH) / 2
    offset_y = (RESIZED_HEIGHT - CROP_HEIGHT) / 2
    return offset_x, offset_y


def transform_point(x: float, y: float, scale_x: float, scale_y: float, 
                    offset_x: float, offset_y: float) -> Tuple[float, float]:
    """
    Transform a single point through resize and center crop.
    
    Args:
        x: Original x coordinate
        y: Original y coordinate
        scale_x: Scale factor for x
        scale_y: Scale factor for y
        offset_x: Crop offset for x
        offset_y: Crop offset for y
        
    Returns:
        Transformed (x, y) coordinates
    """
    # First resize
    new_x = x * scale_x
    new_y = y * scale_y
    
    # Then center crop (subtract offset)
    new_x = new_x - offset_x
    new_y = new_y - offset_y
    
    return new_x, new_y


def clip_to_bounds(x: float, y: float, width: int = CROP_WIDTH, 
                   height: int = CROP_HEIGHT) -> Tuple[float, float]:
    """Clip coordinates to image bounds."""
    x = max(0, min(x, width))
    y = max(0, min(y, height))
    return x, y


def transform_box2d(box: Dict[str, float], scale_x: float, scale_y: float,
                    offset_x: float, offset_y: float) -> Optional[Dict[str, float]]:
    """
    Transform a 2D bounding box.
    
    Returns None if the box is completely outside the cropped area.
    """
    x1, y1 = transform_point(box['x1'], box['y1'], scale_x, scale_y, offset_x, offset_y)
    x2, y2 = transform_point(box['x2'], box['y2'], scale_x, scale_y, offset_x, offset_y)
    
    # Check if box is completely outside the cropped area
    if x2 < 0 or x1 > CROP_WIDTH or y2 < 0 or y1 > CROP_HEIGHT:
        return None
    
    # Clip to bounds
    x1, y1 = clip_to_bounds(x1, y1)
    x2, y2 = clip_to_bounds(x2, y2)
    
    # Check if remaining box has valid dimensions
    if x2 <= x1 or y2 <= y1:
        return None
    
    return {
        'x1': round(x1, 6),
        'y1': round(y1, 6),
        'x2': round(x2, 6),
        'y2': round(y2, 6)
    }


def transform_poly2d(poly_list: List[Dict], scale_x: float, scale_y: float,
                     offset_x: float, offset_y: float) -> List[Dict]:
    """
    Transform polygon annotations.
    
    Note: This clips vertices to bounds but doesn't handle complex polygon clipping.
    """
    transformed_polys = []
    
    for poly in poly_list:
        vertices = poly.get('vertices', [])
        transformed_vertices = []
        
        for vertex in vertices:
            x, y = vertex[0], vertex[1]
            new_x, new_y = transform_point(x, y, scale_x, scale_y, offset_x, offset_y)
            # Clip to bounds
            new_x, new_y = clip_to_bounds(new_x, new_y)
            transformed_vertices.append([round(new_x, 6), round(new_y, 6)])
        
        transformed_poly = {
            'vertices': transformed_vertices,
            'types': poly.get('types', ''),
            'closed': poly.get('closed', False)
        }
        transformed_polys.append(transformed_poly)
    
    return transformed_polys


def transform_label(label: Dict[str, Any], scale_x: float, scale_y: float,
                    offset_x: float, offset_y: float) -> Optional[Dict[str, Any]]:
    """
    Transform a single label's annotations.
    
    Returns None if the annotation is completely outside the cropped area.
    """
    transformed_label = label.copy()
    
    # Transform box2d if present
    if 'box2d' in label:
        new_box = transform_box2d(label['box2d'], scale_x, scale_y, offset_x, offset_y)
        if new_box is None:
            return None  # Box is outside cropped area
        transformed_label['box2d'] = new_box
    
    # Transform poly2d if present
    if 'poly2d' in label:
        transformed_label['poly2d'] = transform_poly2d(
            label['poly2d'], scale_x, scale_y, offset_x, offset_y
        )
    
    return transformed_label


def transform_annotations(data: Dict[str, Any], silent: bool = False) -> Dict[str, Any]:
    """
    Transform all annotations in a JSON annotation file.
    
    Args:
        data: Original annotation data
        silent: If True, suppress output messages
        
    Returns:
        Transformed annotation data
    """
    scale_x, scale_y = calculate_scale_factors()
    offset_x, offset_y = calculate_crop_offset()
    
    if not silent:
        print(f"Transformation parameters:")
        print(f"  Scale: ({scale_x:.6f}, {scale_y:.6f})")
        print(f"  Crop offset: ({offset_x:.1f}, {offset_y:.1f})")
        print(f"  Original size: {ORIGINAL_WIDTH}x{ORIGINAL_HEIGHT}")
        print(f"  Resized size: {RESIZED_WIDTH}x{RESIZED_HEIGHT}")
        print(f"  Final size: {CROP_WIDTH}x{CROP_HEIGHT}")
    
    transformed_data = data.copy()
    transformed_labels = []
    removed_count = 0
    
    for label in data.get('labels', []):
        transformed_label = transform_label(label, scale_x, scale_y, offset_x, offset_y)
        if transformed_label is not None:
            transformed_labels.append(transformed_label)
        else:
            removed_count += 1
            if not silent:
                print(f"  Removed label (outside crop): {label.get('category', 'unknown')} (id: {label.get('id', 'N/A')})")
    
    transformed_data['labels'] = transformed_labels
    
    if not silent:
        print(f"\nLabels: {len(data.get('labels', []))} original -> {len(transformed_labels)} after crop ({removed_count} removed)")
    
    return transformed_data


def process_file(input_path: Path, output_path: Optional[Path] = None, 
                 silent: bool = False) -> Dict[str, Any]:
    """
    Process a single JSON annotation file.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file (optional)
        silent: If True, suppress output messages
        
    Returns:
        Transformed annotation data
    """
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    if not silent:
        print(f"\nProcessing: {input_path}")
    transformed_data = transform_annotations(data, silent=silent)
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(transformed_data, f, indent=2)
        if not silent:
            print(f"Saved to: {output_path}")
    
    return transformed_data


def process_directory(input_dir: Path, output_dir: Path, recursive: bool = True,
                      silent: bool = False) -> None:
    """
    Process all JSON files in a directory, optionally recursively.
    
    Args:
        input_dir: Input directory containing JSON files
        output_dir: Output directory for transformed files
        recursive: If True, scan subdirectories recursively
        silent: If True, show only progress bar
    """
    if recursive:
        json_files = list(input_dir.rglob('*.json'))
    else:
        json_files = list(input_dir.glob('*.json'))
    
    if not silent:
        print(f"Found {len(json_files)} JSON files in {input_dir}" + (" (recursive)" if recursive else ""))
    
    # Use progress bar in silent mode, or when processing many files
    iterator = tqdm(json_files, desc="Processing", unit="file") if silent else json_files
    
    for json_file in iterator:
        # Maintain directory structure: compute relative path from input_dir
        relative_path = json_file.relative_to(input_dir)
        output_file = output_dir / relative_path
        
        # Create parent directories if needed
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        process_file(json_file, output_file, silent=silent)


def main():
    parser = argparse.ArgumentParser(
        description='Adjust bounding boxes after resize and center crop transformation.'
    )
    parser.add_argument(
        'input', type=str,
        help='Input JSON file or directory containing JSON files'
    )
    parser.add_argument(
        '-o', '--output', type=str, default=None,
        help='Output JSON file or directory (default: print to stdout for single file)'
    )
    parser.add_argument(
        '--original-size', type=str, default='1280x720',
        help='Original image size as WxH (default: 1280x720)'
    )
    parser.add_argument(
        '--resized-size', type=str, default='910x512',
        help='Resized image size as WxH (default: 910x512)'
    )
    parser.add_argument(
        '--crop-size', type=str, default='512x512',
        help='Final crop size as WxH (default: 512x512)'
    )
    parser.add_argument(
        '--no-recursive', action='store_true',
        help='Disable recursive directory scanning (default: scan recursively)'
    )
    parser.add_argument(
        '-s', '--silent', action='store_true',
        help='Silent mode: show only progress bar, suppress detailed output'
    )
    
    args = parser.parse_args()
    
    # Parse size arguments
    global ORIGINAL_WIDTH, ORIGINAL_HEIGHT, RESIZED_WIDTH, RESIZED_HEIGHT, CROP_WIDTH, CROP_HEIGHT
    
    orig_w, orig_h = map(int, args.original_size.split('x'))
    ORIGINAL_WIDTH, ORIGINAL_HEIGHT = orig_w, orig_h
    
    resized_w, resized_h = map(int, args.resized_size.split('x'))
    RESIZED_WIDTH, RESIZED_HEIGHT = resized_w, resized_h
    
    crop_w, crop_h = map(int, args.crop_size.split('x'))
    CROP_WIDTH, CROP_HEIGHT = crop_w, crop_h
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        if args.output:
            output_path = Path(args.output)
            transformed = process_file(input_path, output_path, silent=args.silent)
        else:
            transformed = process_file(input_path, silent=args.silent)
            if not args.silent:
                print("\n--- Transformed JSON ---")
                print(json.dumps(transformed, indent=2))
    elif input_path.is_dir():
        if not args.output:
            raise ValueError("Output directory required when processing a directory")
        output_path = Path(args.output)
        process_directory(input_path, output_path, recursive=not args.no_recursive, 
                         silent=args.silent)
    else:
        raise FileNotFoundError(f"Input not found: {input_path}")


if __name__ == '__main__':
    main()
