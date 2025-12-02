#!/usr/bin/env python3
"""
Preprocessing script to create evaluation manifests by matching generated images
to their original counterparts.

Handles:
- Generated images with '_fake' suffix (e.g., 'image_fake.png' -> 'image.jpg')
- Nested directory structures (e.g., 'domain/test_latest/images/')
- Multiple original image directories organized by dataset/domain

Outputs a CSV manifest for use with evaluate_generation.py --pairs csv --manifest
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

from tqdm import tqdm


# Supported image extensions
SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}


def find_image_files(directory: Path, recursive: bool = True) -> List[Path]:
    """Find all supported image files in a directory."""
    image_files = []
    glob_func = directory.rglob if recursive else directory.glob
    for ext in SUPPORTED_EXTENSIONS:
        image_files.extend(glob_func(f"*{ext}"))
        image_files.extend(glob_func(f"*{ext.upper()}"))
    return sorted(image_files)


def normalize_filename(filename: str) -> str:
    """
    Normalize a filename by removing common suffixes added during generation.
    
    Handles:
    - '_fake' suffix (CycleGAN style)
    - '_translated', '_output' suffixes
    - Preserves the base filename for matching
    """
    stem = Path(filename).stem
    
    # Remove common generation suffixes
    suffixes_to_remove = ['_fake', '_translated', '_output', '_gen', '_generated']
    for suffix in suffixes_to_remove:
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break
    
    return stem


def extract_target_domain(domain_name: str) -> str:
    """Extract target domain from translation folder name (e.g., 'clear_day2cloudy' -> 'cloudy')."""
    if '2' in domain_name:
        parts = domain_name.split('2', 1)
        if len(parts) > 1 and parts[1]:
            return parts[1]
    return domain_name


def extract_source_domain(domain_name: str) -> str:
    """Extract source domain from translation folder name (e.g., 'clear_day2cloudy' -> 'clear_day')."""
    if '2' in domain_name:
        parts = domain_name.split('2', 1)
        if parts[0]:
            return parts[0]
    return domain_name


def build_original_index(original_dir: Path, verbose: bool = False) -> Dict[str, Path]:
    """
    Build an index of original images by normalized filename.
    
    Returns:
        Dict mapping normalized filename stem to full path
    """
    if verbose:
        print(f"Indexing original images in {original_dir}...")
    
    original_files = find_image_files(original_dir, recursive=True)
    
    if verbose:
        print(f"  Found {len(original_files)} original images")
    
    # Build index by stem (without extension)
    index: Dict[str, Path] = {}
    duplicates: Dict[str, List[Path]] = defaultdict(list)
    
    for path in original_files:
        stem = path.stem
        if stem in index:
            duplicates[stem].append(path)
            if len(duplicates[stem]) == 1:
                duplicates[stem].insert(0, index[stem])
        else:
            index[stem] = path
    
    if verbose and duplicates:
        print(f"  Warning: {len(duplicates)} filenames appear multiple times")
        for stem, paths in list(duplicates.items())[:3]:
            print(f"    '{stem}': {[str(p) for p in paths[:2]]}...")
    
    return index


def find_generated_images_in_domain(domain_dir: Path) -> List[Path]:
    """
    Find generated images in a domain directory, handling nested structures.
    
    Checks common patterns:
    - domain_dir/images/
    - domain_dir/test_latest/images/
    - domain_dir/*.png (flat)
    """
    # Try common nested patterns first
    nested_patterns = [
        domain_dir / "test_latest" / "images",
        domain_dir / "images",
        domain_dir / "output",
        domain_dir / "results",
    ]
    
    for nested_dir in nested_patterns:
        if nested_dir.exists():
            files = find_image_files(nested_dir, recursive=False)
            if files:
                return files
    
    # Fall back to searching the domain directory itself
    return find_image_files(domain_dir, recursive=True)


def match_domain(
    domain_name: str,
    gen_domain_dir: Path,
    original_index: Dict[str, Path],
    target_dir: Optional[Path] = None,
) -> Tuple[List[Tuple[Path, Path, str]], List[Path], Dict[str, any]]:
    """
    Match generated images to originals for a single domain.
    
    Returns:
        - List of (gen_path, original_path, name) tuples
        - List of unmatched generated paths
        - Stats dict
    """
    gen_files = find_generated_images_in_domain(gen_domain_dir)
    
    matched = []
    unmatched = []
    
    for gen_path in gen_files:
        # Normalize the generated filename
        normalized_stem = normalize_filename(gen_path.name)
        
        if normalized_stem in original_index:
            original_path = original_index[normalized_stem]
            matched.append((gen_path, original_path, normalized_stem))
        else:
            unmatched.append(gen_path)
    
    # Compute target domain info
    target_domain = extract_target_domain(domain_name)
    source_domain = extract_source_domain(domain_name)
    target_domain_dir = target_dir / target_domain if target_dir else None
    
    stats = {
        "domain": domain_name,
        "source_domain": source_domain,
        "target_domain": target_domain,
        "generated_count": len(gen_files),
        "matched_count": len(matched),
        "unmatched_count": len(unmatched),
        "match_rate": len(matched) / len(gen_files) * 100 if gen_files else 0,
        "target_dir": str(target_domain_dir) if target_domain_dir else None,
        "target_exists": target_domain_dir.exists() if target_domain_dir else False,
    }
    
    return matched, unmatched, stats


def create_manifest(
    generated_dir: Path,
    original_dir: Path,
    output_path: Path,
    target_dir: Optional[Path] = None,
    verbose: bool = False,
) -> Dict[str, any]:
    """
    Create a CSV manifest mapping generated images to originals.
    
    Returns:
        Summary statistics
    """
    # Build index of original images
    original_index = build_original_index(original_dir, verbose=verbose)
    
    if not original_index:
        raise ValueError(f"No original images found in {original_dir}")
    
    # Discover domains (subfolders) in generated directory
    domains = [d.name for d in generated_dir.iterdir() if d.is_dir()]
    
    if not domains:
        # Treat as single flat domain
        domains = ["_root"]
    
    if verbose:
        print(f"\nDiscovered {len(domains)} domains: {domains}")
    
    all_matched = []
    all_unmatched = []
    domain_stats = []
    
    for domain in tqdm(domains, desc="Processing domains", disable=not verbose):
        if domain == "_root":
            gen_domain_dir = generated_dir
        else:
            gen_domain_dir = generated_dir / domain
        
        matched, unmatched, stats = match_domain(
            domain, gen_domain_dir, original_index, target_dir
        )
        
        # Add domain prefix to matched entries
        for gen_path, orig_path, name in matched:
            all_matched.append({
                "gen_path": str(gen_path),
                "original_path": str(orig_path),
                "name": name,
                "domain": domain,
                "target_domain": stats["target_domain"],
            })
        
        all_unmatched.extend(unmatched)
        domain_stats.append(stats)
        
        if verbose:
            print(f"  {domain}: {stats['matched_count']}/{stats['generated_count']} matched "
                  f"({stats['match_rate']:.1f}%)")
    
    # Write CSV manifest
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["gen_path", "original_path", "name", "domain", "target_domain"])
        writer.writeheader()
        writer.writerows(all_matched)
    
    # Write summary JSON
    summary = {
        "generated_dir": str(generated_dir),
        "original_dir": str(original_dir),
        "target_dir": str(target_dir) if target_dir else None,
        "manifest_path": str(output_path),
        "total_generated": sum(s["generated_count"] for s in domain_stats),
        "total_matched": len(all_matched),
        "total_unmatched": len(all_unmatched),
        "overall_match_rate": len(all_matched) / sum(s["generated_count"] for s in domain_stats) * 100 
                              if domain_stats else 0,
        "domains": domain_stats,
    }
    
    summary_path = output_path.with_suffix('.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    if verbose:
        print(f"\n=== Summary ===")
        print(f"Total generated: {summary['total_generated']}")
        print(f"Total matched: {summary['total_matched']}")
        print(f"Total unmatched: {summary['total_unmatched']}")
        print(f"Overall match rate: {summary['overall_match_rate']:.1f}%")
        print(f"\nManifest written to: {output_path}")
        print(f"Summary written to: {summary_path}")
    
    # Write unmatched files list if any
    if all_unmatched:
        unmatched_path = output_path.with_name(output_path.stem + "_unmatched.txt")
        with open(unmatched_path, 'w') as f:
            for path in all_unmatched:
                f.write(f"{path}\n")
        if verbose:
            print(f"Unmatched files written to: {unmatched_path}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Create evaluation manifest by matching generated images to originals.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--generated", type=Path, required=True,
        help="Directory containing generated images (with domain subfolders)"
    )
    parser.add_argument(
        "--original", type=Path, required=True,
        help="Directory containing original images (searched recursively)"
    )
    parser.add_argument(
        "--target", type=Path, default=None,
        help="Directory containing target domain images (for FID reference)"
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("manifest.csv"),
        help="Output CSV manifest path"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show detailed progress"
    )
    
    args = parser.parse_args()
    
    if not args.generated.exists():
        raise FileNotFoundError(f"Generated directory not found: {args.generated}")
    if not args.original.exists():
        raise FileNotFoundError(f"Original directory not found: {args.original}")
    
    summary = create_manifest(
        generated_dir=args.generated,
        original_dir=args.original,
        output_path=args.output,
        target_dir=args.target,
        verbose=args.verbose,
    )
    
    return 0 if summary["total_matched"] > 0 else 1


if __name__ == "__main__":
    exit(main())
