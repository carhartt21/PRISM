#!/usr/bin/env python3
"""
Preprocessing script to create evaluation manifests by matching generated images
to their original counterparts.

Handles:
- Generated images with '_fake' suffix (e.g., 'image_fake.png' -> 'image.jpg')
- Nested directory structures (e.g., 'domain/test_latest/images/')
- Multiple original image directories organized by dataset/domain
- Dataset identification from filename patterns for flat directory structures

Outputs a CSV manifest for use with evaluate_generation.py --pairs csv --manifest
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.dataset_mapper import DatasetMapper, build_dataset_index


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


def build_original_index(original_dir: Path, verbose: bool = False) -> Tuple[Dict[str, Path], Dict[str, str]]:
    """
    Build an index of original images by normalized filename.
    
    Returns:
        Tuple of:
        - Dict mapping normalized filename stem to full path
        - Dict mapping filename stem to dataset name
    """
    if verbose:
        print(f"Indexing original images in {original_dir}...")
    
    original_files = find_image_files(original_dir, recursive=True)
    
    if verbose:
        print(f"  Found {len(original_files)} original images")
    
    # Build index by stem (without extension)
    index: Dict[str, Path] = {}
    stem_to_dataset: Dict[str, str] = {}
    duplicates: Dict[str, List[Path]] = defaultdict(list)
    
    # Known datasets
    known_datasets = {'ACDC', 'BDD100k', 'BDD10k', 'IDD-AW', 'MapillaryVistas', 'OUTSIDE15k'}
    
    for path in original_files:
        stem = path.stem
        
        # Extract dataset from path
        dataset = None
        for part in path.parts:
            if part in known_datasets:
                dataset = part
                break
        
        if stem in index:
            duplicates[stem].append(path)
            if len(duplicates[stem]) == 1:
                duplicates[stem].insert(0, index[stem])
        else:
            index[stem] = path
            if dataset:
                stem_to_dataset[stem] = dataset
    
    if verbose and duplicates:
        print(f"  Warning: {len(duplicates)} filenames appear multiple times")
        for stem, paths in list(duplicates.items())[:3]:
            print(f"    '{stem}': {[str(p) for p in paths[:2]]}...")
    
    if verbose:
        # Count per dataset
        dataset_counts = defaultdict(int)
        for ds in stem_to_dataset.values():
            dataset_counts[ds] += 1
        print(f"  Dataset distribution: {dict(dataset_counts)}")
    
    return index, stem_to_dataset


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


def detect_hierarchy_type(domain_dir: Path) -> str:
    """
    Detect if a domain directory has dataset subfolders or flat structure.
    
    Returns:
        'hierarchical' if has dataset subfolders (ACDC, BDD100k, etc.)
        'flat' if images are directly in domain directory
    """
    known_datasets = {'ACDC', 'BDD100k', 'BDD10k', 'IDD-AW', 'MapillaryVistas', 'OUTSIDE15k'}
    
    for item in domain_dir.iterdir():
        if item.is_dir() and item.name in known_datasets:
            return 'hierarchical'
    
    return 'flat'


def match_domain(
    domain_name: str,
    gen_domain_dir: Path,
    original_index: Dict[str, Path],
    stem_to_dataset: Dict[str, str],
    target_dir: Optional[Path] = None,
    dataset_mapper: Optional[DatasetMapper] = None,
) -> Tuple[List[Dict], List[Path], Dict]:
    """
    Match generated images to originals for a single domain.
    
    Handles both hierarchical (with dataset subfolders) and flat structures.
    For flat structures, uses dataset_mapper to identify dataset from filename.
    
    Returns:
        - List of match dicts with gen_path, original_path, name, domain, dataset
        - List of unmatched generated paths
        - Stats dict
    """
    hierarchy = detect_hierarchy_type(gen_domain_dir)
    
    matched = []
    unmatched = []
    dataset_stats = defaultdict(lambda: {"matched": 0, "unmatched": 0})
    
    if hierarchy == 'hierarchical':
        # Process each dataset subfolder
        known_datasets = {'ACDC', 'BDD100k', 'BDD10k', 'IDD-AW', 'MapillaryVistas', 'OUTSIDE15k'}
        
        for item in gen_domain_dir.iterdir():
            if item.is_dir() and item.name in known_datasets:
                dataset = item.name
                gen_files = find_image_files(item, recursive=True)
                
                for gen_path in gen_files:
                    normalized_stem = normalize_filename(gen_path.name)
                    
                    if normalized_stem in original_index:
                        original_path = original_index[normalized_stem]
                        matched.append({
                            "gen_path": gen_path,
                            "original_path": original_path,
                            "name": normalized_stem,
                            "dataset": dataset,
                        })
                        dataset_stats[dataset]["matched"] += 1
                    else:
                        unmatched.append(gen_path)
                        dataset_stats[dataset]["unmatched"] += 1
    else:
        # Flat structure - use filename patterns to identify dataset
        gen_files = find_generated_images_in_domain(gen_domain_dir)
        
        for gen_path in gen_files:
            normalized_stem = normalize_filename(gen_path.name)
            
            if normalized_stem in original_index:
                original_path = original_index[normalized_stem]
                
                # Determine dataset from stem_to_dataset lookup or mapper
                dataset = stem_to_dataset.get(normalized_stem)
                if not dataset and dataset_mapper:
                    dataset = dataset_mapper.get_dataset(gen_path.name)
                if not dataset:
                    dataset = "unknown"
                
                matched.append({
                    "gen_path": gen_path,
                    "original_path": original_path,
                    "name": normalized_stem,
                    "dataset": dataset,
                })
                dataset_stats[dataset]["matched"] += 1
            else:
                unmatched.append(gen_path)
                # Try to identify dataset for stats
                dataset = None
                if dataset_mapper:
                    dataset = dataset_mapper.get_dataset(gen_path.name)
                dataset = dataset or "unknown"
                dataset_stats[dataset]["unmatched"] += 1
    
    # Compute target domain info
    target_domain = extract_target_domain(domain_name)
    source_domain = extract_source_domain(domain_name)
    target_domain_dir = target_dir / target_domain if target_dir else None
    
    total_gen = sum(s["matched"] + s["unmatched"] for s in dataset_stats.values())
    
    stats = {
        "domain": domain_name,
        "source_domain": source_domain,
        "target_domain": target_domain,
        "hierarchy_type": hierarchy,
        "generated_count": total_gen,
        "matched_count": len(matched),
        "unmatched_count": len(unmatched),
        "match_rate": len(matched) / total_gen * 100 if total_gen else 0,
        "target_dir": str(target_domain_dir) if target_domain_dir else None,
        "target_exists": target_domain_dir.exists() if target_domain_dir else False,
        "datasets": {
            ds: {
                "matched": counts["matched"],
                "unmatched": counts["unmatched"],
                "total": counts["matched"] + counts["unmatched"],
            }
            for ds, counts in dataset_stats.items()
        },
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
    
    Includes dataset identification for each image, supporting both
    hierarchical (with dataset subfolders) and flat directory structures.
    
    Returns:
        Summary statistics
    """
    # Build index of original images (now returns tuple with dataset mapping)
    original_index, stem_to_dataset = build_original_index(original_dir, verbose=verbose)
    
    if not original_index:
        raise ValueError(f"No original images found in {original_dir}")
    
    # Create dataset mapper for pattern-based identification
    dataset_mapper = DatasetMapper(original_dir)
    
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
            domain, gen_domain_dir, original_index, stem_to_dataset, 
            target_dir, dataset_mapper
        )
        
        # Add domain info to matched entries
        for match_info in matched:
            all_matched.append({
                "gen_path": str(match_info["gen_path"]),
                "original_path": str(match_info["original_path"]),
                "name": match_info["name"],
                "domain": domain,
                "dataset": match_info["dataset"],
                "target_domain": stats["target_domain"],
            })
        
        all_unmatched.extend(unmatched)
        domain_stats.append(stats)
        
        if verbose:
            print(f"  {domain} ({stats['hierarchy_type']}): {stats['matched_count']}/{stats['generated_count']} matched "
                  f"({stats['match_rate']:.1f}%)")
            # Print dataset breakdown
            for ds, ds_stats in stats.get("datasets", {}).items():
                print(f"    - {ds}: {ds_stats['matched']}/{ds_stats['total']}")
    
    # Write CSV manifest with dataset column
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["gen_path", "original_path", "name", "domain", "dataset", "target_domain"])
        writer.writeheader()
        writer.writerows(all_matched)
    
    # Aggregate dataset statistics across all domains
    aggregate_datasets = defaultdict(lambda: {"matched": 0, "unmatched": 0, "total": 0})
    for stats in domain_stats:
        for ds, ds_stats in stats.get("datasets", {}).items():
            aggregate_datasets[ds]["matched"] += ds_stats["matched"]
            aggregate_datasets[ds]["unmatched"] += ds_stats["unmatched"]
            aggregate_datasets[ds]["total"] += ds_stats["total"]
    
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
        "datasets_aggregate": dict(aggregate_datasets),
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
        print(f"\nDataset breakdown:")
        for ds, ds_stats in aggregate_datasets.items():
            print(f"  {ds}: {ds_stats['matched']}/{ds_stats['total']} matched")
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
