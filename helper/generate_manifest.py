#!/usr/bin/env python3
"""
Unified manifest generation script for image-to-image translation evaluation.

This script creates CSV and JSON manifests that map generated images to their
original counterparts, handling various directory structures and naming conventions.

Modes:
  Single method: Generate manifest for one specific method
  --all: Generate manifests for all methods in GENERATED_IMAGES
  --all-missing: Generate manifests only for methods without existing manifests

Features:
- Handles domain/dataset and dataset/domain hierarchies
- Supports flat directory structures
- Maps various domain naming conventions to canonical forms
- Identifies restoration vs generation tasks
- Handles dataset identification from filename patterns
- Removes common generation suffixes (_fake, _translated, etc.)
- Supports weather domain indices for restoration task matching

Outputs:
- manifest.csv: Paired generated and original images
- manifest.json: Summary statistics and metadata
- manifest_unmatched.txt: Unmatched generated images (if any)
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import DatasetMapper directly to avoid torch dependency
import importlib.util
spec = importlib.util.spec_from_file_location(
    "dataset_mapper", 
    Path(__file__).parent.parent / "utils" / "dataset_mapper.py"
)
dataset_mapper_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset_mapper_module)
DatasetMapper = dataset_mapper_module.DatasetMapper


# =============================================================================
# Constants and Configuration
# =============================================================================

CANONICAL_GENERATION_DOMAINS = {"snowy", "rainy", "foggy", "night", "cloudy", "dawn_dusk", "clear_day"}
CANONICAL_RESTORATION_DOMAINS = {"derained", "dehazed", "desnowed", "night2day"}
RESTORATION_SOURCE_MAPPING = {
    "derained": "rainy",
    "dehazed": "foggy",
    "desnowed": "snowy",
    "night2day": "night",
}

# Common domain name mappings
DOMAIN_MAPPING = {
    "fog": "foggy",
    "rain": "rainy",
    "snow": "snowy",
    "sunny": "clear_day",
    "sunny_day": "clear_day",
    "clear": "clear_day",
    "overcast": "cloudy",
    "dusk": "dawn_dusk",
    "dawn": "dawn_dusk",
}

KNOWN_DATASETS = {'ACDC', 'BDD100k', 'BDD10k', 'IDD-AW', 'MapillaryVistas', 'OUTSIDE15k'}
SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.webp'}


# =============================================================================
# Helper Functions
# =============================================================================

def find_image_files(directory: Path, recursive: bool = True) -> List[Path]:
    """Find all supported image files in a directory."""
    image_files = []
    try:
        glob_func = directory.rglob if recursive else directory.glob
        for ext in SUPPORTED_EXTENSIONS:
            image_files.extend(glob_func(f"*{ext}"))
            image_files.extend(glob_func(f"*{ext.upper()}"))
    except PermissionError:
        pass
    return sorted(image_files)


def normalize_filename(filename: str) -> str:
    """
    Normalize a filename by removing common suffixes and prefixes.
    
    Handles:
    - Dataset prefixes (e.g., 'ACDC_image.png' -> 'image.png')
    - Generation suffixes (_fake, _translated, _output, _gen, _generated)
    - Style transfer suffixes (_lat, _ref, _stylized, _styled)
    - NST pattern (_sa_<number>)
    """
    stem = Path(filename).stem
    
    # Remove dataset prefix if present
    for dataset in KNOWN_DATASETS:
        if stem.startswith(dataset + '_'):
            stem = stem[len(dataset) + 1:]
            break
    
    # NST style: ends with _sa_<number>
    nst_pattern = re.compile(r'_sa_\d+$')
    stem = nst_pattern.sub('', stem)
    
    # Remove common generation suffixes (order matters - check longer suffixes first)
    suffixes_to_remove = [
        '_fake', '_translated', '_output', '_gen', '_generated',
        '_lat', '_ref', '_stylized', '_styled',
    ]
    
    # Keep removing suffixes until none match
    changed = True
    while changed:
        changed = False
        for suffix in suffixes_to_remove:
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
                changed = True
                break
    
    return stem


def normalize_domain(domain_name: str) -> Optional[str]:
    """
    Map a domain name to its canonical form.
    
    Returns None if the domain is not recognized.
    """
    # Direct lookup
    if domain_name in DOMAIN_MAPPING:
        return DOMAIN_MAPPING[domain_name]
    
    # Try lowercase
    lower = domain_name.lower()
    if lower in DOMAIN_MAPPING:
        return DOMAIN_MAPPING[lower]
    
    # Try extracting target from translation pattern (e.g., "sunny_day2foggy")
    if '2' in domain_name:
        parts = domain_name.split('2', 1)
        if len(parts) > 1:
            target = parts[1]
            if target in DOMAIN_MAPPING:
                return DOMAIN_MAPPING[target]
            if target.lower() in DOMAIN_MAPPING:
                return DOMAIN_MAPPING[target.lower()]
            # Check if target is already canonical
            all_canonical = CANONICAL_GENERATION_DOMAINS | CANONICAL_RESTORATION_DOMAINS
            if target in all_canonical:
                return target
            if target.lower() in all_canonical:
                return target.lower()
    
    # Check if it's already a canonical domain
    all_canonical = CANONICAL_GENERATION_DOMAINS | CANONICAL_RESTORATION_DOMAINS
    if domain_name in all_canonical:
        return domain_name
    if domain_name.lower() in all_canonical:
        return domain_name.lower()
    
    return None


def extract_source_domain(domain_name: str) -> Optional[str]:
    """Extract source domain from translation folder name (e.g., 'clear_day2cloudy' -> 'clear_day')."""
    if '2' in domain_name:
        parts = domain_name.split('2', 1)
        if parts[0]:
            return parts[0]
    return None


def extract_target_domain(domain_name: str) -> str:
    """Extract target domain from translation folder name (e.g., 'clear_day2cloudy' -> 'cloudy')."""
    if '2' in domain_name:
        parts = domain_name.split('2', 1)
        if len(parts) > 1 and parts[1]:
            return parts[1]
    return domain_name


def is_restoration_domain(canonical_domain: str) -> bool:
    """Check if a canonical domain is a restoration task."""
    return canonical_domain in CANONICAL_RESTORATION_DOMAINS


def get_restoration_source_domain(canonical_domain: str) -> Optional[str]:
    """Get the source weather domain for a restoration task."""
    return RESTORATION_SOURCE_MAPPING.get(canonical_domain)


def detect_directory_structure(method_dir: Path) -> str:
    """
    Detect the directory structure type for a method.
    
    Returns one of:
    - 'domain_dataset': domain/dataset hierarchy (e.g., foggy/ACDC/)
    - 'dataset_domain': dataset/domain hierarchy (e.g., ACDC/foggy/)
    - 'flat_domain': domain folders with images directly in them
    - 'flat_dataset': dataset folders with images directly in them
    - 'unknown': could not determine structure
    """
    try:
        subdirs = [d for d in method_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    except PermissionError:
        return 'unknown'
    
    if not subdirs:
        return 'unknown'
    
    first_subdir = subdirs[0]
    first_name = first_subdir.name
    
    # Check if first level is datasets
    if first_name in KNOWN_DATASETS:
        try:
            second_level = [d for d in first_subdir.iterdir() if d.is_dir()]
        except PermissionError:
            return 'flat_dataset'
        
        if second_level:
            second_name = second_level[0].name
            if normalize_domain(second_name) is not None:
                return 'dataset_domain'
            elif second_name in KNOWN_DATASETS:
                return 'flat_dataset'
            else:
                images = find_image_files(first_subdir, recursive=False)
                if images:
                    return 'flat_dataset'
        return 'flat_dataset'
    
    # Check if first level is domains
    normalized = normalize_domain(first_name)
    if normalized is not None:
        try:
            second_level = [d for d in first_subdir.iterdir() if d.is_dir()]
        except PermissionError:
            return 'flat_domain'
        
        if second_level:
            second_name = second_level[0].name
            if second_name in KNOWN_DATASETS:
                return 'domain_dataset'
            else:
                for nested in second_level:
                    if nested.name in KNOWN_DATASETS:
                        return 'domain_dataset'
                    try:
                        images_dir = nested / "images"
                        if images_dir.exists():
                            return 'flat_domain'
                    except PermissionError:
                        continue
        
        images = find_image_files(first_subdir, recursive=False)
        if images:
            return 'flat_domain'
        
        return 'flat_domain'
    
    return 'unknown'


# =============================================================================
# Image Entry and Index Building
# =============================================================================

@dataclass
class ImageEntry:
    """Represents a generated image with its metadata."""
    gen_path: Path
    original_path: Optional[Path] = None
    name: str = ""
    dataset: str = ""
    domain_raw: str = ""
    domain_canonical: str = ""
    source_domain: Optional[str] = None
    is_restoration: bool = False
    restoration_source_weather: Optional[str] = None


def build_original_index(original_dir: Path, verbose: bool = False) -> Tuple[Dict[str, Path], Dict[str, str], Dict[str, Dict[str, Path]]]:
    """
    Build an index of original images by normalized filename.
    
    Returns:
        Tuple of:
        - Dict mapping normalized filename stem to full path (for generation tasks)
        - Dict mapping filename stem to dataset name
        - Dict mapping weather domain -> {stem -> path} for restoration source matching
    """
    original_files = find_image_files(original_dir, recursive=True)
    
    if verbose:
        logging.info("Found %d original images", len(original_files))
    
    # Main index for generation tasks
    index: Dict[str, Path] = {}
    stem_to_dataset: Dict[str, str] = {}
    duplicates: Dict[str, List[Path]] = defaultdict(list)
    
    # Weather domain indices for restoration tasks
    weather_indices: Dict[str, Dict[str, Path]] = defaultdict(dict)
    
    for path in original_files:
        stem = path.stem
        
        # Extract dataset and domain from path
        dataset = None
        domain = None
        parts = path.parts
        for i, part in enumerate(parts):
            if part in KNOWN_DATASETS:
                dataset = part
                if i + 1 < len(parts) - 1:
                    domain = parts[i + 1]
                break
        
        # Add to weather domain index if domain is recognized
        if domain:
            canonical = normalize_domain(domain)
            if canonical and canonical in CANONICAL_GENERATION_DOMAINS:
                weather_indices[canonical][stem] = path
        
        # Main index handling
        if stem in index:
            duplicates[stem].append(path)
            if len(duplicates[stem]) == 1:
                duplicates[stem].insert(0, index[stem])
        else:
            index[stem] = path
            if dataset:
                stem_to_dataset[stem] = dataset
    
    if verbose and duplicates:
        logging.warning("  %d filenames appear multiple times", len(duplicates))
        for stem, paths in list(duplicates.items())[:3]:
            logging.warning("    '%s': %s...", stem, [str(p) for p in paths[:2]])
    
    return index, stem_to_dataset, dict(weather_indices)


# =============================================================================
# Processing Functions for Different Structures
# =============================================================================

def process_domain_dataset_structure(
    method_dir: Path,
    original_index: Dict[str, Path],
    stem_to_dataset: Dict[str, str],
    weather_indices: Dict[str, Dict[str, Path]],
    dataset_mapper: DatasetMapper,
) -> Tuple[List[ImageEntry], Dict]:
    """Process a method with domain/dataset hierarchy."""
    entries = []
    stats = defaultdict(lambda: defaultdict(lambda: {"matched": 0, "unmatched": 0}))
    
    try:
        domain_dirs = list(method_dir.iterdir())
    except PermissionError:
        return entries, dict(stats)
    
    for domain_dir in domain_dirs:
        if not domain_dir.is_dir() or domain_dir.name.startswith('.'):
            continue
        
        domain_raw = domain_dir.name
        domain_canonical = normalize_domain(domain_raw)
        source_domain = extract_source_domain(domain_raw)
        
        if domain_canonical is None:
            continue
        
        is_restoration = is_restoration_domain(domain_canonical)
        restoration_source = get_restoration_source_domain(domain_canonical) if is_restoration else None
        
        # Select appropriate index for matching
        if is_restoration and restoration_source and restoration_source in weather_indices:
            match_index = weather_indices[restoration_source]
        else:
            match_index = original_index
        
        try:
            subdirs = list(domain_dir.iterdir())
        except PermissionError:
            continue
        
        for dataset_dir in subdirs:
            if not dataset_dir.is_dir():
                continue
            
            if dataset_dir.name in KNOWN_DATASETS:
                dataset = dataset_dir.name
                images = find_image_files(dataset_dir, recursive=True)
                
                for img_path in images:
                    normalized_stem = normalize_filename(img_path.name)
                    entry = ImageEntry(
                        gen_path=img_path,
                        name=normalized_stem,
                        dataset=dataset,
                        domain_raw=domain_raw,
                        domain_canonical=domain_canonical,
                        source_domain=source_domain,
                        is_restoration=is_restoration,
                        restoration_source_weather=restoration_source,
                    )
                    
                    if normalized_stem in match_index:
                        entry.original_path = match_index[normalized_stem]
                        stats[domain_canonical][dataset]["matched"] += 1
                    else:
                        stats[domain_canonical][dataset]["unmatched"] += 1
                    
                    entries.append(entry)
            else:
                # Nested structure like test_latest/images
                images = find_image_files(dataset_dir, recursive=True)
                for img_path in images:
                    normalized_stem = normalize_filename(img_path.name)
                    
                    dataset = stem_to_dataset.get(normalized_stem)
                    if not dataset:
                        dataset = dataset_mapper.get_dataset(img_path.name) or "unknown"
                    
                    entry = ImageEntry(
                        gen_path=img_path,
                        name=normalized_stem,
                        dataset=dataset,
                        domain_raw=domain_raw,
                        domain_canonical=domain_canonical,
                        source_domain=source_domain,
                        is_restoration=is_restoration,
                        restoration_source_weather=restoration_source,
                    )
                    
                    if normalized_stem in match_index:
                        entry.original_path = match_index[normalized_stem]
                        stats[domain_canonical][dataset]["matched"] += 1
                    else:
                        stats[domain_canonical][dataset]["unmatched"] += 1
                    
                    entries.append(entry)
    
    return entries, dict(stats)


def process_dataset_domain_structure(
    method_dir: Path,
    original_index: Dict[str, Path],
    stem_to_dataset: Dict[str, str],
    weather_indices: Dict[str, Dict[str, Path]],
    dataset_mapper: DatasetMapper,
) -> Tuple[List[ImageEntry], Dict]:
    """Process a method with dataset/domain hierarchy."""
    entries = []
    stats = defaultdict(lambda: defaultdict(lambda: {"matched": 0, "unmatched": 0}))
    
    try:
        dataset_dirs = list(method_dir.iterdir())
    except PermissionError:
        return entries, dict(stats)
    
    for dataset_dir in dataset_dirs:
        if not dataset_dir.is_dir() or dataset_dir.name.startswith('.'):
            continue
        
        if dataset_dir.name not in KNOWN_DATASETS:
            continue
        
        dataset = dataset_dir.name
        
        try:
            domain_dirs = list(dataset_dir.iterdir())
        except PermissionError:
            continue
        
        for domain_dir in domain_dirs:
            if not domain_dir.is_dir():
                continue
            
            domain_raw = domain_dir.name
            domain_canonical = normalize_domain(domain_raw)
            source_domain = extract_source_domain(domain_raw)
            
            if domain_canonical is None:
                continue
            
            is_restoration = is_restoration_domain(domain_canonical)
            restoration_source = get_restoration_source_domain(domain_canonical) if is_restoration else None
            
            # Select appropriate index for matching
            if is_restoration and restoration_source and restoration_source in weather_indices:
                match_index = weather_indices[restoration_source]
            else:
                match_index = original_index
            
            images = find_image_files(domain_dir, recursive=True)
            
            for img_path in images:
                normalized_stem = normalize_filename(img_path.name)
                entry = ImageEntry(
                    gen_path=img_path,
                    name=normalized_stem,
                    dataset=dataset,
                    domain_raw=domain_raw,
                    domain_canonical=domain_canonical,
                    source_domain=source_domain,
                    is_restoration=is_restoration,
                    restoration_source_weather=restoration_source,
                )
                
                if normalized_stem in match_index:
                    entry.original_path = match_index[normalized_stem]
                    stats[domain_canonical][dataset]["matched"] += 1
                else:
                    stats[domain_canonical][dataset]["unmatched"] += 1
                
                entries.append(entry)
    
    return entries, dict(stats)


def process_flat_structure(
    method_dir: Path,
    original_index: Dict[str, Path],
    stem_to_dataset: Dict[str, str],
    weather_indices: Dict[str, Dict[str, Path]],
    dataset_mapper: DatasetMapper,
) -> Tuple[List[ImageEntry], Dict]:
    """Process a method with flat structure (images directly in domain/dataset folders)."""
    entries = []
    stats = defaultdict(lambda: defaultdict(lambda: {"matched": 0, "unmatched": 0}))
    
    try:
        subdirs = list(method_dir.iterdir())
    except PermissionError:
        return entries, dict(stats)
    
    for subdir in subdirs:
        if not subdir.is_dir() or subdir.name.startswith('.'):
            continue
        
        dir_name = subdir.name
        domain_canonical = normalize_domain(dir_name)
        
        if domain_canonical is None:
            continue
        
        domain_raw = dir_name
        source_domain = extract_source_domain(dir_name)
        is_restoration = is_restoration_domain(domain_canonical)
        restoration_source = get_restoration_source_domain(domain_canonical) if is_restoration else None
        
        # Select appropriate index for matching
        if is_restoration and restoration_source and restoration_source in weather_indices:
            match_index = weather_indices[restoration_source]
        else:
            match_index = original_index
        
        images = find_image_files(subdir, recursive=True)
        
        for img_path in images:
            normalized_stem = normalize_filename(img_path.name)
            
            dataset = stem_to_dataset.get(normalized_stem)
            if not dataset:
                dataset = dataset_mapper.get_dataset(img_path.name) or "unknown"
            
            entry = ImageEntry(
                gen_path=img_path,
                name=normalized_stem,
                dataset=dataset,
                domain_raw=domain_raw,
                domain_canonical=domain_canonical,
                source_domain=source_domain,
                is_restoration=is_restoration,
                restoration_source_weather=restoration_source,
            )
            
            if normalized_stem in match_index:
                entry.original_path = match_index[normalized_stem]
                stats[domain_canonical][dataset]["matched"] += 1
            else:
                stats[domain_canonical][dataset]["unmatched"] += 1
            
            entries.append(entry)
    
    return entries, dict(stats)


def process_method(
    method_dir: Path,
    original_index: Dict[str, Path],
    stem_to_dataset: Dict[str, str],
    weather_indices: Dict[str, Dict[str, Path]],
    dataset_mapper: DatasetMapper,
    verbose: bool = False,
) -> Tuple[List[ImageEntry], Dict, str]:
    """
    Process a method directory and return image entries and stats.
    
    Returns:
        Tuple of (entries, stats, structure_type)
    """
    structure = detect_directory_structure(method_dir)
    
    if verbose:
        logging.info("  Detected structure: %s", structure)
    
    if structure == 'domain_dataset':
        entries, stats = process_domain_dataset_structure(
            method_dir, original_index, stem_to_dataset, weather_indices, dataset_mapper
        )
    elif structure == 'dataset_domain':
        entries, stats = process_dataset_domain_structure(
            method_dir, original_index, stem_to_dataset, weather_indices, dataset_mapper
        )
    elif structure in ('flat_domain', 'flat_dataset'):
        entries, stats = process_flat_structure(
            method_dir, original_index, stem_to_dataset, weather_indices, dataset_mapper
        )
    else:
        entries, stats = [], {}
    
    return entries, stats, structure


# =============================================================================
# Manifest Writing
# =============================================================================

def write_manifest(
    entries: List[ImageEntry],
    stats: Dict,
    method_name: str,
    method_dir: Path,
    original_dir: Path,
    target_dir: Optional[Path],
    output_dir: Path,
    structure_type: str,
) -> Dict:
    """Write manifest CSV and JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / "manifest.csv"
    json_path = output_dir / "manifest.json"
    
    # Write CSV
    matched_entries = [e for e in entries if e.original_path is not None]
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "gen_path", "original_path", "name", "domain", "dataset", "target_domain"
        ])
        writer.writeheader()
        for entry in matched_entries:
            writer.writerow({
                "gen_path": str(entry.gen_path),
                "original_path": str(entry.original_path) if entry.original_path else "",
                "name": entry.name,
                "domain": entry.domain_canonical,
                "dataset": entry.dataset,
                "target_domain": extract_target_domain(entry.domain_raw),
            })
    
    # Aggregate statistics
    total_matched = sum(
        s["matched"] for domain_stats in stats.values() 
        for s in domain_stats.values()
    )
    total_unmatched = sum(
        s["unmatched"] for domain_stats in stats.values() 
        for s in domain_stats.values()
    )
    total = total_matched + total_unmatched
    
    # Build domain summary
    domain_summary = {}
    for domain, dataset_stats in stats.items():
        domain_matched = sum(s["matched"] for s in dataset_stats.values())
        domain_unmatched = sum(s["unmatched"] for s in dataset_stats.values())
        domain_total = domain_matched + domain_unmatched
        
        target_exists = False
        if target_dir:
            target_domain_dir = target_dir / domain
            target_exists = target_domain_dir.exists()
        
        is_restoration = is_restoration_domain(domain)
        restoration_source = get_restoration_source_domain(domain) if is_restoration else None
        
        domain_summary[domain] = {
            "total": domain_total,
            "matched": domain_matched,
            "unmatched": domain_unmatched,
            "match_rate": domain_matched / domain_total * 100 if domain_total else 0,
            "target_exists": target_exists,
            "is_restoration": is_restoration,
            "restoration_source_weather": restoration_source,
            "datasets": {
                ds: {
                    "matched": s["matched"],
                    "unmatched": s["unmatched"],
                    "total": s["matched"] + s["unmatched"],
                }
                for ds, s in dataset_stats.items()
            }
        }
    
    # Build dataset summary
    dataset_summary = defaultdict(lambda: {"matched": 0, "unmatched": 0, "total": 0})
    for domain_stats in stats.values():
        for ds, s in domain_stats.items():
            dataset_summary[ds]["matched"] += s["matched"]
            dataset_summary[ds]["unmatched"] += s["unmatched"]
            dataset_summary[ds]["total"] += s["matched"] + s["unmatched"]
    
    # Determine task type
    has_restoration = any(is_restoration_domain(d) for d in domain_summary.keys())
    has_generation = any(d in CANONICAL_GENERATION_DOMAINS for d in domain_summary.keys())
    if has_restoration and has_generation:
        task_type = "mixed"
    elif has_restoration:
        task_type = "restoration"
    else:
        task_type = "generation"
    
    # Write JSON
    summary = {
        "method": method_name,
        "generated_dir": str(method_dir),
        "original_dir": str(original_dir),
        "target_dir": str(target_dir) if target_dir else None,
        "manifest_path": str(csv_path),
        "structure_type": structure_type,
        "task_type": task_type,
        "total_generated": total,
        "total_matched": total_matched,
        "total_unmatched": total_unmatched,
        "overall_match_rate": total_matched / total * 100 if total else 0,
        "domains": domain_summary,
        "datasets_aggregate": dict(dataset_summary),
    }
    
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Write unmatched files if any
    unmatched_entries = [e for e in entries if e.original_path is None]
    if unmatched_entries:
        unmatched_path = output_dir / "manifest_unmatched.txt"
        with open(unmatched_path, 'w') as f:
            for entry in unmatched_entries:
                f.write(f"{entry.gen_path}\n")
    
    return summary


# =============================================================================
# Main Function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate manifest files for image-to-image translation evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--all", action="store_true",
        help="Generate manifests for all methods in GENERATED_IMAGES"
    )
    mode_group.add_argument(
        "--all-missing", action="store_true",
        help="Generate manifests only for methods without existing manifests"
    )
    
    # Directories
    parser.add_argument(
        "--generated", type=Path,
        help="Generated images directory (for single method mode)"
    )
    parser.add_argument(
        "--generated-base", type=Path,
        default=Path("/scratch/aaa_exchange/AWARE/GENERATED_IMAGES"),
        help="Base directory containing method subdirectories (for --all modes)"
    )
    parser.add_argument(
        "--original", type=Path,
        default=Path("/scratch/aaa_exchange/AWARE/FINAL_SPLITS/train/images"),
        help="Directory containing original images"
    )
    parser.add_argument(
        "--target", type=Path,
        default=Path("/scratch/aaa_exchange/AWARE/AWACS/train"),
        help="Directory containing target domain images (for FID reference)"
    )
    parser.add_argument(
        "-o", "--output", type=Path,
        help="Output directory for manifest files (default: generated directory)"
    )
    
    # Method selection
    parser.add_argument(
        "--methods", type=str, nargs="+",
        help="Specific methods to process (for --all modes)"
    )
    
    # Options
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show detailed progress"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without writing files"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='[%(levelname)s] %(message)s'
    )
    
    # Validate arguments
    if not args.all and not args.all_missing and not args.generated:
        parser.error("Either --generated or --all or --all-missing must be specified")
    
    if not args.original.exists():
        raise FileNotFoundError(f"Original directory not found: {args.original}")
    
    # Build original image index
    logging.info("Building original image index...")
    original_index, stem_to_dataset, weather_indices = build_original_index(args.original, args.verbose)
    logging.info("  Indexed %d original images (for generation tasks)", len(original_index))
    for weather_domain, weather_index in weather_indices.items():
        logging.info("  Indexed %d %s images (for restoration tasks)", len(weather_index), weather_domain)
    
    # Create dataset mapper
    dataset_mapper = DatasetMapper(args.original)
    
    # Determine methods to process
    if args.all or args.all_missing:
        if not args.generated_base.exists():
            raise FileNotFoundError(f"Generated base directory not found: {args.generated_base}")
        
        if args.methods:
            method_dirs = [args.generated_base / m for m in args.methods if (args.generated_base / m).exists()]
        else:
            method_dirs = [d for d in args.generated_base.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        # Filter for --all-missing
        if args.all_missing:
            filtered_dirs = []
            for d in method_dirs:
                try:
                    if not (d / "manifest.json").exists():
                        filtered_dirs.append(d)
                except PermissionError:
                    # Skip directories we can't access
                    if args.verbose:
                        logging.warning("Permission denied accessing %s, skipping", d)
                    continue
            method_dirs = filtered_dirs
        
        logging.info("Processing %d methods...", len(method_dirs))
    else:
        # Single method mode
        if not args.generated.exists():
            raise FileNotFoundError(f"Generated directory not found: {args.generated}")
        method_dirs = [args.generated]
    
    # Process methods
    all_summaries = {}
    
    for method_dir in tqdm(method_dirs, desc="Methods", disable=not (args.all or args.all_missing)):
        method_name = method_dir.name
        
        if args.verbose:
            logging.info("\n=== Processing %s ===", method_name)
        
        # Process method
        entries, stats, structure = process_method(
            method_dir, original_index, stem_to_dataset, weather_indices, 
            dataset_mapper, args.verbose
        )
        
        if not entries:
            if args.verbose:
                logging.info("  No images found, skipping")
            continue
        
        # Determine output directory
        if args.output:
            if args.all or args.all_missing:
                output_dir = args.output / method_name
            else:
                output_dir = args.output
        else:
            output_dir = method_dir
        
        if args.dry_run:
            matched = sum(1 for e in entries if e.original_path is not None)
            restoration_count = sum(1 for e in entries if e.is_restoration)
            logging.info("  Would write manifest: %d images, %d matched", len(entries), matched)
            logging.info("  Structure: %s", structure)
            domains = set(e.domain_canonical for e in entries)
            logging.info("  Domains: %s", domains)
            if restoration_count > 0:
                logging.info("  Restoration images: %d", restoration_count)
            continue
        
        # Write manifest
        summary = write_manifest(
            entries, stats, method_name, method_dir,
            args.original, args.target, output_dir, structure
        )
        
        all_summaries[method_name] = summary
        
        if args.verbose:
            logging.info("  Task type: %s", summary['task_type'])
            logging.info("  Total: %d images", summary['total_generated'])
            logging.info("  Matched: %d (%.1f%%)", summary['total_matched'], summary['overall_match_rate'])
            logging.info("  Domains: %s", list(summary['domains'].keys()))
        
        # Single method mode - print summary
        if not (args.all or args.all_missing):
            logging.info("\nManifest created successfully:")
            logging.info("  CSV: %s", output_dir / "manifest.csv")
            logging.info("  JSON: %s", output_dir / "manifest.json")
            logging.info("  Total images: %d", summary['total_generated'])
            logging.info("  Matched: %d (%.1f%%)", summary['total_matched'], summary['overall_match_rate'])
    
    # Write global summary for --all modes
    if (args.all or args.all_missing) and not args.dry_run and all_summaries:
        summary_path = args.generated_base / "all_manifests_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                "generated_base": str(args.generated_base),
                "original_dir": str(args.original),
                "target_dir": str(args.target) if args.target else None,
                "timestamp": datetime.now().isoformat(),
                "methods_processed": len(all_summaries),
                "methods": {
                    name: {
                        "structure_type": s["structure_type"],
                        "task_type": s["task_type"],
                        "total_generated": s["total_generated"],
                        "total_matched": s["total_matched"],
                        "match_rate": s["overall_match_rate"],
                        "domains": list(s["domains"].keys()),
                    }
                    for name, s in all_summaries.items()
                }
            }, f, indent=2)
        logging.info("\nGlobal summary written to: %s", summary_path)
    
    if args.all or args.all_missing:
        logging.info("\nProcessed %d methods successfully.", len(all_summaries))
    
    return 0


if __name__ == "__main__":
    exit(main())
