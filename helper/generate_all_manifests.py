#!/usr/bin/env python3
"""
Generate manifest files for all directories in the GENERATED_IMAGES folder.

This script:
1. Scans all method directories in the generated images folder
2. Maps varied domain naming conventions to canonical domain names
3. Creates manifest CSV and JSON files for each method directory
4. Includes dataset mapping for proper evaluation

Canonical domain names: snowy, rainy, foggy, night, cloudy, dawn_dusk

The script handles various directory structures:
- domain/dataset hierarchy (e.g., foggy/ACDC/)
- dataset/domain hierarchy (e.g., ACDC/foggy/)
- translation naming (e.g., sunny_day2foggy, clear_day2fog)
- restoration naming (e.g., derained, desnowed)
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
from dataclasses import dataclass, field
from tqdm import tqdm

# Add parent directory to path for imports - import dataset_mapper directly
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import DatasetMapper directly to avoid torch dependency in utils/__init__.py
import importlib.util
spec = importlib.util.spec_from_file_location(
    "dataset_mapper", 
    Path(__file__).parent.parent / "utils" / "dataset_mapper.py"
)
dataset_mapper_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset_mapper_module)
DatasetMapper = dataset_mapper_module.DatasetMapper

# =============================================================================
# Domain Mapping Configuration
# =============================================================================

# Canonical domain names for GENERATION tasks (clear -> adverse)
CANONICAL_GENERATION_DOMAINS = {"snowy", "rainy", "foggy", "night", "cloudy", "dawn_dusk"}

# Canonical domain names for RESTORATION tasks (adverse -> clear)
CANONICAL_RESTORATION_DOMAINS = {"derained", "dehazed", "desnowed", "night2day"}

# All canonical domains
CANONICAL_DOMAINS = CANONICAL_GENERATION_DOMAINS | CANONICAL_RESTORATION_DOMAINS

# Mapping from restoration domain to source weather domain (for matching originals)
# e.g., "derained" images come from "rainy" source images
RESTORATION_SOURCE_MAPPING = {
    "derained": "rainy",
    "dehazed": "foggy",
    "desnowed": "snowy",
    "night2day": "night",
}

# Mapping of various domain naming conventions to canonical names
DOMAIN_MAPPING = {
    # =========================================================================
    # GENERATION DOMAINS (clear -> adverse weather)
    # =========================================================================
    
    # Foggy variants
    "fog": "foggy",
    "foggy": "foggy",
    "clearday2foggy": "foggy",
    "clear_day2foggy": "foggy",
    "sunny_day2foggy": "foggy",
    "clear_day2fog": "foggy",
    "sunny2foggy": "foggy",
    
    # Rainy variants
    "rain": "rainy",
    "rainy": "rainy",
    "clearday2rainy": "rainy",
    "clear_day2rainy": "rainy",
    "sunny_day2rainy": "rainy",
    "sunny2rainy": "rainy",
    
    # Snowy variants
    "snow": "snowy",
    "snowy": "snowy",
    "snow_no_flakes": "snowy",
    "clearday2snowy": "snowy",
    "clear_day2snowy": "snowy",
    "sunny_day2snowy": "snowy",
    "sunny2snowy": "snowy",
    
    # Night variants
    "night": "night",
    "clearday2night": "night",
    "clear_day2night": "night",
    "sunny_day2night": "night",
    "sunny2night": "night",
    
    # Cloudy variants
    "cloudy": "cloudy",
    "clouds": "cloudy",
    "clearday2cloudy": "cloudy",
    "clear_day2cloudy": "cloudy",
    "sunny_day2cloudy": "cloudy",
    "sunny2cloudy": "cloudy",
    
    # Dawn/Dusk variants
    "dawn_dusk": "dawn_dusk",
    "dawndusk": "dawn_dusk",
    "sunrisesunset": "dawn_dusk",
    "sunrise_sunset": "dawn_dusk",
    "clearday2dawn_dusk": "dawn_dusk",
    "clear_day2dawn_dusk": "dawn_dusk",
    "sunny_day2dawn_dusk": "dawn_dusk",
    "sunny2dawn_dusk": "dawn_dusk",
    
    # =========================================================================
    # RESTORATION DOMAINS (adverse weather -> clear)
    # =========================================================================
    
    # Derained (rain removal) - source is rainy images
    "derained": "derained",
    "derained2": "derained",
    "derained_drop": "derained",
    "derained_CSD-teacher": "derained",
    "derained_ITS-OTS-teacher": "derained",
    "derained_Rain1400-teacher": "derained",
    "derained_Raindrop": "derained",
    "derained_Rainfog": "derained",
    "derained_Snow100k": "derained",
    "derained_student-setting1": "derained",
    "derained_student-setting2": "derained",
    
    # Dehazed/Defogged (fog/haze removal) - source is foggy images
    "dehazed": "dehazed",
    "defogged": "dehazed",
    "defogged2": "dehazed",
    "defogged_Raindrop": "dehazed",
    "defogged_Rainfog": "dehazed",
    "defog": "dehazed",
    "dehaze": "dehazed",
    
    # Desnowed (snow removal) - source is snowy images
    "desnowed": "desnowed",
    "desnowed_CSD-teacher": "desnowed",
    "desnowed_ITS-OTS-teacher": "desnowed",
    "desnowed_Rain1400-teacher": "desnowed",
    "desnowed_Raindrop": "desnowed",
    "desnowed_Rainfog": "desnowed",
    "desnowed_Snow100k": "desnowed",
    "desnowed_student-setting1": "desnowed",
    "desnowed_student-setting2": "desnowed",
    
    # Night2Day (night enhancement) - source is night images
    "night2day": "night2day",
    "night_to_day": "night2day",
    "nighttoday": "night2day",
}

# Known dataset names
KNOWN_DATASETS = {'ACDC', 'BDD100k', 'BDD10k', 'IDD-AW', 'MapillaryVistas', 'OUTSIDE15k'}

# Supported image extensions
SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}


# =============================================================================
# Utility Functions
# =============================================================================

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
    
    # Check if it's a canonical domain
    if domain_name in CANONICAL_DOMAINS:
        return domain_name
    
    return None


def is_restoration_domain(canonical_domain: str) -> bool:
    """Check if a canonical domain is a restoration task."""
    return canonical_domain in CANONICAL_RESTORATION_DOMAINS


def get_restoration_source_domain(canonical_domain: str) -> Optional[str]:
    """
    Get the source weather domain for a restoration task.
    
    For restoration tasks, the input images come from adverse weather conditions.
    E.g., 'derained' takes 'rainy' images as input.
    
    Returns None if not a restoration domain.
    """
    return RESTORATION_SOURCE_MAPPING.get(canonical_domain)


def extract_source_domain(domain_name: str) -> Optional[str]:
    """Extract source domain from translation folder name."""
    if '2' in domain_name:
        parts = domain_name.split('2', 1)
        if parts[0]:
            return parts[0]
    return None


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
    Normalize a filename by removing common suffixes added during generation.
    """
    stem = Path(filename).stem
    
    # Remove common generation suffixes (order matters - check longer suffixes first)
    suffixes_to_remove = [
        '_fake', '_translated', '_output', '_gen', '_generated',
        '_lat',  # stargan_v2 style
        '_ref',  # reference style
        '_stylized', '_styled',  # style transfer
    ]
    
    # NST style: ends with _sa_<number>
    nst_pattern = re.compile(r'_sa_\d+$')
    stem = nst_pattern.sub('', stem)
    
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
        # Check second level
        try:
            second_level = [d for d in first_subdir.iterdir() if d.is_dir()]
        except PermissionError:
            return 'flat_dataset'
        
        if second_level:
            second_name = second_level[0].name
            if normalize_domain(second_name) is not None:
                return 'dataset_domain'
            elif second_name in KNOWN_DATASETS:
                return 'flat_dataset'  # Nested datasets
            else:
                # Check if there are images directly
                images = find_image_files(first_subdir, recursive=False)
                if images:
                    return 'flat_dataset'
        return 'flat_dataset'
    
    # Check if first level is domains
    normalized = normalize_domain(first_name)
    if normalized is not None:
        # Check second level
        try:
            second_level = [d for d in first_subdir.iterdir() if d.is_dir()]
        except PermissionError:
            return 'flat_domain'
        
        if second_level:
            second_name = second_level[0].name
            if second_name in KNOWN_DATASETS:
                return 'domain_dataset'
            else:
                # Check for nested patterns (e.g., test_latest)
                for nested in second_level:
                    if nested.name in KNOWN_DATASETS:
                        return 'domain_dataset'
                    # Check if this is cycleGAN style (test_latest/images)
                    try:
                        images_dir = nested / "images"
                        if images_dir.exists():
                            return 'flat_domain'
                    except PermissionError:
                        continue
        
        # Check for direct images
        images = find_image_files(first_subdir, recursive=False)
        if images:
            return 'flat_domain'
        
        return 'flat_domain'
    
    return 'unknown'


@dataclass
class ImageEntry:
    """Represents a generated image with its metadata."""
    gen_path: Path
    original_path: Optional[Path] = None
    name: str = ""
    dataset: str = ""
    domain_raw: str = ""  # Original domain name from directory
    domain_canonical: str = ""  # Canonical domain name
    source_domain: Optional[str] = None  # For translation: source domain (e.g., clear_day)
    is_restoration: bool = False  # True if this is a restoration task
    restoration_source_weather: Optional[str] = None  # For restoration: the weather domain of input (e.g., rainy)


def build_original_index(original_dir: Path) -> Tuple[Dict[str, Path], Dict[str, str], Dict[str, Dict[str, Path]]]:
    """
    Build an index of original images by normalized filename.
    
    For restoration tasks, we need to find the source weather images (e.g., rainy images
    for derained outputs). The original directory structure is:
    original_dir/DATASET/DOMAIN/image.png
    
    Returns:
        Tuple of:
        - Dict mapping normalized filename stem to full path (for generation tasks - clear_day)
        - Dict mapping filename stem to dataset name
        - Dict mapping weather domain -> {stem -> path} for restoration source matching
    """
    original_files = find_image_files(original_dir, recursive=True)
    
    # Main index for generation tasks (typically clear_day images)
    index: Dict[str, Path] = {}
    stem_to_dataset: Dict[str, str] = {}
    
    # Weather domain indices for restoration tasks
    # Maps: weather_domain -> {stem -> path}
    weather_indices: Dict[str, Dict[str, Path]] = defaultdict(dict)
    
    for path in original_files:
        stem = path.stem
        
        # Extract dataset and domain from path
        # Path structure: .../DATASET/DOMAIN/image.png
        dataset = None
        domain = None
        parts = path.parts
        for i, part in enumerate(parts):
            if part in KNOWN_DATASETS:
                dataset = part
                # Domain should be next part
                if i + 1 < len(parts) - 1:  # -1 because last part is filename
                    domain = parts[i + 1]
                break
        
        # Add to weather domain index if domain is recognized
        if domain:
            canonical = normalize_domain(domain)
            if canonical and canonical in CANONICAL_GENERATION_DOMAINS:
                # This is a weather domain image - useful for restoration source matching
                weather_indices[canonical][stem] = path
        
        # For the main index, prioritize clear_day images (for generation tasks)
        # But also add other images if stem not already indexed
        if stem not in index:
            index[stem] = path
            if dataset:
                stem_to_dataset[stem] = dataset
        elif domain and domain.lower() in ('clear_day', 'clearday', 'sunny', 'sunny_day'):
            # Prefer clear_day for generation tasks
            index[stem] = path
            if dataset:
                stem_to_dataset[stem] = dataset
    
    return index, stem_to_dataset, dict(weather_indices)


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
            continue  # Skip unrecognized domains
        
        # Check if this is a restoration task
        is_restoration = is_restoration_domain(domain_canonical)
        restoration_source = get_restoration_source_domain(domain_canonical) if is_restoration else None
        
        # Select appropriate index for matching
        if is_restoration and restoration_source and restoration_source in weather_indices:
            match_index = weather_indices[restoration_source]
        else:
            match_index = original_index
        
        # Check for dataset subdirectories
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
                # Might be a nested structure like test_latest/images
                images = find_image_files(dataset_dir, recursive=True)
                for img_path in images:
                    normalized_stem = normalize_filename(img_path.name)
                    
                    # Determine dataset from filename
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
        
        # Check for domain subdirectories
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
            
            # Check if this is a restoration task
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


def process_flat_domain_structure(
    method_dir: Path,
    original_index: Dict[str, Path],
    stem_to_dataset: Dict[str, str],
    weather_indices: Dict[str, Dict[str, Path]],
    dataset_mapper: DatasetMapper,
) -> Tuple[List[ImageEntry], Dict]:
    """Process a method with flat domain structure (images directly in domain folders)."""
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
        
        # Check if this is a restoration task
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
            
            # Determine dataset from filename
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
        print(f"  Detected structure: {structure}")
    
    if structure == 'domain_dataset':
        entries, stats = process_domain_dataset_structure(
            method_dir, original_index, stem_to_dataset, weather_indices, dataset_mapper
        )
    elif structure == 'dataset_domain':
        entries, stats = process_dataset_domain_structure(
            method_dir, original_index, stem_to_dataset, weather_indices, dataset_mapper
        )
    elif structure in ('flat_domain', 'flat_dataset'):
        entries, stats = process_flat_domain_structure(
            method_dir, original_index, stem_to_dataset, weather_indices, dataset_mapper
        )
    else:
        entries, stats = [], {}
    
    return entries, stats, structure


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
            "gen_path", "original_path", "name", "domain_raw", 
            "domain_canonical", "dataset", "source_domain",
            "is_restoration", "restoration_source_weather"
        ])
        writer.writeheader()
        for entry in matched_entries:
            writer.writerow({
                "gen_path": str(entry.gen_path),
                "original_path": str(entry.original_path) if entry.original_path else "",
                "name": entry.name,
                "domain_raw": entry.domain_raw,
                "domain_canonical": entry.domain_canonical,
                "dataset": entry.dataset,
                "source_domain": entry.source_domain or "",
                "is_restoration": entry.is_restoration,
                "restoration_source_weather": entry.restoration_source_weather or "",
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
        
        # Check if target exists
        target_exists = False
        if target_dir:
            target_domain_dir = target_dir / domain
            target_exists = target_domain_dir.exists()
        
        # Check if this is a restoration domain
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
    
    # Build dataset summary (aggregated across domains)
    dataset_summary = defaultdict(lambda: {"matched": 0, "unmatched": 0, "total": 0})
    for domain_stats in stats.values():
        for ds, s in domain_stats.items():
            dataset_summary[ds]["matched"] += s["matched"]
            dataset_summary[ds]["unmatched"] += s["unmatched"]
            dataset_summary[ds]["total"] += s["matched"] + s["unmatched"]
    
    # Determine task type based on domains present
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
        "domain_mapping_used": {
            domain: list(set(
                e.domain_raw for e in entries if e.domain_canonical == domain
            ))
            for domain in domain_summary.keys()
        }
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


def main():
    parser = argparse.ArgumentParser(
        description="Generate manifest files for all methods in GENERATED_IMAGES.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--generated-base", type=Path,
        default=Path("/scratch/aaa_exchange/AWARE/GENERATED_IMAGES"),
        help="Base directory containing method subdirectories"
    )
    parser.add_argument(
        "--original", type=Path,
        default=Path("/scratch/aaa_exchange/AWARE/FINAL_SPLITS/train/images"),
        help="Directory containing original images (searched recursively)"
    )
    parser.add_argument(
        "--target", type=Path,
        default=Path("/scratch/aaa_exchange/AWARE/AWACS/train"),
        help="Directory containing target domain images (for FID reference)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for manifests. If not specified, writes to each method directory."
    )
    parser.add_argument(
        "--methods", type=str, nargs="+", default=None,
        help="Specific methods to process. If not specified, processes all."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show detailed progress"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without writing files"
    )
    
    args = parser.parse_args()
    
    if not args.generated_base.exists():
        raise FileNotFoundError(f"Generated base directory not found: {args.generated_base}")
    if not args.original.exists():
        raise FileNotFoundError(f"Original directory not found: {args.original}")
    
    # Build original image index (includes weather domain indices for restoration)
    print("Building original image index...")
    original_index, stem_to_dataset, weather_indices = build_original_index(args.original)
    print(f"  Indexed {len(original_index)} original images (for generation tasks)")
    for weather_domain, weather_index in weather_indices.items():
        print(f"  Indexed {len(weather_index)} {weather_domain} images (for restoration tasks)")
    
    # Create dataset mapper
    dataset_mapper = DatasetMapper(args.original)
    
    # Discover method directories
    if args.methods:
        method_dirs = [args.generated_base / m for m in args.methods if (args.generated_base / m).exists()]
    else:
        method_dirs = [d for d in args.generated_base.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    print(f"\nProcessing {len(method_dirs)} methods...")
    
    all_summaries = {}
    
    for method_dir in tqdm(method_dirs, desc="Methods"):
        method_name = method_dir.name
        
        if args.verbose:
            print(f"\n=== Processing {method_name} ===")
        
        # Process method
        entries, stats, structure = process_method(
            method_dir, original_index, stem_to_dataset, weather_indices, dataset_mapper, args.verbose
        )
        
        if not entries:
            if args.verbose:
                print(f"  No images found, skipping")
            continue
        
        # Determine output directory
        if args.output_dir:
            output_dir = args.output_dir / method_name
        else:
            output_dir = method_dir
        
        if args.dry_run:
            matched = sum(1 for e in entries if e.original_path is not None)
            restoration_count = sum(1 for e in entries if e.is_restoration)
            print(f"  Would write manifest: {len(entries)} images, {matched} matched")
            print(f"  Structure: {structure}")
            domains = set(e.domain_canonical for e in entries)
            print(f"  Domains: {domains}")
            if restoration_count > 0:
                print(f"  Restoration images: {restoration_count}")
            continue
        
        # Write manifest
        summary = write_manifest(
            entries, stats, method_name, method_dir,
            args.original, args.target, output_dir, structure
        )
        
        all_summaries[method_name] = summary
        
        if args.verbose:
            print(f"  Task type: {summary['task_type']}")
            print(f"  Total: {summary['total_generated']} images")
            print(f"  Matched: {summary['total_matched']} ({summary['overall_match_rate']:.1f}%)")
            print(f"  Domains: {list(summary['domains'].keys())}")
    
    # Write global summary
    if not args.dry_run and all_summaries:
        global_summary_path = args.generated_base / "all_manifests_summary.json"
        with open(global_summary_path, 'w') as f:
            json.dump({
                "generated_base": str(args.generated_base),
                "original_dir": str(args.original),
                "target_dir": str(args.target) if args.target else None,
                "methods_processed": len(all_summaries),
                "canonical_generation_domains": list(CANONICAL_GENERATION_DOMAINS),
                "canonical_restoration_domains": list(CANONICAL_RESTORATION_DOMAINS),
                "restoration_source_mapping": RESTORATION_SOURCE_MAPPING,
                "domain_mapping": DOMAIN_MAPPING,
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
        print(f"\nGlobal summary written to: {global_summary_path}")
    
    print(f"\nProcessed {len(all_summaries)} methods successfully.")
    
    return 0


if __name__ == "__main__":
    exit(main())
