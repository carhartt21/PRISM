#!/usr/bin/env python3
"""
Fix results.json files by removing per_image_details from semantic_consistency.

This script migrates existing results.json files to the new format where
per_image_details are excluded from the main results file (they remain
in the domain-specific *_stats.json files).

Usage:
    python helper/fix_results_format.py /scratch/aaa_exchange/AWARE/STATS
    python helper/fix_results_format.py /scratch/aaa_exchange/AWARE/STATS --dry-run
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)


def get_size_mb(file_path: Path) -> float:
    """Get file size in MB."""
    return file_path.stat().st_size / (1024 * 1024)


def fix_results_file(results_path: Path, dry_run: bool = False) -> Tuple[bool, str, float, float]:
    """
    Fix a single results.json file by removing per_image_details.
    
    Args:
        results_path: Path to the results.json file
        dry_run: If True, don't write changes
        
    Returns:
        Tuple of (was_modified, message, size_before_mb, size_after_mb)
    """
    if not results_path.exists():
        return False, "File does not exist", 0, 0
    
    size_before = get_size_mb(results_path)
    
    try:
        with open(results_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}", size_before, size_before
    
    modified = False
    removed_count = 0
    
    # Check for domains structure (per-domain mode)
    if 'domains' in data:
        for domain_name, domain_data in data['domains'].items():
            if isinstance(domain_data, dict) and 'semantic_consistency' in domain_data:
                sc = domain_data['semantic_consistency']
                if isinstance(sc, dict) and 'per_image_details' in sc:
                    del sc['per_image_details']
                    modified = True
                    removed_count += 1
    
    # Check for flat structure
    if 'semantic_consistency' in data:
        sc = data['semantic_consistency']
        if isinstance(sc, dict) and 'per_image_details' in sc:
            del sc['per_image_details']
            modified = True
            removed_count += 1
    
    if not modified:
        return False, "Already in correct format (no per_image_details found)", size_before, size_before
    
    if dry_run:
        # Calculate estimated new size
        new_json = json.dumps(data, indent=2)
        size_after = len(new_json.encode('utf-8')) / (1024 * 1024)
        return True, f"Would remove per_image_details from {removed_count} section(s)", size_before, size_after
    
    # Write the modified data directly (files are world-writable)
    try:
        with open(results_path, 'w') as f:
            json.dump(data, f, indent=2)
        size_after = get_size_mb(results_path)
        return True, f"Removed per_image_details from {removed_count} section(s)", size_before, size_after
    except Exception as e:
        return False, f"Failed to write: {e}", size_before, size_before


def main():
    parser = argparse.ArgumentParser(
        description="Fix results.json files by removing per_image_details from semantic_consistency."
    )
    parser.add_argument(
        "stats_dir",
        type=Path,
        help="Path to the STATS directory containing method subdirectories"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    stats_dir = args.stats_dir
    if not stats_dir.exists():
        logging.error("Stats directory does not exist: %s", stats_dir)
        sys.exit(1)
    
    # Find all method directories
    method_dirs = sorted([d for d in stats_dir.iterdir() if d.is_dir()])
    
    if not method_dirs:
        logging.warning("No subdirectories found in %s", stats_dir)
        sys.exit(0)
    
    print(f"\n{'='*70}")
    print(f"Results Format Migration Report")
    print(f"{'='*70}")
    print(f"Stats directory: {stats_dir}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"{'='*70}\n")
    
    missing = []
    modified = []
    already_ok = []
    errors = []
    total_saved_mb = 0.0
    
    for method_dir in method_dirs:
        method_name = method_dir.name
        results_path = method_dir / "results.json"
        
        if not results_path.exists():
            missing.append(method_name)
            logging.warning("MISSING: %s/results.json", method_name)
            continue
        
        was_modified, message, size_before, size_after = fix_results_file(results_path, dry_run=args.dry_run)
        
        if was_modified:
            saved = size_before - size_after
            total_saved_mb += saved
            modified.append((method_name, size_before, size_after, saved))
            action = "Would fix" if args.dry_run else "Fixed"
            print(f"✓ {action}: {method_name}")
            print(f"    {message}")
            print(f"    Size: {size_before:.2f} MB → {size_after:.2f} MB (saved {saved:.2f} MB)")
        elif "correct format" in message:
            already_ok.append(method_name)
            logging.debug("OK: %s - %s", method_name, message)
        else:
            errors.append((method_name, message))
            logging.error("ERROR: %s - %s", method_name, message)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total methods scanned: {len(method_dirs)}")
    print(f"  - Modified:    {len(modified)}")
    print(f"  - Already OK:  {len(already_ok)}")
    print(f"  - Missing:     {len(missing)}")
    print(f"  - Errors:      {len(errors)}")
    
    if modified:
        print(f"\nTotal space {'would be ' if args.dry_run else ''}saved: {total_saved_mb:.2f} MB")
    
    if missing:
        print(f"\n{'='*70}")
        print("MISSING results.json FILES:")
        print(f"{'='*70}")
        for name in missing:
            print(f"  - {name}")
    
    if errors:
        print(f"\n{'='*70}")
        print("ERRORS:")
        print(f"{'='*70}")
        for name, msg in errors:
            print(f"  - {name}: {msg}")
    
    if already_ok and args.verbose:
        print(f"\n{'='*70}")
        print("ALREADY IN CORRECT FORMAT:")
        print(f"{'='*70}")
        for name in already_ok:
            print(f"  - {name}")
    
    print()


if __name__ == "__main__":
    main()
