#!/usr/bin/env python3
"""
Script to find strategies that have generated images but are missing quality results.

Compares directories in GENERATED_IMAGES with result files in STATS to identify
which strategies haven't been evaluated yet.
"""

import os
import argparse
from pathlib import Path


def get_strategies_from_generated_images(generated_images_dir: Path) -> set:
    """Get all strategy names from the GENERATED_IMAGES directory."""
    strategies = set()
    
    if not generated_images_dir.exists():
        print(f"Error: GENERATED_IMAGES directory does not exist: {generated_images_dir}")
        return strategies
    
    # List all subdirectories (each represents a strategy)
    for item in generated_images_dir.iterdir():
        if item.is_dir():
            strategies.add(item.name)
    
    return strategies


def get_strategies_from_stats(stats_dir: Path, pattern: str = None) -> set:
    """
    Get all strategy names that have quality results in the STATS directory.
    
    Assumes result files are named like: <strategy_name>_quality.json or similar
    """
    strategies = set()
    
    if not stats_dir.exists():
        print(f"Error: STATS directory does not exist: {stats_dir}")
        return strategies
    
    # List all files and extract strategy names
    for item in stats_dir.iterdir():
        if item.is_file():
            # Try to extract strategy name from filename
            name = item.stem  # filename without extension
            
            # Handle common naming patterns for quality results
            # Adjust these patterns based on your actual file naming convention
            for suffix in ['_quality', '_generation_quality', '_gen_quality', '_results']:
                if name.endswith(suffix):
                    name = name[:-len(suffix)]
                    break
            
            strategies.add(name)
        elif item.is_dir():
            # Also check if strategies are organized as subdirectories
            strategies.add(item.name)
    
    return strategies


def main():
    parser = argparse.ArgumentParser(
        description="Find strategies with generated images but missing quality results"
    )
    parser.add_argument(
        "--generated-images-dir",
        type=Path,
        default=Path("/scratch/aaa_exchange/AWARE/GENERATED_IMAGES"),
        help="Path to GENERATED_IMAGES directory"
    )
    parser.add_argument(
        "--stats-dir",
        type=Path,
        default=Path("/scratch/aaa_exchange/AWARE/STATS"),
        help="Path to STATS directory"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed information about what was found"
    )
    
    args = parser.parse_args()
    
    print(f"Scanning GENERATED_IMAGES: {args.generated_images_dir}")
    print(f"Scanning STATS: {args.stats_dir}")
    print()
    
    # Get strategies from both directories
    generated_strategies = get_strategies_from_generated_images(args.generated_images_dir)
    stats_strategies = get_strategies_from_stats(args.stats_dir)
    
    if args.verbose:
        print(f"Found {len(generated_strategies)} strategies in GENERATED_IMAGES:")
        for s in sorted(generated_strategies):
            print(f"  - {s}")
        print()
        
        print(f"Found {len(stats_strategies)} strategies in STATS:")
        for s in sorted(stats_strategies):
            print(f"  - {s}")
        print()
    
    # Find strategies that are in generated images but not in stats
    missing_stats = generated_strategies - stats_strategies
    
    if missing_stats:
        print(f"=== MISSING QUALITY RESULTS ({len(missing_stats)} strategies) ===")
        for strategy in sorted(missing_stats):
            print(f"  {strategy}")
    else:
        print("All strategies have quality results!")
    
    print()
    
    # Also show strategies in stats that aren't in generated images (might be orphaned)
    orphaned_stats = stats_strategies - generated_strategies
    if orphaned_stats and args.verbose:
        print(f"=== ORPHANED STATS (no corresponding generated images) ({len(orphaned_stats)}) ===")
        for strategy in sorted(orphaned_stats):
            print(f"  {strategy}")
    
    # Summary
    print()
    print("=== SUMMARY ===")
    print(f"Total strategies with generated images: {len(generated_strategies)}")
    print(f"Total strategies with quality results:  {len(stats_strategies)}")
    print(f"Missing quality results:                {len(missing_stats)}")
    
    return len(missing_stats)


if __name__ == "__main__":
    exit(main())
