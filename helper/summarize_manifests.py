#!/usr/bin/env python3
"""
summarize_manifests.py

Iterates through directories in GENERATED_IMAGES, reads manifest files,
and generates a summary of:
- Total number of images in each directory
- Count per weather domain
- Count per source dataset

Usage:
    python helper/summarize_manifests.py [--generated-root PATH] [--output FILE]
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional


def count_images_in_dir(directory: Path) -> int:
    """Count image files in directory (recursive) - optimized version."""
    extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff',
                  '.PNG', '.JPG', '.JPEG', '.WEBP', '.BMP', '.TIFF'}
    count = 0
    try:
        for path in directory.rglob('*'):
            if path.is_file() and path.suffix in extensions:
                count += 1
    except PermissionError:
        pass
    return count


def parse_manifest(manifest_path: Path) -> Dict[str, Any]:
    """Parse a manifest CSV file and return statistics."""
    stats = {
        'total_entries': 0,
        'by_domain': defaultdict(int),
        'by_dataset': defaultdict(int),
        'by_domain_dataset': defaultdict(lambda: defaultdict(int)),
    }
    
    if not manifest_path.exists():
        return None
    
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        
        # Determine which columns to use
        domain_col = 'target_domain' if 'target_domain' in fieldnames else 'domain'
        dataset_col = 'dataset' if 'dataset' in fieldnames else None
        
        for row in reader:
            stats['total_entries'] += 1
            
            # Count by domain
            domain = row.get(domain_col, 'unknown').strip()
            if domain:
                stats['by_domain'][domain] += 1
            
            # Count by dataset
            if dataset_col and dataset_col in row:
                dataset = row.get(dataset_col, 'unknown').strip()
                if dataset:
                    stats['by_dataset'][dataset] += 1
                    stats['by_domain_dataset'][domain][dataset] += 1
    
    # Convert defaultdicts to regular dicts
    stats['by_domain'] = dict(stats['by_domain'])
    stats['by_dataset'] = dict(stats['by_dataset'])
    stats['by_domain_dataset'] = {k: dict(v) for k, v in stats['by_domain_dataset'].items()}
    
    return stats


def summarize_directory(dir_path: Path, fallback_manifest_root: Optional[Path] = None) -> Dict[str, Any]:
    """Summarize a single generated images directory."""
    result = {
        'name': dir_path.name,
        'path': str(dir_path),
        'actual_image_count': 0,
        'manifest_exists': False,
        'manifest_entries': 0,
        'count_match': None,
        'by_domain': {},
        'by_dataset': {},
        'by_domain_dataset': {},
        'manifest_location': None,
    }
    
    # Count actual images
    result['actual_image_count'] = count_images_in_dir(dir_path)
    
    # Try to find manifest in multiple locations
    manifest_locations = [
        dir_path / 'manifest.csv',
        dir_path / 'manifest' / 'manifest.csv',
    ]
    
    # Add fallback location if provided
    if fallback_manifest_root:
        manifest_locations.append(fallback_manifest_root / dir_path.name / 'manifest.csv')
    
    manifest_path = None
    for loc in manifest_locations:
        if loc.exists():
            manifest_path = loc
            break
    
    if manifest_path:
        result['manifest_exists'] = True
        result['manifest_location'] = str(manifest_path)
        manifest_stats = parse_manifest(manifest_path)
        if manifest_stats:
            result['manifest_entries'] = manifest_stats['total_entries']
            result['by_domain'] = manifest_stats['by_domain']
            result['by_dataset'] = manifest_stats['by_dataset']
            result['by_domain_dataset'] = manifest_stats['by_domain_dataset']
            result['count_match'] = result['actual_image_count'] == result['manifest_entries']
    
    return result


def generate_summary(generated_root: Path, output_path: Optional[Path] = None, 
                     fallback_manifest_root: Optional[Path] = None) -> Dict[str, Any]:
    """Generate summary for all directories in generated root."""
    
    # Find all model directories
    model_dirs = sorted([d for d in generated_root.iterdir() if d.is_dir()])
    
    summary = {
        'generated_root': str(generated_root),
        'total_models': len(model_dirs),
        'models': {},
        'aggregate': {
            'total_images': 0,
            'total_manifest_entries': 0,
            'by_domain': defaultdict(int),
            'by_dataset': defaultdict(int),
            'models_with_manifest': 0,
            'models_with_mismatch': 0,
        }
    }
    
    print(f"Scanning {len(model_dirs)} directories in {generated_root}...")
    if fallback_manifest_root:
        print(f"Fallback manifest location: {fallback_manifest_root}")
    print("-" * 80)
    
    for model_dir in model_dirs:
        model_name = model_dir.name
        print(f"Processing {model_name}...", end=" ", flush=True)
        
        result = summarize_directory(model_dir, fallback_manifest_root)
        summary['models'][model_name] = result
        
        # Update aggregates
        summary['aggregate']['total_images'] += result['actual_image_count']
        summary['aggregate']['total_manifest_entries'] += result['manifest_entries']
        
        if result['manifest_exists']:
            summary['aggregate']['models_with_manifest'] += 1
            if result['count_match'] is False:
                summary['aggregate']['models_with_mismatch'] += 1
        
        for domain, count in result['by_domain'].items():
            summary['aggregate']['by_domain'][domain] += count
        for dataset, count in result['by_dataset'].items():
            summary['aggregate']['by_dataset'][dataset] += count
        
        # Print status
        status = "✓" if result['count_match'] else ("⚠ MISMATCH" if result['manifest_exists'] else "✗ NO MANIFEST")
        location_note = " (fallback)" if result.get('manifest_location') and 'manifests/' in result['manifest_location'] else ""
        print(f"{result['actual_image_count']:,} images, {result['manifest_entries']:,} manifest entries [{status}]{location_note}")
    
    # Convert aggregates to regular dicts
    summary['aggregate']['by_domain'] = dict(summary['aggregate']['by_domain'])
    summary['aggregate']['by_dataset'] = dict(summary['aggregate']['by_dataset'])
    
    print("-" * 80)
    print()
    
    # Print summary table
    print_summary_table(summary)
    
    # Save to file if requested
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nFull summary saved to: {output_path}")
    
    return summary
    
    return summary


def print_summary_table(summary: Dict[str, Any]) -> None:
    """Print a formatted summary table."""
    
    print("=" * 100)
    print("GENERATED IMAGES SUMMARY")
    print("=" * 100)
    print()
    
    # Model table
    print(f"{'Model':<30} {'Images':>12} {'Manifest':>12} {'Match':>10} {'Domains':>20}")
    print("-" * 100)
    
    for name, data in sorted(summary['models'].items()):
        domains_str = ', '.join(sorted(data['by_domain'].keys())[:3])
        if len(data['by_domain']) > 3:
            domains_str += f" (+{len(data['by_domain'])-3})"
        
        match_status = "✓" if data['count_match'] else ("⚠" if data['manifest_exists'] else "-")
        manifest_count = data['manifest_entries'] if data['manifest_exists'] else "-"
        
        print(f"{name:<30} {data['actual_image_count']:>12,} {manifest_count:>12} {match_status:>10} {domains_str:>20}")
    
    print("-" * 100)
    agg = summary['aggregate']
    print(f"{'TOTAL':<30} {agg['total_images']:>12,} {agg['total_manifest_entries']:>12,}")
    print()
    
    # Domain breakdown
    print("BY WEATHER DOMAIN (aggregate):")
    print("-" * 50)
    for domain, count in sorted(agg['by_domain'].items(), key=lambda x: -x[1]):
        print(f"  {domain:<20} {count:>15,}")
    print()
    
    # Dataset breakdown
    print("BY SOURCE DATASET (aggregate):")
    print("-" * 50)
    for dataset, count in sorted(agg['by_dataset'].items(), key=lambda x: -x[1]):
        print(f"  {dataset:<20} {count:>15,}")
    print()
    
    # Stats
    print("STATISTICS:")
    print("-" * 50)
    print(f"  Models with manifest:  {agg['models_with_manifest']}/{summary['total_models']}")
    print(f"  Models with mismatch:  {agg['models_with_mismatch']}")
    print()
    
    # Detailed per-model breakdown
    print_detailed_breakdown(summary)


def print_detailed_breakdown(summary: Dict[str, Any]) -> None:
    """Print detailed breakdown per model with domains and datasets."""
    
    print("=" * 120)
    print("DETAILED BREAKDOWN PER MODEL")
    print("=" * 120)
    
    # Get all unique domains and datasets across all models
    all_domains = set()
    all_datasets = set()
    for data in summary['models'].values():
        all_domains.update(data['by_domain'].keys())
        all_datasets.update(data['by_dataset'].keys())
    
    # Sort domains and datasets
    domain_order = ['cloudy', 'dawn_dusk', 'foggy', 'night', 'rainy', 'snowy', 'clear_day']
    domains = [d for d in domain_order if d in all_domains] + sorted(all_domains - set(domain_order))
    datasets = sorted(all_datasets)
    
    # Print domain breakdown header
    print()
    print("BY WEATHER DOMAIN PER MODEL:")
    print("-" * 120)
    
    # Header row
    header = f"{'Model':<25}"
    for domain in domains:
        header += f" {domain[:8]:>10}"
    header += f" {'Total':>12}"
    print(header)
    print("-" * 120)
    
    # Data rows
    for name, data in sorted(summary['models'].items()):
        if not data['manifest_exists']:
            continue
        row = f"{name[:24]:<25}"
        total = 0
        for domain in domains:
            count = data['by_domain'].get(domain, 0)
            total += count
            row += f" {count:>10,}" if count > 0 else f" {'-':>10}"
        row += f" {total:>12,}"
        print(row)
    
    print("-" * 120)
    
    # Totals row
    agg = summary['aggregate']
    row = f"{'TOTAL':<25}"
    grand_total = 0
    for domain in domains:
        count = agg['by_domain'].get(domain, 0)
        grand_total += count
        row += f" {count:>10,}"
    row += f" {grand_total:>12,}"
    print(row)
    print()
    
    # Print dataset breakdown header
    print("BY SOURCE DATASET PER MODEL:")
    print("-" * 120)
    
    # Header row
    header = f"{'Model':<25}"
    for dataset in datasets:
        header += f" {dataset[:12]:>14}"
    header += f" {'Total':>12}"
    print(header)
    print("-" * 120)
    
    # Data rows
    for name, data in sorted(summary['models'].items()):
        if not data['manifest_exists']:
            continue
        row = f"{name[:24]:<25}"
        total = 0
        for dataset in datasets:
            count = data['by_dataset'].get(dataset, 0)
            total += count
            row += f" {count:>14,}" if count > 0 else f" {'-':>14}"
        row += f" {total:>12,}"
        print(row)
    
    print("-" * 120)
    
    # Totals row
    row = f"{'TOTAL':<25}"
    grand_total = 0
    for dataset in datasets:
        count = agg['by_dataset'].get(dataset, 0)
        grand_total += count
        row += f" {count:>14,}"
    row += f" {grand_total:>12,}"
    print(row)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Summarize manifest files in generated images directories"
    )
    parser.add_argument(
        '--generated-root', '-g',
        type=Path,
        default=Path('/scratch/aaa_exchange/AWARE/GENERATED_IMAGES'),
        help='Root directory containing generated image folders'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=None,
        help='Output JSON file for full summary (default: reports/summary_detailed.json)'
    )
    parser.add_argument(
        '--manifest-fallback', '-m',
        type=Path,
        default=Path('/home/mima2416/repositories/PRISM/manifests'),
        help='Fallback directory to check for manifests'
    )
    parser.add_argument(
        '--reports-dir', '-r',
        type=Path,
        default=Path('/home/mima2416/repositories/PRISM/reports'),
        help='Directory to save reports'
    )
    
    args = parser.parse_args()
    
    if not args.generated_root.exists():
        print(f"Error: Directory not found: {args.generated_root}")
        return 1
    
    # Set default output if not specified
    output_path = args.output
    if output_path is None:
        args.reports_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.reports_dir / 'summary_detailed.json'
    
    fallback = args.manifest_fallback if args.manifest_fallback.exists() else None
    generate_summary(args.generated_root, output_path, fallback)
    return 0


if __name__ == '__main__':
    exit(main())
