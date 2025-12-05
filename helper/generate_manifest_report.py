#!/usr/bin/env python3
"""
Generate a comprehensive report from all manifest files.

This script reads all manifest.json files from the manifests directory and
creates a formatted report distinguishing between generation (augmentation)
and restoration tasks.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from datetime import datetime


def load_manifests(manifest_dir: Path) -> Dict[str, Dict]:
    """Load all manifest.json files from the directory."""
    manifests = {}
    
    for method_dir in sorted(manifest_dir.iterdir()):
        if not method_dir.is_dir():
            continue
        
        manifest_path = method_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifests[method_dir.name] = json.load(f)
    
    return manifests


def format_number(n: int) -> str:
    """Format number with thousands separator."""
    return f"{n:,}"


def format_percentage(value: float) -> str:
    """Format percentage with one decimal place."""
    return f"{value:.1f}%"


def get_bar(percentage: float, width: int = 20) -> str:
    """Create a simple ASCII progress bar."""
    filled = int(percentage / 100 * width)
    return "█" * filled + "░" * (width - filled)


def generate_method_section(name: str, data: Dict, detailed: bool = False) -> List[str]:
    """Generate report section for a single method."""
    lines = []
    
    task_type = data.get("task_type", "generation")
    task_label = "🔄 RESTORATION" if task_type == "restoration" else "🎨 GENERATION"
    
    lines.append(f"┌{'─' * 78}┐")
    lines.append(f"│ {name:<50} {task_label:>25} │")
    lines.append(f"├{'─' * 78}┤")
    
    # Basic stats
    total = data.get("total_generated", 0)
    matched = data.get("total_matched", 0)
    match_rate = data.get("overall_match_rate", 0)
    
    lines.append(f"│ {'Total Images:':<20} {format_number(total):>15} │ {'Match Rate:':<15} {format_percentage(match_rate):>8} {get_bar(match_rate):>15} │")
    lines.append(f"│ {'Matched:':<20} {format_number(matched):>15} │ {'Unmatched:':<15} {format_number(total - matched):>8} {'':>15} │")
    lines.append(f"│ {'Structure:':<20} {data.get('structure_type', 'unknown'):>15} │ {'':>40} │")
    
    # Domains
    domains = data.get("domains", {})
    if domains:
        lines.append(f"├{'─' * 78}┤")
        lines.append(f"│ {'DOMAINS':<76} │")
        lines.append(f"├{'─' * 78}┤")
        
        for domain, domain_data in sorted(domains.items()):
            d_total = domain_data.get("total", 0)
            d_matched = domain_data.get("matched", 0)
            d_rate = domain_data.get("match_rate", 0)
            is_restoration = domain_data.get("is_restoration", False)
            source_weather = domain_data.get("restoration_source_weather", "")
            
            domain_label = f"{domain}"
            if is_restoration and source_weather:
                domain_label += f" ← {source_weather}"
            
            target_status = "✓" if domain_data.get("target_exists", False) else "✗"
            
            lines.append(f"│   {domain_label:<25} {format_number(d_total):>10} imgs  {format_percentage(d_rate):>7}  {get_bar(d_rate, 15)}  target:{target_status:<2} │")
    
    # Dataset breakdown (if detailed)
    if detailed:
        datasets = data.get("datasets_aggregate", {})
        if datasets:
            lines.append(f"├{'─' * 78}┤")
            lines.append(f"│ {'DATASETS':<76} │")
            lines.append(f"├{'─' * 78}┤")
            
            for dataset, ds_data in sorted(datasets.items()):
                ds_total = ds_data.get("total", 0)
                ds_matched = ds_data.get("matched", 0)
                ds_rate = (ds_matched / ds_total * 100) if ds_total > 0 else 0
                
                lines.append(f"│   {dataset:<20} {format_number(ds_total):>12} total  {format_number(ds_matched):>10} matched  {format_percentage(ds_rate):>7} │")
    
    lines.append(f"└{'─' * 78}┘")
    lines.append("")
    
    return lines


def generate_summary_table(manifests: Dict[str, Dict], task_type: str) -> List[str]:
    """Generate a summary table for methods of a specific task type."""
    lines = []
    
    # Filter methods by task type
    filtered = {
        name: data for name, data in manifests.items()
        if data.get("task_type", "generation") == task_type
    }
    
    if not filtered:
        return lines
    
    # Header
    lines.append(f"┌{'─' * 35}┬{'─' * 14}┬{'─' * 14}┬{'─' * 12}┐")
    lines.append(f"│ {'Method':<33} │ {'Total':>12} │ {'Matched':>12} │ {'Rate':>10} │")
    lines.append(f"├{'─' * 35}┼{'─' * 14}┼{'─' * 14}┼{'─' * 12}┤")
    
    # Sort by total images descending
    sorted_methods = sorted(filtered.items(), key=lambda x: x[1].get("total_generated", 0), reverse=True)
    
    total_all = 0
    matched_all = 0
    
    for name, data in sorted_methods:
        total = data.get("total_generated", 0)
        matched = data.get("total_matched", 0)
        rate = data.get("overall_match_rate", 0)
        
        total_all += total
        matched_all += matched
        
        lines.append(f"│ {name:<33} │ {format_number(total):>12} │ {format_number(matched):>12} │ {format_percentage(rate):>10} │")
    
    # Footer with totals
    overall_rate = (matched_all / total_all * 100) if total_all > 0 else 0
    lines.append(f"├{'─' * 35}┼{'─' * 14}┼{'─' * 14}┼{'─' * 12}┤")
    lines.append(f"│ {'TOTAL':<33} │ {format_number(total_all):>12} │ {format_number(matched_all):>12} │ {format_percentage(overall_rate):>10} │")
    lines.append(f"└{'─' * 35}┴{'─' * 14}┴{'─' * 14}┴{'─' * 12}┘")
    
    return lines


def generate_domain_summary(manifests: Dict[str, Dict]) -> List[str]:
    """Generate a summary of all domains across methods."""
    lines = []
    
    # Aggregate domain stats
    generation_domains = defaultdict(lambda: {"total": 0, "matched": 0, "methods": set()})
    restoration_domains = defaultdict(lambda: {"total": 0, "matched": 0, "methods": set(), "source": None})
    
    for name, data in manifests.items():
        for domain, domain_data in data.get("domains", {}).items():
            is_restoration = domain_data.get("is_restoration", False)
            
            if is_restoration:
                restoration_domains[domain]["total"] += domain_data.get("total", 0)
                restoration_domains[domain]["matched"] += domain_data.get("matched", 0)
                restoration_domains[domain]["methods"].add(name)
                restoration_domains[domain]["source"] = domain_data.get("restoration_source_weather")
            else:
                generation_domains[domain]["total"] += domain_data.get("total", 0)
                generation_domains[domain]["matched"] += domain_data.get("matched", 0)
                generation_domains[domain]["methods"].add(name)
    
    # Generation domains table
    if generation_domains:
        lines.append("┌─────────────────────────────────────────────────────────────────────────────────┐")
        lines.append("│                           GENERATION DOMAINS SUMMARY                            │")
        lines.append("├─────────────────┬────────────────┬────────────────┬──────────────┬──────────────┤")
        lines.append("│ Domain          │ Total Images   │ Matched        │ Match Rate   │ # Methods    │")
        lines.append("├─────────────────┼────────────────┼────────────────┼──────────────┼──────────────┤")
        
        for domain in sorted(generation_domains.keys()):
            stats = generation_domains[domain]
            rate = (stats["matched"] / stats["total"] * 100) if stats["total"] > 0 else 0
            lines.append(f"│ {domain:<15} │ {format_number(stats['total']):>14} │ {format_number(stats['matched']):>14} │ {format_percentage(rate):>12} │ {len(stats['methods']):>12} │")
        
        lines.append("└─────────────────┴────────────────┴────────────────┴──────────────┴──────────────┘")
        lines.append("")
    
    # Restoration domains table
    if restoration_domains:
        lines.append("┌─────────────────────────────────────────────────────────────────────────────────────────────┐")
        lines.append("│                              RESTORATION DOMAINS SUMMARY                                    │")
        lines.append("├─────────────────┬─────────────────┬────────────────┬────────────────┬──────────┬────────────┤")
        lines.append("│ Domain          │ Source Weather  │ Total Images   │ Matched        │ Rate     │ # Methods  │")
        lines.append("├─────────────────┼─────────────────┼────────────────┼────────────────┼──────────┼────────────┤")
        
        for domain in sorted(restoration_domains.keys()):
            stats = restoration_domains[domain]
            rate = (stats["matched"] / stats["total"] * 100) if stats["total"] > 0 else 0
            source = stats["source"] or "N/A"
            lines.append(f"│ {domain:<15} │ {source:<15} │ {format_number(stats['total']):>14} │ {format_number(stats['matched']):>14} │ {format_percentage(rate):>8} │ {len(stats['methods']):>10} │")
        
        lines.append("└─────────────────┴─────────────────┴────────────────┴────────────────┴──────────┴────────────┘")
        lines.append("")
    
    return lines


def generate_dataset_summary(manifests: Dict[str, Dict]) -> List[str]:
    """Generate a summary of all datasets across methods."""
    lines = []
    
    # Aggregate dataset stats
    dataset_stats = defaultdict(lambda: {"total": 0, "matched": 0, "methods": set()})
    
    for name, data in manifests.items():
        for dataset, ds_data in data.get("datasets_aggregate", {}).items():
            dataset_stats[dataset]["total"] += ds_data.get("total", 0)
            dataset_stats[dataset]["matched"] += ds_data.get("matched", 0)
            dataset_stats[dataset]["methods"].add(name)
    
    if dataset_stats:
        lines.append("┌───────────────────────────────────────────────────────────────────────────────────┐")
        lines.append("│                              DATASET SUMMARY                                      │")
        lines.append("├───────────────────┬────────────────┬────────────────┬──────────────┬──────────────┤")
        lines.append("│ Dataset           │ Total Images   │ Matched        │ Match Rate   │ # Methods    │")
        lines.append("├───────────────────┼────────────────┼────────────────┼──────────────┼──────────────┤")
        
        for dataset in sorted(dataset_stats.keys()):
            stats = dataset_stats[dataset]
            rate = (stats["matched"] / stats["total"] * 100) if stats["total"] > 0 else 0
            lines.append(f"│ {dataset:<17} │ {format_number(stats['total']):>14} │ {format_number(stats['matched']):>14} │ {format_percentage(rate):>12} │ {len(stats['methods']):>12} │")
        
        lines.append("└───────────────────┴────────────────┴────────────────┴──────────────┴──────────────┘")
        lines.append("")
    
    return lines


def generate_report(manifest_dir: Path, detailed: bool = False) -> str:
    """Generate the full report."""
    manifests = load_manifests(manifest_dir)
    
    if not manifests:
        return "No manifests found."
    
    lines = []
    
    # Header
    lines.append("=" * 82)
    lines.append("                    MANIFEST REPORT - GENERATED IMAGES ANALYSIS")
    lines.append("=" * 82)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Manifest Directory: {manifest_dir}")
    lines.append(f"Total Methods: {len(manifests)}")
    lines.append("")
    
    # Count task types
    generation_count = sum(1 for d in manifests.values() if d.get("task_type", "generation") == "generation")
    restoration_count = sum(1 for d in manifests.values() if d.get("task_type") == "restoration")
    
    lines.append(f"  🎨 Generation Methods: {generation_count}")
    lines.append(f"  🔄 Restoration Methods: {restoration_count}")
    lines.append("")
    
    # =========================================================================
    # EXECUTIVE SUMMARY
    # =========================================================================
    lines.append("=" * 82)
    lines.append("                              EXECUTIVE SUMMARY")
    lines.append("=" * 82)
    lines.append("")
    
    # Generation summary table
    lines.append("🎨 GENERATION METHODS (Clear → Adverse Weather)")
    lines.append("-" * 82)
    lines.extend(generate_summary_table(manifests, "generation"))
    lines.append("")
    
    # Restoration summary table
    lines.append("🔄 RESTORATION METHODS (Adverse Weather → Clear)")
    lines.append("-" * 82)
    lines.extend(generate_summary_table(manifests, "restoration"))
    lines.append("")
    
    # =========================================================================
    # DOMAIN SUMMARY
    # =========================================================================
    lines.append("=" * 82)
    lines.append("                              DOMAIN SUMMARY")
    lines.append("=" * 82)
    lines.append("")
    lines.extend(generate_domain_summary(manifests))
    
    # =========================================================================
    # DATASET SUMMARY
    # =========================================================================
    lines.append("=" * 82)
    lines.append("                              DATASET SUMMARY")
    lines.append("=" * 82)
    lines.append("")
    lines.extend(generate_dataset_summary(manifests))
    
    # =========================================================================
    # DETAILED METHOD REPORTS
    # =========================================================================
    lines.append("=" * 82)
    lines.append("                         DETAILED METHOD REPORTS")
    lines.append("=" * 82)
    lines.append("")
    
    # Generation methods first
    lines.append("─" * 82)
    lines.append("                           🎨 GENERATION METHODS")
    lines.append("─" * 82)
    lines.append("")
    
    for name, data in sorted(manifests.items()):
        if data.get("task_type", "generation") == "generation":
            lines.extend(generate_method_section(name, data, detailed))
    
    # Restoration methods
    lines.append("─" * 82)
    lines.append("                           🔄 RESTORATION METHODS")
    lines.append("─" * 82)
    lines.append("")
    
    for name, data in sorted(manifests.items()):
        if data.get("task_type") == "restoration":
            lines.extend(generate_method_section(name, data, detailed))
    
    # =========================================================================
    # NOTES
    # =========================================================================
    lines.append("=" * 82)
    lines.append("                                   NOTES")
    lines.append("=" * 82)
    lines.append("")
    lines.append("• Generation tasks: Transform clear/sunny images to adverse weather conditions")
    lines.append("• Restoration tasks: Remove weather effects from adverse weather images")
    lines.append("• Match Rate: Percentage of generated images matched to original source images")
    lines.append("• For restoration: Original images are from the source weather domain")
    lines.append("  (e.g., 'derained' images are matched against 'rainy' originals)")
    lines.append("• Target exists (✓/✗): Whether target domain exists in AWACS/train for FID")
    lines.append("")
    lines.append("=" * 82)
    
    return "\n".join(lines)


def generate_markdown_report(manifests: Dict, manifest_dir: Path, detailed: bool = False) -> str:
    """Generate a Markdown formatted report."""
    lines = []
    
    # Header
    lines.append("# Manifest Report - Generated Images Analysis")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Manifest Directory:** `{manifest_dir}`")
    lines.append(f"**Total Methods:** {len(manifests)}")
    lines.append("")
    
    # Separate by task type
    generation_methods = {k: v for k, v in manifests.items() 
                         if v.get("task_type", "generation") == "generation"}
    restoration_methods = {k: v for k, v in manifests.items() 
                          if v.get("task_type", "generation") == "restoration"}
    
    lines.append(f"- 🎨 **Generation Methods:** {len(generation_methods)}")
    lines.append(f"- 🔄 **Restoration Methods:** {len(restoration_methods)}")
    lines.append("")
    
    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    
    # Generation table
    if generation_methods:
        lines.append("### 🎨 Generation Methods (Clear → Adverse Weather)")
        lines.append("")
        lines.append("| Method | Total | Matched | Rate |")
        lines.append("|--------|------:|--------:|-----:|")
        
        gen_total = gen_matched = 0
        for name, data in sorted(generation_methods.items()):
            total = data.get("total_generated", 0)
            matched = data.get("total_matched", 0)
            rate = (matched / total * 100) if total > 0 else 0
            lines.append(f"| {name} | {total:,} | {matched:,} | {rate:.1f}% |")
            gen_total += total
            gen_matched += matched
        
        gen_rate = (gen_matched / gen_total * 100) if gen_total > 0 else 0
        lines.append(f"| **TOTAL** | **{gen_total:,}** | **{gen_matched:,}** | **{gen_rate:.1f}%** |")
        lines.append("")
    
    # Restoration table
    if restoration_methods:
        lines.append("### 🔄 Restoration Methods (Adverse Weather → Clear)")
        lines.append("")
        lines.append("| Method | Total | Matched | Rate |")
        lines.append("|--------|------:|--------:|-----:|")
        
        rest_total = rest_matched = 0
        for name, data in sorted(restoration_methods.items()):
            total = data.get("total_generated", 0)
            matched = data.get("total_matched", 0)
            rate = (matched / total * 100) if total > 0 else 0
            lines.append(f"| {name} | {total:,} | {matched:,} | {rate:.1f}% |")
            rest_total += total
            rest_matched += matched
        
        rest_rate = (rest_matched / rest_total * 100) if rest_total > 0 else 0
        lines.append(f"| **TOTAL** | **{rest_total:,}** | **{rest_matched:,}** | **{rest_rate:.1f}%** |")
        lines.append("")
    
    # Domain Summary
    lines.append("## Domain Summary")
    lines.append("")
    
    # Gather domain statistics
    gen_domains = defaultdict(lambda: {"total": 0, "matched": 0, "methods": set()})
    rest_domains = defaultdict(lambda: {"total": 0, "matched": 0, "methods": set(), "source": ""})
    
    source_mapping = {
        "derained": "rainy",
        "dehazed": "foggy",
        "desnowed": "snowy",
        "night2day": "night"
    }
    
    for name, data in manifests.items():
        task_type = data.get("task_type", "generation")
        for domain, domain_data in data.get("domains", {}).items():
            if task_type == "generation":
                gen_domains[domain]["total"] += domain_data.get("total", 0)
                gen_domains[domain]["matched"] += domain_data.get("matched", 0)
                gen_domains[domain]["methods"].add(name)
            else:
                rest_domains[domain]["total"] += domain_data.get("total", 0)
                rest_domains[domain]["matched"] += domain_data.get("matched", 0)
                rest_domains[domain]["methods"].add(name)
                rest_domains[domain]["source"] = source_mapping.get(domain, "unknown")
    
    if gen_domains:
        lines.append("### Generation Domains")
        lines.append("")
        lines.append("| Domain | Total | Matched | Rate | Methods |")
        lines.append("|--------|------:|--------:|-----:|--------:|")
        for domain in sorted(gen_domains.keys()):
            d = gen_domains[domain]
            rate = (d["matched"] / d["total"] * 100) if d["total"] > 0 else 0
            lines.append(f"| {domain} | {d['total']:,} | {d['matched']:,} | {rate:.1f}% | {len(d['methods'])} |")
        lines.append("")
    
    if rest_domains:
        lines.append("### Restoration Domains")
        lines.append("")
        lines.append("| Domain | Source | Total | Matched | Rate | Methods |")
        lines.append("|--------|--------|------:|--------:|-----:|--------:|")
        for domain in sorted(rest_domains.keys()):
            d = rest_domains[domain]
            rate = (d["matched"] / d["total"] * 100) if d["total"] > 0 else 0
            lines.append(f"| {domain} | {d['source']} | {d['total']:,} | {d['matched']:,} | {rate:.1f}% | {len(d['methods'])} |")
        lines.append("")
    
    # Dataset summary
    lines.append("## Dataset Summary")
    lines.append("")
    dataset_stats = defaultdict(lambda: {"total": 0, "matched": 0, "methods": set()})
    
    for name, data in manifests.items():
        # Datasets are nested inside each domain
        for domain, domain_data in data.get("domains", {}).items():
            for dataset, ds_data in domain_data.get("datasets", {}).items():
                dataset_stats[dataset]["total"] += ds_data.get("total", 0)
                dataset_stats[dataset]["matched"] += ds_data.get("matched", 0)
                dataset_stats[dataset]["methods"].add(name)
            dataset_stats[dataset]["methods"].add(name)
    
    lines.append("| Dataset | Total | Matched | Rate | Methods |")
    lines.append("|---------|------:|--------:|-----:|--------:|")
    for dataset in sorted(dataset_stats.keys()):
        d = dataset_stats[dataset]
        rate = (d["matched"] / d["total"] * 100) if d["total"] > 0 else 0
        lines.append(f"| {dataset} | {d['total']:,} | {d['matched']:,} | {rate:.1f}% | {len(d['methods'])} |")
    lines.append("")
    
    # Notes
    lines.append("## Notes")
    lines.append("")
    lines.append("- **Generation tasks:** Transform clear/sunny images to adverse weather conditions")
    lines.append("- **Restoration tasks:** Remove weather effects from adverse weather images")
    lines.append("- **Match Rate:** Percentage of generated images matched to original source images")
    lines.append("- For restoration, original images are from the source weather domain (e.g., 'derained' matched against 'rainy' originals)")
    lines.append("")
    
    return "\n".join(lines)


def generate_json_report(manifests: Dict, manifest_dir: Path) -> str:
    """Generate a JSON formatted report."""
    
    # Separate by task type
    generation_methods = {k: v for k, v in manifests.items() 
                         if v.get("task_type", "generation") == "generation"}
    restoration_methods = {k: v for k, v in manifests.items() 
                          if v.get("task_type", "generation") == "restoration"}
    
    # Calculate totals
    gen_total = sum(d.get("total_generated", 0) for d in generation_methods.values())
    gen_matched = sum(d.get("total_matched", 0) for d in generation_methods.values())
    rest_total = sum(d.get("total_generated", 0) for d in restoration_methods.values())
    rest_matched = sum(d.get("total_matched", 0) for d in restoration_methods.values())
    
    # Domain statistics
    source_mapping = {
        "derained": "rainy",
        "dehazed": "foggy",
        "desnowed": "snowy",
        "night2day": "night"
    }
    
    gen_domains = defaultdict(lambda: {"total": 0, "matched": 0, "methods": []})
    rest_domains = defaultdict(lambda: {"total": 0, "matched": 0, "methods": [], "source_weather": ""})
    dataset_stats = defaultdict(lambda: {"total": 0, "matched": 0, "methods": []})
    
    for name, data in manifests.items():
        task_type = data.get("task_type", "generation")
        for domain, domain_data in data.get("domains", {}).items():
            if task_type == "generation":
                gen_domains[domain]["total"] += domain_data.get("total", 0)
                gen_domains[domain]["matched"] += domain_data.get("matched", 0)
                if name not in gen_domains[domain]["methods"]:
                    gen_domains[domain]["methods"].append(name)
            else:
                rest_domains[domain]["total"] += domain_data.get("total", 0)
                rest_domains[domain]["matched"] += domain_data.get("matched", 0)
                if name not in rest_domains[domain]["methods"]:
                    rest_domains[domain]["methods"].append(name)
                rest_domains[domain]["source_weather"] = source_mapping.get(domain, "unknown")
            
            # Datasets are nested inside each domain
            for dataset, ds_data in domain_data.get("datasets", {}).items():
                dataset_stats[dataset]["total"] += ds_data.get("total", 0)
                dataset_stats[dataset]["matched"] += ds_data.get("matched", 0)
                if name not in dataset_stats[dataset]["methods"]:
                    dataset_stats[dataset]["methods"].append(name)
    
    # Convert defaultdicts to regular dicts
    gen_domains = {k: dict(v) for k, v in gen_domains.items()}
    rest_domains = {k: dict(v) for k, v in rest_domains.items()}
    dataset_stats = {k: dict(v) for k, v in dataset_stats.items()}
    
    report = {
        "generated_at": datetime.now().isoformat(),
        "manifest_directory": str(manifest_dir),
        "summary": {
            "total_methods": len(manifests),
            "generation_methods": len(generation_methods),
            "restoration_methods": len(restoration_methods),
            "generation": {
                "total_images": gen_total,
                "total_matched": gen_matched,
                "match_rate": (gen_matched / gen_total * 100) if gen_total > 0 else 0
            },
            "restoration": {
                "total_images": rest_total,
                "total_matched": rest_matched,
                "match_rate": (rest_matched / rest_total * 100) if rest_total > 0 else 0
            }
        },
        "domains": {
            "generation": gen_domains,
            "restoration": rest_domains
        },
        "datasets": dataset_stats,
        "methods": manifests
    }
    
    return json.dumps(report, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a comprehensive report from manifest files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest-dir", type=Path,
        default=Path(__file__).parent.parent / "manifests",
        help="Directory containing manifest subdirectories"
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output file path. If not specified, prints to stdout."
    )
    parser.add_argument(
        "--detailed", action="store_true",
        help="Include detailed dataset breakdown per method"
    )
    parser.add_argument(
        "--format", choices=["text", "markdown", "json", "all"],
        default="text",
        help="Output format: text, markdown, json, or all"
    )
    
    args = parser.parse_args()
    
    if not args.manifest_dir.exists():
        print(f"Error: Manifest directory not found: {args.manifest_dir}")
        return 1
    
    # Load manifests once
    manifests = load_manifests(args.manifest_dir)
    if not manifests:
        print(f"Error: No manifest files found in {args.manifest_dir}")
        return 1
    
    if args.format == "all":
        formats = ["text", "markdown", "json"]
    else:
        formats = [args.format]
    
    for fmt in formats:
        if fmt == "text":
            report = generate_report(args.manifest_dir, args.detailed)
            ext = ".txt"
        elif fmt == "markdown":
            report = generate_markdown_report(manifests, args.manifest_dir, args.detailed)
            ext = ".md"
        elif fmt == "json":
            report = generate_json_report(manifests, args.manifest_dir)
            ext = ".json"
        
        if args.output:
            if len(formats) > 1:
                out_path = args.output.parent / f"{args.output.stem}{ext}"
            else:
                out_path = args.output
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, 'w') as f:
                f.write(report)
            print(f"Report written to: {out_path}")
        else:
            print(report)
    
    return 0


if __name__ == "__main__":
    exit(main())
