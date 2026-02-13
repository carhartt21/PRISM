#!/usr/bin/env python3
"""
summarize_results.py

Collects and summarizes evaluation results from multiple image generation methods.
Computes a Composite Quality Score (CQS) for ranking different models.

Usage:
    python summarize_results.py --results-dir ./results --output summary_report.json
    python summarize_results.py --results-files result1.json result2.json --output summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


@dataclass
class MethodMetrics:
    """Container for a single method's metrics."""
    method_name: str
    fid: Optional[float] = None
    lpips: Optional[float] = None
    ssim: Optional[float] = None
    psnr: Optional[float] = None
    pixel_accuracy: Optional[float] = None  # semantic_pixel_accuracy
    miou: Optional[float] = None  # semantic_mIoU
    fw_iou: Optional[float] = None  # semantic_fw_IoU (weighted mIoU)
    num_images: int = 0
    
    # Per-domain breakdown
    domain_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # CQS score (computed later)
    cqs: Optional[float] = None
    cqs_rank: Optional[int] = None


@dataclass
class GlobalStats:
    """Global statistics across all methods for z-score normalization."""
    fid_mean: float = 0.0
    fid_std: float = 1.0
    lpips_mean: float = 0.0
    lpips_std: float = 1.0
    pa_mean: float = 0.0  # pixel accuracy
    pa_std: float = 1.0
    wmiou_mean: float = 0.0  # weighted mIoU (fw_IoU)
    wmiou_std: float = 1.0


def extract_metric_value(metrics_dict: Dict[str, Any], key: str) -> Optional[float]:
    """Extract a metric value from various possible structures."""
    if key not in metrics_dict:
        return None
    
    value = metrics_dict[key]
    
    # Handle nested structures
    if isinstance(value, dict):
        # Check for 'value' key (FID format)
        if 'value' in value:
            v = value['value']
        # Check for 'mean' key (aggregate format)
        elif 'mean' in value:
            v = value['mean']
        else:
            return None
    else:
        v = value
    
    # Handle NaN
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    
    return float(v)


def load_result_file(filepath: Path) -> Optional[Dict[str, Any]]:
    """Load and parse a single result JSON file."""
    try:
        with open(filepath) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logging.error(f"Failed to load {filepath}: {e}")
        return None


def extract_method_name(filepath: Path, data: Dict[str, Any]) -> str:
    """Extract method name from filepath or data."""
    # Try to get from data
    if 'generated_root' in data:
        gen_root = Path(data['generated_root'])
        return gen_root.name
    
    # Fall back to filename
    return filepath.stem.replace('_evaluation_results', '').replace('_results', '')


def extract_method_metrics(data: Dict[str, Any], method_name: str) -> MethodMetrics:
    """Extract metrics from a result file into MethodMetrics structure."""
    metrics = MethodMetrics(method_name=method_name)
    
    # Check if this is a per-domain result
    if 'domains' in data:
        # Aggregate metrics from domains
        all_fid = []
        all_lpips = []
        all_ssim = []
        all_psnr = []
        all_pa = []
        all_miou = []
        all_fw_iou = []
        total_images = 0
        
        for domain_name, domain_data in data['domains'].items():
            if 'error' in domain_data:
                continue
            
            domain_metrics = domain_data.get('metrics', {})
            num_pairs = domain_data.get('num_pairs', 0)
            total_images += num_pairs
            
            # Extract domain-level metrics
            dm = {}
            
            fid = extract_metric_value(domain_metrics, 'fid')
            if fid is not None:
                all_fid.append((fid, num_pairs))
                dm['fid'] = fid
            
            lpips = extract_metric_value(domain_metrics, 'lpips')
            if lpips is not None:
                all_lpips.append((lpips, num_pairs))
                dm['lpips'] = lpips
            
            ssim = extract_metric_value(domain_metrics, 'ssim')
            if ssim is not None:
                all_ssim.append((ssim, num_pairs))
                dm['ssim'] = ssim
            
            psnr = extract_metric_value(domain_metrics, 'psnr')
            if psnr is not None:
                all_psnr.append((psnr, num_pairs))
                dm['psnr'] = psnr
            
            # Semantic metrics
            pa = extract_metric_value(domain_metrics, 'semantic_pixel_accuracy')
            if pa is not None:
                all_pa.append((pa, num_pairs))
                dm['pixel_accuracy'] = pa
            
            miou = extract_metric_value(domain_metrics, 'semantic_mIoU')
            if miou is not None:
                all_miou.append((miou, num_pairs))
                dm['mIoU'] = miou
            
            fw_iou = extract_metric_value(domain_metrics, 'semantic_fw_IoU')
            if fw_iou is not None:
                all_fw_iou.append((fw_iou, num_pairs))
                dm['fw_IoU'] = fw_iou
            
            # Check semantic_consistency section as alternative source
            if 'semantic_consistency' in domain_data:
                sc = domain_data['semantic_consistency']
                summary = sc.get('summary', {})
                
                if pa is None and 'average_pixel_accuracy' in summary:
                    pa_val = summary['average_pixel_accuracy']
                    if pa_val is not None:
                        all_pa.append((pa_val, num_pairs))
                        dm['pixel_accuracy'] = pa_val
                
                if miou is None and 'average_mIoU' in summary:
                    miou_val = summary['average_mIoU']
                    if miou_val is not None:
                        all_miou.append((miou_val, num_pairs))
                        dm['mIoU'] = miou_val
                
                if fw_iou is None and 'average_fw_IoU' in summary:
                    fw_iou_val = summary['average_fw_IoU']
                    if fw_iou_val is not None:
                        all_fw_iou.append((fw_iou_val, num_pairs))
                        dm['fw_IoU'] = fw_iou_val
            
            dm['num_images'] = num_pairs
            metrics.domain_metrics[domain_name] = dm
        
        # Compute weighted averages
        metrics.num_images = total_images
        
        if all_fid:
            # For FID, use simple average across domains (not weighted by image count)
            metrics.fid = np.mean([f for f, _ in all_fid])
        
        if all_lpips:
            total_weight = sum(w for _, w in all_lpips)
            metrics.lpips = sum(v * w for v, w in all_lpips) / total_weight if total_weight > 0 else None
        
        if all_ssim:
            total_weight = sum(w for _, w in all_ssim)
            metrics.ssim = sum(v * w for v, w in all_ssim) / total_weight if total_weight > 0 else None
        
        if all_psnr:
            total_weight = sum(w for _, w in all_psnr)
            metrics.psnr = sum(v * w for v, w in all_psnr) / total_weight if total_weight > 0 else None
        
        if all_pa:
            total_weight = sum(w for _, w in all_pa)
            metrics.pixel_accuracy = sum(v * w for v, w in all_pa) / total_weight if total_weight > 0 else None
        
        if all_miou:
            total_weight = sum(w for _, w in all_miou)
            metrics.miou = sum(v * w for v, w in all_miou) / total_weight if total_weight > 0 else None
        
        if all_fw_iou:
            total_weight = sum(w for _, w in all_fw_iou)
            metrics.fw_iou = sum(v * w for v, w in all_fw_iou) / total_weight if total_weight > 0 else None
    
    # Also check aggregate_metrics if present
    if 'aggregate_metrics' in data:
        agg = data['aggregate_metrics']
        
        if metrics.fid is None:
            metrics.fid = extract_metric_value(agg, 'fid')
        if metrics.lpips is None:
            metrics.lpips = extract_metric_value(agg, 'lpips')
        if metrics.ssim is None:
            metrics.ssim = extract_metric_value(agg, 'ssim')
        if metrics.psnr is None:
            metrics.psnr = extract_metric_value(agg, 'psnr')
        if metrics.pixel_accuracy is None:
            metrics.pixel_accuracy = extract_metric_value(agg, 'semantic_pixel_accuracy')
        if metrics.miou is None:
            metrics.miou = extract_metric_value(agg, 'semantic_mIoU')
        if metrics.fw_iou is None:
            metrics.fw_iou = extract_metric_value(agg, 'semantic_fw_IoU')
        
        if metrics.num_images == 0 and 'total_images' in data:
            metrics.num_images = data['total_images']
    
    # Check for flat structure (single domain evaluation)
    if 'metrics' in data and 'domains' not in data:
        m = data['metrics']
        if metrics.fid is None:
            metrics.fid = extract_metric_value(m, 'fid')
        if metrics.lpips is None:
            metrics.lpips = extract_metric_value(m, 'lpips')
        if metrics.ssim is None:
            metrics.ssim = extract_metric_value(m, 'ssim')
        if metrics.psnr is None:
            metrics.psnr = extract_metric_value(m, 'psnr')
        if metrics.pixel_accuracy is None:
            metrics.pixel_accuracy = extract_metric_value(m, 'semantic_pixel_accuracy')
        if metrics.miou is None:
            metrics.miou = extract_metric_value(m, 'semantic_mIoU')
        if metrics.fw_iou is None:
            metrics.fw_iou = extract_metric_value(m, 'semantic_fw_IoU')
        
        if metrics.num_images == 0:
            metrics.num_images = data.get('num_pairs', 0)
    
    return metrics


def compute_global_stats(all_methods: List[MethodMetrics]) -> GlobalStats:
    """Compute global statistics across all methods for z-score normalization."""
    fid_values = [m.fid for m in all_methods if m.fid is not None]
    lpips_values = [m.lpips for m in all_methods if m.lpips is not None]
    pa_values = [m.pixel_accuracy for m in all_methods if m.pixel_accuracy is not None]
    wmiou_values = [m.fw_iou for m in all_methods if m.fw_iou is not None]
    
    stats = GlobalStats()
    
    if fid_values:
        stats.fid_mean = float(np.mean(fid_values))
        stats.fid_std = float(np.std(fid_values, ddof=1)) if len(fid_values) > 1 else 1.0
        if stats.fid_std == 0:
            stats.fid_std = 1.0
    
    if lpips_values:
        stats.lpips_mean = float(np.mean(lpips_values))
        stats.lpips_std = float(np.std(lpips_values, ddof=1)) if len(lpips_values) > 1 else 1.0
        if stats.lpips_std == 0:
            stats.lpips_std = 1.0
    
    if pa_values:
        stats.pa_mean = float(np.mean(pa_values))
        stats.pa_std = float(np.std(pa_values, ddof=1)) if len(pa_values) > 1 else 1.0
        if stats.pa_std == 0:
            stats.pa_std = 1.0
    
    if wmiou_values:
        stats.wmiou_mean = float(np.mean(wmiou_values))
        stats.wmiou_std = float(np.std(wmiou_values, ddof=1)) if len(wmiou_values) > 1 else 1.0
        if stats.wmiou_std == 0:
            stats.wmiou_std = 1.0
    
    return stats


def compute_cqs(
    fid: Optional[float],
    lpips: Optional[float],
    pa: Optional[float],
    wmiou: Optional[float],
    global_stats: GlobalStats,
    weights: Optional[Dict[str, float]] = None,
) -> Optional[float]:
    """
    Compute Composite Quality Score (CQS) using z-score normalization.
    
    The CQS combines multiple metrics into a single score where LOWER is BETTER.
    
    Formula:
        CQS = w_fid * z(FID) + w_lpips * z(LPIPS) + w_pa * (1 - z(PA)) + w_wmiou * (1 - z(wMIoU))
    
    Where:
        - z(x) = (x - mean) / std (z-score normalization)
        - FID and LPIPS: lower is better, so we use z-scores directly
        - PA and wMIoU: higher is better, so we use (1 - z-score) to invert
    
    Default weights: FID=0.4, LPIPS=0.2, PA=0.2, wMIoU=0.2
    
    Args:
        fid: Fréchet Inception Distance
        lpips: Learned Perceptual Image Patch Similarity
        pa: Pixel Accuracy (semantic consistency)
        wmiou: Weighted Mean Intersection over Union (frequency-weighted)
        global_stats: Global statistics for normalization
        weights: Optional custom weights for each metric
    
    Returns:
        Composite Quality Score (lower is better), or None if insufficient metrics
    """
    if weights is None:
        weights = {
            'fid': 0.4,
            'lpips': 0.2,
            'pa': 0.2,
            'wmiou': 0.2,
        }
    
    score = 0.0
    total_weight = 0.0
    
    # FID: lower is better
    if fid is not None and global_stats.fid_std > 0:
        fid_norm = (fid - global_stats.fid_mean) / global_stats.fid_std
        score += weights['fid'] * fid_norm
        total_weight += weights['fid']
    
    # LPIPS: lower is better
    if lpips is not None and global_stats.lpips_std > 0:
        lpips_norm = (lpips - global_stats.lpips_mean) / global_stats.lpips_std
        score += weights['lpips'] * lpips_norm
        total_weight += weights['lpips']
    
    # Pixel Accuracy: higher is better, so invert
    if pa is not None and global_stats.pa_std > 0:
        pa_norm = (pa - global_stats.pa_mean) / global_stats.pa_std
        score += weights['pa'] * (1 - pa_norm)  # Invert: lower z-score is better
        total_weight += weights['pa']
    
    # Weighted mIoU: higher is better, so invert
    if wmiou is not None and global_stats.wmiou_std > 0:
        wmiou_norm = (wmiou - global_stats.wmiou_mean) / global_stats.wmiou_std
        score += weights['wmiou'] * (1 - wmiou_norm)  # Invert: lower z-score is better
        total_weight += weights['wmiou']
    
    if total_weight == 0:
        return None
    
    # Normalize by total weight used (in case some metrics are missing)
    return score / total_weight * sum(weights.values())


def rank_methods(all_methods: List[MethodMetrics]) -> List[MethodMetrics]:
    """Compute CQS and rank methods. Lower CQS is better (rank 1 is best)."""
    # Compute global stats
    global_stats = compute_global_stats(all_methods)
    
    # Compute CQS for each method
    for method in all_methods:
        method.cqs = compute_cqs(
            fid=method.fid,
            lpips=method.lpips,
            pa=method.pixel_accuracy,
            wmiou=method.fw_iou,
            global_stats=global_stats,
        )
    
    # Sort by CQS (lower is better), methods without CQS go to the end
    methods_with_cqs = [m for m in all_methods if m.cqs is not None]
    methods_without_cqs = [m for m in all_methods if m.cqs is None]
    
    methods_with_cqs.sort(key=lambda m: m.cqs)
    
    # Assign ranks
    for i, method in enumerate(methods_with_cqs, 1):
        method.cqs_rank = i
    
    for method in methods_without_cqs:
        method.cqs_rank = None
    
    return methods_with_cqs + methods_without_cqs


def collect_results(
    results_dir: Optional[Path] = None,
    results_files: Optional[List[Path]] = None,
) -> List[MethodMetrics]:
    """Collect results from directory or specific files."""
    all_methods = []
    
    files_to_process = []
    
    if results_files:
        files_to_process.extend(results_files)
    
    if results_dir and results_dir.exists():
        # Find all JSON result files in directory
        for json_file in results_dir.glob("*_results.json"):
            files_to_process.append(json_file)
        for json_file in results_dir.glob("*_evaluation_results.json"):
            if json_file not in files_to_process:
                files_to_process.append(json_file)
        # Also check for results.json files
        for json_file in results_dir.glob("*/results.json"):
            files_to_process.append(json_file)
    
    # Remove duplicates
    files_to_process = list(set(files_to_process))
    
    logging.info(f"Found {len(files_to_process)} result files to process")
    
    for filepath in files_to_process:
        logging.info(f"Processing: {filepath}")
        data = load_result_file(filepath)
        if data is None:
            continue
        
        method_name = extract_method_name(filepath, data)
        metrics = extract_method_metrics(data, method_name)
        
        if metrics.num_images > 0 or metrics.fid is not None or metrics.lpips is not None:
            all_methods.append(metrics)
            fid_str = f"{metrics.fid:.2f}" if metrics.fid is not None else "N/A"
            lpips_str = f"{metrics.lpips:.4f}" if metrics.lpips is not None else "N/A"
            logging.info(
                f"  {method_name}: {metrics.num_images} images, "
                f"FID={fid_str}, LPIPS={lpips_str}"
            )
        else:
            logging.warning(f"  {method_name}: No valid metrics found")
    
    return all_methods


def generate_summary_report(
    all_methods: List[MethodMetrics],
    global_stats: GlobalStats,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Generate a comprehensive summary report."""
    
    # Sort by CQS rank
    ranked_methods = rank_methods(all_methods)
    
    # Recompute global stats after ranking
    global_stats = compute_global_stats(ranked_methods)
    
    report = {
        "summary": {
            "num_methods": len(ranked_methods),
            "methods_with_cqs": len([m for m in ranked_methods if m.cqs is not None]),
        },
        "global_statistics": {
            "fid": {"mean": global_stats.fid_mean, "std": global_stats.fid_std},
            "lpips": {"mean": global_stats.lpips_mean, "std": global_stats.lpips_std},
            "pixel_accuracy": {"mean": global_stats.pa_mean, "std": global_stats.pa_std},
            "fw_iou": {"mean": global_stats.wmiou_mean, "std": global_stats.wmiou_std},
        },
        "cqs_weights": {
            "fid": 0.4,
            "lpips": 0.2,
            "pixel_accuracy": 0.2,
            "fw_iou": 0.2,
        },
        "ranking": [],
        "per_method_details": {},
    }
    
    # Build ranking table
    for method in ranked_methods:
        entry = {
            "rank": method.cqs_rank,
            "method": method.method_name,
            "cqs": round(method.cqs, 4) if method.cqs is not None else None,
            "fid": round(method.fid, 2) if method.fid is not None else None,
            "lpips": round(method.lpips, 4) if method.lpips is not None else None,
            "ssim": round(method.ssim, 4) if method.ssim is not None else None,
            "pixel_accuracy": round(method.pixel_accuracy, 2) if method.pixel_accuracy is not None else None,
            "miou": round(method.miou, 2) if method.miou is not None else None,
            "fw_iou": round(method.fw_iou, 2) if method.fw_iou is not None else None,
            "num_images": method.num_images,
        }
        report["ranking"].append(entry)
        
        # Detailed per-method info
        report["per_method_details"][method.method_name] = {
            "aggregate_metrics": {
                "fid": method.fid,
                "lpips": method.lpips,
                "ssim": method.ssim,
                "psnr": method.psnr,
                "pixel_accuracy": method.pixel_accuracy,
                "miou": method.miou,
                "fw_iou": method.fw_iou,
            },
            "cqs": method.cqs,
            "cqs_rank": method.cqs_rank,
            "num_images": method.num_images,
            "per_domain": method.domain_metrics,
        }
    
    # Save to file if path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        logging.info(f"Summary report saved to {output_path}")
    
    return report


def print_ranking_table(ranked_methods: List[MethodMetrics]) -> None:
    """Print a formatted ranking table to console."""
    print("\n" + "=" * 120)
    print("COMPOSITE QUALITY SCORE (CQS) RANKING")
    print("Lower CQS is better. Rank 1 = Best performing method.")
    print("=" * 120)
    
    # Header
    header = f"{'Rank':>4} | {'Method':<35} | {'CQS':>8} | {'FID':>8} | {'LPIPS':>7} | {'PA':>7} | {'fw-IoU':>7} | {'Images':>10}"
    print(header)
    print("-" * 120)
    
    for method in ranked_methods:
        rank_str = str(method.cqs_rank) if method.cqs_rank else "-"
        cqs_str = f"{method.cqs:.4f}" if method.cqs is not None else "N/A"
        fid_str = f"{method.fid:.2f}" if method.fid is not None else "N/A"
        lpips_str = f"{method.lpips:.4f}" if method.lpips is not None else "N/A"
        pa_str = f"{method.pixel_accuracy:.2f}" if method.pixel_accuracy is not None else "N/A"
        fwiou_str = f"{method.fw_iou:.2f}" if method.fw_iou is not None else "N/A"
        
        row = f"{rank_str:>4} | {method.method_name:<35} | {cqs_str:>8} | {fid_str:>8} | {lpips_str:>7} | {pa_str:>7} | {fwiou_str:>7} | {method.num_images:>10}"
        print(row)
    
    print("=" * 120)
    print("\nMetric Legend:")
    print("  CQS:    Composite Quality Score (lower is better)")
    print("  FID:    Fréchet Inception Distance (lower is better)")
    print("  LPIPS:  Learned Perceptual Image Patch Similarity (lower is better)")
    print("  PA:     Pixel Accuracy - semantic consistency (higher is better)")
    print("  fw-IoU: Frequency-weighted Mean IoU - semantic consistency (higher is better)")
    print()


def generate_markdown_report(
    ranked_methods: List[MethodMetrics],
    global_stats: GlobalStats,
    output_path: Path,
) -> None:
    """Generate a markdown report for documentation."""
    lines = [
        "# Image Generation Methods Evaluation Report",
        "",
        "## Summary",
        "",
        f"- **Total Methods Evaluated**: {len(ranked_methods)}",
        f"- **Methods with CQS Score**: {len([m for m in ranked_methods if m.cqs is not None])}",
        "",
        "## Composite Quality Score (CQS)",
        "",
        "The CQS combines multiple metrics into a single ranking score using z-score normalization:",
        "",
        "```",
        "CQS = 0.4 × z(FID) + 0.2 × z(LPIPS) + 0.2 × (1 - z(PA)) + 0.2 × (1 - z(fw-IoU))",
        "```",
        "",
        "Where **lower CQS is better**.",
        "",
        "### Global Statistics (for z-score normalization)",
        "",
        f"| Metric | Mean | Std |",
        f"|--------|------|-----|",
        f"| FID | {global_stats.fid_mean:.2f} | {global_stats.fid_std:.2f} |",
        f"| LPIPS | {global_stats.lpips_mean:.4f} | {global_stats.lpips_std:.4f} |",
        f"| Pixel Accuracy | {global_stats.pa_mean:.2f} | {global_stats.pa_std:.2f} |",
        f"| fw-IoU | {global_stats.wmiou_mean:.2f} | {global_stats.wmiou_std:.2f} |",
        "",
        "## Ranking Table",
        "",
        "| Rank | Method | CQS | FID ↓ | LPIPS ↓ | SSIM ↑ | PA ↑ | fw-IoU ↑ | Images |",
        "|------|--------|-----|-------|---------|--------|------|----------|--------|",
    ]
    
    for method in ranked_methods:
        rank = method.cqs_rank if method.cqs_rank else "-"
        cqs = f"{method.cqs:.4f}" if method.cqs is not None else "N/A"
        fid = f"{method.fid:.2f}" if method.fid is not None else "N/A"
        lpips = f"{method.lpips:.4f}" if method.lpips is not None else "N/A"
        ssim = f"{method.ssim:.4f}" if method.ssim is not None else "N/A"
        pa = f"{method.pixel_accuracy:.2f}" if method.pixel_accuracy is not None else "N/A"
        fwiou = f"{method.fw_iou:.2f}" if method.fw_iou is not None else "N/A"
        
        lines.append(f"| {rank} | {method.method_name} | {cqs} | {fid} | {lpips} | {ssim} | {pa} | {fwiou} | {method.num_images} |")
    
    lines.extend([
        "",
        "## Metric Descriptions",
        "",
        "- **FID (Fréchet Inception Distance)**: Measures distribution similarity between generated and real images. Lower is better.",
        "- **LPIPS (Learned Perceptual Image Patch Similarity)**: Measures perceptual difference using deep features. Lower is better.",
        "- **SSIM (Structural Similarity Index)**: Measures structural similarity. Higher is better.",
        "- **PA (Pixel Accuracy)**: Semantic segmentation consistency between original and generated. Higher is better.",
        "- **fw-IoU (Frequency-weighted IoU)**: Class-weighted semantic consistency. Higher is better.",
        "",
        "## Per-Domain Results",
        "",
    ])
    
    # Add per-domain breakdown for top 5 methods
    top_methods = [m for m in ranked_methods if m.cqs_rank and m.cqs_rank <= 5]
    for method in top_methods:
        if method.domain_metrics:
            lines.append(f"### {method.method_name}")
            lines.append("")
            lines.append("| Domain | FID | LPIPS | PA | fw-IoU | Images |")
            lines.append("|--------|-----|-------|-------|--------|--------|")
            
            for domain, dm in sorted(method.domain_metrics.items()):
                fid = f"{dm.get('fid', 'N/A'):.2f}" if dm.get('fid') else "N/A"
                lpips = f"{dm.get('lpips', 'N/A'):.4f}" if dm.get('lpips') else "N/A"
                pa = f"{dm.get('pixel_accuracy', 'N/A'):.2f}" if dm.get('pixel_accuracy') else "N/A"
                fwiou = f"{dm.get('fw_IoU', 'N/A'):.2f}" if dm.get('fw_IoU') else "N/A"
                num = dm.get('num_images', 0)
                lines.append(f"| {domain} | {fid} | {lpips} | {pa} | {fwiou} | {num} |")
            
            lines.append("")
    
    output_path.write_text("\n".join(lines))
    logging.info(f"Markdown report saved to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect and summarize evaluation results, compute CQS rankings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="Directory containing result JSON files.",
        default="/scratch/aaa_exchange/AWARE/STATS",
    )
    parser.add_argument(
        "--results-files",
        type=Path,
        nargs="+",
        help="Specific result JSON files to process.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("summary_report.json"),
        help="Output path for the JSON summary report.",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        help="Output path for markdown report (optional).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Output path for CSV ranking table (optional).",
    )
    parser.add_argument(
        "--cqs-weights",
        type=str,
        help="Custom CQS weights as JSON string, e.g., '{\"fid\": 0.3, \"lpips\": 0.3, \"pa\": 0.2, \"wmiou\": 0.2}'",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity.",
    )
    
    return parser.parse_args()


def generate_csv_report(ranked_methods: List[MethodMetrics], output_path: Path) -> None:
    """Generate a CSV ranking table."""
    import csv
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'Rank', 'Method', 'CQS', 'FID', 'LPIPS', 'SSIM', 'PSNR',
            'Pixel_Accuracy', 'mIoU', 'fw_IoU', 'Num_Images'
        ])
        
        # Data rows
        for method in ranked_methods:
            writer.writerow([
                method.cqs_rank if method.cqs_rank else '',
                method.method_name,
                f"{method.cqs:.4f}" if method.cqs is not None else '',
                f"{method.fid:.2f}" if method.fid is not None else '',
                f"{method.lpips:.4f}" if method.lpips is not None else '',
                f"{method.ssim:.4f}" if method.ssim is not None else '',
                f"{method.psnr:.2f}" if method.psnr is not None else '',
                f"{method.pixel_accuracy:.2f}" if method.pixel_accuracy is not None else '',
                f"{method.miou:.2f}" if method.miou is not None else '',
                f"{method.fw_iou:.2f}" if method.fw_iou is not None else '',
                method.num_images,
            ])
    
    logging.info(f"CSV report saved to {output_path}")


def main() -> None:
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not args.results_dir and not args.results_files:
        logging.error("Must specify either --results-dir or --results-files")
        return
    
    # Parse custom CQS weights if provided
    custom_weights = None
    if args.cqs_weights:
        try:
            custom_weights = json.loads(args.cqs_weights)
            logging.info(f"Using custom CQS weights: {custom_weights}")
        except json.JSONDecodeError as e:
            logging.error(f"Invalid CQS weights JSON: {e}")
            return
    
    # Collect results
    all_methods = collect_results(
        results_dir=args.results_dir,
        results_files=args.results_files,
    )
    
    if not all_methods:
        logging.error("No valid results found")
        return
    
    # Compute rankings (with optional custom weights)
    if custom_weights:
        global_stats = compute_global_stats(all_methods)
        for method in all_methods:
            method.cqs = compute_cqs(
                fid=method.fid,
                lpips=method.lpips,
                pa=method.pixel_accuracy,
                wmiou=method.fw_iou,
                global_stats=global_stats,
                weights=custom_weights,
            )
        methods_with_cqs = [m for m in all_methods if m.cqs is not None]
        methods_without_cqs = [m for m in all_methods if m.cqs is None]
        methods_with_cqs.sort(key=lambda m: m.cqs)
        for i, method in enumerate(methods_with_cqs, 1):
            method.cqs_rank = i
        ranked_methods = methods_with_cqs + methods_without_cqs
    else:
        ranked_methods = rank_methods(all_methods)
    
    global_stats = compute_global_stats(ranked_methods)
    
    # Print to console
    print_ranking_table(ranked_methods)
    
    # Generate JSON report
    generate_summary_report(ranked_methods, global_stats, args.output)
    
    # Generate markdown report if requested
    if args.markdown:
        generate_markdown_report(ranked_methods, global_stats, args.markdown)
    
    # Generate CSV report if requested
    if args.csv:
        generate_csv_report(ranked_methods, args.csv)


if __name__ == "__main__":
    main()
