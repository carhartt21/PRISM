#!/usr/bin/env python3
"""
visualize_results.py

Generates comprehensive diagrams and visualizations from PRISM evaluation results
using seaborn and matplotlib. Creates plots for method comparison, metric distributions,
correlations, and domain-wise analysis.

Usage:
    python helper/visualize_results.py --results-dir ./results --output-dir ./plots
    python helper/visualize_results.py --results-files result1.json result2.json --output-dir ./visualizations
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Set seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")

# Metric configurations
METRIC_CONFIG = {
    'fid': {'name': 'FID', 'lower_better': True, 'color': 'red'},
    'lpips': {'name': 'LPIPS', 'lower_better': True, 'color': 'orange'},
    'ssim': {'name': 'SSIM', 'lower_better': False, 'color': 'blue'},
    'psnr': {'name': 'PSNR', 'lower_better': False, 'color': 'green'},
    'semantic_pixel_accuracy': {'name': 'Semantic Pixel Accuracy', 'lower_better': False, 'color': 'purple'},
    'semantic_mIoU': {'name': 'Semantic mIoU', 'lower_better': False, 'color': 'brown'},
    'semantic_fw_IoU': {'name': 'Semantic fw-IoU', 'lower_better': False, 'color': 'pink'},
}

DOMAIN_ORDER = ['clear_day', 'cloudy', 'dawn_dusk', 'foggy', 'night', 'rainy', 'snowy']


def load_results(results_dir: Optional[Path] = None,
                results_files: Optional[List[Path]] = None) -> Dict[str, Dict[str, Any]]:
    """
    Load evaluation results from directory or specific files.

    Returns:
        Dict mapping method names to their result data
    """
    results = {}

    if results_dir:
        for json_file in results_dir.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    method_name = json_file.stem
                    results[method_name] = data
            except Exception as e:
                logging.warning(f"Failed to load {json_file}: {e}")

    if results_files:
        for json_file in results_files:
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    method_name = json_file.stem
                    results[method_name] = data
            except Exception as e:
                logging.warning(f"Failed to load {json_file}: {e}")

    return results


def extract_method_metrics(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Extract metrics for each method into a DataFrame.

    Returns:
        DataFrame with columns: method, metric, value, domain
    """
    rows = []

    for method, data in results.items():
        # Overall metrics
        if 'metrics' in data:
            for metric_key, metric_data in data['metrics'].items():
                if metric_key in METRIC_CONFIG and 'mean' in metric_data:
                    rows.append({
                        'method': method,
                        'metric': metric_key,
                        'value': metric_data['mean'],
                        'domain': 'overall'
                    })

        # Domain-specific metrics
        if 'domains' in data:
            for domain, domain_data in data['domains'].items():
                if 'metrics' in domain_data:
                    for metric_key, metric_data in domain_data['metrics'].items():
                        if metric_key in METRIC_CONFIG and 'mean' in metric_data:
                            rows.append({
                                'method': method,
                                'metric': metric_key,
                                'value': metric_data['mean'],
                                'domain': domain
                            })

    return pd.DataFrame(rows)


def create_method_comparison_plot(df: pd.DataFrame, output_dir: Path) -> None:
    """Create bar plot comparing methods across all metrics."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    metrics = list(METRIC_CONFIG.keys())
    methods = sorted(df['method'].unique())

    for i, metric in enumerate(metrics):
        if i >= len(axes):
            break

        ax = axes[i]
        metric_data = df[(df['metric'] == metric) & (df['domain'] == 'overall')]

        if not metric_data.empty:
            sns.barplot(data=metric_data, x='method', y='value', ax=ax,
                       order=methods, color=METRIC_CONFIG[metric]['color'])
            ax.set_title(f'{METRIC_CONFIG[metric]["name"]} Comparison')
            ax.set_ylabel(METRIC_CONFIG[metric]['name'])
            ax.tick_params(axis='x', rotation=45)

            # Add value labels on bars
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.3f}',
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='center', xytext=(0, 10),
                           textcoords='offset points', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'method_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_metric_distributions_plot(df: pd.DataFrame, output_dir: Path) -> None:
    """Create box plots showing metric distributions across methods."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    metrics = list(METRIC_CONFIG.keys())

    for i, metric in enumerate(metrics):
        if i >= len(axes):
            break

        ax = axes[i]
        metric_data = df[(df['metric'] == metric) & (df['domain'] == 'overall')]

        if not metric_data.empty:
            sns.boxplot(data=metric_data, x='method', y='value', ax=ax,
                       color=METRIC_CONFIG[metric]['color'])
            ax.set_title(f'{METRIC_CONFIG[metric]["name"]} Distribution')
            ax.set_ylabel(METRIC_CONFIG[metric]['name'])
            ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / 'metric_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Create correlation heatmap between different metrics."""
    # Pivot to get methods as rows, metrics as columns
    pivot_df = df[df['domain'] == 'overall'].pivot(
        index='method', columns='metric', values='value'
    )

    # Calculate correlation matrix
    corr_matrix = pivot_df.corr()

    # Check if we have valid data
    if corr_matrix.empty or corr_matrix.isna().all().all():
        logging.warning("Insufficient data for correlation heatmap")
        return

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title('Metric Correlation Matrix')
    plt.tight_layout()
    plt.savefig(output_dir / 'metric_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_domain_comparison_plot(df: pd.DataFrame, output_dir: Path) -> None:
    """Create heatmap showing metrics across domains and methods."""
    # Filter for domains we want to show
    domain_data = df[df['domain'].isin(DOMAIN_ORDER)]

    for metric in METRIC_CONFIG.keys():
        metric_data = domain_data[domain_data['metric'] == metric]

        if metric_data.empty:
            continue

        # Pivot for heatmap
        pivot_df = metric_data.pivot(
            index='method', columns='domain', values='value'
        )

        # Reorder columns
        available_domains = [d for d in DOMAIN_ORDER if d in pivot_df.columns]
        pivot_df = pivot_df[available_domains]

        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_df, annot=True, fmt='.3f',
                   cmap='RdYlBu_r' if METRIC_CONFIG[metric]['lower_better'] else 'RdYlBu',
                   linewidths=0.5)
        plt.title(f'{METRIC_CONFIG[metric]["name"]} by Domain and Method')
        plt.xlabel('Domain')
        plt.ylabel('Method')
        plt.tight_layout()
        plt.savefig(output_dir / f'{metric}_domain_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()


def create_radar_plot(df: pd.DataFrame, output_dir: Path) -> None:
    """Create radar plot for multi-metric comparison."""
    # Normalize metrics to 0-1 scale for radar plot
    overall_data = df[df['domain'] == 'overall']

    # Get methods and metrics
    methods = sorted(overall_data['method'].unique())
    metrics = [m for m in METRIC_CONFIG.keys() if m in overall_data['metric'].unique()]

    if len(methods) < 3 or len(metrics) < 3:
        logging.warning("Need at least 3 methods and 3 metrics for radar plot")
        return

    # Normalize each metric
    normalized_data = {}
    for method in methods:
        method_data = overall_data[overall_data['method'] == method]
        normalized_values = []

        for metric in metrics:
            metric_row = method_data[method_data['metric'] == metric]
            if not metric_row.empty:
                value = metric_row['value'].iloc[0]

                # Normalize based on all values for this metric
                all_values = overall_data[overall_data['metric'] == metric]['value']
                if METRIC_CONFIG[metric]['lower_better']:
                    # For lower-better metrics, invert normalization
                    normalized = 1 - (value - all_values.min()) / (all_values.max() - all_values.min())
                else:
                    normalized = (value - all_values.min()) / (all_values.max() - all_values.min())

                normalized_values.append(normalized)
            else:
                normalized_values.append(0)

        normalized_data[method] = normalized_values

    # Create radar plot
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    for method, values in normalized_data.items():
        values += values[:1]  # Close the loop
        ax.plot(angles, values, 'o-', linewidth=2, label=method)
        ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([METRIC_CONFIG[m]['name'] for m in metrics])
    ax.set_ylim(0, 1)
    ax.set_title('Method Comparison Radar Plot (Normalized Metrics)', size=16)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / 'method_radar_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_scatter_plots(df: pd.DataFrame, output_dir: Path) -> None:
    """Create scatter plots showing relationships between metrics."""
    overall_data = df[df['domain'] == 'overall']

    # Get available metrics
    available_metrics = [m for m in METRIC_CONFIG.keys() if m in overall_data['metric'].unique()]

    if len(available_metrics) < 2:
        logging.warning("Need at least 2 metrics for scatter plots")
        return

    # Create pairwise scatter plots
    fig, axes = plt.subplots(len(available_metrics), len(available_metrics),
                           figsize=(15, 15))

    for i, metric1 in enumerate(available_metrics):
        for j, metric2 in enumerate(available_metrics):
            ax = axes[i, j]

            if i == j:
                # Diagonal: histograms
                data = overall_data[overall_data['metric'] == metric1]
                if not data.empty:
                    sns.histplot(data['value'], ax=ax, color=METRIC_CONFIG[metric1]['color'])
                ax.set_title(METRIC_CONFIG[metric1]['name'])
            else:
                # Off-diagonal: scatter plots
                data1 = overall_data[overall_data['metric'] == metric1].set_index('method')['value']
                data2 = overall_data[overall_data['metric'] == metric2].set_index('method')['value']

                common_methods = data1.index.intersection(data2.index)
                if len(common_methods) > 0:
                    plot_data = pd.DataFrame({
                        'x': data1.loc[common_methods],
                        'y': data2.loc[common_methods],
                        'method': common_methods
                    })
                    sns.scatterplot(data=plot_data, x='x', y='y', ax=ax, hue='method', legend=False)

                if j == 0:
                    ax.set_ylabel(METRIC_CONFIG[metric1]['name'])
                if i == len(available_metrics) - 1:
                    ax.set_xlabel(METRIC_CONFIG[metric2]['name'])

    plt.suptitle('Metric Relationships and Distributions', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / 'metric_scatter_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_summary_report(results: Dict[str, Dict[str, Any]], df: pd.DataFrame, output_dir: Path) -> None:
    """Generate a text summary of the visualizations created."""
    summary_path = output_dir / 'visualization_summary.txt'

    with open(summary_path, 'w') as f:
        f.write("PRISM Results Visualization Summary\n")
        f.write("=" * 40 + "\n\n")

        f.write(f"Total methods analyzed: {len(results)}\n")
        f.write(f"Methods: {', '.join(sorted(results.keys()))}\n\n")

        f.write("Generated visualizations:\n")
        f.write("- method_comparison.png: Bar charts comparing methods across all metrics\n")
        f.write("- metric_distributions.png: Box plots showing metric distributions\n")
        f.write("- metric_correlations.png: Heatmap of correlations between metrics\n")
        f.write("- method_radar_comparison.png: Radar plot for multi-metric comparison\n")
        f.write("- metric_scatter_matrix.png: Scatter plots and histograms of metric relationships\n")

        # Domain-specific plots
        domain_metrics = df[df['domain'] != 'overall']['metric'].unique()
        if len(domain_metrics) > 0:
            f.write(f"- Domain comparison plots: {len(domain_metrics)} metrics across domains\n")
            for metric in sorted(domain_metrics):
                f.write(f"  - {metric}_domain_comparison.png\n")

        f.write("\nMetric Summary:\n")
        overall_data = df[df['domain'] == 'overall']
        for metric in METRIC_CONFIG.keys():
            metric_data = overall_data[overall_data['metric'] == metric]
            if not metric_data.empty:
                mean_val = metric_data['value'].mean()
                std_val = metric_data['value'].std()
                f.write(".3f")

        f.write(f"\nOutput directory: {output_dir}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate comprehensive diagrams from PRISM evaluation results."
    )
    parser.add_argument(
        "--results-dir", type=Path,
        help="Directory containing evaluation result JSON files"
    )
    parser.add_argument(
        "--results-files", type=Path, nargs="+",
        help="Specific JSON result files to visualize"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("plots"),
        help="Output directory for generated plots"
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase verbosity"
    )

    args = parser.parse_args()

    if not args.results_dir and not args.results_files:
        parser.error("Must specify either --results-dir or --results-files")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    logging.info("Loading evaluation results...")
    results = load_results(args.results_dir, args.results_files)

    if not results:
        logging.error("No results found")
        return

    logging.info(f"Loaded {len(results)} result files")

    # Extract metrics into DataFrame
    df = extract_method_metrics(results)
    logging.info(f"Extracted {len(df)} metric data points")

    # Generate plots
    logging.info("Generating method comparison plot...")
    create_method_comparison_plot(df, args.output_dir)

    logging.info("Generating metric distributions plot...")
    create_metric_distributions_plot(df, args.output_dir)

    logging.info("Generating correlation heatmap...")
    create_correlation_heatmap(df, args.output_dir)

    logging.info("Generating domain comparison plots...")
    create_domain_comparison_plot(df, args.output_dir)

    logging.info("Generating radar plot...")
    create_radar_plot(df, args.output_dir)

    logging.info("Generating scatter plots...")
    create_scatter_plots(df, args.output_dir)

    logging.info("Generating summary report...")
    generate_summary_report(results, df, args.output_dir)

    logging.info(f"Visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()