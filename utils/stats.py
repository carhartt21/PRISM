"""
utils.stats: Statistical analysis utilities for metrics
"""
import numpy as np
from typing import Dict, Any, List
from scipy import stats


def compute_confidence_interval(data: List[float], alpha: float = 0.95) -> tuple:
    """
    Compute confidence interval for a list of values.

    Args:
        data: List of numerical values
        alpha: Confidence level (e.g., 0.95 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if len(data) < 2:
        return (np.nan, np.nan)

    data_array = np.array(data)
    mean = np.mean(data_array)
    stderr = stats.sem(data_array)

    # Calculate confidence interval using t-distribution
    confidence_level = alpha
    degrees_freedom = len(data) - 1
    t_value = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)

    margin_error = t_value * stderr
    lower_bound = mean - margin_error
    upper_bound = mean + margin_error

    return (lower_bound, upper_bound)


def compute_basic_stats(data: List[float]) -> Dict[str, float]:
    """
    Compute basic statistical measures for a list of values.

    Args:
        data: List of numerical values

    Returns:
        Dictionary with statistical measures
    """
    if not data:
        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "min": np.nan,
            "max": np.nan,
            "q25": np.nan,
            "q75": np.nan
        }

    data_array = np.array(data)

    return {
        "count": len(data),
        "mean": float(np.mean(data_array)),
        "std": float(np.std(data_array, ddof=1)) if len(data) > 1 else 0.0,
        "median": float(np.median(data_array)),
        "min": float(np.min(data_array)),
        "max": float(np.max(data_array)),
        "q25": float(np.percentile(data_array, 25)),
        "q75": float(np.percentile(data_array, 75))
    }


def perform_normality_test(data: List[float]) -> Dict[str, Any]:
    """
    Perform Shapiro-Wilk normality test on data.

    Args:
        data: List of numerical values

    Returns:
        Dictionary with test results
    """
    if len(data) < 3:
        return {
            "test_name": "Shapiro-Wilk",
            "statistic": np.nan,
            "p_value": np.nan,
            "is_normal": None,
            "note": "Insufficient data for normality test"
        }

    if len(data) > 5000:
        # Use Anderson-Darling test for large samples
        result = stats.anderson(data, dist='norm')
        is_normal = result.statistic < result.critical_values[2]  # 5% significance level
        return {
            "test_name": "Anderson-Darling",
            "statistic": float(result.statistic),
            "p_value": np.nan,  # Anderson-Darling doesn't provide p-value directly
            "is_normal": is_normal,
            "critical_value_5pct": float(result.critical_values[2])
        }
    else:
        # Use Shapiro-Wilk test for smaller samples
        statistic, p_value = stats.shapiro(data)
        return {
            "test_name": "Shapiro-Wilk",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "is_normal": p_value > 0.05,
            "alpha": 0.05
        }


def summarise_metrics(per_image_results: Dict[str, Dict[str, float]], alpha: float = 0.95) -> Dict[str, Any]:
    """
    Compute comprehensive statistical summaries for all metrics.

    Args:
        per_image_results: Dictionary mapping image names to metric scores
        alpha: Confidence level for confidence intervals

    Returns:
        Dictionary with statistical summaries for each metric
    """
    if not per_image_results:
        return {}

    # Organize data by metric
    metric_scores = {}
    for image_name, scores in per_image_results.items():
        for metric_name, score in scores.items():
            if metric_name not in metric_scores:
                metric_scores[metric_name] = []
            metric_scores[metric_name].append(score)

    # Compute statistics for each metric
    summary = {}
    for metric_name, scores in metric_scores.items():
        basic_stats = compute_basic_stats(scores)
        ci_lower, ci_upper = compute_confidence_interval(scores, alpha)
        normality = perform_normality_test(scores)

        summary[metric_name] = {
            **basic_stats,
            "confidence_interval": {
                "alpha": alpha,
                "lower": float(ci_lower) if not np.isnan(ci_lower) else None,
                "upper": float(ci_upper) if not np.isnan(ci_upper) else None
            },
            "normality_test": normality
        }

    return summary


def compare_metric_distributions(scores1: List[float], scores2: List[float], 
                               metric_name: str = "metric") -> Dict[str, Any]:
    """
    Compare two distributions of metric scores using statistical tests.

    Args:
        scores1: First set of scores
        scores2: Second set of scores
        metric_name: Name of the metric being compared

    Returns:
        Dictionary with comparison results
    """
    if len(scores1) < 2 or len(scores2) < 2:
        return {
            "metric": metric_name,
            "error": "Insufficient data for comparison",
            "n1": len(scores1),
            "n2": len(scores2)
        }

    # Basic statistics
    stats1 = compute_basic_stats(scores1)
    stats2 = compute_basic_stats(scores2)

    # Mann-Whitney U test (non-parametric)
    u_statistic, u_p_value = stats.mannwhitneyu(scores1, scores2, alternative='two-sided')

    # Welch's t-test (assuming unequal variances)
    t_statistic, t_p_value = stats.ttest_ind(scores1, scores2, equal_var=False)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(scores1) - 1) * stats1["std"]**2 + 
                         (len(scores2) - 1) * stats2["std"]**2) / 
                        (len(scores1) + len(scores2) - 2))
    cohens_d = (stats1["mean"] - stats2["mean"]) / pooled_std if pooled_std > 0 else 0

    return {
        "metric": metric_name,
        "group1_stats": stats1,
        "group2_stats": stats2,
        "mann_whitney_u": {
            "statistic": float(u_statistic),
            "p_value": float(u_p_value),
            "significant_05": u_p_value < 0.05
        },
        "welch_t_test": {
            "statistic": float(t_statistic),
            "p_value": float(t_p_value),
            "significant_05": t_p_value < 0.05
        },
        "effect_size": {
            "cohens_d": float(cohens_d),
            "interpretation": interpret_cohens_d(cohens_d)
        }
    }


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"
