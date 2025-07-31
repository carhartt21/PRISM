"""
Test statistical analysis utilities
"""
import pytest
import numpy as np
from utils.stats import (
    compute_basic_stats, 
    compute_confidence_interval, 
    summarise_metrics,
    compare_metric_distributions,
    perform_normality_test
)


class TestStatistics:

    def test_basic_stats_normal_data(self):
        """Test basic statistics computation with normal data."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = compute_basic_stats(data)

        assert stats["count"] == 5
        assert stats["mean"] == 3.0
        assert stats["median"] == 3.0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert abs(stats["std"] - np.std(data, ddof=1)) < 1e-10

    def test_basic_stats_empty_data(self):
        """Test basic statistics with empty data."""
        stats = compute_basic_stats([])

        assert stats["count"] == 0
        assert np.isnan(stats["mean"])
        assert np.isnan(stats["median"])
        assert np.isnan(stats["min"])
        assert np.isnan(stats["max"])

    def test_basic_stats_single_value(self):
        """Test basic statistics with single value."""
        data = [42.0]
        stats = compute_basic_stats(data)

        assert stats["count"] == 1
        assert stats["mean"] == 42.0
        assert stats["median"] == 42.0
        assert stats["min"] == 42.0
        assert stats["max"] == 42.0
        assert stats["std"] == 0.0  # Standard deviation of single value

    def test_confidence_interval(self):
        """Test confidence interval computation."""
        # Generate some test data
        np.random.seed(42)
        data = np.random.normal(10, 2, 100).tolist()

        lower, upper = compute_confidence_interval(data, alpha=0.95)

        assert lower < upper
        assert lower < np.mean(data) < upper

        # Test with insufficient data
        lower_small, upper_small = compute_confidence_interval([1.0], alpha=0.95)
        assert np.isnan(lower_small)
        assert np.isnan(upper_small)

    def test_summarise_metrics(self):
        """Test metric summarization functionality."""
        per_image_results = {
            "image1": {"ssim": 0.8, "psnr": 25.0},
            "image2": {"ssim": 0.7, "psnr": 23.0},
            "image3": {"ssim": 0.9, "psnr": 27.0}
        }

        summary = summarise_metrics(per_image_results, alpha=0.95)

        assert "ssim" in summary
        assert "psnr" in summary

        ssim_stats = summary["ssim"]
        assert ssim_stats["count"] == 3
        assert abs(ssim_stats["mean"] - 0.8) < 0.01
        assert "confidence_interval" in ssim_stats
        assert "normality_test" in ssim_stats

    def test_summarise_metrics_empty(self):
        """Test metric summarization with empty data."""
        summary = summarise_metrics({})
        assert summary == {}

    def test_normality_test_normal_data(self):
        """Test normality test with normally distributed data."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 100).tolist()

        result = perform_normality_test(data)

        assert "test_name" in result
        assert "statistic" in result
        assert "p_value" in result
        assert "is_normal" in result
        assert isinstance(result["is_normal"], bool)

    def test_normality_test_insufficient_data(self):
        """Test normality test with insufficient data."""
        result = perform_normality_test([1.0, 2.0])

        assert result["test_name"] == "Shapiro-Wilk"
        assert np.isnan(result["statistic"])
        assert np.isnan(result["p_value"])
        assert result["is_normal"] is None
        assert "Insufficient data" in result["note"]

    def test_compare_distributions(self):
        """Test distribution comparison functionality."""
        np.random.seed(42)
        scores1 = np.random.normal(10, 2, 50).tolist()
        scores2 = np.random.normal(12, 2, 50).tolist()  # Different mean

        comparison = compare_metric_distributions(scores1, scores2, "test_metric")

        assert comparison["metric"] == "test_metric"
        assert "group1_stats" in comparison
        assert "group2_stats" in comparison
        assert "mann_whitney_u" in comparison
        assert "welch_t_test" in comparison
        assert "effect_size" in comparison

        # Check that we detect a significant difference
        assert comparison["mann_whitney_u"]["p_value"] < 0.05
        assert comparison["welch_t_test"]["p_value"] < 0.05

    def test_compare_distributions_insufficient_data(self):
        """Test distribution comparison with insufficient data."""
        comparison = compare_metric_distributions([1.0], [2.0], "test")

        assert "error" in comparison
        assert "Insufficient data" in comparison["error"]
        assert comparison["n1"] == 1
        assert comparison["n2"] == 1


if __name__ == "__main__":
    pytest.main([__file__])
