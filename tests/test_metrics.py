"""
Test metric implementations
"""
import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from metrics.fid import Metric as FIDMetric
from metrics.ssim import Metric as SSIMMetric
from metrics.psnr import Metric as PSNRMetric


class TestMetrics:

    @pytest.fixture
    def sample_images(self):
        """Create sample image tensors for testing."""
        batch_size = 2
        channels, height, width = 3, 64, 64

        # Create random images in [0, 1] range
        gen_images = torch.rand(batch_size, channels, height, width)
        real_images = torch.rand(batch_size, channels, height, width)

        return gen_images, real_images

    def test_ssim_metric(self, sample_images):
        """Test SSIM metric computation."""
        gen_images, real_images = sample_images

        ssim_metric = SSIMMetric(device="cpu")
        scores = ssim_metric(gen_images, real_images)

        assert isinstance(scores, torch.Tensor)
        assert scores.shape == (gen_images.size(0),)
        assert torch.all(scores >= -1) and torch.all(scores <= 1)  # SSIM range [-1, 1]

    def test_psnr_metric(self, sample_images):
        """Test PSNR metric computation."""
        gen_images, real_images = sample_images

        psnr_metric = PSNRMetric(device="cpu")
        scores = psnr_metric(gen_images, real_images)

        assert isinstance(scores, torch.Tensor)
        assert scores.shape == (gen_images.size(0),)
        assert torch.all(scores >= 0)  # PSNR should be positive

    def test_identical_images_ssim(self):
        """Test SSIM with identical images (should give score close to 1)."""
        image = torch.rand(1, 3, 64, 64)

        ssim_metric = SSIMMetric(device="cpu")
        score = ssim_metric(image, image)

        assert torch.allclose(score, torch.tensor([1.0]), atol=1e-3)

    def test_identical_images_psnr(self):
        """Test PSNR with identical images (should give high score)."""
        image = torch.rand(1, 3, 64, 64)

        psnr_metric = PSNRMetric(device="cpu")
        score = psnr_metric(image, image)

        # PSNR of identical images should be very high (approaching infinity)
        assert score > 40  # Typical high PSNR value

    @patch('torchmetrics.image.fid.FrechetInceptionDistance')
    def test_fid_metric_mock(self, mock_fid_class, sample_images):
        """Test FID metric with mocked torchmetrics."""
        gen_images, real_images = sample_images

        # Mock the FID metric
        mock_fid = MagicMock()
        mock_fid.compute.return_value = torch.tensor(15.5)  # Typical FID score
        mock_fid_class.return_value = mock_fid

        fid_metric = FIDMetric(device="cpu")
        scores = fid_metric(gen_images, real_images)

        assert isinstance(scores, torch.Tensor)
        assert scores.shape == (gen_images.size(0),)
        assert torch.all(scores == 15.5)  # Should be the mocked value

        # Verify the mock was called correctly
        mock_fid.update.assert_called()
        mock_fid.compute.assert_called_once()
        mock_fid.reset.assert_called_once()

    def test_metric_device_handling(self):
        """Test that metrics handle device specification correctly."""
        # Test CPU device
        ssim_cpu = SSIMMetric(device="cpu")
        assert str(ssim_cpu.metric.device) == "cpu"

        # Test CUDA device (if available)
        if torch.cuda.is_available():
            ssim_cuda = SSIMMetric(device="cuda")
            assert "cuda" in str(ssim_cuda.metric.device)

    def test_batch_processing(self, sample_images):
        """Test that metrics handle different batch sizes correctly."""
        gen_images, real_images = sample_images

        ssim_metric = SSIMMetric(device="cpu")

        # Test with different batch sizes by slicing
        for batch_size in [1, 2]:
            gen_batch = gen_images[:batch_size]
            real_batch = real_images[:batch_size]

            scores = ssim_metric(gen_batch, real_batch)
            assert scores.shape == (batch_size,)


if __name__ == "__main__":
    pytest.main([__file__])
