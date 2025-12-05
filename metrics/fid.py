from typing import Any, Dict, Optional

import torch


class Metric:
    name = "fid"

    def __init__(self, device="cpu", reference_stats: Optional[Dict[str, Any]] = None):
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance
        except ModuleNotFoundError as exc:  # pragma: no cover - import guard
            raise ModuleNotFoundError(
                "torchmetrics.image.fid requires the optional 'torch-fidelity' dependency."
                " Install via 'pip install torchmetrics[image]' or 'pip install torch-fidelity'."
            ) from exc

        keep_reference = reference_stats is not None
        self.metric = FrechetInceptionDistance(feature=2048, reset_real_features=not keep_reference)
        self.metric.to(device)
        self.device = device
        self.uses_reference_stats = reference_stats is not None
        self._num_fake_samples = 0
        if self.uses_reference_stats and reference_stats:
            self._apply_reference_stats(reference_stats)

    def update(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> None:
        """
        Accumulate features from a batch of images.
        
        FID is a distributional metric - call update() for each batch,
        then compute() once at the end for the final score.
        
        Args:
            x: Generated/fake images (B, C, H, W) in [0, 1] range
            y: Real images (B, C, H, W) in [0, 1] range (only needed if no reference_stats)
        """
        # Ensure images are in [0, 255] range and uint8 for FID
        x_uint8 = (x * 255).clamp(0, 255).to(torch.uint8)
        
        if not self.uses_reference_stats:
            if y is None:
                raise ValueError("FID metric requires target-domain images or reference stats")
            y_uint8 = (y * 255).clamp(0, 255).to(torch.uint8)
            self.metric.update(y_uint8, real=True)
        
        self.metric.update(x_uint8, real=False)
        self._num_fake_samples += x.size(0)

    def compute(self) -> float:
        """
        Compute the final FID score after all batches have been processed.
        
        Returns:
            FID score as a single float value
        """
        if self._num_fake_samples < 2:
            return float('nan')
        
        try:
            fid = self.metric.compute()
            return float(fid.item())
        except RuntimeError as e:
            if "More than one sample is required" in str(e):
                return float('nan')
            raise

    def reset(self) -> None:
        """Reset the metric for a new evaluation."""
        self.metric.reset()
        self._num_fake_samples = 0
        # Re-apply reference stats if using precomputed stats
        if self.uses_reference_stats:
            # The metric was initialized with reset_real_features=False,
            # so real features are preserved across resets
            pass

    def __call__(self, x: torch.Tensor, y: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Legacy per-batch interface (deprecated for FID).
        
        For proper FID computation, use update() for each batch, 
        then compute() once at the end.
        
        This method is kept for backward compatibility but will compute
        FID on just this batch, which is not statistically meaningful.
        """
        # For backward compatibility, update and immediately compute
        # This is NOT the recommended usage for FID
        self.update(x, y)
        fid_score = self.compute()
        self.reset()
        
        # Return FID score for each image in batch (same value repeated)
        batch_size = x.size(0)
        return torch.full((batch_size,), fid_score, device=self.device)

    def _apply_reference_stats(self, stats: Dict[str, Any]) -> None:
        if "mu" not in stats or "sigma" not in stats:
            raise KeyError("Reference stats must contain 'mu' and 'sigma'")

        mu = torch.as_tensor(stats["mu"], dtype=torch.double)
        sigma = torch.as_tensor(stats["sigma"], dtype=torch.double)
        n_value = int(stats.get("n", mu.numel() + 1))
        if n_value <= 1:
            n_value = mu.numel() + 1

        if sigma.shape != (mu.numel(), mu.numel()):
            raise ValueError("Sigma covariance must be square with matching dimension")

        mu = mu.to(self.metric.real_features_sum.device)
        sigma = sigma.to(self.metric.real_features_cov_sum.device)
        mu_outer = torch.outer(mu, mu)

        self.metric.real_features_sum = mu * n_value
        self.metric.real_features_cov_sum = sigma * (n_value - 1) + n_value * mu_outer
        self.metric.real_features_num_samples = torch.tensor(n_value, dtype=torch.long)
