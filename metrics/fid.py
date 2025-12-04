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
        if self.uses_reference_stats and reference_stats:
            self._apply_reference_stats(reference_stats)

    def __call__(self, x: torch.Tensor, y: Optional[torch.Tensor]) -> torch.Tensor:
        # FID requires at least 2 samples to compute distribution statistics
        if x.size(0) < 2:
            # Return NaN for batches too small to compute FID
            return torch.full((x.size(0),), float('nan'), device=x.device)
        
        # Ensure images are in [0, 255] range and uint8 for FID
        x_uint8 = (x * 255).clamp(0, 255).to(torch.uint8)
        if not self.uses_reference_stats:
            if y is None:
                raise ValueError("FID metric requires target-domain images or reference stats")
            if y.size(0) < 2:
                # Return NaN if target batch is too small
                return torch.full((x.size(0),), float('nan'), device=x.device)
            y_uint8 = (y * 255).clamp(0, 255).to(torch.uint8)
            self.metric.update(y_uint8, real=True)

        self.metric.update(x_uint8, real=False)
        
        try:
            fid = self.metric.compute()
        except RuntimeError as e:
            if "More than one sample is required" in str(e):
                # Return NaN if FID computation fails due to insufficient samples
                self.metric.reset()
                return torch.full((x.size(0),), float('nan'), device=self.device)
            raise
        
        self.metric.reset()

        # Return FID score for each image in batch (FID is a global metric)
        # Use repeat to ensure we always get a 1-D tensor
        batch_size = x.size(0)
        return fid.view(1).repeat(batch_size)

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
