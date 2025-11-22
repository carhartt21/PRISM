import torch
from torchmetrics.image import PeakSignalNoiseRatio

class Metric:
    name = "psnr"

    def __init__(self, device="cpu"):
        # Inputs are normalized to [0, 1], so fix the valid data range.
        self.metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.device = device

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # PSNR expects inputs in [0, 1] range
        batch_size = x.size(0)
        psnr_scores = []

        for i in range(batch_size):
            score = self.metric(x[i:i+1], y[i:i+1])
            psnr_scores.append(score)

        return torch.stack(psnr_scores)
