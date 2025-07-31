import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure

class Metric:
    name = "ssim"

    def __init__(self, device="cpu"):
        self.metric = StructuralSimilarityIndexMeasure().to(device)
        self.device = device

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # SSIM expects inputs in [0, 1] range
        batch_size = x.size(0)
        ssim_scores = []

        for i in range(batch_size):
            score = self.metric(x[i:i+1], y[i:i+1])
            ssim_scores.append(score)

        return torch.stack(ssim_scores)
