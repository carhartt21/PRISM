import torch
from torchmetrics.image.inception import InceptionScore

class Metric:
    name = "is"

    def __init__(self, device="cpu"):
        self.metric = InceptionScore().to(device)
        self.device = device

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Inception Score only uses generated images (x), not reference (y)
        # Convert to [0, 255] range and uint8
        x_uint8 = (x * 255).clamp(0, 255).to(torch.uint8)

        self.metric.update(x_uint8)
        is_mean, is_std = self.metric.compute()
        self.metric.reset()

        # Return IS score for each image in batch (IS is a global metric)
        return is_mean.expand(x.size(0))
