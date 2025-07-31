import torch
from torchmetrics.image.fid import FrechetInceptionDistance

class Metric:
    name = "fid"

    def __init__(self, device="cpu"):
        self.metric = FrechetInceptionDistance(feature=2048).to(device)
        self.device = device

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Ensure images are in [0, 255] range and uint8 for FID
        x_uint8 = (x * 255).clamp(0, 255).to(torch.uint8)
        y_uint8 = (y * 255).clamp(0, 255).to(torch.uint8)

        self.metric.update(y_uint8, real=True)
        self.metric.update(x_uint8, real=False)
        fid = self.metric.compute()
        self.metric.reset()

        # Return FID score for each image in batch (FID is a global metric)
        return fid.expand(x.size(0))
