import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

class Metric:
    name = "lpips"

    def __init__(self, device="cpu"):
        self.metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg', reduction='none').to(device)
        self.device = device

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # LPIPS expects inputs in [-1, 1] range
        x_normalized = 2.0 * x - 1.0
        y_normalized = 2.0 * y - 1.0

        # Batched computation with reduction='none' returns per-sample scores
        with torch.no_grad():
            scores = self.metric(x_normalized, y_normalized)
        
        return scores.squeeze()
