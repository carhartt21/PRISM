import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

class Metric:
    name = "lpips"

    def __init__(self, device="cpu"):
        self.metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
        self.device = device

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # LPIPS expects inputs in [-1, 1] range
        x_normalized = 2.0 * x - 1.0
        y_normalized = 2.0 * y - 1.0

        batch_size = x.size(0)
        lpips_scores = []

        for i in range(batch_size):
            score = self.metric(x_normalized[i:i+1], y_normalized[i:i+1])
            lpips_scores.append(score)

        return torch.stack(lpips_scores)
