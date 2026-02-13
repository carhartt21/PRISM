# Image Generation Methods Evaluation Report

## Summary

- **Total Methods Evaluated**: 21
- **Methods with CQS Score**: 21

## Composite Quality Score (CQS)

The CQS combines multiple metrics into a single ranking score using z-score normalization:

```
CQS = 0.4 × z(FID) + 0.2 × z(LPIPS) + 0.2 × (1 - z(PA)) + 0.2 × (1 - z(fw-IoU))
```

Where **lower CQS is better**.

### Global Statistics (for z-score normalization)

| Metric | Mean | Std |
|--------|------|-----|
| FID | 116.36 | 17.70 |
| LPIPS | 0.4716 | 0.1808 |
| Pixel Accuracy | 73.16 | 15.94 |
| fw-IoU | 65.54 | 17.63 |

## Ranking Table

| Rank | Method | CQS | FID ↓ | LPIPS ↓ | SSIM ↑ | PA ↑ | fw-IoU ↑ | Images |
|------|--------|-----|-------|---------|--------|------|----------|--------|
| 1 | cycleGAN | -0.8219 | 92.65 | 0.2126 | 0.8689 | 88.62 | 83.64 | 187398 |
| 2 | albumentations | -0.5753 | N/A | 0.3963 | 0.6864 | 98.01 | 96.39 | 18462 |
| 3 | LANIT | -0.3122 | 106.24 | 0.3175 | 0.7733 | 85.16 | 79.84 | 223300 |
| 4 | StyleID | -0.0868 | 89.49 | 0.4456 | 0.6101 | 67.49 | 58.64 | 15533 |
| 5 | VisualCloze | -0.0485 | N/A | 0.3752 | 0.6088 | 86.14 | 79.60 | 9232 |
| 6 | flux_kontext | 0.1039 | N/A | 0.3985 | 0.6246 | 83.12 | 77.15 | 62537 |
| 7 | step1x_v1p2 | 0.1190 | 91.63 | 0.5969 | 0.4020 | 68.21 | 58.71 | 119050 |
| 8 | automold | 0.1382 | 121.12 | 0.3188 | 0.6706 | 80.94 | 74.57 | 95700 |
| 9 | SUSTechGAN | 0.3797 | 147.49 | 0.1943 | 0.8894 | 89.21 | 84.54 | 127699 |
| 10 | IP2P | 0.3820 | 114.22 | 0.4696 | 0.5504 | 72.10 | 63.83 | 187398 |
| 11 | cyclediffusion | 0.3911 | 138.77 | 0.2220 | 0.8158 | 83.08 | 75.65 | 110883 |
| 12 | EDICT | 0.5838 | 145.93 | 0.2393 | 0.8029 | 82.67 | 75.09 | 40058 |
| 13 | step1x_new | 0.6041 | N/A | 0.4939 | 0.5101 | 75.88 | 68.01 | 77343 |
| 14 | Attribute_Hallucination | 0.6174 | 117.95 | 0.6076 | 0.3751 | 72.30 | 63.74 | 191400 |
| 15 | UniControl | 0.7051 | 114.90 | 0.7041 | 0.3254 | 70.34 | 61.51 | 187398 |
| 16 | CUT | 0.7510 | 119.38 | 0.5644 | 0.3448 | 66.87 | 56.62 | 60381 |
| 17 | Img2Img | 0.8037 | 120.25 | 0.5573 | 0.4334 | 65.03 | 55.04 | 124932 |
| 18 | Qwen-Image-Edit | 0.8765 | N/A | 0.5854 | 0.4056 | 73.93 | 64.68 | 21253 |
| 19 | cnet_seg | 1.3813 | 120.77 | 0.7346 | 0.2273 | 49.43 | 39.72 | 187398 |
| 20 | CNetSeg | 1.3813 | 120.77 | 0.7346 | 0.2273 | 49.43 | 39.72 | 187398 |
| 21 | stargan_v2 | 1.4110 | 100.28 | 0.7355 | 0.2023 | 28.45 | 19.57 | 187398 |

## Metric Descriptions

- **FID (Fréchet Inception Distance)**: Measures distribution similarity between generated and real images. Lower is better.
- **LPIPS (Learned Perceptual Image Patch Similarity)**: Measures perceptual difference using deep features. Lower is better.
- **SSIM (Structural Similarity Index)**: Measures structural similarity. Higher is better.
- **PA (Pixel Accuracy)**: Semantic segmentation consistency between original and generated. Higher is better.
- **fw-IoU (Frequency-weighted IoU)**: Class-weighted semantic consistency. Higher is better.

## Per-Domain Results

### cycleGAN

| Domain | FID | LPIPS | PA | fw-IoU | Images |
|--------|-----|-------|-------|--------|--------|
| clear_day2cloudy | 101.57 | 0.2826 | 89.34 | 83.45 | 31233 |
| clear_day2dawn_dusk | 115.21 | 0.1823 | 91.48 | 86.82 | 31233 |
| clear_day2fog | N/A | 0.1994 | 86.81 | 81.71 | 31233 |
| clear_day2night | 66.62 | 0.2391 | 87.17 | 82.13 | 31233 |
| clear_day2rainy | 89.18 | 0.1781 | 86.49 | 81.87 | 31233 |
| clear_day2snowy | 90.69 | 0.1940 | 90.41 | 85.89 | 31233 |

### albumentations

| Domain | FID | LPIPS | PA | fw-IoU | Images |
|--------|-----|-------|-------|--------|--------|
| fog | N/A | 0.3964 | N/A | N/A | 18452 |
| rain | N/A | 0.1926 | 98.16 | 96.65 | 5 |
| snow | N/A | 0.1675 | 97.86 | 96.13 | 5 |

### LANIT

| Domain | FID | LPIPS | PA | fw-IoU | Images |
|--------|-----|-------|-------|--------|--------|
| clear_day | 113.01 | 0.3178 | 85.15 | 79.83 | 31900 |
| cloudy | 120.48 | 0.3177 | 85.14 | 79.82 | 31900 |
| dawn_dusk | 114.72 | 0.3178 | 85.13 | 79.81 | 31900 |
| foggy | 115.55 | 0.3171 | 85.18 | 79.87 | 31900 |
| night | 77.61 | 0.3178 | 85.14 | 79.82 | 31900 |
| rainy | 87.14 | 0.3172 | 85.18 | 79.86 | 31900 |
| snowy | 115.14 | 0.3174 | 85.18 | 79.85 | 31900 |

### StyleID

| Domain | FID | LPIPS | PA | fw-IoU | Images |
|--------|-----|-------|-------|--------|--------|
| cloudy | 94.47 | 0.4262 | 65.91 | 56.07 | 2589 |
| dawn_dusk | 95.79 | 0.4964 | 67.92 | 58.69 | 2588 |
| foggy | 91.44 | 0.4359 | 62.01 | 53.92 | 2589 |
| night | 74.20 | 0.4845 | 66.42 | 57.15 | 2589 |
| rainy | 99.30 | 0.4125 | 71.82 | 63.64 | 2589 |
| snowy | 81.78 | 0.4180 | 70.88 | 62.39 | 2589 |

### VisualCloze

| Domain | FID | LPIPS | PA | fw-IoU | Images |
|--------|-----|-------|-------|--------|--------|
| ACDC | N/A | 0.3952 | 86.06 | 79.09 | 4206 |
| MapillaryVistas | N/A | 0.3585 | 86.21 | 80.02 | 5026 |
