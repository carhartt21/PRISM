# Generated Images Summary

**Generated:** December 23, 2024

## Overview

| Metric | Value |
|--------|-------|
| Total Models | 26 |
| Total Images | 2,984,281 |
| Models with Manifests | 26/26 |
| Manifest Mismatches | 4 (active generation) |

## Models Summary

| Model | Images | Manifest | Status | Notes |
|-------|--------|----------|--------|-------|
| AOD-Net | 547 | 270 | ⚠️ | Restoration task (defogging) |
| Attribute_Hallucination | 191,400 | 191,400 | ✅ | |
| CNetSeg | 187,398 | 187,398 | ✅ | |
| CUT | 191,400 | 191,400 | ✅ | |
| EDICT | 64,187 | 64,187 | ✅ | |
| IP2P | 187,398 | 187,398 | ✅ | |
| Img2Img | 187,398 | 187,398 | ✅ | |
| LANIT | 223,300 | 223,300 | ✅ | Includes clear_day domain |
| NST | 41,498 | 41,498 | ✅ | Only cloudy, dawn_dusk |
| Qwen-Image-Edit | 9,404 | 9,324 | ⚠️ | Still generating |
| SUSTechGAN | 127,700 | 127,700 | ✅ | |
| StyleID | 15,533 | 15,533 | ✅ | |
| TSIT | 191,400 | 191,400 | ✅ | |
| UniControl | 187,398 | 187,398 | ✅ | |
| VisualCloze | 33,707 | 14,151 | ⚠️ | Still generating |
| Weather_Effect_Generator | 82,179 | 82,179 | ✅ | foggy, rainy, snowy only |
| albumentations_weather | 95,700 | 95,700 | ✅ | foggy, rain, snow |
| augmenters | 159,500 | 159,500 | ✅ | Unknown domain label |
| automold | 95,700 | 95,700 | ✅ | foggy, rainy, snowy |
| cycleGAN | 187,398 | 187,398 | ✅ | |
| cyclediffusion | 110,883 | 110,883 | ✅ | |
| flux2 | 10,851 | 4,206 | ⚠️ | Still generating |
| flux_kontext | 69,900 | 69,900 | ✅ | |
| stargan_v2 | 187,398 | 187,398 | ✅ | |
| step1x_new | 77,343 | 77,343 | ✅ | |
| step1x_v1p2 | 67,761 | 67,761 | ✅ | |

## By Weather Domain (Aggregate)

| Domain | Images |
|--------|--------|
| rainy | 468,718 |
| snowy | 450,236 |
| cloudy | 441,375 |
| dawn_dusk | 436,732 |
| foggy | 424,398 |
| night | 386,428 |
| unknown | 159,500 |
| fog | 94,366 |
| clear_day | 31,900 |
| rain | 31,900 |
| snow | 31,900 |
| defogged | 270 |

## By Source Dataset (Aggregate)

| Dataset | Images |
|---------|--------|
| BDD100k | 1,258,993 |
| MapillaryVistas | 691,381 |
| OUTSIDE15k | 394,893 |
| IDD-AW | 351,209 |
| BDD10k | 170,871 |
| ACDC | 90,376 |

## Detailed Per-Model Breakdown

### Weather Domains by Model

| Model | cloudy | dawn_dusk | foggy | night | rainy | snowy | Other |
|-------|--------|-----------|-------|-------|-------|-------|-------|
| AOD-Net | - | - | - | - | - | - | defogged: 270 |
| Attribute_Hallucination | 31,900 | 31,900 | 31,900 | 31,900 | 31,900 | 31,900 | - |
| CNetSeg | 31,233 | 31,233 | 31,233 | 31,233 | 31,233 | 31,233 | - |
| CUT | 31,900 | 31,900 | fog: 31,900 | 31,900 | 31,900 | 31,900 | - |
| EDICT | 10,690 | 10,664 | 10,721 | 10,651 | 10,807 | 10,654 | - |
| IP2P | 31,233 | 31,233 | 31,233 | 31,233 | 31,233 | 31,233 | - |
| Img2Img | 31,233 | 31,233 | 31,233 | 31,233 | 31,233 | 31,233 | - |
| LANIT | 31,900 | 31,900 | 31,900 | 31,900 | 31,900 | 31,900 | clear_day: 31,900 |
| NST | 23,058 | 18,440 | - | - | - | - | - |
| Qwen-Image-Edit | 770 | 770 | 5,478 | 769 | 769 | 768 | - |
| SUSTechGAN | 31,900 | 31,900 | 31,900 | 50 | 31,900 | 50 | - |
| StyleID | 2,589 | 2,588 | 2,589 | 2,589 | 2,589 | 2,589 | - |
| TSIT | 31,900 | 31,900 | 31,900 | 31,900 | 31,900 | 31,900 | - |
| UniControl | 31,233 | 31,233 | 31,233 | 31,233 | 31,233 | 31,233 | - |
| VisualCloze | 2,358 | 2,358 | 2,359 | 2,358 | 2,359 | 2,359 | - |
| Weather_Effect_Generator | - | - | 31,900 | - | 18,379 | 31,900 | - |
| albumentations_weather | - | - | 31,900 | - | rain: 31,900 | snow: 31,900 | - |
| augmenters | - | - | - | - | - | - | unknown: 159,500 |
| automold | - | - | 31,900 | - | 31,900 | 31,900 | - |
| cycleGAN | 31,233 | 31,233 | fog: 31,233 | 31,233 | 31,233 | 31,233 | - |
| cyclediffusion | 18,480 | 18,480 | 18,481 | 18,480 | 18,481 | 18,481 | - |
| flux2 | 701 | 701 | 701 | 701 | 701 | 701 | - |
| flux_kontext | 11,650 | 11,650 | 11,650 | 11,650 | 11,650 | 11,650 | - |
| stargan_v2 | 31,233 | 31,233 | fog: 31,233 | 31,233 | 31,233 | 31,233 | - |
| step1x_new | 12,889 | 12,890 | 12,892 | 12,890 | 12,891 | 12,891 | - |
| step1x_v1p2 | 11,292 | 11,293 | 11,295 | 11,292 | 11,294 | 11,295 | - |

### Source Datasets by Model

| Model | ACDC | BDD100k | BDD10k | IDD-AW | MapillaryVistas | OUTSIDE15k |
|-------|------|---------|--------|--------|-----------------|------------|
| AOD-Net | 96 | - | - | 174 | - | - |
| Attribute_Hallucination | 4,206 | 79,992 | 14,220 | 23,082 | 45,162 | 24,738 |
| CNetSeg | 4,206 | 79,992 | 10,218 | 23,082 | 45,162 | 24,738 |
| CUT | 4,206 | 79,992 | 14,220 | 23,082 | 45,162 | 24,738 |
| EDICT | 4,206 | 59,981 | - | - | - | - |
| IP2P | 4,206 | 79,992 | 10,218 | 23,082 | 45,162 | 24,738 |
| Img2Img | 4,206 | 79,992 | 10,218 | 23,082 | 45,162 | 24,738 |
| LANIT | 4,907 | 97,993 | 11,921 | 26,929 | 52,689 | 28,861 |
| NST | 1,013 | 20,523 | 2,402 | 5,298 | 9,164 | 3,098 |
| Qwen-Image-Edit | 4,358 | - | 2,370 | 333 | 2,263 | - |
| SUSTechGAN | 2,904 | 55,996 | 6,812 | 15,388 | 30,108 | 16,492 |
| StyleID | 4,206 | 11,327 | - | - | - | - |
| TSIT | 4,206 | 83,994 | 10,218 | 23,082 | 45,162 | 24,738 |
| UniControl | 4,206 | 79,992 | 10,218 | 23,082 | 45,162 | 24,738 |
| VisualCloze | 4,206 | - | - | 9,945 | - | - |
| Weather_Effect_Generator | 2,103 | 39,996 | 7,110 | 9,670 | 15,054 | 8,246 |
| albumentations_weather | 2,103 | 39,996 | 7,110 | 11,541 | 22,581 | 12,369 |
| augmenters | 3,505 | 66,660 | 11,850 | 19,235 | 37,635 | 20,615 |
| automold | 2,103 | 39,996 | 7,110 | 11,541 | 22,581 | 12,369 |
| cycleGAN | 4,206 | 79,992 | 10,218 | 23,082 | 45,162 | 24,738 |
| cyclediffusion | 4,206 | 79,992 | 14,220 | 12,465 | - | - |
| flux2 | 4,206 | - | - | - | - | - |
| flux_kontext | - | - | - | - | 45,162 | 24,738 |
| stargan_v2 | 4,206 | 79,992 | 10,218 | 23,082 | 45,162 | 24,738 |
| step1x_new | 4,206 | 10,325 | - | 20,952 | 20,959 | 20,901 |
| step1x_v1p2 | 4,194 | 12,278 | - | - | 26,727 | 24,562 |

## Notes

### Models with Special Characteristics

1. **AOD-Net**: Restoration task (defogging only), not weather generation
2. **LANIT**: Includes clear_day as a domain (bidirectional)
3. **NST**: Only covers cloudy and dawn_dusk domains
4. **Weather_Effect_Generator, automold**: Only foggy, rainy, snowy
5. **albumentations_weather**: Uses "rain", "snow" instead of "rainy", "snowy"
6. **augmenters**: Unknown domain labels
7. **SUSTechGAN**: Very few night and snowy samples (50 each)

### Active Generation (as of Dec 23, 2024)

- **VisualCloze**: 33,707 actual, 14,151 in manifest (still generating)
- **flux2**: 10,851 actual, 4,206 in manifest (still generating)
- **Qwen-Image-Edit**: 9,404 actual, 9,324 in manifest (still generating)

### Dataset Coverage

- **Full coverage (all 6 datasets)**: Attribute_Hallucination, CNetSeg, CUT, IP2P, Img2Img, LANIT, NST, TSIT, UniControl, SUSTechGAN, Weather_Effect_Generator, albumentations_weather, augmenters, automold, cycleGAN, stargan_v2
- **Partial coverage**: EDICT (ACDC, BDD100k), StyleID (ACDC, BDD100k), cyclediffusion (4), step1x_new (5), step1x_v1p2 (4), flux_kontext (2), VisualCloze (2), Qwen-Image-Edit (4), flux2 (1)

---

## Analysis

### Coverage Heatmap

Models with best coverage (6 weather domains × 6 datasets = 36 combinations):

| Model | Domain Coverage | Dataset Coverage | Combinations Covered |
|-------|-----------------|------------------|---------------------|
| Attribute_Hallucination | 6/6 | 6/6 | 36/36 |
| TSIT | 6/6 | 6/6 | 36/36 |
| IP2P | 6/6 | 6/6 | 36/36 |
| Img2Img | 6/6 | 6/6 | 36/36 |
| UniControl | 6/6 | 6/6 | 36/36 |
| CNetSeg | 6/6 | 6/6 | 36/36 |
| CUT | 6/6 | 6/6 | 36/36 |
| cycleGAN | 6/6 | 6/6 | 36/36 |
| stargan_v2 | 6/6 | 6/6 | 36/36 |
| LANIT | 7/6 | 6/6 | 42/36+ |
| SUSTechGAN | 6/6 | 6/6 | 36/36* |

*SUSTechGAN has very few samples for night (50) and snowy (50)

### Model Categories

#### Generation Methods (clear_day → weather)
- **GAN-based**: cycleGAN, stargan_v2, CUT, LANIT, TSIT, CNetSeg
- **Diffusion-based**: EDICT, cyclediffusion, IP2P, Img2Img, flux_kontext, flux2, step1x_new, step1x_v1p2, VisualCloze, Qwen-Image-Edit
- **Other**: UniControl, Attribute_Hallucination, StyleID, NST

#### Restoration Methods (weather → clear_day)
- **Defogging**: AOD-Net

#### Augmentation Methods
- **Physics-based**: Weather_Effect_Generator, automold
- **Learned**: albumentations_weather, augmenters

### Data Files

For detailed analysis, CSV files are available:
- `summary_by_domain.csv` - Images per domain per model
- `summary_by_dataset.csv` - Images per dataset per model  
- `summary_models.csv` - Model-level summary with primary domain/dataset

### Observations

1. **Domain Naming Inconsistency**: Some models use "fog" vs "foggy", "rain" vs "rainy"
   - Recommendation: Standardize to "foggy", "rainy", "snowy" in evaluation pipeline

2. **Unbalanced Datasets**: BDD100k dominates with 1.26M images (43% of total)
   - ACDC is smallest with 90K images (3% of total)

3. **Model Completeness**:
   - 22/26 models have complete manifests (100% match)
   - 4/26 are actively generating (manifests need regeneration)

4. **Domain Coverage Gaps**:
   - NST: Only cloudy, dawn_dusk (missing 4 domains)
   - Weather_Effect_Generator, automold: Only foggy, rainy, snowy (missing 3 domains)
   - SUSTechGAN: night and snowy have only 50 samples each
