#!/usr/bin/env python3
"""
Semantic Consistency Evaluation for Image-to-Image Translation
Uses DeepLabV3 (ResNet backbone) to compare segmentation masks between
source and translated images.

Author: Research Script
Date: November 2025
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import warnings
import logging

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


# Cityscapes color palette (19 classes + background) for visualization
CITYSCAPES_COLORS = np.array([
    [128,  64, 128], [244,  35, 232], [ 70,  70,  70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170,  30], [220, 220,   0],
    [107, 142,  35], [152, 251, 152], [ 70, 130, 180], [220,  20,  60],
    [255,   0,   0], [  0,   0, 142], [  0,   0,  70], [  0,  60, 100],
    [  0,  80, 100], [  0,   0, 230], [119,  11,  32], [  0,   0,   0]
], dtype=np.uint8)

CITYSCAPES_CLASS_NAMES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic_light', 'traffic_sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
    'bicycle'
]

def colorize_mask(mask: np.ndarray, palette: np.ndarray = CITYSCAPES_COLORS) -> Image.Image:
    """Convert a segmentation mask to a colorized PIL image using the palette."""
    if mask.ndim != 2:
        raise ValueError("Segmentation mask must be 2D for colorization")
    palette_len = palette.shape[0]
    clipped = np.clip(mask, 0, palette_len - 1)
    colored = palette[clipped]
    return Image.fromarray(colored, mode='RGB')


def build_output_path(
    root: Optional[Path],
    base_dir: Path,
    image_path: Path,
    suffix: str
) -> Optional[Path]:
    """Construct an output path mirroring the image path under the provided root."""
    if root is None:
        return None
    relative_path = image_path.relative_to(base_dir)
    return (root / relative_path).with_suffix(suffix)


def load_suffixes(config_path: Optional[Path]) -> Tuple[str, ...]:
    """Load default suffixes from a JSON config file."""
    if config_path is None:
        return ()
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        logging.warning("Suffix config not found at %s. Using empty defaults.", config_path)
        return ()
    except json.JSONDecodeError as err:
        logging.warning("Failed to parse suffix config %s: %s. Using empty defaults.", config_path, err)
        return ()

    suffixes = data.get('suffixes', [])
    if not isinstance(suffixes, list):
        logging.warning("'suffixes' must be a list in %s. Using empty defaults.", config_path)
        return ()

    normalized = []
    for entry in suffixes:
        if isinstance(entry, str):
            cleaned = entry.strip()
            if cleaned:
                normalized.append(cleaned)
    return tuple(normalized)

class SegFormerEvaluator:
    """
    Evaluator for semantic consistency using SegFormer.
    """
    
    def __init__(
        self,
        model_name: str = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
        device: Optional[str] = None,
        class_names: Optional[List[str]] = None,
        cache_dir: Optional[Path] = None,
        segmentation_cache_dir: Optional[Path] = None,
    ):
        """
        Initialize SegFormer model and processor.
        
        Args:
            model_name: HuggingFace model identifier
            device: Computation device ('cuda' or 'cpu'). Auto-detect if None.
            cache_dir: Optional directory to cache downloaded models
            segmentation_cache_dir: Optional directory with precalculated segmentation masks.
                Expected structure: {segmentation_cache_dir}/{image_stem}.npy
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = class_names if class_names else CITYSCAPES_CLASS_NAMES
        self.segmentation_cache_dir = Path(segmentation_cache_dir) if segmentation_cache_dir else None
        
        logging.info("Initializing SegFormer (%s) on %s...", model_name, self.device)
        if self.segmentation_cache_dir:
            logging.info("Using precalculated segmentation cache: %s", self.segmentation_cache_dir)
        
        # Load processor and model
        try:
            logging.debug("Attempting to load %s from cache...", model_name)
            self.processor = SegformerImageProcessor.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)
            self.model = SegformerForSemanticSegmentation.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)
            logging.debug("Successfully loaded from cache!")
        except Exception as e:
            logging.debug("Cache miss or error (%s). Downloading model...", e)
            self.processor = SegformerImageProcessor.from_pretrained(model_name, cache_dir=cache_dir)
            self.model = SegformerForSemanticSegmentation.from_pretrained(model_name, cache_dir=cache_dir)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Get number of classes from config
        self.num_classes = self.model.config.num_labels
        
        logging.info("Model loaded with %d classes!", self.num_classes)

    def load_cached_mask(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Try to load a precalculated segmentation mask from cache.
        
        Args:
            image_path: Path to the original image
            
        Returns:
            Segmentation mask as numpy array, or None if not found
        """
        if self.segmentation_cache_dir is None:
            return None
        
        cache_path = self.segmentation_cache_dir / f"{image_path.stem}.npy"
        if cache_path.exists():
            try:
                return np.load(cache_path, allow_pickle=False)
            except Exception as e:
                logging.warning("Failed to load cached mask %s: %s", cache_path, e)
                return None
        return None

    def segment_image(self, image_path: Path, use_cache: bool = True) -> np.ndarray:
        """Perform segmentation for a single image path, using cache if available."""
        if use_cache:
            cached = self.load_cached_mask(image_path)
            if cached is not None:
                return cached
        return self.segment_images([image_path], use_cache=False)[0]

    def segment_images(self, image_paths: List[Path], use_cache: bool = True) -> List[np.ndarray]:
        """
        Batch-segment multiple images to amortize preprocessing and inference.
        
        Args:
            image_paths: List of paths to images to segment
            use_cache: Whether to check for precalculated masks first
            
        Returns:
            List of segmentation masks as numpy arrays
        """
        if not image_paths:
            return []

        # Check cache first if enabled
        results: List[Optional[np.ndarray]] = [None] * len(image_paths)
        paths_to_compute: List[Tuple[int, Path]] = []
        
        if use_cache and self.segmentation_cache_dir:
            for idx, path in enumerate(image_paths):
                cached = self.load_cached_mask(path)
                if cached is not None:
                    results[idx] = cached
                else:
                    paths_to_compute.append((idx, path))
        else:
            paths_to_compute = list(enumerate(image_paths))
        
        # If all masks were cached, return early
        if not paths_to_compute:
            return [r for r in results if r is not None]
        
        # Compute remaining masks
        images: List[Image.Image] = []
        original_sizes: List[Tuple[int, int]] = []
        for _, path in paths_to_compute:
            image = Image.open(path).convert('RGB')
            images.append(image)
            original_sizes.append(image.size)  # (W, H)

        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        computed_masks: List[np.ndarray] = []
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

            for idx, (width, height) in enumerate(original_sizes):
                sample_logits = logits[idx:idx + 1]
                upsampled_logits = torch.nn.functional.interpolate(
                    sample_logits,
                    size=(height, width),  # (H, W)
                    mode='bilinear',
                    align_corners=False
                )
                prediction = upsampled_logits.argmax(dim=1).squeeze(0)
                computed_masks.append(prediction.cpu().numpy())
        
        # Place computed masks back into results
        for (original_idx, _), mask in zip(paths_to_compute, computed_masks):
            results[original_idx] = mask

        return [r for r in results if r is not None]
    
    @staticmethod
    def compute_pixel_accuracy(mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute pixel-wise accuracy between two segmentation masks.
        
        If masks have different shapes, resize mask2 to match mask1 using
        nearest-neighbor interpolation (to preserve class labels).
        """
        if mask1.shape != mask2.shape:
            # Resize mask2 to match mask1 using nearest-neighbor interpolation
            from scipy.ndimage import zoom
            zoom_factors = (mask1.shape[0] / mask2.shape[0], mask1.shape[1] / mask2.shape[1])
            mask2 = zoom(mask2, zoom_factors, order=0)  # order=0 = nearest neighbor
        
        correct_pixels = np.sum(mask1 == mask2)
        total_pixels = mask1.size
        return (correct_pixels / total_pixels) * 100.0
    
    @staticmethod
    def _resize_mask_to_match(mask1: np.ndarray, mask2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Resize mask2 to match mask1's shape if they differ.
        
        Uses nearest-neighbor interpolation to preserve class labels.
        """
        if mask1.shape == mask2.shape:
            return mask1, mask2
        
        from scipy.ndimage import zoom
        zoom_factors = (mask1.shape[0] / mask2.shape[0], mask1.shape[1] / mask2.shape[1])
        mask2_resized = zoom(mask2, zoom_factors, order=0)  # order=0 = nearest neighbor
        return mask1, mask2_resized
    
    @staticmethod
    def compute_iou(
        mask1: np.ndarray,
        mask2: np.ndarray,
        num_classes: int,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compute IoU metrics.
        
        If masks have different shapes, resize mask2 to match mask1.
        """
        if mask1.shape != mask2.shape:
            from scipy.ndimage import zoom
            zoom_factors = (mask1.shape[0] / mask2.shape[0], mask1.shape[1] / mask2.shape[1])
            mask2 = zoom(mask2, zoom_factors, order=0)
        
        ious = []
        class_ious = {}
        
        for cls in range(num_classes):
            mask1_cls = (mask1 == cls)
            mask2_cls = (mask2 == cls)
            
            intersection = np.logical_and(mask1_cls, mask2_cls).sum()
            union = np.logical_or(mask1_cls, mask2_cls).sum()
            
            if union > 0:
                iou = intersection / union
                ious.append(iou)
                label = (
                    class_names[cls]
                    if class_names and cls < len(class_names)
                    else f'class_{cls}'
                )
                class_ious[label] = iou
        
        miou = np.mean(ious) if ious else 0.0
        
        return {
            'mIoU': miou,
            'class_IoUs': class_ious
        }
        
    @staticmethod
    def compute_frequency_weighted_iou(
        mask1: np.ndarray,
        mask2: np.ndarray,
        num_classes: int,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compute frequency-weighted IoU, accounting for class imbalance.
        
        Args:
            mask1: First segmentation mask (H, W)
            mask2: Second segmentation mask (H, W)
            num_classes: Number of semantic classes
            class_names: Optional list of class labels for reporting
            
        Returns:
            Dictionary with fw-IoU, mIoU, and class frequencies (values may be nested)
        """
        if mask1.shape != mask2.shape:
            from scipy.ndimage import zoom
            zoom_factors = (mask1.shape[0] / mask2.shape[0], mask1.shape[1] / mask2.shape[1])
            mask2 = zoom(mask2, zoom_factors, order=0)
        
        class_ious = []
        class_frequencies = []
        class_iou_dict = {}
        
        total_pixels = mask1.size
        
        for cls in range(num_classes):
            # Get binary masks for current class
            mask1_cls = (mask1 == cls)
            mask2_cls = (mask2 == cls)
            
            # Compute intersection and union
            intersection = np.logical_and(mask1_cls, mask2_cls).sum()
            union = np.logical_or(mask1_cls, mask2_cls).sum()
            
            # Frequency of class in ground truth (mask1)
            frequency = mask1_cls.sum() / total_pixels
            
            # Compute IoU (skip if class not present in either mask)
            if union > 0:
                iou = intersection / union
                class_ious.append(iou)
                class_frequencies.append(frequency)
                label = (
                    class_names[cls]
                    if class_names and cls < len(class_names)
                    else f'class_{cls}'
                )
                class_iou_dict[label] = {
                    'IoU': float(iou),
                    'frequency': float(frequency)
                }
        
        # Compute metrics
        miou = float(np.mean(class_ious)) if class_ious else 0.0
        
        # Frequency-weighted IoU
        if class_frequencies:
            fw_iou = float(
                np.sum(np.array(class_frequencies) * np.array(class_ious)) / np.sum(class_frequencies)
            )
        else:
            fw_iou = 0.0
        
        return {
            'mIoU': miou * 100.0,
            'fw_IoU': fw_iou * 100.0,
            'class_details': class_iou_dict
        }

        
    def evaluate_pair(
        self,
        source_path: Path,
        translated_path: Path
    ) -> Dict[str, float]:
        """Evaluate semantic consistency for a single image pair."""
        # Segment both images in a single batch for efficiency
        masks = self.segment_images([source_path, translated_path])
        source_mask, translated_mask = masks[0], masks[1]
        
        # Compute metrics
        pixel_acc = self.compute_pixel_accuracy(source_mask, translated_mask)
        iou_metrics = self.compute_iou(
            source_mask,
            translated_mask,
            self.num_classes,
            self.class_names
        )
        weighted_iou_metrics = self.compute_frequency_weighted_iou(
            source_mask,
            translated_mask,
            self.num_classes,
            self.class_names
        )
        
        return {
            'pixel_accuracy': pixel_acc,
            'mIoU': iou_metrics['mIoU'] * 100.0,
            'class_IoUs': iou_metrics['class_IoUs'],
            'fw_IoU': weighted_iou_metrics['fw_IoU'],
            'class_details': weighted_iou_metrics['class_details']
        }

    def evaluate_pairs_batched(
        self,
        pairs: List[Tuple[Path, Path]],
        batch_size: int = 8,
        original_cache_dir: Optional[Path] = None,
        generated_cache_dir: Optional[Path] = None,
        original_cache_dirs: Optional[Dict[str, Path]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate semantic consistency for multiple pairs with batched inference.
        
        This is significantly faster than calling evaluate_pair() in a loop
        because it batches the GPU inference and supports precalculated masks.
        
        Args:
            pairs: List of (source_path, translated_path) tuples
            batch_size: Number of pairs to process at once
            original_cache_dir: Single directory with precalculated masks for original images
                Expected structure: {original_cache_dir}/{image_stem}.npy
            generated_cache_dir: Directory with precalculated masks for generated images
                Expected structure: {generated_cache_dir}/{image_stem}.npy
            original_cache_dirs: Dict mapping dataset names to cache directories.
                The dataset is inferred from the image path (e.g., .../BDD100k/... -> BDD100k)
                Expected structure: {cache_dir}/{image_stem}.npy
            
        Returns:
            List of result dictionaries, one per pair
        """
        results = []
        cache_hits = 0
        cache_misses = 0
        
        def find_dataset_from_path(image_path: Path) -> Optional[str]:
            """Extract dataset name from image path (e.g., .../BDD100k/cloudy/image.png -> BDD100k)"""
            parts = image_path.parts
            # Known dataset names
            known_datasets = {'ACDC', 'BDD100k', 'BDD10k', 'IDD-AW', 'MapillaryVistas', 'OUTSIDE15k'}
            for part in parts:
                if part in known_datasets:
                    return part
            return None
        
        def load_cached_mask_multi(image_path: Path, single_cache: Optional[Path], multi_cache: Optional[Dict[str, Path]]) -> Optional[np.ndarray]:
            """Try to load mask from single cache dir or multi-cache dict based on dataset."""
            # Try single cache first
            if single_cache:
                cache_path = single_cache / f"{image_path.stem}.npy"
                if cache_path.exists():
                    try:
                        return np.load(cache_path, allow_pickle=False)
                    except Exception:
                        pass
            
            # Try multi-cache based on dataset
            if multi_cache:
                dataset = find_dataset_from_path(image_path)
                if dataset and dataset in multi_cache:
                    cache_path = multi_cache[dataset] / f"{image_path.stem}.npy"
                    if cache_path.exists():
                        try:
                            return np.load(cache_path, allow_pickle=False)
                        except Exception:
                            pass
            
            # Also check instance's segmentation_cache_dir
            if self.segmentation_cache_dir:
                cache_path = self.segmentation_cache_dir / f"{image_path.stem}.npy"
                if cache_path.exists():
                    try:
                        return np.load(cache_path, allow_pickle=False)
                    except Exception:
                        pass
            
            return None
        
        for batch_start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[batch_start:batch_start + batch_size]
            
            # Try to load cached masks first
            source_masks: List[Optional[np.ndarray]] = []
            translated_masks: List[Optional[np.ndarray]] = []
            paths_to_compute: List[Tuple[int, str, Path]] = []  # (pair_idx, 'source'|'translated', path)
            
            for i, (source_path, translated_path) in enumerate(batch_pairs):
                # Check original/source cache (supports multi-dataset structure)
                source_mask = load_cached_mask_multi(source_path, original_cache_dir, original_cache_dirs)
                
                if source_mask is not None:
                    cache_hits += 1
                else:
                    paths_to_compute.append((i, 'source', source_path))
                    cache_misses += 1
                source_masks.append(source_mask)
                
                # Check generated cache (typically single directory)
                translated_mask = None
                if generated_cache_dir:
                    cache_path = generated_cache_dir / f"{translated_path.stem}.npy"
                    if cache_path.exists():
                        try:
                            translated_mask = np.load(cache_path, allow_pickle=False)
                            cache_hits += 1
                        except Exception:
                            pass
                
                if translated_mask is None:
                    paths_to_compute.append((i, 'translated', translated_path))
                    cache_misses += 1
                translated_masks.append(translated_mask)
            
            # Compute any missing masks in a single batch
            if paths_to_compute:
                compute_paths = [path for _, _, path in paths_to_compute]
                computed_masks = self.segment_images(compute_paths, use_cache=False)
                
                for (pair_idx, mask_type, _), mask in zip(paths_to_compute, computed_masks):
                    if mask_type == 'source':
                        source_masks[pair_idx] = mask
                    else:
                        translated_masks[pair_idx] = mask
            
            # Compute metrics for each pair
            for i, (source_path, translated_path) in enumerate(batch_pairs):
                source_mask = source_masks[i]
                translated_mask = translated_masks[i]
                
                if source_mask is None or translated_mask is None:
                    # This shouldn't happen, but handle gracefully
                    results.append({
                        'pixel_accuracy': 0.0,
                        'mIoU': 0.0,
                        'class_IoUs': {},
                        'fw_IoU': 0.0,
                        'class_details': {},
                        'error': 'Failed to compute segmentation masks'
                    })
                    continue
                
                pixel_acc = self.compute_pixel_accuracy(source_mask, translated_mask)
                iou_metrics = self.compute_iou(
                    source_mask,
                    translated_mask,
                    self.num_classes,
                    self.class_names
                )
                weighted_iou_metrics = self.compute_frequency_weighted_iou(
                    source_mask,
                    translated_mask,
                    self.num_classes,
                    self.class_names
                )
                
                results.append({
                    'pixel_accuracy': pixel_acc,
                    'mIoU': iou_metrics['mIoU'] * 100.0,
                    'class_IoUs': iou_metrics['class_IoUs'],
                    'fw_IoU': weighted_iou_metrics['fw_IoU'],
                    'class_details': weighted_iou_metrics['class_details']
                })
        
        # Log cache statistics at the end if cache hit rate is low
        total = cache_hits + cache_misses
        if total > 0:
            hit_rate = cache_hits / total
            # Avoid spamming the console for healthy cache rates; only warn when coverage < 50%
            if hit_rate < 0.5:
                logging.info(
                    "Segmentation cache: %d/%d hits (%.1f%%), %d computed",
                    cache_hits,
                    total,
                    100 * hit_rate,
                    cache_misses,
                )
        
        return results

class DeepLabV3Evaluator:
    """
    Evaluator for semantic consistency using DeepLabV3 segmentation.
    """
    
    def __init__(
        self,
        backbone: str = 'resnet101',
        device: Optional[str] = None,
        num_classes: int = 19,  # Cityscapes has 19 classes + background
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize the DeepLabV3 model and preprocessing pipeline.
        
        Args:
            backbone: Model backbone ('resnet50' or 'resnet101')
            device: Computation device ('cuda' or 'cpu'). Auto-detect if None.
            num_classes: Number of segmentation classes (20 for Cityscapes)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.class_names = class_names if class_names else CITYSCAPES_CLASS_NAMES
        
        logging.info("Initializing DeepLabV3 with %s backbone on %s...", backbone, self.device)
        
        # Load pre-trained DeepLabV3 model
        if backbone == 'resnet50':
            self.model = models.segmentation.deeplabv3_resnet50(
                pretrained=False,
                progress=True, 
                num_classes=self.num_classes                
            )
            state_dict = torch.load(
                'weights/deeplabv3_resnet50_cityscapes.bin',
                map_location=self.device
            )
            # Some checkpoints include auxiliary heads; load non-matching keys loosely.
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if unexpected:
                logging.warning("Ignored unexpected keys in checkpoint: %s", unexpected)
            if missing:
                logging.warning("Missing keys when loading checkpoint: %s", missing)
            self.model.eval()
        elif backbone == 'resnet101':
            self.model = models.segmentation.deeplabv3_resnet101(
                pretrained=True,
                progress=True, 
                num_classes=self.num_classes
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose 'resnet50' or 'resnet101'.")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Define preprocessing transforms (ImageNet normalization)
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        logging.info("Model loaded successfully!")
    
    def segment_image(self, image_path: Path) -> np.ndarray:
        """
        Perform semantic segmentation on a single image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Segmentation mask as numpy array (H, W) with class indices
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size  # (W, H)
        
        input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Perform inference
        with torch.no_grad():
            output = self.model(input_tensor)['out']
            
            # Get class predictions (argmax over channels)
            predictions = torch.argmax(output, dim=1).squeeze(0)
            
            # Resize to original image size if needed
            if predictions.shape != (original_size[1], original_size[0]):
                predictions = F.interpolate(
                    predictions.unsqueeze(0).unsqueeze(0).float(),
                    size=(original_size[1], original_size[0]),
                    mode='nearest'
                ).squeeze().long()
        
        return predictions.cpu().numpy()
    
    @staticmethod
    def compute_pixel_accuracy(mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        Compute pixel-wise accuracy between two segmentation masks.
        
        If masks have different shapes, resize mask2 to match mask1.
        
        Args:
            mask1: First segmentation mask (H, W)
            mask2: Second segmentation mask (H, W)
            
        Returns:
            Pixel accuracy as percentage (0-100)
        """
        if mask1.shape != mask2.shape:
            from scipy.ndimage import zoom
            zoom_factors = (mask1.shape[0] / mask2.shape[0], mask1.shape[1] / mask2.shape[1])
            mask2 = zoom(mask2, zoom_factors, order=0)
        
        correct_pixels = np.sum(mask1 == mask2)
        total_pixels = mask1.size
        return (correct_pixels / total_pixels) * 100.0
    
    @staticmethod
    def compute_iou(
        mask1: np.ndarray,
        mask2: np.ndarray,
        num_classes: int,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compute Intersection-over-Union (IoU) metrics.
        
        If masks have different shapes, resize mask2 to match mask1.
        
        Args:
            mask1: First segmentation mask (H, W)
            mask2: Second segmentation mask (H, W)
            num_classes: Number of segmentation classes
            
        Returns:
            Dictionary with mIoU and class-wise IoU values
        """
        if mask1.shape != mask2.shape:
            from scipy.ndimage import zoom
            zoom_factors = (mask1.shape[0] / mask2.shape[0], mask1.shape[1] / mask2.shape[1])
            mask2 = zoom(mask2, zoom_factors, order=0)
        
        ious = []
        class_ious = {}
        
        for cls in range(num_classes):
            # Get binary masks for current class
            mask1_cls = (mask1 == cls)
            mask2_cls = (mask2 == cls)
            
            # Compute intersection and union
            intersection = np.logical_and(mask1_cls, mask2_cls).sum()
            union = np.logical_or(mask1_cls, mask2_cls).sum()
            
            # Compute IoU (skip if class not present in either mask)
            if union > 0:
                iou = intersection / union
                ious.append(iou)
                label = (
                    class_names[cls]
                    if class_names and cls < len(class_names)
                    else f'class_{cls}'
                )
                class_ious[label] = iou
        
        # Compute mean IoU
        miou = np.mean(ious) if ious else 0.0
        
        return {
            'mIoU': miou,
            'class_IoUs': class_ious
        }
    
    @staticmethod
    def compute_frequency_weighted_iou(
        mask1: np.ndarray,
        mask2: np.ndarray,
        num_classes: int,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compute frequency-weighted IoU, accounting for class imbalance.
        
        If masks have different shapes, resize mask2 to match mask1.
        
        Args:
            mask1: First segmentation mask (H, W)
            mask2: Second segmentation mask (H, W)
            num_classes: Number of semantic classes
            class_names: Optional list of class labels for reporting
            
        Returns:
            Dictionary with fw-IoU, mIoU, and class frequencies (values may be nested)
        """
        if mask1.shape != mask2.shape:
            from scipy.ndimage import zoom
            zoom_factors = (mask1.shape[0] / mask2.shape[0], mask1.shape[1] / mask2.shape[1])
            mask2 = zoom(mask2, zoom_factors, order=0)
        
        class_ious = []
        class_frequencies = []
        class_iou_dict = {}
        
        total_pixels = mask1.size
        
        for cls in range(num_classes):
            # Get binary masks for current class
            mask1_cls = (mask1 == cls)
            mask2_cls = (mask2 == cls)
            
            # Compute intersection and union
            intersection = np.logical_and(mask1_cls, mask2_cls).sum()
            union = np.logical_or(mask1_cls, mask2_cls).sum()
            
            # Frequency of class in ground truth (mask1)
            frequency = mask1_cls.sum() / total_pixels
            
            # Compute IoU (skip if class not present in either mask)
            if union > 0:
                iou = intersection / union
                class_ious.append(iou)
                class_frequencies.append(frequency)
                label = (
                    class_names[cls]
                    if class_names and cls < len(class_names)
                    else f'class_{cls}'
                )
                class_iou_dict[label] = {
                    'IoU': float(iou),
                    'frequency': float(frequency)
                }
        
        # Compute metrics
        miou = float(np.mean(class_ious)) if class_ious else 0.0
        
        # Frequency-weighted IoU
        if class_frequencies:
            fw_iou = float(
                np.sum(np.array(class_frequencies) * np.array(class_ious)) / np.sum(class_frequencies)
            )
        else:
            fw_iou = 0.0
        
        return {
            'mIoU': miou * 100.0,
            'fw_IoU': fw_iou * 100.0,
            'class_details': class_iou_dict
        }

    def evaluate_pair(
        self,
        source_path: Path,
        translated_path: Path
    ) -> Dict[str, Any]:
        """
        Evaluate semantic consistency for a single image pair.
        
        Args:
            source_path: Path to source image
            translated_path: Path to translated image
            
        Returns:
            Dictionary with consistency metrics
        """
        # Segment both images
        source_mask = self.segment_image(source_path)
        translated_mask = self.segment_image(translated_path)
        
        # Compute metrics
        pixel_acc = self.compute_pixel_accuracy(source_mask, translated_mask)
        iou_metrics = self.compute_iou(
            source_mask,
            translated_mask,
            self.num_classes,
            self.class_names
        )
        
        return {
            'pixel_accuracy': pixel_acc,
            'mIoU': iou_metrics['mIoU'] * 100.0,  # Convert to percentage
            'class_IoUs': iou_metrics['class_IoUs']
        }


def get_image_pairs(
    source_dir: Path,
    translated_dir: Path,
    extensions: Tuple[str, ...] = ('.png',),
    strip_suffixes: Tuple[str, ...] = (),
    default_suffixes: Tuple[str, ...] = ()
) -> List[Tuple[Path, Path]]:
    """
    Find matching image pairs by mirroring directory structures and
    tolerating filename suffixes (e.g., *_ref_anon).
    
    Args:
        source_dir: Directory with source images
        translated_dir: Directory with translated images
        extensions: Allowed image file extensions
        strip_suffixes: Filename suffixes to drop before matching
        
    Returns:
        List of (source_path, translated_path) tuples
    """
    extensions = tuple(sorted({ext.lower() for ext in extensions})) if extensions else ('.png',)
    combined_suffixes = (strip_suffixes or ()) + default_suffixes
    suffix_order = [suffix for suffix in dict.fromkeys(combined_suffixes) if suffix]
    suffixes: Tuple[str, ...] = tuple(sorted(suffix_order, key=len, reverse=True))
    separators = ('_', '-', '.', ' ')

    def normalize_stem(stem: str) -> str:
        """Strip configured suffixes (and adjoining separators) from a filename stem."""
        base = stem
        changed = True
        while changed:
            changed = False
            for suffix in suffixes:
                if not suffix or not base.endswith(suffix):
                    continue
                trimmed = base[:-len(suffix)]
                while trimmed and trimmed[-1] in separators:
                    trimmed = trimmed[:-1]
                base = trimmed
                changed = True
                break
        return base

    def iter_images(root: Path) -> List[Path]:
        if not root.exists():
            return []
        return sorted(
            path for path in root.rglob('*')
            if path.is_file() and path.suffix.lower() in extensions
        )

    def dir_key(base: Path, path: Path) -> str:
        rel_parent = path.relative_to(base).parent
        return rel_parent.as_posix() if rel_parent != Path('.') else '.'

    def build_entries(root: Path) -> List[Tuple[Path, str, str]]:
        entries: List[Tuple[Path, str, str]] = []
        for img_path in iter_images(root):
            entries.append((
                img_path,
                dir_key(root, img_path),
                normalize_stem(img_path.stem)
            ))
        return entries

    source_entries = build_entries(source_dir)
    translated_entries = build_entries(translated_dir)

    translated_by_dir: Dict[Tuple[str, str], List[Path]] = {}
    translated_by_stem: Dict[str, List[Path]] = {}
    for path, rel_dir, norm_stem in translated_entries:
        translated_by_dir.setdefault((rel_dir, norm_stem), []).append(path)
        translated_by_stem.setdefault(norm_stem, []).append(path)

    logging.info("Found %d source images for pairing.", len(source_entries))

    pairs: List[Tuple[Path, Path]] = []
    missing = 0
    fallback_matches = 0

    for source_path, rel_dir, norm_stem in source_entries:
        key = (rel_dir, norm_stem)
        candidates = translated_by_dir.get(key)
        used_fallback = False

        if not candidates:
            candidates = translated_by_stem.get(norm_stem, [])
            used_fallback = bool(candidates)

        if not candidates:
            missing += 1
            logging.warning("No matching translated image for %s", source_path.relative_to(source_dir))
            continue

        match = next(
            (candidate for candidate in candidates
             if candidate.suffix.lower() == source_path.suffix.lower()),
            candidates[0]
        )

        if used_fallback:
            fallback_matches += 1
            logging.debug(
                "Matched %s using fallback search in translated tree.", source_path.relative_to(source_dir)
            )

        pairs.append((source_path, match))

    logging.info(
        "Pairing complete: %d matches, %d missing, %d fallback matches.", len(pairs), missing, fallback_matches
    )

    return pairs


def aggregate_results(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate metrics across all image pairs.
    
    Args:
        all_results: List of result dictionaries from each pair
        
    Returns:
        Dictionary with averaged metrics
    """
    if not all_results:
        return {}
    
    # Aggregate pixel accuracy, mIoU, and fw-IoU
    avg_pixel_acc = np.mean([r['pixel_accuracy'] for r in all_results])
    avg_miou = np.mean([r['mIoU'] for r in all_results])
    fw_ious = [r['fw_IoU'] for r in all_results if 'fw_IoU' in r]
    avg_fw_iou = np.mean(fw_ious) if fw_ious else 0.0
    
    # Aggregate class-wise IoU (average across pairs where class appears)
    all_class_ious = {}
    fw_class_details: Dict[str, Dict[str, List[float]]] = {}
    for result in all_results:
        for cls, iou in result['class_IoUs'].items():
            if cls not in all_class_ious:
                all_class_ious[cls] = []
            all_class_ious[cls].append(iou)
        for cls, stats in result.get('class_details', {}).items():
            entry = fw_class_details.setdefault(cls, {'IoU': [], 'frequency': []})
            entry['IoU'].append(stats.get('IoU', 0.0))
            entry['frequency'].append(stats.get('frequency', 0.0))
    
    avg_class_ious = {
        cls: np.mean(ious) * 100.0  # Convert to percentage
        for cls, ious in all_class_ious.items()
    }

    avg_fw_class_details = {
        cls: {
            'average_IoU': np.mean(details['IoU']) * 100.0,
            'average_frequency': np.mean(details['frequency']) * 100.0
        }
        for cls, details in fw_class_details.items()
        if details['IoU']
    }
    
    return {
        'average_pixel_accuracy': avg_pixel_acc,
        'average_mIoU': avg_miou,
        'average_fw_IoU': avg_fw_iou,
        'average_class_IoUs': avg_class_ious,
        'average_fw_class_details': avg_fw_class_details,
        'num_pairs_evaluated': len(all_results)
    }


def save_results(
    results: Dict[str, Any],
    output_path: Path,
    detailed_results: Optional[List[Dict[str, Any]]] = None,
    source_dir: Optional[Path] = None,
    translated_dir: Optional[Path] = None
):
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Aggregated results dictionary
        output_path: Path to output JSON file
        detailed_results: Optional list of per-pair results
    """
    output_data: Dict[str, Any] = {
        'summary': results,
        'meta': {
            'source_dir': str(source_dir) if source_dir else None,
            'translated_dir': str(translated_dir) if translated_dir else None,
            'generated_at': datetime.now().isoformat()
        }
    }
    
    if detailed_results:
        output_data['detailed_results'] = detailed_results
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logging.info("Results saved to: %s", output_path)


def print_results(
    results: Dict[str, Any],
    source_dir: Optional[Path] = None,
    translated_dir: Optional[Path] = None
):
    """
    Print formatted evaluation results.
    
    Args:
        results: Aggregated results dictionary
    """
    logging.info("\n" + "="*60)
    logging.info("SEMANTIC CONSISTENCY EVALUATION RESULTS")
    logging.info("="*60)
    if source_dir:
        logging.info("Source directory: %s", source_dir)
    if translated_dir:
        logging.info("Translated directory: %s", translated_dir)
    logging.info("Number of image pairs evaluated: %d", results['num_pairs_evaluated'])
    logging.info("Average Pixel Accuracy: %.2f%%", results['average_pixel_accuracy'])
    logging.info("Average mIoU: %.2f%%", results['average_mIoU'])
    if 'average_fw_IoU' in results:
        logging.info("Average Frequency-Weighted IoU: %.2f%%", results['average_fw_IoU'])
    
    if results.get('average_class_IoUs'):
        logging.info("Class-wise IoU (%%):")
        for cls, iou in sorted(results['average_class_IoUs'].items()):
            logging.info("  %s: %.2f%%", cls, iou)
    if results.get('average_fw_class_details'):
        logging.info("Frequency-weighted class details (%%):")
        for cls, stats in sorted(results['average_fw_class_details'].items()):
            avg_iou = stats.get('average_IoU', 0.0)
            avg_freq = stats.get('average_frequency', 0.0)
            logging.info("  %s: IoU %.2f%%, freq %.2f%%", cls, avg_iou, avg_freq)
    
    logging.info("="*60)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Evaluate semantic consistency in image-to-image translation using SegFormer',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--source_dir',
        type=str,
        required=True,
        help='Directory containing source images'
    )
    parser.add_argument(
        '--translated_dir',
        type=str,
        required=True,
        help='Directory containing translated images (matching filenames)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--backbone',
        type=str,
        choices=['resnet50', 'resnet101'],
        default='resnet50',
        help='DeepLabV3 backbone architecture (default: resnet101)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results',
        help='Directory to save evaluation results (default: ./results)'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        default=None,
        help='Computation device (default: auto-detect)'
    )
    parser.add_argument(
        '--save_detailed',
        action='store_true',
        help='Save detailed per-pair results in output JSON'
    )
    parser.add_argument(
        '--extensions',
        type=str,
        nargs='+',
        default=['.png', '.jpg', '.jpeg', '.bmp'],
        help='Image file extensions to process (default: .png .jpg .jpeg .bmp)'
    )
    parser.add_argument(
        '--strip_suffixes',
        type=str,
        nargs='+',
        default=None,
        help='Filename suffixes to strip from both source and translated stems before matching'
    )
    parser.add_argument(
        '--suffix_config',
        type=str,
        default='configs/suffixes.json',
        help='Path to JSON file listing default suffixes to strip when pairing filenames'
    )
    parser.add_argument(
        '--source_cache_dir',
        type=str,
        default='cache/',
        help='Directory to cache source segmentation masks (*.npy) for reuse'
    )
    parser.add_argument(
        '--reuse_cached_source',
        action='store_true',
        help='Load cached source masks from --source_cache_dir when available'
    )
    parser.add_argument(
        '--save_segmentations_dir',
        type=str,
        default=None,
        help='Directory to save raw segmentation masks (*.npy) for source and translated images'
    )
    parser.add_argument(
        '--save_color_segmentations_dir',
        type=str,
        default=None,
        help='Directory to save colorized segmentation previews (*.png)'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=[
            'segformer-b0',
            'segformer-b1', 
            'segformer-b2',
            'segformer-b3',
            'segformer-b4',
            'segformer-b5'
        ],
        default='segformer-b5',
        help='SegFormer model size (default: b5 for best accuracy)'
    )
    
    args = parser.parse_args()
    
    # Map model choice to HuggingFace model name
    model_mapping = {
        'segformer-b0': 'nvidia/segformer-b0-finetuned-cityscapes-768-768',
        'segformer-b1': 'nvidia/segformer-b1-finetuned-cityscapes-1024-1024',
        'segformer-b2': 'nvidia/segformer-b2-finetuned-cityscapes-1024-1024',
        'segformer-b3': 'nvidia/segformer-b3-finetuned-cityscapes-1024-1024',
        'segformer-b4': 'nvidia/segformer-b4-finetuned-cityscapes-1024-1024',
        'segformer-b5': 'nvidia/segformer-b5-finetuned-cityscapes-1024-1024',
    }
    
    
    # Convert paths
    source_dir = Path(args.source_dir)
    translated_dir = Path(args.translated_dir)
    output_dir = Path(args.output_dir)
    suffix_config_path = Path(args.suffix_config) if args.suffix_config else None
    source_cache_dir = Path(args.source_cache_dir) if args.source_cache_dir else None
    seg_save_dir = Path(args.save_segmentations_dir) if args.save_segmentations_dir else None
    color_save_dir = Path(args.save_color_segmentations_dir) if args.save_color_segmentations_dir else None
    seg_source_root = (seg_save_dir / 'source') if seg_save_dir else None
    seg_translated_root = (seg_save_dir / 'translated') if seg_save_dir else None
    color_source_root = (color_save_dir / 'source') if color_save_dir else None
    color_translated_root = (color_save_dir / 'translated') if color_save_dir else None
    
    # Validate directories
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    if not translated_dir.exists():
        raise FileNotFoundError(f"Translated directory not found: {translated_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    for optional_dir in (
        source_cache_dir,
        seg_save_dir,
        seg_source_root,
        seg_translated_root,
        color_save_dir,
        color_source_root,
        color_translated_root,
    ):
        if optional_dir:
            optional_dir.mkdir(parents=True, exist_ok=True)

    def save_mask_array(mask: np.ndarray, root: Optional[Path], base_dir: Path, image_path: Path):
        if root is None:
            return
        target_path = build_output_path(root, base_dir, image_path, '.npy')
        if target_path is None:
            return
        target_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(target_path, mask.astype(np.uint8))

    def save_color_preview(mask: np.ndarray, root: Optional[Path], base_dir: Path, image_path: Path):
        if root is None:
            return
        target_path = build_output_path(root, base_dir, image_path, '.png')
        if target_path is None:
            return
        target_path.parent.mkdir(parents=True, exist_ok=True)
        colorize_mask(mask).save(target_path)
    
    # Initialize evaluator

    model_name = model_mapping[args.model]    
    evaluator = SegFormerEvaluator(
        model_name=model_name,
        device=args.device
    )
    
    # Get image pairs
    logging.info("Finding image pairs...")
    default_suffixes = load_suffixes(suffix_config_path)

    image_pairs = get_image_pairs(
        source_dir,
        translated_dir,
        extensions=tuple(args.extensions),
        strip_suffixes=tuple(args.strip_suffixes) if args.strip_suffixes else (),
        default_suffixes=default_suffixes
    )
    
    if not image_pairs:
        logging.error("No matching image pairs found!")
        return
    
    logging.info("Found %d image pairs", len(image_pairs))
    
    # Evaluate all pairs
    all_results = []
    cache_hits = 0
    cache_writes = 0
    
    logging.info("Evaluating semantic consistency...")
    for source_path, translated_path in tqdm(image_pairs, desc="Processing"):
        try:
            cache_file = build_output_path(source_cache_dir, source_dir, source_path, '.npy')
            source_mask: Optional[np.ndarray] = None
            translated_mask: Optional[np.ndarray] = None

            pending: List[Tuple[str, Path]] = []

            if cache_file and args.reuse_cached_source and cache_file.exists():
                source_mask = np.load(cache_file, allow_pickle=False)
                cache_hits += 1
            else:
                pending.append(('source', source_path))

            pending.append(('translated', translated_path))

            if pending:
                computed_masks = evaluator.segment_images([path for _, path in pending])
                for (label, path), mask in zip(pending, computed_masks):
                    if label == 'source':
                        source_mask = mask
                        if cache_file:
                            cache_file.parent.mkdir(parents=True, exist_ok=True)
                            np.save(cache_file, source_mask.astype(np.uint8))
                            cache_writes += 1
                    else:
                        translated_mask = mask

            if source_mask is None or translated_mask is None:
                raise RuntimeError("Failed to generate both source and translated masks.")

            save_mask_array(source_mask, seg_source_root, source_dir, source_path)
            save_mask_array(translated_mask, seg_translated_root, translated_dir, translated_path)
            save_color_preview(source_mask, color_source_root, source_dir, source_path)
            save_color_preview(translated_mask, color_translated_root, translated_dir, translated_path)

            pixel_acc = evaluator.compute_pixel_accuracy(source_mask, translated_mask)
            iou_metrics = evaluator.compute_iou(source_mask, translated_mask, evaluator.num_classes, evaluator.class_names)
            weighted_iou_metrics = evaluator.compute_frequency_weighted_iou(
                source_mask,
                translated_mask,
                evaluator.num_classes, 
                evaluator.class_names
            )
            result = {
                'pixel_accuracy': pixel_acc,
                'mIoU': iou_metrics['mIoU'] * 100.0,
                'class_IoUs': iou_metrics['class_IoUs'],
                'fw_IoU': weighted_iou_metrics['fw_IoU'],
                'class_details': weighted_iou_metrics['class_details'],
                'source_image': source_path.name,
                'translated_image': translated_path.name
            }
            all_results.append(result)
        except Exception as e:
            logging.error("Error processing %s: %s", source_path.name, str(e))
            continue

    if source_cache_dir:
        logging.info(
            "Source cache summary -> hits: %d, writes: %d, directory: %s", cache_hits, cache_writes, source_cache_dir
        )
    
    # Aggregate results
    aggregated_results = aggregate_results(all_results)
    
    # Print results
    print_results(
        aggregated_results,
        source_dir=source_dir,
        translated_dir=translated_dir
    )
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = output_dir / f'semantic_consistency_results_{timestamp}.json'
    save_results(
        aggregated_results,
        output_path,
        detailed_results=all_results if args.save_detailed else None,
        source_dir=source_dir,
        translated_dir=translated_dir
    )
    
    logging.info("Evaluation complete!")


if __name__ == '__main__':
    main()
