#!/usr/bin/env python3
"""
evaluate_generation.py

End-to-end image-quality evaluation for image-to-image translation and
weather-synthesis models.

Supports: FID, IS, SSIM, LPIPS, PSNR  →  easily add more via plugins.
"""

from __future__ import annotations
import argparse, json, logging, sys, time, os
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

import yaml  # ← safe_load used; see README for security notes
try:  # pragma: no cover - supports both package and script execution
    from .utils.image_io import (
        LoadedImagePair,
        find_image_files,
        load_and_pair_images_with_paths,
        load_image,
    )
    from .utils.stats import summarise_metrics         # mean/std/CI
    from .utils.logging_setup import configure_logger  # coloured logging
except ImportError:  # pragma: no cover
    from utils.image_io import (
        LoadedImagePair,
        find_image_files,
        load_and_pair_images_with_paths,
        load_image,
    )
    from utils.stats import summarise_metrics         # mean/std/CI
    from utils.logging_setup import configure_logger  # coloured logging


SEGFORMER_MODEL_MAP: Dict[str, str] = {
    "segformer-b0": "nvidia/segformer-b0-finetuned-cityscapes-768-768",
    "segformer-b1": "nvidia/segformer-b1-finetuned-cityscapes-1024-1024",
    "segformer-b2": "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
    "segformer-b3": "nvidia/segformer-b3-finetuned-cityscapes-1024-1024",
    "segformer-b4": "nvidia/segformer-b4-finetuned-cityscapes-1024-1024",
    "segformer-b5": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
}
DEFAULT_SEGFORMER_MODEL = "segformer-b5"


def build_metric_plugins(
    names: Sequence[str],
    device: str,
    fid_options: Optional[Dict[str, Any]] = None,
):
    """Instantiate metric plugins, injecting optional FID-specific settings."""
    from metrics import registry  # imported lazily to dodge costly deps during tests

    plugins = []
    for name in names:
        if name not in registry:
            logging.warning("Metric '%s' not found in registry; skipping", name)
            continue
        kwargs: Dict[str, Any] = {"device": device}
        if name == "fid" and fid_options:
            kwargs.update(fid_options)
        plugins.append(registry[name](**kwargs))
    return plugins


def aggregate_semantic_results(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate semantic-consistency metrics across all pairs."""
    if not results:
        return {}

    avg_pixel_acc = float(np.mean([r["pixel_accuracy"] for r in results]))
    avg_miou = float(np.mean([r["mIoU"] for r in results]))
    fw_ious = [r.get("fw_IoU") for r in results if r.get("fw_IoU") is not None]
    avg_fw_iou = float(np.mean(fw_ious)) if fw_ious else 0.0

    class_iou_values: Dict[str, List[float]] = {}
    fw_class_details: Dict[str, Dict[str, List[float]]] = {}
    for result in results:
        for cls, score in result.get("class_IoUs", {}).items():
            class_iou_values.setdefault(cls, []).append(score)
        for cls, stats in result.get("class_details", {}).items():
            entry = fw_class_details.setdefault(cls, {"IoU": [], "frequency": []})
            entry["IoU"].append(stats.get("IoU", 0.0))
            entry["frequency"].append(stats.get("frequency", 0.0))

    avg_class_ious = {
        cls: float(np.mean(values) * 100.0)
        for cls, values in class_iou_values.items()
    }
    avg_fw_class_details = {
        cls: {
            "average_IoU": float(np.mean(details["IoU"]) * 100.0) if details["IoU"] else 0.0,
            "average_frequency": float(np.mean(details["frequency"]) * 100.0) if details["frequency"] else 0.0,
        }
        for cls, details in fw_class_details.items()
    }

    return {
        "average_pixel_accuracy": avg_pixel_acc,
        "average_mIoU": avg_miou,
        "average_fw_IoU": avg_fw_iou,
        "average_class_IoUs": avg_class_ious,
        "average_fw_class_details": avg_fw_class_details,
        "num_pairs_evaluated": len(results),
    }


def compute_semantic_consistency(
    pairs: Sequence[LoadedImagePair],
    model_variant: str,
    device: str,
    show_progress: bool = True,
    evaluator_factory: Optional[Callable[..., Any]] = None,
    cache_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """Run SegFormer-based semantic consistency on paired images."""
    if not pairs:
        return {
            "scalars": {},
            "metadata": {
                "enabled": False,
                "summary": {},
                "per_image_details": {},
            },
        }

    model_name = SEGFORMER_MODEL_MAP[model_variant]

    if evaluator_factory is None:
        from semantic_consistency import SegFormerEvaluator  # lazy import

        evaluator = SegFormerEvaluator(model_name=model_name, device=device, cache_dir=cache_dir)
    else:
        evaluator = evaluator_factory(model_name=model_name, device=device)

    iterator = (
        tqdm(pairs, desc="Semantic consistency", leave=False)
        if show_progress
        else pairs
    )
    per_image_scalars: Dict[str, Dict[str, float]] = {}
    detailed_results: Dict[str, Dict[str, Any]] = {}
    raw_results: List[Dict[str, Any]] = []

    for pair in iterator:
        result = evaluator.evaluate_pair(pair.original_path, pair.gen_path)
        raw_results.append(result)
        per_image_scalars[pair.name] = {
            "semantic_pixel_accuracy": float(result.get("pixel_accuracy", 0.0)),
            "semantic_mIoU": float(result.get("mIoU", 0.0)),
            "semantic_fw_IoU": float(result.get("fw_IoU", 0.0)),
        }
        detailed_results[pair.name] = {
            **result,
            "generated_image": str(pair.gen_path),
            "original_image": str(pair.original_path),
        }

    summary = aggregate_semantic_results(raw_results)
    metadata = {
        "enabled": True,
        "model_variant": model_variant,
        "model_name": model_name,
        "device": device,
        "summary": summary,
        "per_image_details": detailed_results,
    }

    return {"scalars": per_image_scalars, "metadata": metadata}


def load_target_images(target_dir: Path, image_size: Tuple[int, int]) -> List[torch.Tensor]:
    """Load all images from target directory for FID computation."""
    image_paths = find_image_files(target_dir)
    tensors: List[torch.Tensor] = []
    for path in image_paths:
        tensors.append(load_image(path, image_size))
    return tensors


def cycle_batches(tensors: Sequence[torch.Tensor], batch_size: int) -> Iterator[torch.Tensor]:
    """Yield stacked batches endlessly by cycling through provided tensors."""
    if not tensors:
        raise ValueError("Target tensor collection must not be empty")
    total = len(tensors)
    index = 0
    while True:
        batch: List[torch.Tensor] = []
        for _ in range(batch_size):
            batch.append(tensors[index])
            index = (index + 1) % total
        yield torch.stack(batch)


def load_fid_stats(stats_path: Path) -> Dict[str, np.ndarray]:
    """Load precomputed FID statistics (mu, sigma[, n]) from NPZ file."""
    with np.load(stats_path) as data:
        if "mu" not in data or "sigma" not in data:
            raise KeyError("FID stats file must contain 'mu' and 'sigma'")
        stats = {
            "mu": data["mu"],
            "sigma": data["sigma"],
        }
        if "n" in data:
            stats["n"] = int(data["n"])  # type: ignore[assignment]
        return stats


def discover_domains(root_dir: Path) -> List[str]:
    """Discover domain subfolders within a root directory.

    Returns list of domain names (subfolder names) that contain images.
    If root_dir contains images directly (no subfolders), returns ['_root'].
    """
    subfolders = [d for d in root_dir.iterdir() if d.is_dir()]
    domains = []
    for sub in subfolders:
        # Check if subfolder contains any images
        if find_image_files(sub):
            domains.append(sub.name)
    if not domains:
        # No subfolders with images; treat root as single domain
        if find_image_files(root_dir):
            return ["_root"]
    return sorted(domains)


def save_domain_stats(stats: Dict[str, Any], output_path: Path) -> None:
    """Save per-domain metrics to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats, indent=2))
    logging.info("Saved domain stats to %s", output_path)


def load_domain_stats(stats_path: Path) -> Dict[str, Any]:
    """Load per-domain metrics from a JSON file."""
    if not stats_path.exists():
        raise FileNotFoundError(f"Domain stats file not found: {stats_path}")
    with open(stats_path) as f:
        return json.load(f)


def save_fid_stats(mu: np.ndarray, sigma: np.ndarray, n: int, output_path: Path) -> None:
    """Save precomputed FID statistics to NPZ file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, mu=mu, sigma=sigma, n=n)
    logging.info("Saved FID stats to %s", output_path)


def compute_fid_statistics(
    image_dir: Path,
    image_size: Tuple[int, int],
    batch_size: int,
    device: str,
) -> Dict[str, np.ndarray]:
    """Compute FID statistics (mu, sigma, n) for a directory of images.

    Uses Inception-v3 to extract features and computes mean/covariance.
    """
    from torchmetrics.image.fid import FrechetInceptionDistance

    fid_metric = FrechetInceptionDistance(feature=2048, reset_real_features=False)
    fid_metric.to(device)

    image_paths = find_image_files(image_dir)
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")

    logging.info("Computing FID statistics for %d images in %s", len(image_paths), image_dir)

    # Load and process images in batches
    for batch_start in tqdm(
        range(0, len(image_paths), batch_size),
        desc=f"Processing {image_dir.name}",
        unit_scale=True,
    ):
        batch_paths = image_paths[batch_start : batch_start + batch_size]
        batch_tensors = [load_image(p, image_size) for p in batch_paths]
        batch = torch.stack(batch_tensors).to(device, non_blocking=True)

        # Convert to uint8 [0, 255] as required by FID
        batch_uint8 = (batch * 255).clamp(0, 255).to(torch.uint8)

        with torch.no_grad():
            fid_metric.update(batch_uint8, real=True)

    # Extract statistics from the metric's internal state
    n = int(fid_metric.real_features_num_samples.item())
    if n == 0:
        raise ValueError("No features computed; check image directory")

    # Recover mu and sigma from accumulated sums
    mu = fid_metric.real_features_sum / n
    # Covariance: E[X^T X] - mu^T mu
    # The metric stores sum of outer products adjusted for covariance calculation
    mu_outer = torch.outer(mu, mu)
    sigma = fid_metric.real_features_cov_sum / (n - 1) - (n / (n - 1)) * mu_outer

    return {
        "mu": mu.cpu().numpy(),
        "sigma": sigma.cpu().numpy(),
        "n": n,
    }


def precalculate_stats(
    original_dir: Path,
    target_dir: Path,
    stats_dir: Path,
    image_size: Tuple[int, int],
    batch_size: int,
    device: str,
    per_domain: bool = False,
    semantic_model: str = "segformer-b5",
    compute_segmentation: bool = True,
    compute_lpips_features: bool = True,
    cache_dir: Optional[Path] = None,
) -> None:
    """Precalculate and store FID statistics, segmentation maps, and LPIPS features.

    If per_domain is True, computes stats for each subdomain folder.
    
    Args:
        original_dir: Directory with original/reference images
        target_dir: Directory with target-domain images
        stats_dir: Output directory for precomputed statistics
        image_size: Image size for FID computation
        batch_size: Batch size for processing
        device: Computation device
        per_domain: Whether to process per subdomain
        semantic_model: SegFormer model variant for segmentation
        compute_segmentation: Whether to precompute segmentation maps
        compute_lpips_features: Whether to precompute LPIPS features
    """
    stats_dir.mkdir(parents=True, exist_ok=True)

    # Initialize segmentation model if needed
    segmentation_evaluator = None
    if compute_segmentation:
        from semantic_consistency import SegFormerEvaluator
        model_name = SEGFORMER_MODEL_MAP[semantic_model]
        logging.info("Initializing SegFormer (%s) for segmentation precomputation", semantic_model)
        segmentation_evaluator = SegFormerEvaluator(model_name=model_name, device=device, cache_dir=cache_dir)

    # Initialize LPIPS model if needed
    lpips_model = None
    if compute_lpips_features:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        logging.info("Initializing LPIPS VGG model for feature extraction")
        lpips_model = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)

    def process_directory(
        image_dir: Path,
        output_prefix: str,
        domain_stats_dir: Path,
    ) -> None:
        """Process a single directory: compute FID stats, segmentation, and LPIPS features."""
        image_paths = find_image_files(image_dir)
        if not image_paths:
            logging.warning("No images found in %s", image_dir)
            return

        # --- FID Statistics ---
        try:
            fid_stats = compute_fid_statistics(image_dir, image_size, batch_size, device)
            fid_path = domain_stats_dir / f"{output_prefix}_fid.npz"
            save_fid_stats(fid_stats["mu"], fid_stats["sigma"], fid_stats["n"], fid_path)
        except Exception as e:
            logging.error("Failed to compute FID stats for %s: %s", image_dir, e)

        # --- Segmentation Maps ---
        if segmentation_evaluator is not None:
            seg_dir = domain_stats_dir / f"{output_prefix}_segmentation"
            seg_dir.mkdir(parents=True, exist_ok=True)
            logging.info("Computing segmentation maps for %d images in %s", len(image_paths), image_dir)
            
            for batch_start in tqdm(
                range(0, len(image_paths), batch_size),
                desc=f"Segmentation {output_prefix}",
                leave=False,
            ):
                batch_paths = image_paths[batch_start : batch_start + batch_size]
                try:
                    masks = segmentation_evaluator.segment_images(batch_paths)
                    for path, mask in zip(batch_paths, masks):
                        output_path = seg_dir / f"{path.stem}.npy"
                        np.save(output_path, mask.astype(np.uint8))
                except Exception as e:
                    logging.error("Failed to segment batch starting at %s: %s", batch_paths[0], e)

        # --- LPIPS Features ---
        if lpips_model is not None:
            lpips_dir = domain_stats_dir / f"{output_prefix}_lpips_features"
            lpips_dir.mkdir(parents=True, exist_ok=True)
            logging.info("Computing LPIPS features for %d images in %s", len(image_paths), image_dir)
            
            # Access the underlying VGG network for feature extraction
            vgg_net = lpips_model.net
            
            for batch_start in tqdm(
                range(0, len(image_paths), batch_size),
                desc=f"LPIPS features {output_prefix}",
                leave=False,
            ):
                batch_paths = image_paths[batch_start : batch_start + batch_size]
                batch_tensors = [load_image(p, image_size) for p in batch_paths]
                batch = torch.stack(batch_tensors).to(device, non_blocking=True)
                
                # Normalize to [-1, 1] as expected by LPIPS
                batch_normalized = 2.0 * batch - 1.0
                
                try:
                    with torch.no_grad():
                        # Extract features from VGG layers
                        if hasattr(vgg_net, 'net'):
                            features = vgg_net.net(batch_normalized)
                        else:
                            features = vgg_net(batch_normalized)
                        # features is a list of tensors from different VGG layers
                        for i, (path, feat_list) in enumerate(zip(batch_paths, zip(*[f for f in features]))):
                            # Concatenate features from all layers
                            combined_features = torch.cat([f.flatten() for f in feat_list]).cpu().numpy()
                            output_path = lpips_dir / f"{path.stem}.npy"
                            np.save(output_path, combined_features)
                except Exception as e:
                    logging.error("Failed to extract LPIPS features for batch starting at %s: %s", batch_paths[0], e)

    if per_domain:
        # Discover domains from original directory
        domains = discover_domains(original_dir)
        if not domains:
            logging.error("No domains found in %s", original_dir)
            return
        logging.info("Precalculating stats for %d domains: %s", len(domains), domains)

        for domain in tqdm(domains, desc="Precalculating domains"):
            if domain == "_root":
                original_domain_dir = original_dir
                target_domain_dir = target_dir
                domain_prefix = "root"
            else:
                original_domain_dir = original_dir / domain
                target_domain_dir = target_dir / domain if target_dir else None
                domain_prefix = domain

            domain_stats_dir = stats_dir / domain_prefix
            domain_stats_dir.mkdir(parents=True, exist_ok=True)

            # Process original directory
            if original_domain_dir.exists() and find_image_files(original_domain_dir):
                process_directory(original_domain_dir, "original", domain_stats_dir)

            # Process target directory
            if target_domain_dir and target_domain_dir.exists() and find_image_files(target_domain_dir):
                process_directory(target_domain_dir, "target", domain_stats_dir)
    else:
        # Flat mode: compute stats for entire directories
        # Original stats
        if original_dir.exists() and find_image_files(original_dir):
            process_directory(original_dir, "original", stats_dir)

        # Target stats
        if target_dir and target_dir.exists() and find_image_files(target_dir):
            process_directory(target_dir, "target", stats_dir)

    logging.info("Precalculation complete. Stats saved to %s", stats_dir)


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate generated images against original reference images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--generated", type=Path,
                   help="Folder with generated images.")
    p.add_argument("--original", type=Path,
                   help="Folder with original/reference images (required for --precalculate).")
    p.add_argument("--target", type=Path,
                   help="Folder with target-domain images (required for FID unless --fid-stats is provided).")
    p.add_argument("-m", "--metrics", nargs="+",
                   default=["fid", "ssim", "lpips", "psnr", "is"],
                   help="Metrics to compute (plugins).")
    p.add_argument("-b", "--batch-size", type=int, default=32)
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto",
                   help="Computation device.")
    p.add_argument("--pairs", choices=["auto", "csv"], default="auto",
                   help="Pairing strategy: filename matching or CSV manifest.")
    p.add_argument("--manifest", type=Path,
                   help="CSV with gen_path,original_path if --pairs csv.")
    p.add_argument("--output", type=Path, default=Path("results.json"))
    p.add_argument("--config", type=Path,
                   help="YAML/JSON file with additional options.")
    p.add_argument("--fid-stats", type=Path,
                   help="Path to NPZ file with precomputed FID statistics (mu, sigma, optional n).")
    p.add_argument("-v", "--verbose", action="count", default=0,
                   help="Increase logging verbosity (-v, -vv).")
    p.add_argument("--semantic-consistency", action="store_true",
                   help="Compute semantic consistency via SegFormer.")
    p.add_argument("--semantic-model",
                   choices=sorted(SEGFORMER_MODEL_MAP.keys()),
                   help="SegFormer backbone variant (default segformer-b5).")
    p.add_argument("--semantic-device", choices=["cpu", "cuda", "auto"],
                   help="Device for semantic consistency evaluator (defaults to --device or auto).")
    p.add_argument("--per-domain", action="store_true",
                   help="Calculate and save metrics per domain (subfolder).")
    p.add_argument("--stats-dir", type=Path, default=Path("stats"),
                   help="Directory to save/load per-domain FID stats.")
    p.add_argument("--precalculate", action="store_true",
                   help="Precalculate and store FID statistics for original and target directories. "
                        "Only requires --original and --target; --generated is not needed.")
    p.add_argument("--no-segmentation", action="store_true",
                   help="Skip precomputing segmentation maps during --precalculate.")
    p.add_argument("--no-lpips", action="store_true",
                   help="Skip precomputing LPIPS features during --precalculate.")
    p.add_argument("--cache-dir", type=Path, default=Path("/scratch/chge7185/models"),
                   help="Directory to cache downloaded models (default: /scratch/chge7185/models).")
    return p.parse_args()


# --------------------------------------------------------------------------- #
def evaluate_domain(
    *,
    domain_name: str,
    gen_dir: Path,
    original_dir: Path,
    target_dir: Optional[Path],
    fid_stats_path: Optional[Path],
    metric_names: Sequence[str],
    batch_size: int,
    device: str,
    image_size: Tuple[int, int],
    pairs_strategy: str,
    manifest: Optional[Path],
    semantic_enabled: bool,
    semantic_model: str,
    semantic_device: str,
    verbose: int,
    stats_dir: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Evaluate a single domain and return results dict."""
    fid_requested = "fid" in metric_names

    # Load or compute FID reference stats
    fid_reference_stats: Optional[Dict[str, np.ndarray]] = None
    if fid_stats_path and fid_stats_path.exists():
        fid_reference_stats = load_fid_stats(fid_stats_path)
        logging.info("[%s] Loaded precomputed FID stats from %s", domain_name, fid_stats_path)

    fid_target_tensors: Optional[List[torch.Tensor]] = None
    if fid_requested and fid_reference_stats is None:
        if target_dir is None:
            logging.warning("[%s] FID metric requires --target or --fid-stats; skipping FID", domain_name)
            metric_names = [m for m in metric_names if m != "fid"]
        elif not target_dir.exists():
            logging.warning("[%s] Target directory not found: %s; skipping FID", domain_name, target_dir)
            metric_names = [m for m in metric_names if m != "fid"]
        else:
            fid_target_tensors = load_target_images(target_dir, image_size)
            if not fid_target_tensors:
                logging.warning("[%s] No images found in target directory %s; skipping FID", domain_name, target_dir)
                metric_names = [m for m in metric_names if m != "fid"]
            else:
                logging.info("[%s] Loaded %d target-domain images for FID", domain_name, len(fid_target_tensors))

    fid_requested = "fid" in metric_names

    # Load and pair images
    pairs = load_and_pair_images_with_paths(
        gen_dir=gen_dir,
        original_dir=original_dir,
        strategy=pairs_strategy,
        manifest=manifest,
        image_size=image_size,
    )
    if not pairs:
        logging.warning("[%s] No valid image pairs found; skipping domain.", domain_name)
        return {"domain": domain_name, "error": "No valid image pairs found"}
    logging.info("[%s] Found %d paired images", domain_name, len(pairs))

    # Build metrics
    fid_plugin_options = {"reference_stats": fid_reference_stats} if fid_reference_stats else None
    metrics = build_metric_plugins(metric_names, device=device, fid_options=fid_plugin_options)
    if not metrics:
        logging.warning("[%s] No valid metrics; skipping domain.", domain_name)
        return {"domain": domain_name, "error": "No valid metrics"}

    fid_target_iterator: Optional[Iterator[torch.Tensor]] = None
    if fid_requested and fid_reference_stats is None and fid_target_tensors:
        fid_target_iterator = cycle_batches(fid_target_tensors, batch_size)

    # Batch processing loop
    per_image_results: Dict[str, Dict[str, float]] = {}
    for batch_start in tqdm(
        list(range(0, len(pairs), batch_size)),
        desc=f"Evaluating {domain_name}",
        unit_scale=True,
        leave=False,
    ):
        batch_pairs = pairs[batch_start : batch_start + batch_size]
        gens = torch.stack([p.gen_tensor for p in batch_pairs]).to(device, non_blocking=True)
        originals = torch.stack([p.original_tensor for p in batch_pairs]).to(device, non_blocking=True)
        names = [p.name for p in batch_pairs]

        with torch.no_grad():
            for m in metrics:
                if m.name == "fid":
                    target_batch = None
                    if fid_reference_stats is None:
                        if fid_target_iterator is None:
                            continue
                        target_batch = next(fid_target_iterator).to(device, non_blocking=True)
                    scores = m(gens, target_batch)
                else:
                    scores = m(gens, originals)
                for n, s in zip(names, scores):
                    per_image_results.setdefault(n, {})[m.name] = float(s)

    # Semantic consistency evaluation
    semantic_payload: Optional[Dict[str, Any]] = None
    if semantic_enabled:
        logging.info("[%s] Running semantic consistency (model=%s, device=%s)", domain_name, semantic_model, semantic_device)
        try:
            semantic_payload = compute_semantic_consistency(
                pairs=pairs,
                model_variant=semantic_model,
                device=semantic_device,
                show_progress=verbose == 0,
                cache_dir=cache_dir,
            )
        except Exception as exc:  # noqa: BLE001
            logging.error("[%s] Semantic consistency evaluation failed: %s", domain_name, exc)
        else:
            for name, scores in semantic_payload["scalars"].items():
                per_image_results.setdefault(name, {}).update(scores)

    # Summary
    summary = summarise_metrics(per_image_results, alpha=0.95)

    result: Dict[str, Any] = {
        "domain": domain_name,
        "generated": str(gen_dir),
        "original": str(original_dir),
        "num_pairs": len(pairs),
        "metrics": summary,
        "per_image": per_image_results,
    }

    if semantic_payload and semantic_payload["metadata"].get("enabled"):
        result["semantic_consistency"] = semantic_payload["metadata"]

    # Save per-domain stats file
    if stats_dir:
        domain_stats_path = stats_dir / f"{domain_name}.json"
        save_domain_stats(result, domain_stats_path)

    return result


# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    
    # Set TORCH_HOME to cache_dir to control where models are downloaded
    if args.cache_dir:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ['TORCH_HOME'] = str(args.cache_dir)
        logging.info("Set TORCH_HOME to %s", args.cache_dir)

    configure_logger(args.verbose)

    # Load external configuration -------------------------------------------------
    cfg: Dict = {}
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f) if args.config.suffix in {".yml", ".yaml"} \
                 else json.load(f)
        logging.info("Merged configuration from %s", args.config)

    image_size_cfg = cfg.get("image_size")
    if image_size_cfg is not None:
        if isinstance(image_size_cfg, (list, tuple)) and len(image_size_cfg) == 2:
            image_size = (int(image_size_cfg[0]), int(image_size_cfg[1]))
        else:
            raise ValueError("image_size config value must be a 2-element list/tuple")
    else:
        image_size = (299, 299)

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else args.device
    logging.info("Using device=%s", device)

    target_dir_value = args.target or cfg.get("target")
    target_dir = Path(target_dir_value) if target_dir_value else None
    fid_stats_value = args.fid_stats or cfg.get("fid_stats")
    fid_stats_path = Path(fid_stats_value) if fid_stats_value else None

    semantic_cfg: Dict[str, Any] = cfg.get("semantic_consistency", {})
    semantic_enabled = bool(args.semantic_consistency or semantic_cfg.get("enabled"))
    semantic_model = (
        args.semantic_model
        or semantic_cfg.get("model")
        or DEFAULT_SEGFORMER_MODEL
    )
    semantic_device = (
        args.semantic_device
        or semantic_cfg.get("device")
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if semantic_device == "auto":
        semantic_device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------------------------------
    # Precalculate mode: compute and store FID statistics only
    # --------------------------------------------------------------------------
    if args.precalculate:
        if target_dir is None:
            logging.error("--precalculate requires --target to be specified")
            sys.exit(1)
        if args.original is None:
            logging.error("--precalculate requires --original to be specified")
            sys.exit(1)
        precalculate_stats(
            original_dir=args.original,
            target_dir=target_dir,
            stats_dir=args.stats_dir,
            image_size=image_size,
            batch_size=args.batch_size,
            device=device,
            per_domain=args.per_domain,
            semantic_model=semantic_model,
            compute_segmentation=not args.no_segmentation,
            compute_lpips_features=not args.no_lpips,
            cache_dir=args.cache_dir,
        )
        return

    # --------------------------------------------------------------------------
    # Validation: --original is required for evaluation mode
    # --------------------------------------------------------------------------
    if args.original is None:
        logging.error("--original is required for evaluation")
        sys.exit(1)

    # --------------------------------------------------------------------------
    # Per-domain or flat evaluation
    # --------------------------------------------------------------------------
    if args.per_domain:
        # Discover domains (subfolders) in generated directory
        domains = discover_domains(args.generated)
        if not domains:
            logging.error("No domains found in %s", args.generated)
            sys.exit(1)
        logging.info("Discovered %d domains: %s", len(domains), domains)

        all_domain_results: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "generated_root": str(args.generated),
            "original_root": str(args.original),
            "domains": {},
        }

        for domain in tqdm(domains, desc="Domains"):
            # Resolve directories for this domain
            if domain == "_root":
                gen_domain_dir = args.generated
                original_domain_dir = args.original
                target_domain_dir = target_dir
                domain_fid_stats = fid_stats_path
            else:
                gen_domain_dir = args.generated / domain
                original_domain_dir = args.original / domain
                target_domain_dir = target_dir / domain if target_dir else None
                # Try domain-specific FID stats first
                domain_fid_stats = args.stats_dir / f"{domain}_fid.npz" if args.stats_dir else None
                if domain_fid_stats and not domain_fid_stats.exists():
                    domain_fid_stats = fid_stats_path  # fall back to global

            result = evaluate_domain(
                domain_name=domain,
                gen_dir=gen_domain_dir,
                original_dir=original_domain_dir,
                target_dir=target_domain_dir,
                fid_stats_path=domain_fid_stats,
                metric_names=list(args.metrics),
                batch_size=args.batch_size,
                device=device,
                image_size=image_size,
                pairs_strategy=args.pairs,
                manifest=args.manifest,
                semantic_enabled=semantic_enabled,
                semantic_model=semantic_model,
                semantic_device=semantic_device,
                verbose=args.verbose,
                stats_dir=args.stats_dir,
                cache_dir=args.cache_dir,
            )
            all_domain_results["domains"][domain] = result

        # Aggregate summary across all domains
        all_per_image: Dict[str, Dict[str, float]] = {}
        for domain, res in all_domain_results["domains"].items():
            if "per_image" in res:
                for img_name, scores in res["per_image"].items():
                    all_per_image[f"{domain}/{img_name}"] = scores

        all_domain_results["aggregate_metrics"] = summarise_metrics(all_per_image, alpha=0.95)
        all_domain_results["total_images"] = len(all_per_image)

        args.output.write_text(json.dumps(all_domain_results, indent=2))
        logging.info("Results written to %s", args.output)

    else:
        # Original flat evaluation
        fid_requested = "fid" in args.metrics

        fid_reference_stats: Optional[Dict[str, np.ndarray]] = None
        if fid_stats_path:
            if not fid_stats_path.exists():
                raise FileNotFoundError(f"FID stats file not found: {fid_stats_path}")
            fid_reference_stats = load_fid_stats(fid_stats_path)
            logging.info("Loaded precomputed FID stats from %s", fid_stats_path)

        fid_target_tensors: Optional[List[torch.Tensor]] = None
        if fid_requested and fid_reference_stats is None:
            if target_dir is None:
                logging.error("FID metric requires --target or --fid-stats")
                sys.exit(1)
            if not target_dir.exists():
                raise FileNotFoundError(f"Target directory not found: {target_dir}")
            fid_target_tensors = load_target_images(target_dir, image_size)
            if not fid_target_tensors:
                logging.error("No images found in target directory %s", target_dir)
                sys.exit(1)
            logging.info("Loaded %d target-domain images for FID from %s", len(fid_target_tensors), target_dir)
        elif target_dir and not fid_requested:
            logging.info("Target directory specified but FID metric not requested; ignoring %s", target_dir)

        # Robust pairing of images
        pairs = load_and_pair_images_with_paths(
            gen_dir=args.generated,
            original_dir=args.original,
            strategy=args.pairs,
            manifest=args.manifest,
            image_size=image_size,
        )
        if not pairs:
            logging.error("No valid image pairs found; aborting.")
            sys.exit(1)
        logging.info("Found %d paired images", len(pairs))

        # Dynamically instantiate metrics
        fid_plugin_options = {"reference_stats": fid_reference_stats} if fid_reference_stats else None
        metrics = build_metric_plugins(args.metrics, device=device, fid_options=fid_plugin_options)
        if not metrics:
            logging.error("No valid metrics requested (%s).", args.metrics)
            sys.exit(1)

        fid_target_iterator: Optional[Iterator[torch.Tensor]] = None
        if fid_requested and fid_reference_stats is None:
            if fid_target_tensors is None:
                logging.error("FID metric requires target-domain images when stats are not provided")
                sys.exit(1)
            fid_target_iterator = cycle_batches(fid_target_tensors, args.batch_size)

        # Batch processing loop
        per_image_results: Dict[str, Dict[str, float]] = {}
        for batch in tqdm(list(range(0, len(pairs), args.batch_size)),
                          desc="Evaluating", unit_scale=True):
            batch_pairs = pairs[batch: batch + args.batch_size]
            gens = torch.stack([p.gen_tensor for p in batch_pairs]).to(device, non_blocking=True)
            originals = torch.stack([p.original_tensor for p in batch_pairs]).to(device, non_blocking=True)
            names = [p.name for p in batch_pairs]

            with torch.no_grad():
                for m in metrics:
                    if m.name == "fid":
                        target_batch = None
                        if fid_reference_stats is None:
                            if fid_target_iterator is None:
                                raise RuntimeError("FID target iterator not initialized")
                            target_batch = next(fid_target_iterator).to(device, non_blocking=True)
                        scores = m(gens, target_batch)
                    else:
                        scores = m(gens, originals)
                    for n, s in zip(names, scores):
                        per_image_results.setdefault(n, {})[m.name] = float(s)

        # Semantic consistency evaluation
        semantic_payload: Optional[Dict[str, Any]] = None
        if semantic_enabled:
            logging.info(
                "Running semantic consistency evaluation (model=%s, device=%s)",
                semantic_model,
                semantic_device,
            )
            try:
                semantic_payload = compute_semantic_consistency(
                    pairs=pairs,
                    model_variant=semantic_model,
                    device=semantic_device,
                    show_progress=args.verbose == 0,
                )
            except Exception as exc:  # noqa: BLE001
                logging.error("Semantic consistency evaluation failed: %s", exc)
            else:
                for name, scores in semantic_payload["scalars"].items():
                    per_image_results.setdefault(name, {}).update(scores)

        # Statistical summaries
        summary = summarise_metrics(per_image_results, alpha=0.95)

        # Serialize results
        out = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "generated": str(args.generated),
            "original": str(args.original),
            "metrics": summary,
            "per_image": per_image_results,
        }

        if semantic_payload and semantic_payload["metadata"].get("enabled"):
            out["semantic_consistency"] = semantic_payload["metadata"]

        args.output.write_text(json.dumps(out, indent=2))
        logging.info("Results written to %s", args.output)


if __name__ == "__main__":
    main()
