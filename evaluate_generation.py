#!/usr/bin/env python3
"""
evaluate_generation.py

End-to-end image-quality evaluation for image-to-image translation and
weather-synthesis models.

Supports: FID, IS, SSIM, LPIPS, PSNR  →  easily add more via plugins.
"""

from __future__ import annotations
import argparse, json, logging, sys, time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

import yaml  # ← safe_load used; see README for security notes
try:  # pragma: no cover - supports both package and script execution
    from .utils.image_io import (
        LoadedImagePair,
        load_and_pair_images_with_paths,
    )
    from .utils.stats import summarise_metrics         # mean/std/CI
    from .utils.logging_setup import configure_logger  # coloured logging
except ImportError:  # pragma: no cover
    from utils.image_io import (
        LoadedImagePair,
        load_and_pair_images_with_paths,
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


def build_metric_plugins(names: Sequence[str], device: str):
    """Lazily import the metric registry to avoid heavy deps during import."""
    from metrics import registry  # imported here to dodge costly deps during tests

    return registry.build(list(names), device=device)


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

        evaluator = SegFormerEvaluator(model_name=model_name, device=device)
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
        result = evaluator.evaluate_pair(pair.real_path, pair.gen_path)
        raw_results.append(result)
        per_image_scalars[pair.name] = {
            "semantic_pixel_accuracy": float(result.get("pixel_accuracy", 0.0)),
            "semantic_mIoU": float(result.get("mIoU", 0.0)),
            "semantic_fw_IoU": float(result.get("fw_IoU", 0.0)),
        }
        detailed_results[pair.name] = {
            **result,
            "generated_image": str(pair.gen_path),
            "real_image": str(pair.real_path),
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

# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate generated images against real images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--generated", required=True, type=Path,
                   help="Folder with generated images.")
    p.add_argument("--real",       required=True, type=Path,
                   help="Folder with reference images.")
    p.add_argument("-m", "--metrics", nargs="+",
                   default=["fid", "ssim", "lpips", "psnr", "is"],
                   help="Metrics to compute (plugins).")
    p.add_argument("-b", "--batch-size", type=int, default=32)
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto",
                   help="Computation device.")
    p.add_argument("--pairs", choices=["auto", "csv"], default="auto",
                   help="Pairing strategy: filename matching or CSV manifest.")
    p.add_argument("--manifest", type=Path,
                   help="CSV with gen_path,real_path if --pairs csv.")
    p.add_argument("--output", type=Path, default=Path("results.json"))
    p.add_argument("--config", type=Path,
                   help="YAML/JSON file with additional options.")
    p.add_argument("-v", "--verbose", action="count", default=0,
                   help="Increase logging verbosity (-v, -vv).")
    p.add_argument("--semantic-consistency", action="store_true",
                   help="Compute semantic consistency via SegFormer.")
    p.add_argument("--semantic-model",
                   choices=sorted(SEGFORMER_MODEL_MAP.keys()),
                   help="SegFormer backbone variant (default segformer-b5).")
    p.add_argument("--semantic-device", choices=["cpu", "cuda", "auto"],
                   help="Device for semantic consistency evaluator (defaults to --device or auto).")
    return p.parse_args()


# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    configure_logger(args.verbose)

    # Load external configuration -------------------------------------------------
    cfg: Dict = {}
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f) if args.config.suffix in {".yml", ".yaml"} \
                 else json.load(f)
        logging.info("Merged configuration from %s", args.config)

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else args.device
    logging.info("Using device=%s", device)

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

    # Robust pairing of images ----------------------------------------------------
    pairs = load_and_pair_images_with_paths(
        gen_dir=args.generated,
        real_dir=args.real,
        strategy=args.pairs,
        manifest=args.manifest,
    )
    if not pairs:
        logging.error("No valid image pairs found; aborting.")
        sys.exit(1)
    logging.info("Found %d paired images", len(pairs))

    # Dynamically instantiate metrics --------------------------------------------
    metrics = build_metric_plugins(args.metrics, device=device)
    if not metrics:
        logging.error("No valid metrics requested (%s).", args.metrics)
        sys.exit(1)

    # Batch processing loop -------------------------------------------------------
    per_image_results: Dict[str, Dict[str, float]] = {}
    for batch in tqdm(list(range(0, len(pairs), args.batch_size)),
                      desc="Evaluating", unit_scale=True):
        batch_pairs = pairs[batch: batch + args.batch_size]
        gens = torch.stack([p.gen_tensor for p in batch_pairs]).to(device, non_blocking=True)
        reals = torch.stack([p.real_tensor for p in batch_pairs]).to(device, non_blocking=True)
        names = [p.name for p in batch_pairs]

        with torch.no_grad():
            for m in metrics:
                scores = m(gens, reals)                          # size=B
                for n, s in zip(names, scores):
                    per_image_results.setdefault(n, {})[m.name] = float(s)

    # Semantic consistency evaluation --------------------------------------------
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

    # Statistical summaries -------------------------------------------------------
    summary = summarise_metrics(per_image_results, alpha=0.95)

    # Serialize results -----------------------------------------------------------
    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "generated": str(args.generated),
        "real": str(args.real),
        "metrics": summary,
        "per_image": per_image_results,
    }

    if semantic_payload and semantic_payload["metadata"].get("enabled"):
        out["semantic_consistency"] = semantic_payload["metadata"]

    args.output.write_text(json.dumps(out, indent=2))
    logging.info("Results written to %s", args.output)


if __name__ == "__main__":
    main()
