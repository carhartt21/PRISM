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
from typing import Callable, Dict, List, Sequence, Tuple

import torch
from tqdm.auto import tqdm

import yaml  # ← safe_load used; see README for security notes
from metrics import registry                      # plugin registry
from utils.image_io import load_and_pair_images   # robust pairing
from utils.stats import summarise_metrics         # mean/std/CI
from utils.logging_setup import configure_logger  # coloured logging

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

    # Robust pairing of images ----------------------------------------------------
    pairs = load_and_pair_images(
        gen_dir=args.generated,
        real_dir=args.real,
        strategy=args.pairs,
        manifest=args.manifest,
        warn_unpaired=True,
    )
    if not pairs:
        logging.error("No valid image pairs found; aborting.")
        sys.exit(1)
    logging.info("Found %d paired images", len(pairs))

    # Dynamically instantiate metrics --------------------------------------------
    metrics = registry.build(args.metrics, device=device)
    if not metrics:
        logging.error("No valid metrics requested (%s).", args.metrics)
        sys.exit(1)

    # Batch processing loop -------------------------------------------------------
    per_image_results = {}
    for batch in tqdm(list(range(0, len(pairs), args.batch_size)),
                      desc="Evaluating", unit_scale=True):
        batch_pairs = pairs[batch: batch + args.batch_size]
        gen, real, names = zip(*batch_pairs)                     # tensors on CPU
        gen = torch.stack(gen).to(device, non_blocking=True)
        real = torch.stack(real).to(device, non_blocking=True)

        with torch.no_grad():
            for m in metrics:
                scores = m(gen, real)                            # size=B
                for n, s in zip(names, scores):
                    per_image_results.setdefault(n, {})[m.name] = float(s)

    # Statistical summaries -------------------------------------------------------
    summary = summarise_metrics(per_image_results, alpha=0.95)

    # Serialize results -----------------------------------------------------------
    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "generated": str(args.generated),
           "real": str(args.real),
           "metrics": summary,
           "per_image": per_image_results}

    args.output.write_text(json.dumps(out, indent=2))
    logging.info("Results written to %s", args.output)


if __name__ == "__main__":
    main()
