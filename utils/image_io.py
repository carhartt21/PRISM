"""
utils.image_io: Robust image loading and pairing utilities
"""
import csv
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union

import torch
import torchvision.transforms as T
from PIL import Image, ImageFile

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm.auto import tqdm as _tqdm


# Supported image extensions
SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}


def load_image(path: Union[str, Path], size: Tuple[int, int] = (299, 299)) -> torch.Tensor:
    """
    Load and preprocess an image to tensor format.

    Args:
        path: Path to image file
        size: Target size (height, width) for resizing

    Returns:
        Tensor of shape (C, H, W) with values in [0, 1]
    """
    try:
        with Image.open(path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Resize and convert to tensor
            transform = T.Compose([
                T.Resize(size),
                T.ToTensor(),  # Converts to [0, 1] range
            ])

            return transform(img)
    except Exception as e:
        logging.error(f"Failed to load image {path}: {e}")
        raise


def find_image_files(directory: Path) -> List[Path]:
    """Find all supported image files in a directory."""
    image_files = []
    for ext in SUPPORTED_EXTENSIONS:
        image_files.extend(directory.rglob(f"*{ext}"))
        image_files.extend(directory.rglob(f"*{ext.upper()}"))
    return sorted(image_files)


def match_by_filename(gen_files: List[Path], original_files: List[Path]) -> List[Tuple[Path, Path, str]]:
    """
    Match generated and original images by filename (ignoring extension).

    Returns:
        List of (gen_path, original_path, name) tuples
    """
    original_dict = {f.stem: f for f in original_files}
    pairs = []
    unmatched_gen = []

    for gen_file in gen_files:
        stem = gen_file.stem
        if stem in original_dict:
            pairs.append((gen_file, original_dict[stem], stem))
        else:
            unmatched_gen.append(gen_file)

    if unmatched_gen:
        logging.warning(f"Found {len(unmatched_gen)} unmatched generated images")
        for f in unmatched_gen[:5]:  # Show first 5
            logging.warning(f"  Unmatched: {f.name}")
        if len(unmatched_gen) > 5:
            logging.warning(f"  ... and {len(unmatched_gen) - 5} more")

    unmatched_original = set(original_dict.keys()) - {stem for _, _, stem in pairs}
    if unmatched_original:
        logging.warning(f"Found {len(unmatched_original)} unmatched original images")

    return pairs


def load_pairs_from_csv(manifest_path: Path) -> List[Tuple[Path, Path, str]]:
    """
    Load image pairs from CSV manifest.

    CSV format: gen_path,original_path
    """
    pairs = []
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            gen_path = Path(row['gen_path'])
            original_path = Path(row['original_path'])
            name = f"pair_{i:04d}"

            if not gen_path.exists():
                logging.warning(f"Generated image not found: {gen_path}")
                continue
            if not original_path.exists():
                logging.warning(f"Original image not found: {original_path}")
                continue

            pairs.append((gen_path, original_path, name))

    return pairs


@dataclass
class LoadedImagePair:
    """Container for paired tensors plus their source paths."""

    gen_tensor: torch.Tensor
    original_tensor: torch.Tensor
    name: str
    gen_path: Path
    original_path: Path


def pair_image_paths(
    gen_dir: Path,
    original_dir: Path,
    strategy: str = "auto",
    manifest: Optional[Path] = None
) -> List[Tuple[Path, Path, str]]:
    """Return matched generated/original image paths without loading pixels."""
    if strategy == "csv":
        if not manifest:
            raise ValueError("Manifest file required for CSV pairing strategy")
        return load_pairs_from_csv(manifest)

    gen_files = find_image_files(gen_dir)
    original_files = find_image_files(original_dir)

    if not gen_files:
        raise ValueError(f"No images found in generated directory: {gen_dir}")
    if not original_files:
        raise ValueError(f"No images found in original directory: {original_dir}")

    return match_by_filename(gen_files, original_files)


def _load_single_pair(
    args: Tuple[Path, Path, str, Tuple[int, int]]
) -> Optional[LoadedImagePair]:
    """
    Load a single image pair. Used for parallel loading.
    
    Args:
        args: Tuple of (gen_path, original_path, name, image_size)
    
    Returns:
        LoadedImagePair or None if loading failed
    """
    gen_path, original_path, name, image_size = args
    try:
        gen_tensor = load_image(gen_path, image_size)
        original_tensor = load_image(original_path, image_size)
        return LoadedImagePair(
            gen_tensor=gen_tensor,
            original_tensor=original_tensor,
            name=name,
            gen_path=gen_path,
            original_path=original_path,
        )
    except Exception as e:
        logging.error(f"Failed to load pair {name}: {e}")
        return None


def load_and_pair_images_with_paths(
    gen_dir: Path,
    original_dir: Path,
    strategy: str = "auto",
    manifest: Optional[Path] = None,
    image_size: Tuple[int, int] = (299, 299),
    num_workers: int = 0,
) -> List[LoadedImagePair]:
    """
    Load paired images and keep track of their originating file paths.
    
    Args:
        gen_dir: Directory containing generated images
        original_dir: Directory containing original images
        strategy: Pairing strategy ("auto", "csv")
        manifest: Path to CSV manifest (required if strategy="csv")
        image_size: Target size (height, width) for loaded images
        num_workers: Number of parallel workers for loading.
            0 = sequential (default), >0 = use ThreadPoolExecutor
    
    Returns:
        List of LoadedImagePair objects
    """
    path_pairs = pair_image_paths(gen_dir, original_dir, strategy=strategy, manifest=manifest)

    if not path_pairs:
        return []

    tensor_pairs: List[LoadedImagePair] = []
    failed_loads = 0
    total = len(path_pairs)

    if num_workers > 0:
        # Parallel loading using ThreadPoolExecutor
        # Image I/O is I/O-bound, so threading works well
        load_args = [(gen_path, orig_path, name, image_size) 
                     for gen_path, orig_path, name in path_pairs]
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(_load_single_pair, args): args[2] 
                       for args in load_args}
            
            # Collect results with progress bar
            for future in _tqdm(
                as_completed(futures), 
                total=total, 
                unit="pair", 
                desc=f"Loading pairs ({num_workers} workers)"
            ):
                result = future.result()
                if result is not None:
                    tensor_pairs.append(result)
                else:
                    failed_loads += 1
    else:
        # Sequential loading (original behavior)
        iterator = enumerate(
            _tqdm(path_pairs, total=total, unit="pair", desc="Loading pairs"),
            start=1,
        )

        for i, (gen_path, original_path, name) in iterator:
            try:
                gen_tensor = load_image(gen_path, image_size)
                original_tensor = load_image(original_path, image_size)
                tensor_pairs.append(
                    LoadedImagePair(
                        gen_tensor=gen_tensor,
                        original_tensor=original_tensor,
                        name=name,
                        gen_path=gen_path,
                        original_path=original_path,
                    )
                )
            except Exception as e:
                logging.error(f"Failed to load pair {name}: {e}")
                failed_loads += 1

    if failed_loads > 0:
        logging.warning(f"Failed to load {failed_loads} image pairs")

    logging.info(f"Successfully loaded {len(tensor_pairs)} image pairs")
    return tensor_pairs


def load_and_pair_images(
    gen_dir: Path,
    original_dir: Path,
    strategy: str = "auto",
    manifest: Optional[Path] = None,
    warn_unpaired: bool = True,
    image_size: Tuple[int, int] = (299, 299),
    num_workers: int = 0,
) -> List[Tuple[torch.Tensor, torch.Tensor, str]]:
    """
    Backwards-compatible wrapper that returns tensors without path metadata.
    
    Args:
        gen_dir: Directory containing generated images
        original_dir: Directory containing original images
        strategy: Pairing strategy ("auto", "csv")
        manifest: Path to CSV manifest (required if strategy="csv")
        warn_unpaired: Deprecated, kept for API compatibility
        image_size: Target size (height, width) for loaded images
        num_workers: Number of parallel workers for loading.
            0 = sequential (default), >0 = use ThreadPoolExecutor
    
    Returns:
        List of (gen_tensor, original_tensor, name) tuples
    """
    _ = warn_unpaired  # retained for API compatibility
    pairs_with_paths = load_and_pair_images_with_paths(
        gen_dir,
        original_dir,
        strategy=strategy,
        manifest=manifest,
        image_size=image_size,
        num_workers=num_workers,
    )
    return [(p.gen_tensor, p.original_tensor, p.name) for p in pairs_with_paths]
