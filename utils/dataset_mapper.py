#!/usr/bin/env python3
"""
Dataset Mapper - Efficient filename-to-dataset mapping.

Maps image filenames to their source datasets based on naming patterns.
This is useful when generated images are in a flat structure without
dataset subdirectories.

Dataset patterns:
- ACDC: GOPR*_frame_*_rgb_ref_anon.png  (e.g., GOPR0122_frame_000161_rgb_ref_anon.png)
- BDD100k/BDD10k: [8hex]-[8hex].jpg  (need lookup to distinguish)
- IDD-AW: [digits]_leftImg8bit.png  (e.g., 000128_leftImg8bit.png)
- MapillaryVistas: ~22 char alphanumeric with dashes/underscores .jpg
- OUTSIDE15k: ADE_train_*.jpg  (e.g., ADE_train_00000134.jpg)
"""

import re
from pathlib import Path
from typing import Dict, Optional, Set, Tuple
import logging


# Regex patterns for dataset identification
DATASET_PATTERNS = {
    'ACDC': re.compile(r'^GOPR\d+_frame_\d+.*\.png$', re.IGNORECASE),
    'IDD-AW': re.compile(r'^\d+_leftImg8bit\.png$', re.IGNORECASE),
    'OUTSIDE15k': re.compile(r'^ADE_train_\d+\.jpg$', re.IGNORECASE),
    # BDD pattern: 8 hex chars, dash, 8 hex chars
    'BDD': re.compile(r'^[0-9a-f]{8}-[0-9a-f]{8}\.(jpg|png)$', re.IGNORECASE),
    # MapillaryVistas: ~22 char with special chars, but NOT matching other patterns
    'MapillaryVistas': re.compile(r'^[A-Za-z0-9_-]{15,30}\.(jpg|png)$', re.IGNORECASE),
}


class DatasetMapper:
    """
    Efficiently maps image filenames to their source datasets.
    
    For datasets that can't be distinguished by pattern alone (BDD100k vs BDD10k),
    builds a lookup table from the original image directory.
    """
    
    def __init__(
        self,
        original_dir: Optional[Path] = None,
        known_datasets: Optional[Set[str]] = None,
    ):
        """
        Initialize the dataset mapper.
        
        Args:
            original_dir: Path to original images directory with dataset subfolders.
                         Used to build lookup for ambiguous patterns (BDD100k/BDD10k).
            known_datasets: Set of dataset names to include. If None, uses all known.
        """
        self.known_datasets = known_datasets or {
            'ACDC', 'BDD100k', 'BDD10k', 'IDD-AW', 'MapillaryVistas', 'OUTSIDE15k'
        }
        
        # Lookup tables for ambiguous datasets
        self._bdd100k_stems: Set[str] = set()
        self._bdd10k_stems: Set[str] = set()
        self._stem_to_dataset: Dict[str, str] = {}
        
        # Build lookup if original_dir provided
        if original_dir:
            self._build_lookup(original_dir)
    
    def _build_lookup(self, original_dir: Path) -> None:
        """Build filename -> dataset lookup from original images directory."""
        logging.info("Building dataset lookup from %s...", original_dir)
        
        import os
        
        for dataset in self.known_datasets:
            dataset_path = original_dir / dataset
            if not dataset_path.exists():
                continue
            
            # Walk through dataset directory
            for root, _, files in os.walk(dataset_path):
                for filename in files:
                    stem = Path(filename).stem
                    self._stem_to_dataset[stem] = dataset
                    
                    # Track BDD stems separately for pattern-based fallback
                    if dataset == 'BDD100k':
                        self._bdd100k_stems.add(stem)
                    elif dataset == 'BDD10k':
                        self._bdd10k_stems.add(stem)
        
        logging.info("  Indexed %d filenames across datasets", len(self._stem_to_dataset))
        if self._bdd100k_stems:
            logging.info("  BDD100k: %d files, BDD10k: %d files",
                        len(self._bdd100k_stems), len(self._bdd10k_stems))
    
    def get_dataset(self, filename: str) -> Optional[str]:
        """
        Determine the dataset for a given filename.
        
        Args:
            filename: Image filename (with or without extension)
            
        Returns:
            Dataset name or None if unknown
        """
        # Normalize: get stem without extension
        stem = Path(filename).stem
        name = Path(filename).name if '.' in filename else filename
        
        # First, try exact lookup
        if stem in self._stem_to_dataset:
            return self._stem_to_dataset[stem]
        
        # Fall back to pattern matching
        return self._get_dataset_by_pattern(name, stem)
    
    def _get_dataset_by_pattern(self, filename: str, stem: str) -> Optional[str]:
        """Identify dataset using regex patterns."""
        
        # Check definitive patterns first (most specific)
        if DATASET_PATTERNS['ACDC'].match(filename):
            return 'ACDC'
        
        if DATASET_PATTERNS['IDD-AW'].match(filename):
            return 'IDD-AW'
        
        if DATASET_PATTERNS['OUTSIDE15k'].match(filename):
            return 'OUTSIDE15k'
        
        # Check BDD pattern
        if DATASET_PATTERNS['BDD'].match(filename):
            # Try to distinguish BDD100k from BDD10k
            if stem in self._bdd10k_stems:
                return 'BDD10k'
            elif stem in self._bdd100k_stems:
                return 'BDD100k'
            else:
                # Default to BDD100k if can't distinguish (larger dataset)
                return 'BDD100k'
        
        # MapillaryVistas has varied naming, check last after excluding others
        if DATASET_PATTERNS['MapillaryVistas'].match(filename):
            # Make sure it's not matching other patterns
            if not any([
                DATASET_PATTERNS['ACDC'].match(filename),
                DATASET_PATTERNS['IDD-AW'].match(filename),
                DATASET_PATTERNS['OUTSIDE15k'].match(filename),
                DATASET_PATTERNS['BDD'].match(filename),
            ]):
                return 'MapillaryVistas'
        
        return None
    
    def get_dataset_from_path(self, path: Path) -> Optional[str]:
        """
        Get dataset from a full path.
        
        First checks if path contains a known dataset folder name,
        then falls back to filename pattern matching.
        """
        # Check path components for dataset name
        parts = path.parts
        for part in parts:
            if part in self.known_datasets:
                return part
        
        # Fall back to filename matching
        return self.get_dataset(path.name)
    
    def batch_get_datasets(self, filenames: list) -> Dict[str, str]:
        """
        Get datasets for multiple filenames efficiently.
        
        Returns:
            Dict mapping filename to dataset name
        """
        return {f: self.get_dataset(f) for f in filenames}


def build_dataset_index(original_dir: Path) -> Dict[str, Tuple[str, Path]]:
    """
    Build a comprehensive index mapping filename stems to (dataset, full_path).
    
    This is more detailed than DatasetMapper and includes full paths.
    
    Args:
        original_dir: Root directory containing dataset subfolders
        
    Returns:
        Dict mapping filename stem to (dataset_name, full_path)
    """
    import os
    
    index: Dict[str, Tuple[str, Path]] = {}
    known_datasets = ['ACDC', 'BDD100k', 'BDD10k', 'IDD-AW', 'MapillaryVistas', 'OUTSIDE15k']
    
    for dataset in known_datasets:
        dataset_path = original_dir / dataset
        if not dataset_path.exists():
            continue
        
        for root, _, files in os.walk(dataset_path):
            for filename in files:
                # Check if it's an image file
                ext = Path(filename).suffix.lower()
                if ext not in {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}:
                    continue
                
                stem = Path(filename).stem
                full_path = Path(root) / filename
                
                # If stem already exists, prefer the one from larger dataset
                # or keep first occurrence
                if stem not in index:
                    index[stem] = (dataset, full_path)
    
    return index


# Convenience function
def identify_dataset(filename: str, mapper: Optional[DatasetMapper] = None) -> Optional[str]:
    """
    Quick function to identify dataset from filename.
    
    For repeated calls, use DatasetMapper directly for better performance.
    """
    if mapper:
        return mapper.get_dataset(filename)
    
    # Pattern-only matching (no lookup)
    name = Path(filename).name if '/' in filename or '\\' in filename else filename
    
    if DATASET_PATTERNS['ACDC'].match(name):
        return 'ACDC'
    if DATASET_PATTERNS['IDD-AW'].match(name):
        return 'IDD-AW'
    if DATASET_PATTERNS['OUTSIDE15k'].match(name):
        return 'OUTSIDE15k'
    if DATASET_PATTERNS['BDD'].match(name):
        return 'BDD100k'  # Default, can't distinguish without lookup
    if DATASET_PATTERNS['MapillaryVistas'].match(name):
        return 'MapillaryVistas'
    
    return None


if __name__ == "__main__":
    # Test the mapper
    import sys
    
    test_files = [
        "GOPR0122_frame_000161_rgb_ref_anon.png",  # ACDC
        "0000f77c-6257be58.jpg",  # BDD
        "000128_leftImg8bit.png",  # IDD-AW
        "00qclUcInksIYnm19b1Xfw.jpg",  # MapillaryVistas
        "ADE_train_00000134.jpg",  # OUTSIDE15k
    ]
    
    print("Testing pattern-based identification:")
    for f in test_files:
        dataset = identify_dataset(f)
        print(f"  {f} -> {dataset}")
    
    if len(sys.argv) > 1:
        original_dir = Path(sys.argv[1])
        print(f"\nTesting with lookup from {original_dir}:")
        mapper = DatasetMapper(original_dir)
        for f in test_files:
            dataset = mapper.get_dataset(f)
            print(f"  {f} -> {dataset}")
