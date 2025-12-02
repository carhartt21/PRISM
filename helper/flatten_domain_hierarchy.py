#!/usr/bin/env python3
"""
Flatten directory hierarchy by moving all images for each domain into the domain folder,
and then delete empty subdirectories.

Usage:
    python flatten_domain_hierarchy.py --root /path/to/root [--dry-run] [--add-prefix] [--extensions jpg png]

Examples:
    python flatten_domain_hierarchy.py --root /scratch/aaa_exchange/AWARE/GENERATED_IMAGES/cycleGAN --dry-run
    python flatten_domain_hierarchy.py --root /scratch/aaa_exchange/AWARE/FINAL_SPLITS/train/images

Defaults:
- The script processes every immediate subfolder of root (each domain)
- It moves only image files (.jpg, .jpeg, .png, .tif, .bmp, .tiff)
- When filename collisions happen, the script appends "-1", "-2", etc., unless --add-prefix is used

"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import shutil
from typing import List, Tuple
from tqdm import tqdm


SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}


def is_image(p: Path, extensions: List[str]) -> bool:
    return p.suffix.lower() in extensions


def move_file(src: Path, dst: Path, dry_run: bool = False) -> bool:
    """Move file from src to dst. If dst exists, append a numeric suffix to avoid collision.
    Returns True if moved (or would be moved in dry-run), False otherwise.
    """
    if dst.exists():
        base = dst.stem
        ext = dst.suffix
        dirpath = dst.parent
        # find next available name
        n = 1
        candidate = dirpath / f"{base}-{n}{ext}"
        while candidate.exists():
            n += 1
            candidate = dirpath / f"{base}-{n}{ext}"
        new_dst = candidate
        if dry_run:
            logging.debug("Would move %s -> %s", src, new_dst)
            return True
        else:
            shutil.move(str(src), str(new_dst))
            logging.debug("Moved %s -> %s (collision resolved)", src, new_dst)
            return True
    else:
        if dry_run:
            logging.debug("Would move %s -> %s", src, dst)
            return True
        else:
            # ensure parent exists
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            logging.debug("Moved %s -> %s", src, dst)
            return True


def flatten_domain(root_domain_dir: Path, extensions: List[str], add_prefix: bool = False, dry_run: bool = False) -> Tuple[int, int, List[Path]]:
    """
    Move images from nested subfolders into root_domain_dir. Returns tuple: (moved, skipped, files_that_failed)
    - moved: count of moved files
    - skipped: count of non-image files skipped
    - failed: list of files couldn't be moved
    """
    moved = 0
    skipped = 0
    failed = []

    # Walk nested directories (excluding the domain root itself when listing files)
    # We want to include files under the domain root if they are further nested; files directly in domain root are left in place
    for sub in root_domain_dir.rglob('*'):
        if sub.is_file() and sub.parent != root_domain_dir:
            if is_image(sub, extensions):
                # target path: root_domain_dir / (optional prefix) / filename
                filename = sub.name
                if add_prefix:
                    prefix = sub.parent.name
                    # sanitize prefix (replace path separators) by nothing
                    filename = f"{prefix}_{filename}"
                target = root_domain_dir / filename
                try:
                    if move_file(sub, target, dry_run=dry_run):
                        moved += 1
                except Exception as e:
                    logging.error("Failed to move '%s' -> '%s': %s", sub, target, e)
                    failed.append(sub)
            else:
                skipped += 1

    # Remove empty directories under domain root
    # Walk bottom-up and remove directories that are empty
    for dirpath, dirnames, filenames in os.walk(root_domain_dir, topdown=False):
        if dirpath == str(root_domain_dir):
            continue
        if not os.listdir(dirpath):
            if dry_run:
                logging.debug("Would remove empty directory: %s", dirpath)
            else:
                try:
                    os.rmdir(dirpath)
                    logging.debug("Removed empty directory: %s", dirpath)
                except Exception as e:
                    logging.debug("Could not remove directory %s: %s", dirpath, e)

    return moved, skipped, failed


def flatten_root(root: Path, extensions: List[str], add_prefix: bool = False, dry_run: bool = False) -> dict:
    """Process all immediate subfolders of root (domains)."""
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(root)

    domains = [d for d in root.iterdir() if d.is_dir()]
    if not domains:
        logging.info("No subfolders found in %s; nothing to flatten.", root)
        return {}

    results = {}
    for domain_dir in domains:
        logging.info("Processing domain: %s", domain_dir)
        moved, skipped, failed = flatten_domain(domain_dir, extensions, add_prefix, dry_run)
        results[domain_dir.name] = {
            'moved': moved,
            'skipped_non_images': skipped,
            'failed': [str(p) for p in failed]
        }
        logging.info("Domain %s: moved=%d, skipped=%d, failed=%d", domain_dir.name, moved, skipped, len(failed))
    return results


def parse_args():
    parser = argparse.ArgumentParser(description='Flatten image hierarchy per domain.')
    parser.add_argument('--root', required=True, help='Root directory containing domain subfolders')
    parser.add_argument('--extensions', nargs='+', default=['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'], help='File extensions to consider images')
    parser.add_argument('--add-prefix', action='store_true', help='Add source subdirectory name as prefix to filename to avoid collisions')
    parser.add_argument('--dry-run', action='store_true', help='Do not move files; only show changes')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='[%(levelname)s] %(message)s')

    root = Path(args.root)
    extensions = [e if e.startswith('.') else f'.{e}' for e in args.extensions]

    print(f"Flattening domains in: {root}")
    print(f"Extensions: {extensions}")
    print(f"Add prefix: {args.add_prefix}")
    print(f"Dry run: {args.dry_run}")

    results = flatten_root(root, extensions, add_prefix=args.add_prefix, dry_run=args.dry_run)

    print('\nSummary:')
    for domain, stats in results.items():
        print(f"{domain}: moved={stats['moved']}, skipped_non_images={stats['skipped_non_images']}, failed={len(stats['failed'])}")

    if any(stats['failed'] for stats in results.values()):
        print('\nFailed files:')
        for domain, stats in results.items():
            if stats['failed']:
                print(f"\nDomain: {domain}")
                for p in stats['failed']:
                    print(f"  {p}")
