#!/usr/bin/env python3
"""
Count image files per subfolder in a directory.

Usage:
    python count_images_per_folder.py --root /path/to/directory [--recursive] [--extensions jpg png] [--format csv|json|plain]

Example:
    python count_images_per_folder.py --root /scratch/aaa_exchange/AWARE/GENERATED_IMAGES/cycleGAN --recursive --format csv
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List
import csv

DEFAULT_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']


def count_images_in_dir(root: Path, recursive: bool = False, extensions: List[str] = None) -> Dict[str, int]:
    if extensions is None:
        extensions = DEFAULT_EXTENSIONS
    extensions = {e if e.startswith('.') else '.' + e for e in extensions}

    counts = {}

    if recursive:
        # Walk all subdirectories recursively and count per directory
        try:
            for dirpath in root.rglob('*'):
                try:
                    if dirpath.is_dir():
                        try:
                            files = [f for f in dirpath.iterdir() if f.is_file() and f.suffix.lower() in extensions]
                            if files:
                                counts[str(dirpath)] = len(files)
                        except PermissionError as e:
                            logging.debug(f"Permission denied listing {dirpath}: {e}")
                except OSError as e:
                    logging.debug(f"OS error checking {dirpath}: {e}")
        except PermissionError as e:
            logging.error(f"Permission error accessing {root}: {e}")
            return {}
    else:
        # Only count in immediate subfolders
        for sub in root.iterdir():
            try:
                if sub.is_dir():
                    files = [f for f in sub.iterdir() if f.is_file() and f.suffix.lower() in extensions]
                    counts[str(sub)] = len(files)
            except PermissionError as e:
                logging.error(f"Permission error accessing {e.filename}: {e.strerror}")
    return counts


def output_counts(counts: Dict[str, int], fmt: str, output: Path = None) -> None:
    if fmt == 'json':
        s = json.dumps(counts, indent=2)
        if output:
            output.write_text(s)
            print(f"Wrote JSON to {output}")
        else:
            print(s)
    elif fmt == 'csv':
        if output:
            with open(output, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['folder', 'count'])
                for folder, cnt in sorted(counts.items(), key=lambda x: x[0]):
                    writer.writerow([folder, cnt])
            print(f"Wrote CSV to {output}")
        else:
            print('folder,count')
            for folder, cnt in sorted(counts.items(), key=lambda x: x[0]):
                print(f"{folder},{cnt}")
    else:
        for folder, cnt in sorted(counts.items(), key=lambda x: x[0]):
            print(f"{folder}: {cnt}")


def parse_args():
    parser = argparse.ArgumentParser(description="Count image files per subfolder")
    parser.add_argument('--root', required=True, help='Root directory to scan')
    parser.add_argument('--recursive', action='store_true', help='Scan subfolders recursively')
    parser.add_argument('--aggregate', action='store_true', help='Aggregate counts per immediate subfolder (sum across subtrees)')
    parser.add_argument('--extensions', nargs='+', default=DEFAULT_EXTENSIONS, help='Extensions to count')
    parser.add_argument('--format', choices=['plain', 'csv', 'json'], default='plain', help='Output format')
    parser.add_argument('--output', help='Write output to path (for csv/json)')
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format='[%(levelname)s] %(message)s')
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(root)
    counts = count_images_in_dir(root, recursive=args.recursive, extensions=args.extensions)
    # If aggregate option is set, sum counts for each immediate child under root
    if args.aggregate:
        aggregated = {}
        for sub in root.iterdir():
            if sub.is_dir():
                total = 0
                # Sum all counts under this subfolder
                for dirpath, cnt in counts.items():
                    try:
                        if Path(dirpath).is_relative_to(sub):
                            total += cnt
                    except Exception:
                        # is_relative_to may raise on some paths or if sub is not a prefix
                        continue
                aggregated[str(sub)] = total
        counts = aggregated
    output_counts(counts, args.format, Path(args.output) if args.output else None)
