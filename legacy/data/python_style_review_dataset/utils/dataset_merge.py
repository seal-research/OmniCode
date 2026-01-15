#!/usr/bin/env python3
"""
merge_sweagent_instances.py

Merge all SWE-agent style review datasets (JSON arrays) under a folder into one file.

Usage:
  python merge_sweagent_instances.py --input-dir path/to/folder --output merged.json

What it does:
- Finds all .json files inside the input directory (non-recursive by default).
- Loads each JSON file (must be a top-level list of instances).
- Concatenates them into a single list.
- Deduplicates by instance_id unless --no-dedup is specified.
- Saves merged output.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict


def load_json_array(path: Path) -> List[Dict]:
    """Load a JSON array from a file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f"[!] {path} does not contain a JSON array, skipping.", file=sys.stderr)
            return []
        return data
    except Exception as e:
        print(f"[!] Failed to load {path}: {e}", file=sys.stderr)
        return []


def merge_datasets(paths: List[Path], deduplicate: bool = True) -> List[Dict]:
    """Merge multiple JSON array datasets into one."""
    merged = []
    seen_ids = set()

    for p in paths:
        dataset = load_json_array(p)
        count = 0
        for inst in dataset:
            if deduplicate:
                iid = inst.get("instance_id")
                if iid in seen_ids:
                    continue
                seen_ids.add(iid)
            merged.append(inst)
            count += 1
        print(f'Added {count} instances from {p}')
        

    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge SWE-agent datasets from a folder (all JSON files).")
    parser.add_argument("--input-dir", default="python_style_review_dataset", help="Folder containing JSON files to merge")
    parser.add_argument("--output", default="style_review_python_instances.json", help="Output JSON file")
    parser.add_argument("--recursive", action="store_true", help="Search JSON files recursively")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[!] Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    if args.recursive:
        json_files = list(input_dir.rglob("*.json"))
    else:
        json_files = list(input_dir.glob("*.json"))

    if not json_files:
        print(f"[!] No JSON files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[+] Found {len(json_files)} JSON files in {input_dir}")
    merged = merge_datasets(json_files)
    print(len(merged), "instances merged. ")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    print(f"[+] Merged {len(merged)} total instances into {args.output}")


if __name__ == "__main__":
    main()
