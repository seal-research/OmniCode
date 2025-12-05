#!/usr/bin/env python3
"""
Extract `instance_id` and `patch` from one or more CodeArena/MSWE dataset files.

Usage examples:
  python scripts/extract_instance_patches.py \
    data/codearena_instances_python.json \
    data/codearena_instances_java.json \
    data/codearena_instances_cpp.json \
    -o data/instances_patches.jsonl

The script supports `.json` arrays and `.jsonl` files. Output is a JSON Lines
file with one object per line: {"instance_id": ..., "patch": ...}
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Iterable, List


def read_json_or_jsonl(path: str) -> List[dict]:
    """Return list of items from a JSON array, single JSON object, or JSONL file."""
    results = []
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Try JSON (array or object)
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # if top-level dict contains a list of items under a known key
            # try common keys, else wrap single object
            for k in ('instances', 'data', 'items'):
                if k in data and isinstance(data[k], list):
                    return data[k]
            return [data]
    except json.JSONDecodeError:
        # fall through to jsonl parsing
        pass

    # Try JSONL: iterate lines and parse each JSON object
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                results.append(obj)
            except json.JSONDecodeError:
                # skip malformed lines
                continue
    return results


def extract_from_files(paths: Iterable[str], out_path: str) -> int:
    seen = set()
    written = 0

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as out_f:
        final = {}
        for p in paths:
            if not os.path.exists(p):
                print(f"Warning: input file not found: {p}", file=sys.stderr)
                continue

            items = read_json_or_jsonl(p)
            for item in items:
                if not isinstance(item, dict):
                    continue

                # common field names
                instance_id = (
                    item.get('instance_id')
                    or item.get('id')
                    or item.get('instanceId')
                    or item.get('instance')
                )

                patch = (
                    item.get('patch')
                    or item.get('fix_patch')
                    or item.get('model_patch')
                    or ''
                )

                # If instance_id missing, try to derive from repo + number
                if not instance_id:
                    repo = item.get('repo') or item.get('repository')
                    number = item.get('pull_number') or item.get('number')
                    if isinstance(repo, str) and number is not None:
                        instance_id = f"{repo.replace('/', '__')}_{number}"

                if not instance_id:
                    # Skip entries without any reasonable instance id
                    continue

                if instance_id in seen:
                    continue

                seen.add(instance_id)
                final[instance_id] = patch
                written += 1
        json.dump(final, out_f, indent=2)

    return written


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Extract instance_id and patch from dataset files')
    parser.add_argument('input_files', nargs='+', help='Input paths (JSON array or JSONL)')
    parser.add_argument('-o', '--output', default='data/instances_patches.json', help='Output JSONL path')
    args = parser.parse_args(argv)

    written = extract_from_files(args.input_files, args.output)
    print(f'Wrote {written} entries to {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
