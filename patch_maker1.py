#!/usr/bin/env python3
"""
fill_patches.py

Reads a sweagent JSON file with items that include "instance_id".
For each item, tries to read DS_<instance_id>/<instance_id>/<instance_id>.patch
and fill the "patch" field with that file's contents (if present).

Writes an output JSON and prints repo-level patch-found statistics.

Usage examples:
  python fill_patches.py \
      --infile sweagent_pmd_results_deepseek.json \
      --out sweagent_pmd_results_deepseek_with_patches.json

  python fill_patches.py --infile in.json --out out.json --base-dir /path/to/ds_dirs --stats-out stats.csv
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict

def find_patch_path(base_dir: str, instance_id: str) -> str:
    """Return expected patch file path for an instance_id under base_dir."""
    # Pattern: DS_<instance_id>/<instance_id>/<instance_id>.patch
    return os.path.join(base_dir, f"{instance_id}","fix.patch")

def safe_read_text(path: str) -> str:
    """Read a text file robustly, returning a string. Returns '' on error."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except Exception as e:
        # return empty string to indicate failure but don't crash the run
        print(f"[WARN] Failed to read '{path}': {e}", file=sys.stderr)
        return ""

def main(argv=None):
    p = argparse.ArgumentParser(description="Fill patch fields in sweagent JSON from .patch files.")
    p.add_argument("--infile", "-i", required=True, help="Input JSON file (list of objects).")
    p.add_argument("--out", "-o", default=None, help="Output JSON file. If not provided, adds suffix '_with_patches.json'.")
    p.add_argument("--base-dir", "-b", default="/share/dutta/ays57/omnicode/baselines/aider_gemini_sr_java", help="Base directory containing DS_<instance_id> subdirs. Default: current dir.")
    p.add_argument("--stats-out", default=None, help="Optional path to write repo statistics as CSV.")
    p.add_argument("--backup", action="store_true", help="If set and --out equals --infile, create a backup of infile.")
    args = p.parse_args(argv)

    infile = args.infile
    base_dir = args.base_dir

    if not os.path.isfile(infile):
        print(f"[ERROR] infile '{infile}' does not exist or is not a file.", file=sys.stderr)
        sys.exit(2)

    # Read input JSON
    with open(infile, "r", encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except Exception as e:
            print(f"[ERROR] Failed to parse JSON from '{infile}': {e}", file=sys.stderr)
            sys.exit(3)

    if not isinstance(data, list):
        print(f"[ERROR] Expected top-level JSON to be a list of objects.", file=sys.stderr)
        sys.exit(4)

    # Stats per repo
    stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "patches_found": 0})
    total_entries = 0
    patched_entries = 0
    missing_instance_id = 0

    for idx, item in enumerate(data):
        total_entries += 1
        repo = item.get("repo", "<unknown>")
        instance_id = item.get("instance_id")

        stats[repo]["total"] += 1

        if not instance_id:
            missing_instance_id += 1
            # leave patch field as-is (or add empty string)
            if "patch" not in item:
                item["patch"] = ""
            continue

        patch_path = find_patch_path(base_dir, instance_id)
        if os.path.isfile(patch_path):
            patch_text = safe_read_text(patch_path)
            if patch_text is None:
                patch_text = ""
            # Place exact content into the patch field
            item["patch"] = patch_text
            stats[repo]["patches_found"] += 1
            patched_entries += 1
        else:
            # Ensure field exists and is empty
            item.setdefault("patch", "")
            # Optionally, we could also try alternative paths (not required by user)
            # but keep it simple and deterministic.

    # Determine output filename
    out = args.out
    if not out:
        base, ext = os.path.splitext(infile)
        out = f"{base}{ext or '.json'}"

    # Backup if requested and overwriting
    if args.backup and os.path.abspath(out) == os.path.abspath(infile):
        backup_path = infile + ".bak"
        print(f"[INFO] Backing up '{infile}' -> '{backup_path}'")
        with open(infile, "rb") as rf, open(backup_path, "wb") as wf:
            wf.write(rf.read())

    # Write updated JSON (pretty-printed)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)

    # Print statistics
    print("\n=== Patch extraction summary ===")
    print(f"Input file: {infile}")
    print(f"Output file: {out}")
    print(f"Base dir for DS_ folders: {os.path.abspath(base_dir)}")
    print(f"Total entries processed : {total_entries}")
    print(f"Entries patched         : {patched_entries}")
    print(f"Entries missing inst_id : {missing_instance_id}")
    print()

    # Per-repo lines
    print("Per-repo patch stats:")
    # sort repos by total desc
    for repo, rstat in sorted(stats.items(), key=lambda x: (-x[1]["total"], x[0])):
        total = rstat["total"]
        found = rstat["patches_found"]
        pct = (found / total * 100) if total else 0.0
        print(f"  {repo:30}  total: {total:5d}  patches_found: {found:5d}  ({pct:5.1f}%)")

    # Optional: write stats CSV
    if args.stats_out:
        try:
            import csv
            with open(args.stats_out, "w", newline="", encoding="utf-8") as csvf:
                writer = csv.writer(csvf)
                writer.writerow(["repo", "total_instances", "patches_found", "percent_found"])
                for repo, rstat in sorted(stats.items(), key=lambda x: (-x[1]["total"], x[0])):
                    total = rstat["total"]
                    found = rstat["patches_found"]
                    pct = (found / total * 100) if total else 0.0
                    writer.writerow([repo, total, found, f"{pct:.2f}"])
            print(f"\n[INFO] Wrote stats to '{args.stats_out}'")
        except Exception as e:
            print(f"[WARN] Failed to write stats CSV to '{args.stats_out}': {e}", file=sys.stderr)

    print("\nDone.")

if __name__ == "__main__":
    main()
