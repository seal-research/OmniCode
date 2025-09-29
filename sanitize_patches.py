#!/usr/bin/env python3
"""
normalize_patches_endings_real_newline.py

Ensure every item in a JSON list file (default: sweagent_pmd_results.json) has its
'patch' field end with a single real newline character '\n' (not the two-character sequence '\\n').

Usage:
  python normalize_patches_endings_real_newline.py \
      --infile sweagent_pmd_results.json \
      [--outfile out.json] \
      [--backup] \
      [--dry-run]
"""
import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from tempfile import NamedTemporaryFile

def normalize_patch_str(s: str) -> str:
    """
    Ensure s ends with exactly one real newline character '\n'.
    Steps:
      1. Remove any trailing literal backslash+'n' sequences (i.e., the two chars '\\' and 'n').
      2. Remove trailing real newlines/carriage returns.
      3. Append a single real newline '\n'.
    """
    if not isinstance(s, str):
        return s
    # Remove trailing literal backslash + 'n' pairs (only at the very end)
    while s.endswith("\\n"):
        s = s[:-2]
    # Remove trailing real newline/carriage-return characters
    s = s.rstrip("\r\n")
    # Append a single real newline
    return s + "\n"

def main():
    p = argparse.ArgumentParser(description="Normalize 'patch' fields to end with a real newline")
    p.add_argument("--infile", "-i", default="sweagent_pmd_results_gpt5.json", help="Input JSON file (list of objects).")
    p.add_argument("--outfile", "-o", default=None, help="Output JSON file. If omitted, infile is overwritten.")
    p.add_argument("--backup", "-b", action="store_true", help="Create a timestamped backup of the input file before overwriting.")
    p.add_argument("--dry-run", action="store_true", help="Show what would change but do not write any file.")
    args = p.parse_args()

    infile = args.infile
    outfile = args.outfile or infile

    if not os.path.isfile(infile):
        print(f"ERROR: infile '{infile}' does not exist.", file=sys.stderr)
        sys.exit(2)

    with open(infile, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("ERROR: expected top-level JSON array (list of objects).", file=sys.stderr)
        sys.exit(3)

    changed = []
    total = 0

    for idx, item in enumerate(data):
        total += 1
        if not isinstance(item, dict):
            continue
        patch = item.get("patch", None)
        if patch is None or not isinstance(patch, str):
            continue

        new_patch = normalize_patch_str(patch)
        if new_patch != patch:
            repo = item.get("repo")
            pr = item.get("pull_number")
            ident = f"{repo or 'unknown_repo'}:{pr if pr is not None else idx}"
            changed.append((idx, ident))
            item["patch"] = new_patch

    print(f"Scanned {total} items. Modified {len(changed)} patches.")
    if changed:
        print("Modified entries (index, repo:pull_number):")
        for idx, ident in changed:
            print(f"  - [{idx}] {ident}")

    if args.dry_run or len(changed) == 0:
        if args.dry_run:
            print("Dry-run enabled; no files written.")
        else:
            print("No changes required; file not modified.")
        return

    if args.backup and infile == outfile:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{infile}.bak.{ts}"
        shutil.copy2(infile, backup_name)
        print(f"Backup created: {backup_name}")

    dirout = os.path.dirname(os.path.abspath(outfile)) or "."
    with NamedTemporaryFile("w", delete=False, dir=dirout, encoding="utf-8") as tmp:
        json.dump(data, tmp, ensure_ascii=False, indent=2)
        tmp_name = tmp.name

    shutil.move(tmp_name, outfile)
    print(f"Wrote updated JSON to: {outfile}")

if __name__ == "__main__":
    main()
