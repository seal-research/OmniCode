#!/usr/bin/env python3
"""
filter_flat_pylint.py

Usage:
  python filter_flat_pylint.py --instance-id instance_id --input raw_pylint.json

What it does:
 - Reads flat pylint JSON (list of message dicts)
 - Keeps only certain types (error, warning, fatal by default)
 - Drops ignored message-ids (style noise, env/import errors)
 - Writes filtered list back to output
"""

import argparse
import json
import os
import sys
import re
from json import JSONDecodeError

# -------------------------------
# Customize filters here
# -------------------------------

# Style / refactor messages you don't care about
STYLE_IGNORE_IDS = {
    "R0917",  # too-many-positional-arguments
}

# Environment-related errors to ignore
ENV_IGNORE_IDS = {
    "E0401",  # import-error
    "E0611",  # no-name-in-module
    "E1101",  # no-member
}

# Combine them
IGNORE_IDS = STYLE_IGNORE_IDS | ENV_IGNORE_IDS

# Keep only these severities
KEEP_TYPES = {"error", "warning", "fatal"}


def filter_pylint_output(pylint_output, keep_types=KEEP_TYPES):
    filtered = []
    for msg in pylint_output:
        if msg.get("type") not in keep_types:
            continue
        if msg.get("message-id") in IGNORE_IDS:
            continue
        filtered.append(msg)
    return filtered


def parse_pylint_output(out_json_path, label):
    try:
        with open(out_json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[!] pylint did not produce valid JSON: {e}", file=sys.stderr)
        return [], {"global_score": 0.0, "total_errors": 0, "total_files": 0}

    # filter messages before parsing into per-file structure
    filtered_msgs = filter_pylint_output(raw)

    with open(f'pylint_results/filtered_pylint_data/filtered_{label}.json', "w", encoding="utf-8") as f:
        json.dump(filtered_msgs, f, indent=2)

    file_map = {}
    for msg in filtered_msgs:
        path = msg.get("path")
        entry = file_map.setdefault(path, {"file": path, "messages": []})
        entry["messages"].append({
            "line": msg.get("line"),
            "column": msg.get("column"),
            "symbol": msg.get("symbol"),
            "message": msg.get("message"),
            "type": msg.get("type")
        })

    files = []
    total_errors = 0
    for path, rec in file_map.items():
        errs = sum(1 for m in rec["messages"] if m["type"] in ("error", "fatal"))
        total_errors += errs
        score = max(0.0, 10.0 - errs / 5.0)
        rec["score"] = round(score, 2)
        rec["error_count"] = errs
        files.append(rec)

    global_score = round((sum(f["score"] for f in files) / len(files)) if files else 10.0, 2)
    overview = {"global_score": global_score, "total_errors": total_errors, "total_files": len(files)}
    obj = {"label": label, "files": files, "overview": overview}

    return obj


def main():
    ap = argparse.ArgumentParser(description="Filter flat pylint JSON output")
    ap.add_argument("--instance-id", required=True, help="Instance ID")
    ap.add_argument("--input", required=True, help="Raw pylint JSON file")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    filtered = parse_pylint_output(args.input, args.instance_id)

    with open(f'pylint_results/{args.instance_id}.json', "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2)

    print(f"[+] Filtered {len(filtered)} messages (from {len(args.input)}) → {args.instance_id}.json")


if __name__ == "__main__":
    main()
