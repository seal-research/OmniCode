#!/usr/bin/env python3
"""
Produce deepseek_java_results.csv from a sweagent style-review results file.

Usage:
  python make_deepseek_csv.py sweagent_style_review_results [--out deepseek_java_results.csv]

The input file can be:
 - a JSON array: [ { ... }, { ... }, ... ]
 - or newline-delimited JSON (JSONL): one JSON object per line

Output CSV columns:
  org, repo, instance_id, missing_violations_count, problem_violations_count
"""
from __future__ import annotations
import json
import csv
import argparse
import logging
from typing import Iterable, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def iter_json_objects(path: str) -> Iterable[Dict[str, Any]]:
    """
    Yield JSON objects from `path`. Handles:
      - full JSON array (loads the whole file)
      - JSONL (one object per line)
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
        if not text:
            return
        # Try to parse as whole JSON first (array or single object)
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            # fallback: try JSONL (one JSON object per non-empty line)
            logging.info("Input not a single JSON blob — trying JSONL (one JSON object per line).")
            with open(path, "r", encoding="utf-8") as fh:
                for ln in fh:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        yield json.loads(ln)
                    except json.JSONDecodeError as e:
                        logging.warning("Skipping invalid JSON line: %s", e)
            return

        # If we got here, obj is a parsed JSON object.
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    yield item
                else:
                    logging.warning("Skipping non-object item in JSON array.")
        elif isinstance(obj, dict):
            # If it's a dict, maybe it's already the single object, or contains a list in some key.
            # Common pattern: {"results": [ ... ]}
            # We'll try to find a top-level list value if present, else yield the dict itself.
            list_val = None
            for v in obj.values():
                if isinstance(v, list):
                    list_val = v
                    break
            if list_val is not None:
                for item in list_val:
                    if isinstance(item, dict):
                        yield item
                    else:
                        logging.warning("Skipping non-object item in nested JSON list.")
            else:
                yield obj
        else:
            logging.error("Top-level JSON is neither array nor object; nothing to do.")


def extract_counts(item: Dict[str, Any], count_key: str, fallback_list_keys=()) -> int:
    """
    Extract an integer count from item[count_key] if present and int-like.
    Otherwise, try to compute from any list fields provided in fallback_list_keys.
    Returns 0 if none found or invalid.
    """
    val = item.get(count_key)
    if isinstance(val, int):
        return val
    # Sometimes the count might be a string containing digits
    if isinstance(val, str) and val.isdigit():
        try:
            return int(val)
        except ValueError:
            pass
    # Fallback: check for list fields that represent the violations
    for lk in fallback_list_keys:
        lst = item.get(lk)
        if isinstance(lst, list):
            return len(lst)
    # last resort: try to infer from similarly named keys
    for k in item:
        if k.lower().startswith(count_key.replace("_count", "")) and isinstance(item[k], int):
            return item[k]
    return 0


def row_from_item(item: Dict[str, Any]) -> Dict[str, object]:
    org = item.get("org", "")
    repo = item.get("repo", "")
    instance_id = item.get("instance_id", item.get("instance", ""))
    missing = extract_counts(item, "missing_violations_count", ("missing_violations",))
    problem = extract_counts(item, "problem_violations_count", ("problem_violations", "violations"))
    return {
        "org": org,
        "repo": repo,
        "instance_id": instance_id,
        "missing_violations_count": missing,
        "problem_violations_count": problem,
    }


def main():
    p = argparse.ArgumentParser(description="Create deepseek_java_results.csv from sweagent results.")
    p.add_argument("infile", help="Input file (JSON array or JSONL). Example: sweagent_style_review_results")
    p.add_argument("--out", "-o", default="aider_java_results.csv",
                   help="Output CSV filename (default: deepseek_java_results.csv)")
    args = p.parse_args()

    infile = args.infile
    outfile = args.out

    logging.info("Reading input: %s", infile)
    objs = iter_json_objects(infile)

    header = ["org", "repo", "instance_id", "missing_violations_count", "problem_violations_count"]
    count = 0
    with open(outfile, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=header)
        writer.writeheader()
        for item in objs:
            try:
                row = row_from_item(item)
            except Exception as e:
                logging.warning("Skipping an item due to error: %s", e)
                continue
            writer.writerow(row)
            count += 1

    logging.info("Wrote %d rows to %s", count, outfile)
    print(f"Done — {outfile} created with {count} rows.")


if __name__ == "__main__":
    main()
