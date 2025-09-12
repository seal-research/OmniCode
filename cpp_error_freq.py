#!/usr/bin/env python3
"""
count_sources.py

Count frequency of unique errors grouped by their 'source' string
and write a CSV with two columns: source,frequency

Usage:
    python count_sources.py /path/to/results.json [--out source_freq.csv] [--skip-empty]
"""
import argparse
import json
import csv
import os
import sys
from collections import Counter

def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)

def extract_files(parsed):
    # Prefer parsed["files"], else if top-level list treat as files list,
    # else if top-level has "messages" treat as single file wrapper.
    if isinstance(parsed, dict):
        if "files" in parsed and isinstance(parsed["files"], list):
            return parsed["files"]
        if "messages" in parsed and isinstance(parsed["messages"], list):
            return [ {"file": parsed.get("file", "<unknown>"), "messages": parsed["messages"]} ]
    if isinstance(parsed, list):
        return parsed
    return []

def main():
    p = argparse.ArgumentParser(description="Count source frequencies from results.json")
    p.add_argument("results", help="Path to results.json")
    p.add_argument("--out", help="CSV output path (default: stdout)", default=None)
    p.add_argument("--skip-empty", action="store_true",
                   help="Skip messages missing a 'source' field (i.e. don't count '<no-source>')")
    args = p.parse_args()

    try:
        parsed = load_json(args.results)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    files = extract_files(parsed)
    counter = Counter()
    total_messages = 0

    for f in files:
        # tolerate a few common alternative keys
        messages = f.get("messages") or f.get("warnings") or f.get("errors") or []
        if not isinstance(messages, list):
            continue
        for m in messages:
            total_messages += 1
            src = m.get("source")
            if src is None:
                if args.skip_empty:
                    continue
                src = "<no-source>"
            # normalize to string
            counter[str(src)] += 1

    # prepare rows sorted by descending frequency, then source lexicographically
    rows = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))

    # write CSV (source,frequency)
    if args.out:
        try:
            with open(args.out, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["source", "frequency"])
                for src, freq in rows:
                    writer.writerow([src, freq])
            print(f"Wrote {len(rows)} rows to {args.out} (total messages scanned: {total_messages})")
        except Exception as e:
            print(f"ERROR writing CSV: {e}", file=sys.stderr)
            sys.exit(3)
    else:
        writer = csv.writer(sys.stdout)
        writer.writerow(["source", "frequency"])
        for src, freq in rows:
            writer.writerow([src, freq])

if __name__ == "__main__":
    main()
