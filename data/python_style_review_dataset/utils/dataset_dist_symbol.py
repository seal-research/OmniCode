#!/usr/bin/env python3
"""
Summarize distribution of pylint symbols across dataset.
"""

import json
import argparse
from pathlib import Path
from collections import Counter

def main():
    parser = argparse.ArgumentParser(description="Summarize distribution of pylint symbols across dataset")
    parser.add_argument("--input", required=True, help="Path to dataset JSON file")
    parser.add_argument("--output", default="symbol_distribution.json", help="Path to save symbol distribution JSON")
    args = parser.parse_args()

    # Load dataset
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))

    # Count symbols
    symbol_counts = Counter()
    for inst in data:
        files = inst.get("style_review", {}).get("files", {})
        for fdata in files.values():
            for msg in fdata.get("messages", []):
                sym = msg.get("symbol", "<unknown>")
                symbol_counts[sym] += 1

    # Prepare summary
    summary = {
        "total_instances": len(data),
        "unique_symbols": len(symbol_counts),
        "distribution": dict(symbol_counts.most_common())
    }

    # Save
    Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Print preview
    print(f"Processed {len(data)} instances")
    print(f"Found {len(symbol_counts)} unique symbols")
    print("Top 10 most common symbols:")
    for sym, count in symbol_counts.most_common(10):
        print(f"  {sym}: {count}")

    print(f"\nSaved full distribution to {args.output}")

if __name__ == "__main__":
    main()
