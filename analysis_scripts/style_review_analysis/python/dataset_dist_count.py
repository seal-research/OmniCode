#!/usr/bin/env python3
"""
Summarize distribution of instances by total error count
and save results to a JSON file.
"""

import json
import argparse
from collections import Counter
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Summarize dataset distribution by error count")
    parser.add_argument("--input", required=True, help="Path to dataset JSON file")
    parser.add_argument("--output", default="count_distribution.json", help="Path to save summary JSON")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    totals = 0
    # Count instances by their error totals
    counts = Counter()
    for inst in data:
        total = inst.get("style_review", {}).get("total_messages", 0)
        totals += total
        counts[total] += 1

    # Prepare summary dictionary
    summary = {
        "total_instances": len(data),
        "average_instances": totals/len(data),
        "distribution": {str(total): count for total, count in sorted(counts.items())}
    }

    # Print summary
    print("Distribution of dataset instances by total error count:\n")
    for total_errors, num_instances in sorted(counts.items()):
        print(f"  {total_errors:3d} errors: {num_instances} instances")
    print("\nTotal instances:", len(data))
    print(f"\nSaving summary to {args.output}")
    print(totals/len(data))

    # Save JSON summary
    Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
