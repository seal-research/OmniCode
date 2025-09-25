#!/usr/bin/env python3
import re
import json
import argparse
from pathlib import Path

def analyze_patch(patch_text):
    stats = {
        "files_changed": 0,
        "hunks_changed": 0,
        "lines_added": 0,
        "lines_removed": 0,
        "net_change": 0,
    }

    for line in patch_text.splitlines():
        if line.startswith("diff --git"):
            stats["files_changed"] += 1
        elif line.startswith("@@"):
            stats["hunks_changed"] += 1
        elif line.startswith("+") and not line.startswith("+++"):
            stats["lines_added"] += 1
        elif line.startswith("-") and not line.startswith("---"):
            stats["lines_removed"] += 1

    stats["net_change"] = stats["lines_added"] - stats["lines_removed"]
    stats["complexity_score"] = (
        stats["files_changed"]
        + stats["hunks_changed"]
        + (stats["lines_added"] + stats["lines_removed"]) / 10.0
    )
    return stats


def main():
    ap = argparse.ArgumentParser(description="Analyze complexity of git diff patches")
    ap.add_argument("input_json", help="JSON file containing patches (list of dicts with 'patch', 'repo', 'instance_id')")
    ap.add_argument("output_jsonl", help="Output JSONL with patch complexity stats")
    ap.add_argument("--summary", default="patches_summary.json",
                    help="Output JSON file for global summary")
    args = ap.parse_args()

    data = json.load(open(args.input_json, "r"))
    out_path = Path(args.output_jsonl)

    global_stats = {
        "total_patches": 0,
        "total_files_changed": 0,
        "total_hunks_changed": 0,
        "total_lines_added": 0,
        "total_lines_removed": 0,
        "total_complexity_score": 0.0,
    }

    with out_path.open("w", encoding="utf-8") as outf:
        for obj in data:
            patch = obj.get("patch", "")
            if not patch:
                continue
            stats = analyze_patch(patch)

            record = {
                "repo": obj.get("repo"),
                "instance_id": obj.get("instance_id"),
                "patch_complexity": stats
            }
            outf.write(json.dumps(record, ensure_ascii=False) + "\n")

            # update global stats
            global_stats["total_patches"] += 1
            global_stats["total_files_changed"] += stats["files_changed"]
            global_stats["total_hunks_changed"] += stats["hunks_changed"]
            global_stats["total_lines_added"] += stats["lines_added"]
            global_stats["total_lines_removed"] += stats["lines_removed"]
            global_stats["total_complexity_score"] += stats["complexity_score"]

    if global_stats["total_patches"] > 0:
        global_stats["avg_complexity_score"] = round(
            global_stats["total_complexity_score"] / global_stats["total_patches"], 3
        )
        global_stats["avg_lines_added_per_patch"] = round(
            global_stats["total_lines_added"] / global_stats["total_patches"], 3
        )
        global_stats["avg_lines_removed_per_patch"] = round(
            global_stats["total_lines_removed"] / global_stats["total_patches"], 3
        )

    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump(global_stats, f, indent=2)

    print(f"[+] Wrote per-patch results to {out_path}")
    print(f"[+] Wrote global summary to {args.summary}")


if __name__ == "__main__":
    main()
