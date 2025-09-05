#!/usr/bin/env python3
"""
jsons_to_style_csv.py

Process SWE agent style-review JSON outputs and produce a CSV summary.

Output CSV columns:
  Instance number, Org, Repo, Pull Number, Style Errors Fixed, Total Style Errors, Score

Behavior:
 - If --dir is provided: the script will look for the exact filenames listed
   in HARDCODED_FILES inside that directory and process files in that same order.
 - If none of the HARDCODED_FILES are found in the directory, it falls back to
   a glob search using --pattern (and --recurse if requested).
 - If --dir is not given, the script looks for HARDCODED_FILES in the current
   working directory (same order) and will skip missing files (with a notice).
"""

import json
import csv
import sys
from pathlib import Path
from typing import List, Any
import argparse

# 🔒 Hardcoded list of filenames (kept for backward compatibility and ordering).
HARDCODED_FILES = [
    "sweagent_style_review_results_apache_dubbo_10638.json",
    "sweagent_style_review_results_elastic_logstash_17021.json",
    "sweagent_style_review_results_alibaba_fastjson2_2775.json",
    "sweagent_style_review_results_fasterxml_jackson-core_1309.json",
    "sweagent_style_review_results_fasterxml_jackson-dataformat-xml_644.json",
    "sweagent_style_review_results_fasterxml_jackson-databind_4641.json",
    "sweagent_style_review_results_google_gson_1787.json",
    "sweagent_style_review_results_googlecontainertools_jib_4144.json",
    "sweagent_style_review_results_mockito_mockito_3424.json",
    "sweagent_style_review_results_google_guava_6586_1.json",
    "sweagent_style_review_results_google_guava_6586_2.json",
    "sweagent_style_review_results_spring-projects_spring-boot_45267.json",
]


def load_entries_from_file(path: Path) -> List[dict]:
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"Failed reading/parsing JSON '{path}': {e}")
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise RuntimeError(f"Unexpected JSON top-level type in '{path}': {type(data)}")


def safe_int(x: Any) -> Any:
    if x is None:
        return ""
    if isinstance(x, int):
        return x
    s = str(x).strip()
    if s == "":
        return ""
    try:
        return int(s)
    except Exception:
        return s


def compute_score(fixed: int, total: int) -> float:
    try:
        fixed = int(fixed)
    except Exception:
        fixed = 0
    try:
        total = int(total)
    except Exception:
        total = 0
    if total == 0:
        return 0.0
    return float(fixed) / float(total)


def gather_from_files(files: List[Path]) -> List[dict]:
    rows = []
    for p in files:
        try:
            entries = load_entries_from_file(p)
        except Exception as e:
            print(f"Warning: skipping '{p}': {e}", file=sys.stderr)
            continue
        for ent in entries:
            org = ent.get("org", "") or ""
            repo = ent.get("repo", "") or ""
            pull = safe_int(ent.get("pull_number", ""))
            # resolved fixed count
            fixed = ent.get("missing_violations_count", None)
            if fixed is None:
                mv = ent.get("missing_violations", None)
                if isinstance(mv, list):
                    fixed = len(mv)
                else:
                    fixed = 0
            if isinstance(fixed, list):
                fixed = len(fixed)
            try:
                fixed_i = int(fixed)
            except Exception:
                fixed_i = 0
            total = ent.get("problem_violations_count", 0)
            try:
                total_i = int(total)
            except Exception:
                total_i = 0
            score = compute_score(fixed_i, total_i)
            rows.append({
                "org": org,
                "repo": repo,
                "pull_number": pull,
                "fixed": fixed_i,
                "total": total_i,
                "score": score,
                "source_file": str(p),
            })
    return rows


def find_files_in_dir_by_pattern(directory: Path, pattern: str = "sweagent_style_review_results_*.json", recursive: bool = False) -> List[Path]:
    if recursive:
        return sorted(directory.rglob(pattern))
    else:
        return sorted(directory.glob(pattern))


def main():
    parser = argparse.ArgumentParser(description="Produce CSV summary from SWE agent style-review JSON outputs.")
    parser.add_argument("--dir", "-d", help="Directory to search for JSON files (optional).")
    parser.add_argument("--pattern", "-p", default="sweagent_style_review_results_*.json", help="Glob pattern to search for (default: sweagent_style_review_results_*.json).")
    parser.add_argument("--recurse", "-r", action="store_true", help="Search directories recursively (only used when --dir is provided and HARDCODED names weren't found).")
    parser.add_argument("--out", "-o", default="style_summary.csv", help="Output CSV path (default: style_summary.csv).")
    parser.add_argument("--use-hardcoded", action="store_true", help="Force using HARDCODED_FILES from the current working directory instead of searching a directory.")
    args = parser.parse_args()

    files_to_process: List[Path] = []

    # If directory provided, prefer to collect HARDCODED files from that directory in order.
    if args.dir:
        dirp = Path(args.dir)
        if not dirp.exists() or not dirp.is_dir():
            print(f"Error: directory not found or not a directory: {dirp}", file=sys.stderr)
            sys.exit(2)

        # Try to locate each HARDCODED filename inside the provided directory (non-recursive).
        # This enforces the exact ordering required by the user.
        found_any = False
        for name in HARDCODED_FILES:
            candidate = dirp / name
            if candidate.exists():
                files_to_process.append(candidate)
                found_any = True
            else:
                print(f"Notice: expected file not found in --dir (skipping): {candidate}", file=sys.stderr)

        if not found_any:
            # fallback to pattern search (honor recurse flag)
            print(f"No HARDCODED filenames were found in '{dirp}'. Falling back to pattern search '{args.pattern}'.", file=sys.stderr)
            matched = find_files_in_dir_by_pattern(dirp, pattern=args.pattern, recursive=args.recurse)
            if matched:
                files_to_process = matched
            else:
                print(f"No files matching pattern '{args.pattern}' found in directory '{dirp}'.", file=sys.stderr)

    # If no --dir (or fallback situation) - try HARDCODED_FILES in current working directory,
    # unless user explicitly asked to use hardcoded in cwd (use-hardcoded) — behavior is the same.
    if not files_to_process:
        cwd = Path.cwd()
        for name in HARDCODED_FILES:
            p = cwd / name
            if p.exists():
                files_to_process.append(p)
            else:
                print(f"Notice: hardcoded file not found (skipping): {p}", file=sys.stderr)

    if not files_to_process:
        print("Error: No JSON files to process. Provide --dir with matching files or place HARDCODED_FILES in the current directory.", file=sys.stderr)
        sys.exit(2)

    print(f"Processing {len(files_to_process)} file(s) in the requested order...")

    rows = gather_from_files(files_to_process)

    # Keep ordering stable: rows already correspond to files in the requested order; however
    # if you still want to sort by score, uncomment the following line. Currently we keep
    # file-derived ordering (HARDCODED order) as the primary ordering.
    # rows.sort(key=lambda r: (-r["score"], r["org"], r["repo"], str(r["pull_number"])))

    out_path = Path(args.out)
    try:
        with out_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["Instance number", "Org", "Repo", "Pull Number",
                             "Style Errors Fixed", "Total Style Errors", "Score", "Source File"])
            for i, r in enumerate(rows, start=1):
                score_str = f"{r['score']:.4f}"
                writer.writerow([i, r["org"], r["repo"], r["pull_number"],
                                 r["fixed"], r["total"], score_str, r.get("source_file", "")])
    except Exception as e:
        print(f"Error writing CSV '{out_path}': {e}", file=sys.stderr)
        sys.exit(3)

    print(f"Wrote {len(rows)} rows to {out_path.resolve()}")
    total_files = len({r.get("source_file") for r in rows})
    print(f"Summary: {len(rows)} instances from {total_files} input file(s).")


if __name__ == "__main__":
    main()
