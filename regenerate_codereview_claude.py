#!/usr/bin/env python3
"""
Resume/regenerate interrupted ACR CodeReview for Claude by re-running only the
corrupted/tail instances from an existing codereview_results.jsonl.

Usage example:
  python regenerate_codereview_claude.py \
    --results-dir results/claude_sonnet_acr_results \
    --task-file data/codearena_instances.jsonl \
    --start-line 20 \
    --run

If --start-line is not provided, the script will attempt to detect the first
corrupted line by scanning for the first JSON decoding error or malformed entry.

By default, it prepares the acr_runner command for model
  openrouter/anthropic/claude-sonnet-4
and mode codereview. You can override the model via --model-name.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def detect_first_bad_line(jsonl_path: Path) -> Optional[int]:
    """Return 1-based index of the first bad/malformed line, else None."""
    with jsonl_path.open("r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if not isinstance(obj, dict) or "instance_id" not in obj:
                    return idx
            except Exception:
                return idx
    return None


def collect_instance_ids_from(jsonl_path: Path, start_line: int) -> List[str]:
    ids: List[str] = []
    with jsonl_path.open("r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f, 1):
            if idx < start_line:
                continue
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if isinstance(obj, dict) and "instance_id" in obj:
                    ids.append(str(obj["instance_id"]))
            except Exception:
                # treat malformed line as needing regeneration; cannot extract id
                # so we skip it to avoid breaking the command; user can add manually if needed
                continue
    # deduplicate but keep order
    seen = set()
    out = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def build_acr_command(
    acr_root: Path,
    task_file: Path,
    output_dir: Path,
    model_name: str,
    instance_ids: List[str],
) -> list[str]:
    return [
        sys.executable,
        str(acr_root / "acr_runner.py"),
        "-i",
        str(task_file),
        "-o",
        str(output_dir),
        "-m",
        model_name,
        "--instance-ids",
        ",".join(instance_ids),
        "--mode",
        "codereview",
        "--agentic",
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate interrupted CodeReview runs for Claude")
    ap.add_argument("--results-dir", type=Path, required=True, help="ACR results dir (e.g., results/claude_sonnet_acr_results)")
    ap.add_argument("--task-file", type=Path, required=True, help="Dataset tasks file (e.g., data/codearena_instances.jsonl)")
    ap.add_argument("--acr-root", type=Path, default=Path("baselines/AutoCodeRover"), help="Path to baselines/AutoCodeRover")
    ap.add_argument("--model-name", type=str, default="openrouter/anthropic/claude-sonnet-4", help="Model name for ACR")
    ap.add_argument("--start-line", type=int, default=None, help="1-based line to start regenerating from in codereview_results.jsonl")
    ap.add_argument("--run", action="store_true", help="Actually run the regeneration command (otherwise just print it)")

    args = ap.parse_args()

    results_dir: Path = args.results_dir
    results_file = results_dir / "codereview" / "codereview_results.jsonl"
    if not results_file.exists():
        raise SystemExit(f"codereview_results.jsonl not found: {results_file}")

    if not args.task_file.exists():
        raise SystemExit(f"Task file not found: {args.task_file}")

    if not args.acr_root.exists():
        raise SystemExit(f"ACR root not found: {args.acr_root}")

    print("=== Regenerate CodeReview for Claude ===")
    print(f"results_dir: {results_dir}")
    print(f"results_file: {results_file}")
    print(f"task_file:   {args.task_file}")
    print(f"acr_root:    {args.acr_root}")
    print(f"model:       {args.model_name}")

    start_line = args.start_line
    if start_line is None:
        start_line = detect_first_bad_line(results_file) or 1_000_000_000  # effectively none
        if start_line == 1_000_000_000:
            print("No corruption detected; nothing to regenerate.")
            return
        print(f"Detected first corrupted/malformed line at: {start_line}")
    else:
        print(f"Using user-provided start_line: {start_line}")

    instance_ids = collect_instance_ids_from(results_file, start_line)
    if not instance_ids:
        print("No instance IDs collected for regeneration.")
        return

    print(f"Collected {len(instance_ids)} instance_ids starting from line {start_line}.")
    cmd = build_acr_command(args.acr_root, args.task_file, results_dir, args.model_name, instance_ids)

    print("\nCommand:\n  " + " ".join(cmd))
    if not args.run:
        print("\nPreview only. Re-run with --run to execute.")
        return

    try:
        print("\nRunning regeneration...\n")
        res = subprocess.run(cmd, cwd=args.acr_root, capture_output=True, text=True, check=True)
        print(res.stdout)
        if res.stderr:
            print(res.stderr)
        print("\nRegeneration completed.")
    except subprocess.CalledProcessError as e:
        print("Regeneration failed:")
        print(e.stdout)
        print(e.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()


