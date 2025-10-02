#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path.cwd()
SWEAGENT_FILE = ROOT / "sweagent_pmd_results.json"
OUTPUT_FILE = ROOT / "sweagent_style_review.json"

# --- Helpers -----------------------------------------------------------------


def run(cmd: List[str], cwd: Path | None = None) -> Tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)."""
    try:
        proc = subprocess.run(
            cmd,
            cwd=(None if cwd is None else str(cwd)),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except FileNotFoundError as e:
        return 127, "", str(e)


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# --- Problem statement parsing ----------------------------------------------

VIOLATION_LINE_RE = re.compile(
    r"Line\s*(?P<line>\d+)\s*,\s*Column\s*(?P<col>\d+)\s*:\s*(?P<msg>.+?)\s*(?:\[(?P<source>[^\]]+)\])?\s*$"
)


def parse_problem_statement(text: str) -> List[Dict[str, Any]]:
    """
    Parse the problem_statement text and return list of violations:
      [{ 'file': ..., 'line': int, 'column': int, 'message': ..., 'source': ... }, ...]
    """
    if not text:
        return []
    entries: List[Dict[str, Any]] = []
    # Split by occurrences of "File:" (keeps repo-relative file path on first line of each block)
    blocks = re.split(r"\n\s*File:\s*", text)
    for block in (blocks[1:] if len(blocks) > 1 else blocks):
        lines = block.splitlines()
        if not lines:
            continue
        file_line = lines[0].strip()
        file_path = file_line
        for ln in lines[1:]:
            ln = ln.strip()
            if not ln:
                continue
            m = VIOLATION_LINE_RE.search(ln)
            if not m:
                continue
            entries.append(
                {
                    "file": file_path,
                    "line": int(m.group("line")),
                    "column": int(m.group("col")),
                    "message": m.group("msg").strip(),
                    "source": (m.group("source") or "").strip(),
                }
            )
    return entries


# --- original_style_errors loader -------------------------------------------


def load_original_style_errors(path: Path) -> List[Dict[str, Any]]:
    """
    Load original_style_errors.json and flatten messages to list of dicts:
      [{ 'file_base': basename, 'file_path': full_path, 'line': int|None, 'column': int|None, 'message':..., 'source':...}, ...]
    """
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return []
    flat: List[Dict[str, Any]] = []
    for fileobj in data:
        file_path = fileobj.get("file") or fileobj.get("filename") or ""
        base = os.path.basename(file_path)
        for msg in fileobj.get("messages", []) or []:
            try:
                line = int(msg.get("line"))
            except Exception:
                line = None
            try:
                col = int(msg.get("column"))
            except Exception:
                col = None
            # normalize source to be a stripped string (avoid None)
            src = (msg.get("source") or "").strip()
            flat.append(
                {
                    "file_base": base,
                    "file_path": file_path,
                    "line": line,
                    "column": col,
                    "message": msg.get("message"),
                    "source": src,
                }
            )
    return flat


# --- compare function -------------------------------------------------------


def compare_violations(problem_entries: List[Dict[str, Any]], original_msgs: List[Dict[str, Any]]):
    """
    Return (missing_count, missing_entries) where missing_entries are problem_entries not present in original_msgs.
    Matching is done on (basename, line, column).
    """
    lookup = set((m["file_base"], m["line"], m["column"]) for m in original_msgs)
    missing = []
    for p in problem_entries:
        base = os.path.basename(p["file"])
        key = (base, p["line"], p["column"])
        if key not in lookup:
            missing.append(p)
    return len(missing), missing


# --- additional errors computation ------------------------------------------


def compute_additional_errors(problem_entries: List[Dict[str, Any]], original_msgs: List[Dict[str, Any]]):
    """
    Compute "additional errors" with the following rules:
      - Consider only original_msgs that belong to the same files (by basename) mentioned in problem_entries.
      - Exclude any original_msg whose source matches any source present in the problem_entries for that same file.
        (i.e., additional errors should NOT be the same source as any in the problem statement)
      - Exclude any original_msg whose source or message contains the substrings 'whitespace' or 'newline' (case-insensitive).
      - Consider an original_msg "additional" if its (line, column) is different from ALL (line, column) locations present
        in problem_entries for that same file (regardless of source).
    Returns (additional_count, additional_entries)
    """
    # Build mapping: file_base -> set of (line, column) from problem_entries (all sources)
    problems_linecols: Dict[str, set] = {}
    # Build mapping: file_base -> set of sources present in problem_entries
    problem_sources: Dict[str, set] = {}

    for p in problem_entries:
        base = os.path.basename(p["file"])
        line = p.get("line")
        col = p.get("column")
        problems_linecols.setdefault(base, set()).add((line, col))
        src = (p.get("source") or "").strip()
        if src:
            problem_sources.setdefault(base, set()).add(src)

    additional: List[Dict[str, Any]] = []
    seen = set()
    for m in original_msgs:
        base = m.get("file_base")
        if not base:
            continue
        # only consider original messages from files mentioned in the problem statement
        if base not in problems_linecols:
            continue

        src = (m.get("source") or "").strip()
        msg_text = (m.get("message") or "")
        # skip if source matches any source present in problem entries for this file (we WANT different sources)
        if src and src in problem_sources.get(base, set()):
            continue

        # skip messages/sources that mention 'whitespace' or 'newline' (case-insensitive)
        low_src = src.lower()
        low_msg = msg_text.lower()
        if "whitespace" in low_src or "newline" in low_src or "whitespace" in low_msg or "newline" in low_msg:
            continue

        msg_line = m.get("line")
        msg_col = m.get("column")
        # If the original message's (line, column) is not one of the problem entries for that file,
        # it's considered an "additional" error (i.e., same file but different location and different source).
        if (msg_line, msg_col) not in problems_linecols[base]:
            uniq = (base, src, msg_line, msg_col)
            if uniq not in seen:
                seen.add(uniq)
                additional.append(m)

    return len(additional), additional


# --- per-item processing ----------------------------------------------------


def process_item(item: Dict[str, Any], work_root: Path) -> Dict[str, Any]:
    """
    Process a single entry from sweagent_pmd_results.json.
    - clone repo if needed
    - checkout base_commit
    - apply patch
    - copy repo to data/java_style_review/{org}/{repo}/style_review/style-review-{pull_number}/repo
    - run codearena.py (from ROOT)
    - load original_style_errors.json and compare violations
    - compute additional_errors_count and list (same file but different source/location; excludes whitespace/newline)
    """
    repo_field = item.get("repo")
    if not repo_field or "/" not in repo_field:
        raise ValueError("repo field missing or invalid: " + str(repo_field))
    org, repo = repo_field.split("/", 1)
    pull_number = str(item.get("pull_number"))
    instance_id = item.get("instance_id")
    base_commit = item.get("base_commit") or ""
    patch_text = item.get("patch") or ""
    problem_statement = item.get("problem_statement") or ""

    print(f"\n=== Processing {org}/{repo} pull {pull_number} instance {instance_id} ===")

    # clone repo (under work_root/clones)
    clones_root = work_root / "clones"
    safe_mkdir(clones_root)
    clone_path = clones_root / f"{org}__{repo}"
    if not clone_path.exists():
        print(f"Cloning https://github.com/{org}/{repo}.git -> {clone_path}")
        code, out, err = run(["git", "clone", f"https://github.com/{org}/{repo}.git", str(clone_path)])
        if code != 0:
            print(f"git clone failed: {code}\n{err}")
            return {
                "org": org,
                "repo": repo,
                "pull_number": pull_number,
                "instance_id": instance_id,
                "error": "git_clone_failed",
                "git_error": err,
            }
    else:
        print(f"Using existing clone at {clone_path}")
        # fetch latest just in case
        run(["git", "fetch", "--all"], cwd=clone_path)

    # checkout base_commit if provided
    checkout_failed = False
    if base_commit:
        print(f"Checking out base commit {base_commit}")
        code, out, err = run(["git", "checkout", base_commit], cwd=clone_path)
        if code != 0:
            print(f"Checkout failed (attempting to fetch commit): {err}")
            run(["git", "fetch", "--all"], cwd=clone_path)
            code, out, err = run(["git", "checkout", base_commit], cwd=clone_path)
            if code != 0:
                print(f"Still failed to checkout {base_commit}: {err}")
                checkout_failed = True
            else:
                checkout_failed = False

    # apply patch (robust attempts)
    applied_ok = False
    if patch_text and patch_text.strip():
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tf:
            tf.write(patch_text)
            patch_file = Path(tf.name)
        try:
            print(f"Attempting to apply patch via git apply: {patch_file}")
            code, out, err = run(["git", "apply", str(patch_file)], cwd=clone_path)
            if code == 0:
                applied_ok = True
                print("git apply succeeded")
            else:
                print(f"git apply failed: {err}\nTrying git apply --3way")
                code, out, err = run(["git", "apply", "--3way", "--whitespace=nowarn", str(patch_file)], cwd=clone_path)
                if code == 0:
                    applied_ok = True
                    print("git apply --3way succeeded")
                else:
                    print(f"git apply --3way failed: {err}\nTrying git apply --index")
                    code, out, err = run(["git", "apply", "--index", str(patch_file)], cwd=clone_path)
                    if code == 0:
                        run(["git", "add", "-A"], cwd=clone_path)
                        run(["git", "commit", "-m", "Apply patch from sweagent_pmd_results_deepseek.json"], cwd=clone_path)
                        applied_ok = True
                    else:
                        print(f"All patch apply attempts failed. git apply --index err: {err}")
        finally:
            try:
                patch_file.unlink()
            except Exception:
                pass
    else:
        print("No patch provided; skipping patch apply.")

    # copy repository to target path (style_review directory)
    target_root = work_root / "data" / "java_style_review" / org / repo / "style_review" / f"style-review-{pull_number}"
    repo_copy_path = target_root / "repo"
    print(f"Copying repo to {repo_copy_path}")
    if repo_copy_path.exists():
        print("Target repo copy already exists; removing it first")
        shutil.rmtree(repo_copy_path)
    safe_mkdir(target_root)
    try:
        shutil.copytree(clone_path, repo_copy_path)
    except Exception as e:
        print(f"Failed to copy tree: {e}")
        return {
            "org": org,
            "repo": repo,
            "pull_number": pull_number,
            "instance_id": instance_id,
            "error": "copy_failed",
            "exception": str(e),
        }

    codearena_cmd = [
        sys.executable,
        "multiswebench_local/multi_swe_bench/harness/style_review/pmd_runner.py",
        "--org",
        f"{org}",
        "--repo",
        f"{repo}",
        "--pull",
        f"{pull_number}",
        "--base-commit",
        f"{base_commit}",
    ]
    print(f"Running codearena in {ROOT}: {' '.join(codearena_cmd)}")
    codearena_exit_code, codearena_out, codearena_err = run(codearena_cmd, cwd=ROOT)
    print(f"codearena exit {codearena_exit_code}\nstdout:\n{codearena_out}\nstderr:\n{codearena_err}")

    # load original_style_errors.json from target_root (codearena should write here)
    original_errors_path = target_root / "original_style_errors.json"
    original_msgs = load_original_style_errors(original_errors_path)

    # parse and compare problem_statement violations
    problem_entries = parse_problem_statement(problem_statement)
    missing_count, missing_entries = compare_violations(problem_entries, original_msgs)

    # compute additional errors (same file as problem entries, different source than problem entries,
    # exclude messages/sources containing 'whitespace' or 'newline')
    additional_count, additional_entries = compute_additional_errors(problem_entries, original_msgs)
    if additional_count:
        print(f"Found {additional_count} additional error(s) (same file, different source, different location) in original_style_errors.json")

    result = {
        "org": org,
        "repo": repo,
        "pull_number": pull_number,
        "instance_id": instance_id,
        "base_commit": base_commit,
        "applied_patch": applied_ok,
        "checkout_failed": checkout_failed,
        "codearena_exit_code": codearena_exit_code,
        "codearena_stdout": codearena_out[:5000],
        "codearena_stderr": codearena_err[:5000],
        "problem_violations_count": len(problem_entries),
        "missing_violations_count": missing_count,
        "missing_violations": missing_entries,
        "additional_errors_count": additional_count,
        "additional_errors": additional_entries,
    }

    return result


# --- top-level / filtering / CLI --------------------------------------------


def load_input_items(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        print(f"ERROR: {path} not found in {ROOT}")
        sys.exit(2)
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        sys.exit(2)
    return data


def filter_items(
    data: List[Dict[str, Any]], org: str | None = None, repo: str | None = None, pull_number: str | None = None
) -> List[Dict[str, Any]]:
    if not (org or repo or pull_number):
        return data

    def match_item(it: Dict[str, Any]) -> bool:
        repo_field = it.get("repo", "")
        if "/" not in repo_field:
            return False
        org0, repo0 = repo_field.split("/", 1)
        pr0 = str(it.get("pull_number", ""))
        if org and org0 != org:
            return False
        if repo and repo0 != repo:
            return False
        if pull_number and pr0 != str(pull_number):
            return False
        return True

    return [it for it in data if match_item(it)]


def main_filterable(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    work_root = ROOT
    for item in items:
        try:
            res = process_item(item, work_root)
        except Exception as e:
            print(f"Exception while processing item: {e}")
            res = {"error": "exception", "exception": str(e), "item": item}
        results.append(res)
        # incremental write
        try:
            with OUTPUT_FILE.open("w", encoding="utf-8") as of:
                json.dump(results, of, indent=2)
        except Exception as e:
            print(f"Failed to write output file: {e}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sweagent style review workflow (supports filtering by org/repo/pull).")
    parser.add_argument("--org", type=str, help="Organization (e.g., apache)")
    parser.add_argument("--repo", type=str, help="Repository name (e.g., dubbo)")
    parser.add_argument("--pull_number", type=str, help="Pull request number (e.g., 10638)")
    parser.add_argument("--swe-file", type=str, default=str(SWEAGENT_FILE), help=f"Input file (default: {SWEAGENT_FILE})")
    parser.add_argument("--instance", type=str, help='Combined instance identifier in form org/repo:pull_number')
    args = parser.parse_args()

    # support --instance "org/repo:pr"
    if args.instance:
        try:
            repo_part, pr_part = args.instance.split(":", 1)
            org_part, repo_part2 = repo_part.split("/", 1)
            args.org = org_part
            args.repo = repo_part2
            args.pull_number = pr_part
        except Exception:
            print('Invalid --instance format. Expected org/repo:pull_number')
            sys.exit(2)

    data = load_input_items(Path(args.swe_file))
    items_to_process = filter_items(data, org=args.org, repo=args.repo, pull_number=args.pull_number)
    if (args.org or args.repo or args.pull_number) and not items_to_process:
        print(f"No entries found in {args.swe_file} matching the provided filters (org={args.org}, repo={args.repo}, pull_number={args.pull_number}).")
        sys.exit(3)

    results = main_filterable(items_to_process)

    print("\nAll done. Summary:")
    for r in results:
        print(json.dumps(r, indent=2)[:1000])

    print(f"\nFull results written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
