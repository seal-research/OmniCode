#!/usr/bin/env python3
"""
sweagent_style_review_runner.py

Usage:
  Place this script in the directory that contains sweagent_pmd_results.json,
  sweagent_pmd_results_gemini.json and codearena.py.

  Process everything in the JSON (default):
    python3 sweagent_style_review_runner.py

  Process a specific org/repo/pull_number (filters sweagent_pmd_results.json):
    python3 sweagent_style_review_runner.py --org apache --repo dubbo --pull_number 10638

  Or use combined instance identifier:
    python3 sweagent_style_review_runner.py --instance "apache/dubbo:10638"

Important:
 - This script will run `python codearena.py ...` **from the same directory the script is run in** (ROOT).
 - Ensure codearena.py is present and runnable from that directory.
 - This modified version applies the patch from sweagent_pmd_results_gemini.json
   (matched by instance_id) **first**, then applies the patch from
   sweagent_pmd_results.json (the original behaviour).
"""
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
GEMINI_FILE = ROOT / "sweagent_pmd_results_gemini.json"
OUTPUT_FILE = ROOT / "sweagent_style_review_results1.json"

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
            flat.append(
                {
                    "file_base": base,
                    "file_path": file_path,
                    "line": line,
                    "column": col,
                    "message": msg.get("message"),
                    "source": msg.get("source"),
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


# --- helper to load gemini patches ------------------------------------------


def load_gemini_map(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load gemini JSON and return a mapping from instance_id -> item (dict).
    This allows us to look up the gemini patch for a given instance_id.
    """
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return {}
    mapping: Dict[str, Dict[str, Any]] = {}
    for item in data:
        iid = item.get("instance_id")
        if not iid:
            continue
        mapping[iid] = item
    return mapping


# --- per-item processing ----------------------------------------------------


def _attempt_apply_patch_to_repo(patch_text: str, clone_path: Path) -> Tuple[bool, str]:
    """
    Attempt to apply a patch to the repo at clone_path.
    Returns (applied_ok, combined_error_output).
    This encapsulates the multiple attempts used previously.
    """
    if not patch_text or not patch_text.strip():
        return False, "no_patch"
    patch_file = None
    combined_err = ""
    try:
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tf:
            tf.write(patch_text)
            patch_file = Path(tf.name)
        # try git apply
        code, out, err = run(["git", "apply", "--whitespace=nowarn", str(patch_file)], cwd=clone_path)
        combined_err += f"git apply exit {code}\n{err}\n"
        if code == 0:
            return True, combined_err
        # try git apply --3way
        code, out, err = run(["git", "apply", "--3way", "--whitespace=nowarn", str(patch_file)], cwd=clone_path)
        combined_err += f"git apply --3way exit {code}\n{err}\n"
        if code == 0:
            return True, combined_err
        # try git apply --index then add commit
        code, out, err = run(["git", "apply", "--index", str(patch_file)], cwd=clone_path)
        combined_err += f"git apply --index exit {code}\n{err}\n"
        if code == 0:
            run(["git", "add", "-A"], cwd=clone_path)
            run(["git", "commit", "-m", "Apply patch from sweagent style runner"], cwd=clone_path)
            return True, combined_err
        # all attempts failed
        return False, combined_err
    finally:
        if patch_file is not None:
            try:
                patch_file.unlink()
            except Exception:
                pass


def process_item(item: Dict[str, Any], work_root: Path, gemini_map: Dict[str, Dict[str, Any]] | None = None) -> Dict[str, Any]:
    """
    Process a single entry from sweagent_pmd_results.json.
    - clone repo if needed
    - checkout base_commit
    - apply gemini patch (if present) first
    - apply patch from sweagent_pmd_results.json
    - copy repo to data/java_style_review/{org}/{repo}/style_review/style-review-{pull_number}/repo
    - run codearena.py (from ROOT)
    - load original_style_errors.json and compare violations
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

    # apply gemini patch first (if provided in gemini_map)
    applied_gemini_ok = False
    gemini_apply_err = ""
    if gemini_map and instance_id:
        gemini_item = gemini_map.get(instance_id)
        if gemini_item:
            gemini_patch_text = gemini_item.get("patch") or ""
            if gemini_patch_text and gemini_patch_text.strip():
                print(f"Attempting to apply GEMINI patch for instance {instance_id}")
                applied_gemini_ok, gemini_apply_err = _attempt_apply_patch_to_repo(gemini_patch_text, clone_path)
                if applied_gemini_ok:
                    print("GEMINI patch applied successfully")
                else:
                    print(f"GEMINI patch apply failed: {gemini_apply_err}")
            else:
                print("GEMINI entry found but no patch provided; skipping GEMINI patch apply.")
        else:
            print("No GEMINI entry for this instance; skipping GEMINI patch apply.")
    else:
        print("No GEMINI map provided or instance_id missing; skipping GEMINI patch apply.")

    # Now apply the original patch from sweagent_pmd_results.json (same logic as before)
    applied_ok = False
    original_apply_err = ""
    if patch_text and patch_text.strip():
        print(f"Attempting to apply original patch for instance {instance_id}")
        applied_ok, original_apply_err = _attempt_apply_patch_to_repo(patch_text, clone_path)
        if applied_ok:
            print("Original patch applied successfully")
        else:
            print(f"Original patch apply failed: {original_apply_err}")
    else:
        print("No original patch provided; skipping original patch apply.")

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

    # --- RUN codearena.py from ROOT (script dir) as requested -----------------
    # This is the key change: run from ROOT, not inside the copied repo.
    codearena_cmd = [
        sys.executable,
        "codearena.py",
        "--StyleReview",
        "--predictions_path",
        "gold",
        "--run_id",
        "mswe_java_style_review",
        "--max_workers",
        "1",
        "--instance_ids",
        f"{org}/{repo}:{pull_number}",
        "--mswe_phase",
        "all",
        "--force_rebuild",
        "True",
        "--review_type",
        "pmd",
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

    result = {
        "org": org,
        "repo": repo,
        "pull_number": pull_number,
        "instance_id": instance_id,
        "base_commit": base_commit,
        "applied_gemini_patch": applied_gemini_ok,
        "gemini_apply_error": gemini_apply_err[:4000] if gemini_apply_err else "",
        "applied_patch": applied_ok,
        "original_apply_error": original_apply_err[:4000] if original_apply_err else "",
        "checkout_failed": checkout_failed,
        "codearena_exit_code": codearena_exit_code,
        "codearena_stdout": codearena_out[:5000],
        "codearena_stderr": codearena_err[:5000],
        "problem_violations_count": len(problem_entries),
        "missing_violations_count": missing_count,
        "missing_violations": missing_entries,
    }

    return result


# --- top-level / filtering / CLI --------------------------------------------


def load_input_items() -> List[Dict[str, Any]]:
    if not SWEAGENT_FILE.exists():
        print(f"ERROR: {SWEAGENT_FILE} not found in {ROOT}")
        sys.exit(2)
    try:
        with SWEAGENT_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to load {SWEAGENT_FILE}: {e}")
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


def main_filterable(items: List[Dict[str, Any]], gemini_map: Dict[str, Dict[str, Any]] | None = None) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    work_root = ROOT
    for item in items:
        try:
            res = process_item(item, work_root, gemini_map=gemini_map)
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

    data = load_input_items()
    items_to_process = filter_items(data, org=args.org, repo=args.repo, pull_number=args.pull_number)
    if (args.org or args.repo or args.pull_number) and not items_to_process:
        print(f"No entries found in {SWEAGENT_FILE} matching the provided filters (org={args.org}, repo={args.repo}, pull_number={args.pull_number}).")
        sys.exit(3)

    # load gemini map (optional - if file absent we'll skip gemini patches)
    gemini_map = load_gemini_map(GEMINI_FILE)
    if gemini_map:
        print(f"Loaded GEMINI mappings for {len(gemini_map)} instances from {GEMINI_FILE}")
    else:
        print(f"No GEMINI mapping loaded from {GEMINI_FILE}; proceeding without GEMINI patches.")

    results = main_filterable(items_to_process, gemini_map=gemini_map)

    print("\nAll done. Summary:")
    for r in results:
        print(json.dumps(r, indent=2)[:1000])

    print(f"\nFull results written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
