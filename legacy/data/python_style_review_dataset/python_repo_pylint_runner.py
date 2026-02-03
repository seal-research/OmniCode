#!/usr/bin/env python3
"""
What it does:
 - clones repo
 - checks out PR (fetches pull/<PR>/head)
 - optionally applies a patch from codeareana_instances matching an instance_id
 - runs pylint across repo
 - filters out style/env noise (e.g. too-many-args, import errors)
 - parses pylint JSON output
 - emits JSON summary to stdout and to --out
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------- Filtering configuration ----------
STYLE_IGNORE_IDS = {
    "R0917",  # too-many-positional-arguments
}
ENV_IGNORE_IDS = {
    "E0401",  # import-error
    "E0611",  # no-name-in-module
    "E1101",  # no-member
}
IGNORE_IDS = STYLE_IGNORE_IDS | ENV_IGNORE_IDS
KEEP_TYPES = {"error", "warning", "fatal"}


def filter_pylint_output(pylint_output, keep_types=KEEP_TYPES):
    """Filter out unwanted pylint messages before parsing."""
    filtered = []
    for msg in pylint_output:
        if msg.get("type") not in keep_types:
            continue
        if msg.get("message-id") in IGNORE_IDS:
            continue
        filtered.append(msg)
    return filtered


# ---------- Helpers ----------
def run(cmd, cwd=None, capture=False, env=None, check=True):
    if capture:
        res = subprocess.run(cmd, cwd=cwd, env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if check and res.returncode != 0:
            raise subprocess.CalledProcessError(res.returncode, cmd, res.stdout)
        return res.stdout
    else:
        subprocess.run(cmd, cwd=cwd, env=env, check=check)

def git_checkout_commit(repo_dir, commit_hash):
    # Ensure the commit is present locally
    run(["git", "fetch", "origin", commit_hash], cwd=repo_dir)
    # Checkout the exact commit (detached HEAD state)
    run(["git", "checkout", commit_hash], cwd=repo_dir)
    print(f"[+] Checked out commit {commit_hash}", file=sys.stderr)

def load_swe_results(swe_results_path):
    p = Path(swe_results_path).expanduser()
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else None
    except Exception as e:
        print(f"[!] Failed to load swe results {p}: {e}", file=sys.stderr)
        return None

def apply_patch_to_repo(repo_dir, patch_text, work_base):
    try:
        # First attempt: apply patch directly to working tree (no index)
        proc = subprocess.run(
            ["git", "apply", "--whitespace=fix", "-"],
            input=patch_text,
            text=True,
            cwd=repo_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if proc.returncode == 0:
            print("[+] Patch applied with 'git apply -'.", file=sys.stderr)
            return True

        # Second attempt: apply and stage
        proc2 = subprocess.run(
            ["git", "apply", "--whitespace=fix", "--index", "-"],
            input=patch_text,
            text=True,
            cwd=repo_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if proc2.returncode == 0:
            print("[+] Patch applied with 'git apply --index'.", file=sys.stderr)
            return True

        # Fallback: write patch to file and try again
        tmp_patch = Path(work_base) / "swe_patch.diff"
        tmp_patch.write_text(patch_text, encoding="utf-8")
        proc3 = subprocess.run(
            ["git", "apply", "--whitespace=fix", str(tmp_patch)],
            cwd=repo_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if proc3.returncode == 0:
            print(f"[+] Patch applied with {tmp_patch}.", file=sys.stderr)
            return True

        # Failed, save for debugging
        failed = Path(work_base) / "failed_patch.diff"
        failed.write_text(patch_text, encoding="utf-8")
        print(f"[!] Patch failed, wrote to {failed}", file=sys.stderr)
        print(proc.stdout or proc2.stdout or proc3.stdout, file=sys.stderr)
        return False

    except Exception as e:
        print(f"[!] Exception applying patch: {e}", file=sys.stderr)
        return False


# ---------- Run pylint ----------
def run_pylint(repo_dir, out_json_path, jobs=None):
    pylint_bin = shutil.which("pylint")
    if not pylint_bin:
        raise FileNotFoundError("pylint not found in PATH.")

    cmd = [pylint_bin, repo_dir, "-f", "json", "--score=n"]
    if jobs:
        cmd += ["-j", str(jobs)]

    log_path = out_json_path + ".log"
    with open(out_json_path, "w", encoding="utf-8") as outf, \
         open(log_path, "w", encoding="utf-8") as logf:
        subprocess.run(cmd, cwd=repo_dir, stdout=outf, stderr=logf, check=False)


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
    return files, overview


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Run pylint style review on a PR and emit JSON summary.")
    ap.add_argument("--repo-url", required=True)
    ap.add_argument("--pr", required=True, type=int)
    ap.add_argument("--out", default="pylint_results/results.json")
    ap.add_argument("--work-dir", default='.')
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--instance-id", required=False)
    ap.add_argument("--apply-patch", default=True)
    args = ap.parse_args()

    if args.work_dir:
        work_base = Path(args.work_dir).expanduser().resolve()
        work_base.mkdir(parents=True, exist_ok=True)
    
    repo_name = Path(args.repo_url).stem
    repo_dir = (work_base / 'repo').resolve()
    print(f"[+] working dir: {work_base}", file=sys.stderr)

    try:
        run(["git", "clone", args.repo_url, str(repo_dir)])
        with open('../codearena_instances_python.json') as f:
            data = json.load(f)
        for i in data:
            if i["instance_id"] == args.instance_id:
                patch = i["patch"]
                commit = i["base_commit"]
        if args.apply_patch:
            git_checkout_commit(str(repo_dir), commit)
            apply_patch_to_repo(str(repo_dir), patch, work_base)
        else: 
            git_checkout_commit(str(repo_dir), commit)

        out_json_path = str(work_base / f"pylint_results/full_pylint_run_data/{repo_name}_pylint.json")
        print("[+] Running pylint ...", file=sys.stderr)
        run_pylint(str(repo_dir), out_json_path, jobs=args.jobs)

        files, overview = parse_pylint_output(out_json_path, args.instance_id)

        obj = {"label": args.instance_id, "files": files, "overview": overview}

        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)

        print(json.dumps(obj, indent=2))
    finally:
        print(f"[+] Done. Results written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
