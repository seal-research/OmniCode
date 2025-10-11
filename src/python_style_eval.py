#!/usr/bin/env python3
"""
Batch evaluate model-generated patches against dataset style review instances.

Example command: 
python src/python_style_eval.py \
  --dataset-json data/python_style_review_dataset/chosen_python_style_review_subset.json \
  --instances-file data/python_style_review_dataset/chosen_python_style_review_instances.txt \
  --patches-jsonl path-to-output-model-patches.jsonl \
  --work-base /scratch/$USER/style_eval \
  --out style_review_results.json \
  
Workflow:
 - Read dataset JSON (list of instances with "repo", "instance_id", "base_commit", "style_review", etc.)
 - Read instance_ids from a text file (each line is one id to evaluate)
 - Read patches from a JSONL file (one per line, with instance_id + model_patch.diff)
 - For each instance:
     - Clone repo once into work-base/shared_repos/<repo>/repo (cached)
     - Checkout base_commit (reset --hard + clean -fdx)
     - Apply model patch
     - Run pylint on affected files
     - Filter out noisy IDs and types
     - Compare errors before vs after
     - Summarize {dataset_total_errors, solved_errors, newly_created_errors, solve_rate}
 - Save all results to output JSON
 - Cleanup work-base unless --keep-work is set
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from collections import Counter

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
    """Filter out unwanted pylint messages before comparing."""
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
    run(["git", "fetch", "origin", commit_hash], cwd=repo_dir, check=False)
    run(["git", "reset", "--hard"], cwd=repo_dir)
    run(["git", "clean", "-fdx"], cwd=repo_dir)
    run(["git", "checkout", "--force", commit_hash], cwd=repo_dir)
    print(f"[+] Checked out commit {commit_hash}", file=sys.stderr)


def get_repo_dir(repo_url, repo_name, work_base):
    repo_root = Path(work_base) / "shared_repos" / repo_name
    repo_dir = repo_root / "repo"
    if not repo_dir.exists():
        repo_root.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", repo_url, str(repo_dir)])
    return repo_dir


def apply_patch_to_repo(repo_dir, patch_text, work_base):
    try:
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

        failed = Path(work_base) / "failed_patch.diff"
        failed.write_text(patch_text, encoding="utf-8")
        print(f"[!] Patch failed, wrote to {failed}", file=sys.stderr)
        print(proc.stdout or proc2.stdout or proc3.stdout, file=sys.stderr)
        return False

    except Exception as e:
        print(f"[!] Exception applying patch: {e}", file=sys.stderr)
        return False


def run_pylint(repo_dir, files, out_json_path):
    pylint_bin = shutil.which("pylint")
    if not pylint_bin:
        raise FileNotFoundError("pylint not found in PATH.")

    cmd = [pylint_bin, "-f", "json", "--score=n"] + files
    log_path = out_json_path + ".log"
    with open(out_json_path, "w", encoding="utf-8") as outf, \
         open(log_path, "w", encoding="utf-8") as logf:
        subprocess.run(cmd, cwd=repo_dir, stdout=outf, stderr=logf, check=False)

    try:
        with open(out_json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        print(f"[!] Failed to parse pylint output: {e}", file=sys.stderr)
        return []
    return filter_pylint_output(raw)


# ---------- Core Evaluation ----------
def evaluate_instance(instance, patch_text, work_base):
    repo = instance["repo"]
    repo_url = f"https://github.com/{repo}.git"
    commit = instance["base_commit"]
    instance_id = instance["instance_id"]
    repo_name = repo.split("/")[-1]

    style_review = instance.get("style_review", {})
    baseline_msgs = []
    for fpath, rec in style_review.get("files", {}).items():
        for m in rec.get("messages", []):
            # keep same tuple structure as filtered pylint output
            baseline_msgs.append((fpath, m["line"], m["column"], m["symbol"], m["message"]))

    if not patch_text:
        baseline_set = set(baseline_msgs)
        return {
            "instance_id": instance_id,
            "repo": repo,
            "dataset_total_errors": len(baseline_set),
            "solved_errors": 0,
            "newly_created_errors": 0,
            "solve_rate": 0
        }

    # Instance-specific workspace
    inst_base = Path(work_base) / instance_id
    inst_base.mkdir(parents=True, exist_ok=True)

    repo_dir = get_repo_dir(repo_url, repo_name, work_base)
    git_checkout_commit(str(repo_dir), commit)

    if not apply_patch_to_repo(str(repo_dir), patch_text, inst_base):
        return None

    affected_files = list(style_review.get("files", {}).keys())
    out_json_path = str(inst_base / "patched_pylint.json")
    patched_msgs = run_pylint(str(repo_dir), affected_files, out_json_path)
    patched_tuples = [(m["path"], m["line"], m["column"], m["symbol"], m["message"]) for m in patched_msgs]

    baseline_set = set(baseline_msgs)
    patched_set = set(patched_tuples)
    solved = baseline_set - patched_set
    newly_created = patched_set - baseline_set

    # Symbol-level analysis
    solved_symbols = [m[3] for m in solved]
    newly_created_symbols = [m[3] for m in newly_created]
    unsolved = baseline_set & patched_set
    unsolved_symbols = [m[3] for m in unsolved]

    return {
        "instance_id": instance_id,
        "repo": repo,
        "dataset_total_errors": len(baseline_set),
        "solved_errors": len(solved),
        "newly_created_errors": len(newly_created),
        "solve_rate": max(0, (len(solved) - len(newly_created)) / len(baseline_set)),
        # Unique symbols
        "solved_symbols": sorted(set(solved_symbols)),
        "unsolved_symbols": sorted(set(unsolved_symbols)),
        "newly_created_symbols": sorted(set(newly_created_symbols)),
        # Symbol counts
        "solved_symbol_counts": dict(Counter(solved_symbols)),
        "unsolved_symbol_counts": dict(Counter(unsolved_symbols)),
        "newly_created_symbol_counts": dict(Counter(newly_created_symbols)),
    }



# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Batch evaluate patches from JSONL for multiple dataset instances.")
    ap.add_argument("--dataset-json", required=True)
    ap.add_argument("--instances-file", required=True)
    ap.add_argument("--patches-jsonl", required=True)
    ap.add_argument("--work-base", required=True, help="Base directory for repos + workdirs")
    ap.add_argument("--out", default="batch_eval_results.json")
    args = ap.parse_args()

    with open(args.dataset_json, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        print("[!] Dataset must be a list of instances", file=sys.stderr)
        sys.exit(1)

    with open(args.instances_file, "r") as f:
        instance_ids = [line.strip() for line in f if line.strip()]

    patches = {}
    with open(args.patches_jsonl, "r") as f:
        for line in f:
            obj = json.loads(line)
            iid = obj.get("instance_id")
            patch_text = obj.get("model_patch", {}).get("model_patch")
            if iid and patch_text:
                patches[iid] = patch_text

    id_to_instance = {i["instance_id"]: i for i in data}
    results = []

    for iid in instance_ids:
        inst = id_to_instance.get(iid)
        if not inst:
            print(f"[!] instance_id {iid} not found in dataset", file=sys.stderr)
            continue

        patch_text = patches.get(iid)
        if not patch_text:
            print(f"[!] patch for {iid} not found in {args.patches_jsonl}", file=sys.stderr)
            res = evaluate_instance(inst, None, args.work_base)
            results.append(res)
            continue

        print(f"\n=== Evaluating {iid} ===", file=sys.stderr)
        res = evaluate_instance(inst, patch_text, args.work_base)
        if res:
            results.append(res)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[+] Wrote {len(results)} results to {args.out}")

    if results:
        avg_solve_rate = sum(r["solve_rate"] for r in results) / len(results)
        avg_newly_created = sum(r["newly_created_errors"] for r in results) / len(results)

        solved_counter = Counter()
        unsolved_counter = Counter()
        newly_created_counter = Counter()

        for r in results:
            solved_counter.update(r.get("solved_symbol_counts", {}))
            unsolved_counter.update(r.get("unsolved_symbol_counts", {}))
            newly_created_counter.update(r.get("newly_created_symbol_counts", {}))

        global_summary = {
            "total_instances": len(results),
            "average_solve_rate": round(avg_solve_rate, 3),
            "average_newly_created_errors": round(avg_newly_created, 3),
            "solved_symbol_counts": dict(solved_counter),
            "unsolved_symbol_counts": dict(unsolved_counter),
            "newly_created_symbol_counts": dict(newly_created_counter),
        }

        summary_path = Path(args.out).with_name("global_summary.json")
        with open(summary_path, "w") as f:
            json.dump(global_summary, f, indent=2)

        print(f"[+] Wrote global summary to {summary_path}")

    print(f"[+] Cleaning up work-base {args.work_base}", file=sys.stderr)
    shutil.rmtree(args.work_base, ignore_errors=True)


if __name__ == "__main__":
    main()
