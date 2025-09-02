import argparse
import json
import subprocess
import shutil
import os
import re
from pathlib import Path
import sys

SAFE_NAME_RE = re.compile(r'^[A-Za-z0-9._\-]+$')  # repo/org/pr must match this

def _safe_name(name: str):
    if not SAFE_NAME_RE.match(name):
        raise ValueError(f"Unsafe name: {name!r}")
    return name

def load_json_or_jsonl(path: Path):
    """
    Load a file that may be a JSON array or JSON Lines (JSONL).
    Returns a list of objects.
    """
    text = path.read_text(encoding="utf-8")
    s = text.lstrip()
    if not s:
        return []
    if s[0] == "[":
        return json.loads(text)
    else:
        objs = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            objs.append(json.loads(line))
        return objs

def load_instances(instances_path: Path):
    return load_json_or_jsonl(instances_path)

def find_base_commit(instances, org: str, repo: str, pr_number: int):
    """Find base_commit for matching repo and pull_number. repo field is expected as 'org/repo'."""
    target_repo_field = f"{org}/{repo}"
    for entry in instances:
        entry_repo = entry.get("repo")
        entry_pull = entry.get("pull_number") if "pull_number" in entry else entry.get("pull_number")
        try:
            entry_pull_int = int(entry_pull) if entry_pull is not None else None
        except Exception:
            entry_pull_int = None

        if entry_repo == target_repo_field and entry_pull_int == pr_number:
            base = entry.get("base_commit")
            if base:
                return base
    return None

def run_commands(org, repo, pr_number, base_commit, workdir: str = "."):
    # basic validation
    org = _safe_name(org)
    repo = _safe_name(repo)
    pr_number = _safe_name(pr_number)
    # base_commit can be any hash/branch, but ensure not empty
    if not base_commit:
        raise ValueError("base_commit is empty")

    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    repo_dir = workdir / repo
    # If repo already exists, remove it to ensure a clean clone
    if repo_dir.exists():
        print(f"[INFO] Removing existing repo dir: {repo_dir}")
        shutil.rmtree(repo_dir)

    git_url = f"https://github.com/{org}/{repo}.git"

    try:
        # clone
        print(f"[INFO] Cloning {git_url} into {workdir}")
        subprocess.run(["git", "clone", git_url], check=True, cwd=workdir)

        # fetch PR into local branch pr-<pr_number>
        print(f"[INFO] Fetching PR {pr_number} into branch pr-{pr_number}")
        subprocess.run(
            ["git", "fetch", "origin", f"pull/{pr_number}/head:pr-{pr_number}"],
            check=True,
            cwd=repo_dir
        )

        # checkout base commit/branch
        print(f"[INFO] Checking out base commit/branch: {base_commit}")
        subprocess.run(["git", "checkout", base_commit], check=True, cwd=repo_dir)

        # build patch path (expand ~)
        patch_path = Path(os.path.expanduser(
            f"~/OmniCode/server/sweagent_pmd_{org}_{repo}_{pr_number}_results/{org}__{repo}_{pr_number}/{org}__{repo}_{pr_number}.patch"
        ))

        if not patch_path.exists():
            raise FileNotFoundError(f"Patch file not found: {patch_path}")

        print(f"[INFO] Applying patch: {patch_path}")
        subprocess.run(["git", "apply", str(patch_path)], check=True, cwd=repo_dir)

        # prepare destination path and copy
        dst = workdir / "data" / "java_style_review" / org / repo / "style_review" / f"style-review-{pr_number}" / "repo"
        if dst.exists():
            print(f"[INFO] Removing existing destination: {dst}")
            shutil.rmtree(dst)

        print(f"[INFO] Copying repo from {repo_dir} -> {dst}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(repo_dir, dst)

        print("[SUCCESS] Done run_commands.")
        return True

    except subprocess.CalledProcessError as e:
        print("[ERROR] Command failed:", e)
        raise
    except Exception:
        raise

def execute_style_review(org, repo, pr_number):
    """
    Run the style review using the same Python interpreter (safer).
    Uses list form for subprocess to avoid shell injection.
    """
    instance_id = f"{org}/{repo}:{pr_number}"
    cmd = [
        sys.executable, "codearena.py",
        "--StyleReview",
        "--predictions_path", "gold",
        "--run_id", "mswe_java_style_review",
        "--max_workers", "1",
        "--instance_ids", instance_id,
        "--mswe_phase", "all",
        "--force_rebuild", "True",
        "--review_type", "pmd",
    ]
    print("[INFO] Executing style review:", " ".join(cmd))
    subprocess.run(cmd, check=True)

# -------------------------
# New helpers for comparison
# -------------------------
def gather_error_tuples_from_original(original_path: Path):
    """
    Load original_style_errors.json (or JSON array) which is a list of file dicts:
      [{ "file": "...", "score": ..., "error_count": N, "messages": [ {line, column, type, message, source}, ... ] }, ...]
    Return set of tuples: (filename, source, line, column, message)
    """
    objs = load_json_or_jsonl(original_path)
    s = set()
    for fileobj in objs:
        filename = fileobj.get("file") or fileobj.get("filename") or ""
        messages = fileobj.get("messages") or []
        for m in messages:
            line = m.get("line")
            column = m.get("column")
            source = m.get("source") or m.get("rule") or ""
            message = m.get("message") or ""
            s.add((filename, source, int(line) if line is not None else None, int(column) if column is not None else None, message))
    return s

def gather_error_tuples_from_filtered_entry(filtered_entry: dict):
    """
    filtered_entry is expected to be a single entry from unique_results_pmd.jsonl with keys like:
    { "label": "apache/dubbo:pr-10638", "overview": {...}, "files": [ { "file": "...", "score":..., "error_count":..., "messages":[{line,column,type,message,source}, ...] }, ... ] }
    Return set of tuples: (filename, source, line, column, message)
    """
    s = set()
    files = filtered_entry.get("files") or []
    for f in files:
        filename = f.get("file") or f.get("filename") or ""
        messages = f.get("messages") or []
        for m in messages:
            line = m.get("line")
            column = m.get("column")
            source = m.get("source") or m.get("rule") or ""
            message = m.get("message") or ""
            s.add((filename, source, int(line) if line is not None else None, int(column) if column is not None else None, message))
    return s

def check_style_errors(org: str, repo: str, pr_number: str, workdir: Path, filtered_results_path: Path = None):
    """
    Compare original errors (from data/java_style_review/<org>/<repo>/style_review/style-review-<pr_number>/original_style_errors.json)
    with post-run results in unique_results_pmd.jsonl and report unresolved errors.

    RETURNS:
      (remaining_original_count, extra_filtered_count)

      - remaining_original_count: number of original errors NOT present in filtered_results (i.e., unresolved original errors)
      - extra_filtered_count: number of errors present in filtered_results but NOT in original (i.e., filtered-only)
    """
    # locate original file
    original_path = (workdir / "data" / "java_style_review" / org / repo / "style_review" / f"style-review-{pr_number}" / "original_style_errors.json")
    if not original_path.exists():
        print(f"[ERROR] original_style_errors file not found at: {original_path}")
        return (1, 0)  # treat as unresolved

    # locate unique_results_pmd.jsonl
    if filtered_results_path is None:
        filtered_results_path = workdir / "unique_results_pmd.jsonl"
    if not filtered_results_path.exists():
        print(f"[ERROR] unique_results_pmd.jsonl not found at: {filtered_results_path}")
        return (1, 0)

    # load filtered results and find the matching label
    filtered_entries = load_json_or_jsonl(filtered_results_path)
    target_label = f"{org}/{repo}:pr-{pr_number}"
    matched = None
    for e in filtered_entries:
        if e.get("label") == target_label:
            matched = e
            #break

    if matched is None:
        print(f"[ERROR] No entry with label {target_label} found in {filtered_results_path}")
        return (1, 0)

    # build sets
    original_set = gather_error_tuples_from_original(original_path)
    filtered_set = gather_error_tuples_from_filtered_entry(matched)

    # keyed sets by (file, source, line, column)
    orig_keyed = {(f, src, ln, col) for (f, src, ln, col, msg) in original_set}
    filt_keyed = {(f, src, ln, col) for (f, src, ln, col, msg) in filtered_set}

    remaining_orig_keys = orig_keyed - filt_keyed          # original but not in filtered (unresolved original errors)
    extra_filtered_keys = filt_keyed - orig_keyed          # in filtered but not in original (filtered-only)

    print(f"[INFO] Original unique errors: {len(orig_keyed)}")
    print(f"[INFO] Unique errors: {len(filt_keyed)}")
    print(f"[INFO] Extra errors in filtered (not in original): {len(extra_filtered_keys)}")

    if not orig_keyed:
        print("[INFO] No original style errors found (original file empty).")

    if remaining_orig_keys:
        # Print details of remaining original errors
        print("[WARN] The following original errors remain unresolved:")
        '''msg_map = {}
        for (f, src, ln, col, msg) in original_set:
            key = (f, src, ln, col)
            msg_map.setdefault(key, []).append(msg)
        for key in sorted(remaining_orig_keys):
            msgs = msg_map.get(key, ["(no message)"])
            file_path, source, line, column = key
            for m in msgs:
                print(f" - file={file_path} source={source} line={line} column={column} msg={m}")'''

    # Also optionally print details of extra filtered-only errors (brief)
    if extra_filtered_keys:
        print("[INFO] Note: There are errors present in filtered_results that were not in the original errors.")
        # don't spam full list here by default; print count only. If you want details, we can print them.

    # Return the tuple as documented above
    return (len(remaining_orig_keys), len(extra_filtered_keys))

# -------------------------
# main
# -------------------------
def main():
    p = argparse.ArgumentParser(description="Clone repo, apply patch, copy, run style review, and verify errors resolved.")
    p.add_argument("org", help="GitHub org (e.g. apache)")
    p.add_argument("repo", help="Repository name (e.g. dubbo)")
    p.add_argument("pr_number", help="Pull request number (e.g. 10638)")
    p.add_argument("--workdir", default=".", help="Working directory (default: current dir)")
    p.add_argument("--instances", default="data/multiswebench_data/mswebench_instances.json",
                   help="Path to instances JSONL/JSON file")
    p.add_argument("--filtered_results", default=None,
                   help="Path to unique_results_pmd.jsonl (defaults to <workdir>/unique_results_pmd.jsonl)")
    args = p.parse_args()

    try:
        org = _safe_name(args.org)
        repo = _safe_name(args.repo)
        pr_number_str = _safe_name(args.pr_number)
        pr_number_int = int(pr_number_str)
    except Exception as e:
        print("[ERROR] Invalid input:", e)
        sys.exit(2)

    instances_path = Path(args.instances)
    if not instances_path.exists():
        print(f"[ERROR] Instances file not found: {instances_path}")
        sys.exit(3)

    try:
        instances = load_instances(instances_path)
    except Exception as e:
        print(f"[ERROR] Failed to load instances file: {e}")
        sys.exit(4)

    base_commit = find_base_commit(instances, org, repo, pr_number_int)
    if not base_commit:
        print(f"[ERROR] Could not find base_commit for {org}/{repo}#{pr_number_int} in {instances_path}")
        sys.exit(5)

    print(f"[INFO] Found base_commit: {base_commit}")

    workdir = Path(args.workdir).resolve()

    # run the operations
    try:
        run_commands(org, repo, pr_number_str, base_commit, workdir=args.workdir)
        execute_style_review(org, repo, pr_number_str)
    except subprocess.CalledProcessError as e:
        print("[ERROR] A subprocess failed:", e)
        sys.exit(6)
    except Exception as e:
        print("[ERROR] Unhandled error:", e)
        sys.exit(7)

    # After style review, check if errors were resolved
    filtered_results_path = Path(args.filtered_results) if args.filtered_results else None
    remaining_original_count, extra_filtered_count = check_style_errors(org, repo, pr_number_str, workdir, filtered_results_path)

    # Fail if original errors remain (preserve previous strict behavior)
    if remaining_original_count > 0:
        print("[ERROR] Style review did NOT resolve all original errors.")
        sys.exit(8)

    # At this point original errors are resolved.
    # Print the extra_filtered_count (this is the number you requested)
    print(f"[INFO] Number of errors present in filtered_results but not in original: {extra_filtered_count}")

    # succeed
    print("[SUCCESS] Style review resolved all original errors.")
    sys.exit(0)

if __name__ == "__main__":
    main()
