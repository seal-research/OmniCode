import os
import json
import subprocess
from pathlib import Path
import tempfile
import argparse
import sys

# Paths
JSON_PATH = "data/multiswebench_data/mswebench_instances.json"
BASE_DIR = Path("data/java_style_review")


def run_cmd(cmd, cwd=None):
    print(f"[CMD] {' '.join(cmd)} (cwd={cwd})")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERR] Command failed: {result.stderr}")
        raise RuntimeError(result.stderr)
    return result.stdout

def ensure_repo_cloned(org, repo, pull_number, base_commit):
    repo_dir = BASE_DIR / org / repo / "style_review" / f"style-review-{pull_number}" / "repo"
    if not repo_dir.exists():
        print(f"[INFO] Cloning repo {org}/{repo} at commit {base_commit}")
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        run_cmd([
            "git", "clone",
            f"https://github.com/{org}/{repo}.git",
            str(repo_dir)
        ])
        run_cmd(["git", "checkout", base_commit], cwd=repo_dir)
    else:
        print(f"[INFO] Repo already exists: {repo_dir}")
    return repo_dir

def apply_patch(repo_dir, patch_str):
    print(f"[INFO] Applying patch in {repo_dir}")
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp_patch:
        tmp_patch.write(patch_str)
        tmp_patch_path = tmp_patch.name
    try:
        run_cmd(["git", "apply", tmp_patch_path], cwd=repo_dir)
        print("[INFO] Patch applied successfully")
    except RuntimeError:
        print("[WARN] Patch failed to apply, attempting with --3way")
        run_cmd(["git", "apply", "--3way", tmp_patch_path], cwd=repo_dir)
    finally:
        os.remove(tmp_patch_path)

def apply_patch_from_file(repo_dir, patch_file_path):
    if not patch_file_path.exists():
        print(f"[WARN] Sweagent patch file does not exist: {patch_file_path}")
        return
    print(f"[INFO] Applying sweagent patch from file {patch_file_path}")
    try:
        run_cmd(["git", "apply", str(patch_file_path)], cwd=repo_dir)
        print("[INFO] Sweagent patch applied successfully")
    except RuntimeError:
        print("[WARN] Sweagent patch failed to apply, attempting with --3way")
        run_cmd(["git", "apply", "--3way", str(patch_file_path)], cwd=repo_dir)

def main():
    parser = argparse.ArgumentParser(description="Apply gold patch and sweagent patch for specific org/repo/pr_number")
    parser.add_argument("org", help="Organization name (e.g., alibaba)")
    parser.add_argument("repo", help="Repository name (e.g., fastjson2)")
    parser.add_argument("pr_number", type=int, help="Pull request number")
    args = parser.parse_args()

    with open(JSON_PATH, "r") as f:
        instances = json.load(f)

    target_entry = None
    for inst in instances:
        if inst["repo"] == f"{args.org}/{args.repo}" and inst["pull_number"] == args.pr_number:
            target_entry = inst
            break

    if not target_entry:
        print(f"[ERR] No entry found for {args.org}/{args.repo} PR {args.pr_number}")
        sys.exit(1)

    repo_dir = ensure_repo_cloned(args.org, args.repo, args.pr_number, target_entry["base_commit"])
    

    # Compose sweagent patch path
    # Pattern: sweagent_pmd_org_repo_pr_results/org/repo:pr/org/repo:pr.patch
    SWEAGENT_PATCH_BASE = Path(f"sweagent_pmd_{args.org}_{args.repo}_{args.pr_number}_results")

    sweagent_patch_path = Path("/home/debjitd/OmniCode/patch1.patch")#SWEAGENT_PATCH_BASE / args.org / f"{args.repo}:{args.pr_number}" / f"{args.org}/{args.repo}:{args.pr_number}.patch"
    apply_patch_from_file(repo_dir, sweagent_patch_path)
    apply_patch(repo_dir, target_entry["patch"])

if __name__ == "__main__":
    main()
