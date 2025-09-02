#!/usr/bin/env python3
"""
pr_clang_tidy_review.py

Usage:
  python multiswebench_local/multi_swe_bench/harness/CPP_style_review/style_reviewcpp.py --repo-url https://github.com/catchorg/Catch2.git --pr 2849 --clang-tidy-config /path/to/.clang-tidy --out results.json

What it does:
 - clones repo
 - checks out PR (fetches pull/<PR>/head)
 - copies .clang-tidy if provided
 - runs cmake to produce compile_commands.json
 - runs run-clang-tidy (or clang-tidy per-file fallback)
 - parses clang-tidy output
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
from multiprocessing.pool import ThreadPool
from pathlib import Path

# ---------- Helpers ----------
def run(cmd, cwd=None, capture=False, env=None, check=True):
    if capture:
        res = subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if check and res.returncode != 0:
            raise subprocess.CalledProcessError(res.returncode, cmd, res.stdout)
        return res.stdout
    else:
        subprocess.run(cmd, cwd=cwd, env=env, check=check)

def git_checkout_pr(repo_dir, pr):
    # safe fetch + checkout style that works on GitHub
    run(["git", "fetch", "origin", f"pull/{pr}/head:pr/{pr}"], cwd=repo_dir)
    run(["git", "checkout", f"pr/{pr}"], cwd=repo_dir)

def ensure_cmake_build(repo_dir, build_dir):
    os.makedirs(build_dir, exist_ok=True)
    # generate compile commands
    run(["cmake", "-S", repo_dir, "-B", build_dir, "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", "-DCMAKE_BUILD_TYPE=Debug"])
    # Build (fast) to ensure compile_commands is generated for complex projects; users can override by prebuilt build
    try:
        run(["cmake", "--build", build_dir, "--", "-j", str(os.cpu_count() or 2)])
    except subprocess.CalledProcessError:
        # build may fail for some projects; compile_commands.json is often created even if build fails
        pass

def find_run_clang_tidy():
    # check common names
    for name in ("run-clang-tidy", "run-clang-tidy.py"):
        path = shutil.which(name)
        if path:
            return path
    return None

def run_run_clang_tidy(run_clang_tidy_path, repo_dir, build_dir, out_txt_path, jobs=None):
    cmd = [run_clang_tidy_path, "-p", build_dir, "-header-filter=.*"]
    if jobs:
        cmd += ["-j", str(jobs)]
    # run and capture stdout+stderr into file
    with open(out_txt_path, "w", encoding="utf-8") as outf:
        subprocess.run(cmd, cwd=repo_dir, stdout=outf, stderr=subprocess.STDOUT, check=False)

def clang_tidy_per_file(repo_dir, build_dir, out_txt_path, jobs=None):
    compile_db = Path(build_dir) / "compile_commands.json"
    if not compile_db.exists():
        raise FileNotFoundError(f"compile_commands.json missing at {compile_db}. Can't run clang-tidy per-file.")

    with open(compile_db, "r", encoding="utf-8") as f:
        cb = json.load(f)

    # collect unique files
    files = sorted({os.path.abspath(entry["file"]) for entry in cb})
    clang_tidy_bin = shutil.which("clang-tidy")
    if not clang_tidy_bin:
        raise FileNotFoundError("clang-tidy not found in PATH.")

    # define worker to run clang-tidy for a file
    def worker(src):
        # clang-tidy respects -p build directory
        try:
            out = subprocess.run([clang_tidy_bin, "-p", build_dir, src], cwd=repo_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            return out.stdout
        except Exception as e:
            return f"ERROR running clang-tidy on {src}: {e}\n"

    pool_size = jobs or (os.cpu_count() or 2)
    pool = ThreadPool(pool_size)
    results = pool.map(worker, files)
    pool.close()
    pool.join()
    # write concatenated output
    with open(out_txt_path, "w", encoding="utf-8") as f:
        for chunk in results:
            f.write(chunk)
            if not chunk.endswith("\n"):
                f.write("\n")

# ---------- Parsing clang-tidy output ----------
CLANG_LINE_RE = re.compile(r'^(.*?):(\d+):(\d+):\s+(warning|error|note):\s+(.*?)\s+\[(.*?)\]\s*$')

def parse_clang_tidy_output(txt_path):
    """
    Convert the textual clang-tidy output into a dictionary shaped like:
    {
      "label": "org/repo:pr-123",
      "files": [
         {
            "file": "/abs/path/file.cpp",
            "score": 9.0,
            "error_count": 1,
            "messages": [ {line, column, type, message, source} ... ]
         },
         ...
      ],
      "overview": { "global_score": X, "total_errors": N, "total_warnings": M, "total_files": K }
    }
    """
    file_map = {}
    with open(txt_path, "r", errors="ignore", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            m = CLANG_LINE_RE.match(line.strip())
            if not m:
                continue
            path, line_no, col, sev, msg, checker = m.groups()
            if sev == "note":
                # skip notes for summary (they're usually explanatory)
                continue
            rec = file_map.setdefault(path, {"file": path, "score": 0.0, "error_count": 0, "messages": []})
            if sev == "error":
                rec["error_count"] += 1
            rec["messages"].append({
                "line": int(line_no),
                "column": int(col),
                "type": "error" if sev == "error" else "warning",
                "message": msg,
                "source": checker
            })

    files = []
    total_errors = 0
    total_warnings = 0
    for path, rec in file_map.items():
        errs = sum(1 for m in rec["messages"] if m["type"] == "error")
        warns = sum(1 for m in rec["messages"] if m["type"] == "warning")
        total_errors += errs
        total_warnings += warns
        # simple score: start from 10 and subtract errors/10 (clamped)
        score = max(0.0, 10.0 - (errs / 10.0))
        rec["score"] = round(score, 2)
        files.append(rec)

    # global score is mean of file scores (or 10 if no files)
    global_score = round((sum(f["score"] for f in files) / len(files)) if files else 10.0, 2)
    overview = {
        "global_score": global_score,
        "total_errors": total_errors,
        "total_warnings": total_warnings,
        "total_files": len(files)
    }

    return files, overview

# ---------- Main flow ----------
def main():
    ap = argparse.ArgumentParser(description="Run clang-tidy style review on a PR and emit JSON summary.")
    ap.add_argument("--repo-url", required=True, help="Git repo clone URL (https://github.com/owner/repo.git)")
    ap.add_argument("--pr", required=True, type=int, help="PR number to fetch & checkout")
    ap.add_argument("--clang-tidy-config", required=False, help="Path to .clang-tidy config (will be copied into repo root)")
    ap.add_argument("--out", default="results.json", help="Output JSON file (summary)")
    ap.add_argument("--work-dir", default=None, help="Optional working dir (defaults to tempdir)")
    ap.add_argument("--jobs", type=int, default=None, help="Parallel jobs for clang-tidy (default: cpu count)")
    args = ap.parse_args()

    work_base = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="clang_tidy_review_"))
    repo_dir = work_base / "repo"
    build_dir = work_base / "build"

    print(f"[+] working dir: {work_base}", file=sys.stderr)
    try:
        # clone
        print("[+] Cloning repo...", file=sys.stderr)
        run(["git", "clone", args.repo_url, str(repo_dir)])
        # checkout pr
        print(f"[+] Fetching and checking out PR {args.pr}...", file=sys.stderr)
        git_checkout_pr(str(repo_dir), args.pr)

        # copy .clang-tidy if provided
        if args.clang_tidy_config:
            cfg_src = Path(args.clang_tidy_config)
            if cfg_src.exists():
                dst = Path(repo_dir) / ".clang-tidy"
                shutil.copy(cfg_src, dst)
                print(f"[+] Copied {cfg_src} -> {dst}", file=sys.stderr)
            else:
                print(f"[!] Provided clang-tidy config not found: {cfg_src}", file=sys.stderr)

        # ensure CMake build + compile_commands
        print("[+] Running CMake to create compile_commands.json (may take a while)...", file=sys.stderr)
        ensure_cmake_build(str(repo_dir), str(build_dir))

        # run clang-tidy: prefer run-clang-tidy wrapper if available
        out_txt = str(work_base / "clang-tidy.txt")
        rct = find_run_clang_tidy()
        if rct:
            print(f"[+] Running {rct} (parallelized) ...", file=sys.stderr)
            run_run_clang_tidy(rct, str(repo_dir), str(build_dir), out_txt, jobs=args.jobs)
        else:
            print("[!] run-clang-tidy not found; falling back to per-file clang-tidy (slower).", file=sys.stderr)
            clang_tidy_per_file(str(repo_dir), str(build_dir), out_txt, jobs=args.jobs)

        print(f"[+] Parsing clang-tidy output at {out_txt} ...", file=sys.stderr)
        files, overview = parse_clang_tidy_output(out_txt)

        # label from repo url and pr
        repo_name = Path(args.repo_url).stem
        owner = None
        # try to parse owner from URL like github.com/owner/repo.git
        m = re.search(r'[:/](?P<owner>[^/]+)/' + re.escape(repo_name), args.repo_url)
        if m:
            owner = m.group("owner")
        label = f"{owner or 'unknown'}/{repo_name}:pr-{args.pr}"

        obj = {
            "label": label,
            "files": files,
            "overview": overview
        }

        # write output
        out_path = Path(args.out)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)

        # also print summary to stdout
        print(json.dumps(obj, indent=2))
        print(f"[+] Done. Results written to {out_path}", file=sys.stderr)

    finally:
        if args.work_dir is None:
            # keep temp dir for inspection? remove it to be tidy. Comment the next line if you want to keep it.
            try:
                shutil.rmtree(work_base)
            except Exception:
                pass

if __name__ == "__main__":
    main()
