#!/usr/bin/env python3
"""
pr_clang_tidy_review.py

Modified to require an explicit --clang-tidy-bin argument and to always use that binary.
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

def run_run_clang_tidy(run_clang_tidy_path, repo_dir, build_dir, out_txt_path, jobs=None):
    """
    Keep this helper in case you want to explicitly run a wrapper that itself is configured
    to use a specific clang-tidy. Note: when using this wrapper, the wrapper's behavior
    determines which clang-tidy binary is invoked (it may still use PATH).
    This script's guaranteed path to the clang-tidy binary is enforced by requiring
    --clang-tidy-bin and using clang_tidy_per_file below.
    """
    cmd = [run_clang_tidy_path, "-p", build_dir, "-header-filter=.*"]
    if jobs:
        cmd += ["-j", str(jobs)]
    # run and capture stdout+stderr into file
    with open(out_txt_path, "w", encoding="utf-8") as outf:
        subprocess.run(cmd, cwd=repo_dir, stdout=outf, stderr=subprocess.STDOUT, check=False)

def clang_tidy_per_file(repo_dir, build_dir, out_txt_path, clang_tidy_bin, jobs=None):
    """
    Run the explicit clang-tidy binary on each file listed in compile_commands.json.
    This function uses the `clang_tidy_bin` argument exclusively (no PATH lookups).
    """
    compile_db = Path(build_dir) / "compile_commands.json"
    if not compile_db.exists():
        raise FileNotFoundError(f"compile_commands.json missing at {compile_db}. Can't run clang-tidy per-file.")

    with open(compile_db, "r", encoding="utf-8") as f:
        cb = json.load(f)

    # collect unique files
    files = sorted({os.path.abspath(entry["file"]) for entry in cb})

    # verify clang_tidy_bin again
    if not clang_tidy_bin:
        raise FileNotFoundError("No clang-tidy binary provided.")
    if not os.path.exists(clang_tidy_bin) and not shutil.which(clang_tidy_bin):
        raise FileNotFoundError(f"clang-tidy binary not found: {clang_tidy_bin}")

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
def resolve_executable(name_or_path):
    """
    Resolve an executable given either an absolute/relative path or a name in PATH.
    Returns the resolved absolute path or None if not found.
    """
    # Try shutil.which first (works for names and absolute paths)
    found = shutil.which(name_or_path)
    if found:
        return found
    # If that didn't find it, check if it's a path that exists
    p = Path(name_or_path)
    if p.exists():
        return str(p.resolve())
    return None

def main():
    ap = argparse.ArgumentParser(description="Run clang-tidy style review on a PR and emit JSON summary (requires explicit clang-tidy binary).")
    ap.add_argument("--repo-url", required=True, help="Git repo clone URL (https://github.com/owner/repo.git)")
    ap.add_argument("--pr", required=True, type=int, help="PR number to fetch & checkout")
    ap.add_argument("--clang-tidy-config", required=False, help="Path to .clang-tidy config (will be copied into repo root)")
    ap.add_argument("--clang-tidy-bin", required=True, help="Explicit path or executable name for clang-tidy (required)")
    ap.add_argument("--out", default="results.json", help="Output JSON file (summary)")
    ap.add_argument("--work-dir", default=None, help="Optional working dir (defaults to tempdir)")
    ap.add_argument("--jobs", type=int, default=None, help="Parallel jobs for clang-tidy (default: cpu count)")
    # Note: we keep run-clang-tidy wrapper support only if explicitly provided; otherwise we always
    # run the explicit clang-tidy binary per-file to guarantee the binary used is the one specified.
    ap.add_argument("--run-clang-tidy-bin", required=False, help="Optional explicit run-clang-tidy wrapper to use (if provided, wrapper will be used; otherwise explicit clang-tidy will run per-file).")

    args = ap.parse_args()

    # Resolve clang-tidy binary (must be provided and resolvable)
    resolved_clang_tidy = resolve_executable(args.clang_tidy_bin)
    if not resolved_clang_tidy:
        print(f"[ERROR] clang-tidy binary not found: {args.clang_tidy_bin}", file=sys.stderr)
        sys.exit(2)
    if not os.access(resolved_clang_tidy, os.X_OK):
        print(f"[ERROR] clang-tidy binary is not executable: {resolved_clang_tidy}", file=sys.stderr)
        sys.exit(3)

    resolved_run_wrapper = None
    if args.run_clang_tidy_bin:
        resolved_run_wrapper = resolve_executable(args.run_clang_tidy_bin)
        if not resolved_run_wrapper:
            print(f"[ERROR] run-clang-tidy wrapper not found: {args.run_clang_tidy_bin}", file=sys.stderr)
            sys.exit(4)
        if not os.access(resolved_run_wrapper, os.X_OK):
            print(f"[ERROR] run-clang-tidy wrapper is not executable: {resolved_run_wrapper}", file=sys.stderr)
            sys.exit(5)

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

        # run clang-tidy: require explicit clang-tidy binary; optionally use provided run-clang-tidy wrapper
        out_txt = str(work_base / "clang-tidy.txt")
        if resolved_run_wrapper:
            print(f"[+] Running wrapper {resolved_run_wrapper} (note: wrapper may choose clang-tidy itself) ...", file=sys.stderr)
            run_run_clang_tidy(resolved_run_wrapper, str(repo_dir), str(build_dir), out_txt, jobs=args.jobs)
        else:
            print(f"[+] Running clang-tidy per-file using explicit binary: {resolved_clang_tidy} ...", file=sys.stderr)
            clang_tidy_per_file(str(repo_dir), str(build_dir), out_txt, clang_tidy_bin=resolved_clang_tidy, jobs=args.jobs)

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
            # remove temp dir to be tidy. Comment the next line if you want to keep it for inspection.
            try:
                shutil.rmtree(work_base)
            except Exception:
                pass

if __name__ == "__main__":
    main()
