#!/usr/bin/env python3
"""
pmd_runner.py

Run PMD on a repo for style review and produce:
 - original_style_errors.json
 - original_style_report.json

Usage:
 python pmd_runner.py --org <org> --repo <repo> --pull <pull_number> --base-commit <base_commit>

This version:
 - collects all .java files and writes a temporary --file-list for PMD (so only .java files are analyzed)
 - tries modern PMD 'pmd check' with --force-language, and falls back to legacy 'pmd -d ... -language'
 - accepts PMD exit codes 0 and 4 (4 means violations found)
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List
from shutil import which

def run_cmd(cmd: List[str], cwd: Path = None, capture_output: bool = False):
    proc = subprocess.run(cmd, cwd=(cwd or None),
                          stdout=(subprocess.PIPE if capture_output else None),
                          stderr=(subprocess.PIPE if capture_output else None),
                          text=True)
    return proc

def is_dir_nonempty(path: Path) -> bool:
    return path.exists() and path.is_dir() and any(path.iterdir())

def ensure_parent(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def clone_repo(git_url: str, target_dir: Path, base_commit: str):
    if target_dir.exists() and any(target_dir.iterdir()):
        raise RuntimeError(f"Target directory {target_dir} already exists and is non-empty.")
    ensure_parent(target_dir.parent)
    print(f"Cloning {git_url} into {target_dir} ...")
    run_cmd(["git", "clone", git_url, str(target_dir)])
    try:
        run_cmd(["git", "fetch", "--all"], cwd=target_dir)
    except Exception as e:
        print("Warning: git fetch --all failed:", e)
    print(f"Checking out {base_commit} ...")
    run_cmd(["git", "checkout", base_commit], cwd=target_dir)

def find_java_files(repo_dir: Path) -> List[Path]:
    java_files = []
    for root, _, files in os.walk(repo_dir):
        for f in files:
            if f.endswith(".java"):
                java_files.append(Path(root) / f)
    return java_files

def detect_pmd_presence() -> bool:
    return which("pmd") is not None

def try_pmd_with_variants(repo_dir: Path, file_list_path: Path, report_json_path: Path, ruleset: str) -> subprocess.CompletedProcess:
    """
    Try multiple PMD invocations until one produces a report or returns acceptable exit code.
    Returns the CompletedProcess of the successful invocation (or raises).
    Acceptable returncodes: 0 (no violations) or 4 (violations found).
    """
    candidates = []

    # Modern 'pmd check' variant using --file-list and --force-language
    candidates.append([
        "pmd", "check",
        "--file-list", str(file_list_path),
        "-f", "json",
        "-r", str(report_json_path),
        "--force-language", "java",
        "-R", ruleset
    ])

    # Modern: include -d too (some installs accept both)
    candidates.append([
        "pmd", "check",
        "-d", str(repo_dir),
        "--file-list", str(file_list_path),
        "-f", "json",
        "-r", str(report_json_path),
        "--force-language", "java",
        "-R", ruleset
    ])

    # Legacy form that older PMD versions accept (use file list)
    candidates.append([
        "pmd",
        "-d", str(repo_dir),
        "--file-list", str(file_list_path),
        "-f", "json",
        "-r", str(report_json_path),
        "-language", "java",
        "-R", ruleset
    ])

    # Legacy without file-list (as a fallback) - uses dir scan but we prefer file-list
    candidates.append([
        "pmd",
        "-d", str(repo_dir),
        "-R", ruleset,
        "-f", "json",
        "-r", str(report_json_path),
        "-language", "java"
    ])

    last_proc = None
    for cmd in candidates:
        try:
            print("Trying PMD command:", " ".join(cmd))
            proc = run_cmd_capture(cmd)
            last_proc = proc
            # Accept 0 or 4 as success; also accept if report_json_path was created
            if proc.returncode in (0, 4) or report_json_path.exists():
                print(f"PMD command finished with return code {proc.returncode}")
                if proc.stdout:
                    print("PMD stdout (first 400 chars):\n", (proc.stdout[:400] + ("..." if len(proc.stdout) > 400 else "")))
                if proc.stderr:
                    print("PMD stderr (first 400 chars):\n", (proc.stderr[:400] + ("..." if len(proc.stderr) > 400 else "")))
                # If PMD wrote JSON to stdout but not to file, capture it
                if not report_json_path.exists() and proc.stdout and (proc.stdout.strip().startswith("{") or proc.stdout.strip().startswith("[")):
                    try:
                        report_json_path.write_text(proc.stdout)
                        print("Saved PMD JSON from stdout into", report_json_path)
                    except Exception as e:
                        print("Failed to write PMD stdout to file:", e)
                return proc
            else:
                # Not acceptable; print why and try next
                print(f"PMD returned code {proc.returncode}. stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
        except FileNotFoundError:
            raise RuntimeError("`pmd` not found on PATH. Install PMD or ensure it's accessible.")
        except Exception as e:
            print("PMD invocation raised exception:", e)
            last_proc = None
            # try next candidate

    # If we exit loop without success, raise with diagnostics
    raise RuntimeError("All PMD invocation variants failed. Last process result:\n"
                       f"proc: {last_proc}\n"
                       f"Ensure PMD is installed and supports JSON output and the flags attempted.")

def run_cmd_capture(cmd: List[str]):
    """Run command and capture stdout/stderr, returning CompletedProcess (not raising)."""
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def normalize_pmd_json_and_write_outputs(pmd_report_path: Path, out_errors_path: Path, out_summary_path: Path, repo_dir: Path):
    if not pmd_report_path.exists():
        print("No PMD report found at", pmd_report_path)
        out_errors_path.write_text(json.dumps([], indent=2))
        out_summary_path.write_text(json.dumps({"global_score": 10.0, "total_errors": 0, "total_warnings": 0}, indent=2))
        return

    txt = pmd_report_path.read_text()
    try:
        raw = json.loads(txt)
    except Exception as e:
        raise RuntimeError(f"Failed to parse PMD JSON at {pmd_report_path}: {e}\nStart:\n{txt[:400]}")

    files = raw.get("files") or raw.get("fileReports") or []
    # fallback: top-level "violations"
    if not files and isinstance(raw, dict) and "violations" in raw:
        grouped = {}
        for v in raw["violations"]:
            fname = v.get("filename") or v.get("file") or v.get("fileName") or "unknown"
            grouped.setdefault(fname, []).append(v)
        files = [{"filename": k, "violations": vv} for k, vv in grouped.items()]

    results = []
    total_errors = 0
    file_scores = []

    for f in files:
        filename = f.get("filename") or f.get("file") or f.get("fileName")
        if not filename:
            continue
        abs_path = Path(filename)
        if not abs_path.is_absolute():
            abs_path = (repo_dir / filename).resolve()
        else:
            abs_path = abs_path.resolve()

        violations = f.get("violations") or []
        msgs = []
        for v in violations:
            line = v.get("beginline") or v.get("line") or 0
            column = v.get("begincolumn") or v.get("column") or 0
            message = v.get("message") or v.get("description") or ""
            source = v.get("rule") or v.get("ruleName") or v.get("ruleId") or v.get("ruleset") or "PMD"
            msgs.append({
                "line": int(line) if line is not None else 0,
                "column": int(column) if column is not None else 0,
                "type": "error",
                "message": message,
                "source": source
            })
        error_count = len(msgs)
        total_errors += error_count
        score = 0.0 if error_count > 0 else 10.0
        file_scores.append(score)
        results.append({
            "file": str(abs_path),
            "score": float(score),
            "error_count": error_count,
            "messages": msgs
        })

    if not results:
        java_files = find_java_files(repo_dir)
        if java_files:
            for jf in java_files:
                results.append({
                    "file": str(jf.resolve()),
                    "score": 10.0,
                    "error_count": 0,
                    "messages": []
                })
            total_errors = 0
            file_scores = [10.0] * len(java_files)

    global_score = round(sum(file_scores) / len(file_scores), 2) if file_scores else 10.0
    summary = {"global_score": global_score, "total_errors": total_errors, "total_warnings": 0}

    out_errors_path.write_text(json.dumps(results, indent=2))
    out_summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote:\n - {out_errors_path}\n - {out_summary_path}")

def main():
    parser = argparse.ArgumentParser(description="Run PMD on a GitHub repo for style review")
    parser.add_argument("--org", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--pull", required=True)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--git-url", default=None)
    parser.add_argument("--ruleset", default="rulesets/java/quickstart.xml")
    args = parser.parse_args()

    base_out_dir = Path("data") / "java_style_review" / args.org / args.repo / "style_review" / f"style-review-{args.pull}"
    repo_dir = base_out_dir / "repo"
    ensure_parent(base_out_dir)

    if is_dir_nonempty(repo_dir):
        print(f"Repository directory {repo_dir} already exists and is non-empty. Using existing checkout.")
    else:
        git_url = args.git_url or f"https://github.com/{args.org}/{args.repo}.git"
        try:
            clone_repo(git_url, repo_dir, args.base_commit)
        except Exception as e:
            print("Failed to clone repo:", e)
            sys.exit(2)

    java_files = find_java_files(repo_dir)
    out_errors_path = base_out_dir / "original_style_errors.json"
    out_summary_path = base_out_dir / "original_style_report.json"

    if not java_files:
        print("No .java files found in repo. Writing empty outputs.")
        out_errors_path.write_text(json.dumps([], indent=2))
        out_summary_path.write_text(json.dumps({"global_score": 10.0, "total_errors": 0, "total_warnings": 0}, indent=2))
        return

    if not detect_pmd_presence():
        print("`pmd` not found on PATH. Install PMD before running this script.")
        sys.exit(3)

    # write file-list containing absolute paths to .java files (one per line)
    with tempfile.NamedTemporaryFile(mode="w", delete=False, prefix="pmd_file_list_", suffix=".txt") as tf:
        file_list_path = Path(tf.name)
        for jf in java_files:
            tf.write(str(jf.resolve()) + "\n")
    print("Wrote PMD file list to", file_list_path)

    pmd_report_path = base_out_dir / "pmd_report.json"

    try:
        proc = try_pmd_with_variants(repo_dir, file_list_path, pmd_report_path, args.ruleset)
    except Exception as e:
        # Clean up file list
        try:
            file_list_path.unlink(missing_ok=True)
        except Exception:
            pass
        print("Error running PMD:", e)
        sys.exit(4)
    finally:
        # remove file list (best-effort)
        try:
            file_list_path.unlink(missing_ok=True)
        except Exception:
            pass

    # If PMD wrote JSON to stdout and we didn't capture it earlier, try capture from proc
    if not pmd_report_path.exists() and 'proc' in locals():
        out = proc.stdout or ""
        if out.strip().startswith("{") or out.strip().startswith("["):
            pmd_report_path.write_text(out)
            print("Saved PMD JSON from stdout to", pmd_report_path)

    try:
        normalize_pmd_json_and_write_outputs(pmd_report_path, out_errors_path, out_summary_path, repo_dir)
    except Exception as e:
        print("Error normalizing PMD JSON:", e)
        sys.exit(5)

if __name__ == "__main__":
    main()
