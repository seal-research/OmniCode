#!/usr/bin/env python3
"""
compare_clang_tidy_runs_parallel.py

Parallelized, safe version of compare_clang_tidy_runs_final.py.

Usage:
  python compare_clang_tidy_runs_parallel.py \
    --org apache --repo dubbo --pr 10638 \
    --swe-file sweagent_clang-tidy_results.json \
    --style-script multiswebench_local/multi_swe_bench/harness/CPP_style_review/style_reviewcpp.py \
    --clang-config multiswebench_local/multi_swe_bench/harness/CPP_style_review/.clang-tidy \
    --max-workers 6 \
    [--write-diffs diffs_dir]

Notes:
 - Each instance is executed in parallel but writes to a unique workdir and unique results file:
     --workdir <workdir_prefix>_<instance_id>
     --out results_<instance_id>.json
 - Default concurrency is min(8, (cpu_count or 4) * 2). Tune with --max-workers.
 - Use --write-diffs <dir> to write a per-instance JSON with parsed/results/missing/additional lists for inspection.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
import os
import re
import csv
import shlex
import subprocess
import sys
import time
from datetime import datetime

# --- parsing helpers (same robust parser as before) ------------------------

KNOWN_SOURCE_EXTS_RE = re.compile(r'\.(c|cc|cpp|cxx|h|hpp|hh|ipp|inl|m|mm)\b', re.IGNORECASE)

def normalize_problem_statement_text(ps):
    if ps is None:
        return ""
    if "\\n" in ps and ps.count("\\n") >= ps.count("\n"):
        ps = ps.replace("\\r\\n", "\\n").replace("\\n", "\n")
    ps = ps.replace("\r\n", "\n").replace("\r", "\n")
    return ps

def parse_problem_statement(ps_text):
    """
    Returns (violations_list, reported_total)
    violations_list: list of (basename, line:int, column:int)
    reported_total: int or None
    """
    ps = normalize_problem_statement_text(ps_text)
    violations = []
    reported_total = None

    m1 = re.search(r'Total\s+Unique\s+Violations\s*:\s*(\d+)', ps, flags=re.IGNORECASE)
    if not m1:
        m1 = re.search(r'Violations\s*:\s*(\d+)', ps, flags=re.IGNORECASE)
    if m1:
        try:
            reported_total = int(m1.group(1))
        except Exception:
            reported_total = None

    # File blocks
    file_block_re = re.compile(r'^\s*File:\s*(?P<path>.+?)\s*\n(?P<body>.*?)(?=(?:^\s*File:\s*)|\Z)', flags=re.MULTILINE | re.DOTALL)
    blocks = list(file_block_re.finditer(ps))
    if blocks:
        for b in blocks:
            fullpath = b.group("path").strip()
            basename = os.path.basename(fullpath)
            body = b.group("body") or ""
            for lm in re.finditer(r'Line\s+(\d+)\s*,\s*Column\s+(\d+)', body, flags=re.IGNORECASE):
                try:
                    line = int(lm.group(1)); col = int(lm.group(2))
                    violations.append((basename, line, col))
                except Exception:
                    pass
        # dedupe preserving order
        seen = set(); dedup = []
        for v in violations:
            if v not in seen:
                dedup.append(v); seen.add(v)
        return dedup, reported_total

    # fallback scanning
    current_basename = None
    lines = ps.splitlines()
    for i, raw in enumerate(lines):
        line = raw.rstrip()
        if '/' in line and KNOWN_SOURCE_EXTS_RE.search(line):
            current_basename = os.path.basename(line.strip()); continue
        mfile = re.match(r'^\s*File\s*:\s*(.+)$', line, flags=re.IGNORECASE)
        if mfile:
            candidate = mfile.group(1).strip()
            current_basename = os.path.basename(candidate); continue
        mline = re.search(r'Line\s+(\d+)\s*,\s*Column\s+(\d+)', line, flags=re.IGNORECASE)
        if mline:
            ln = int(mline.group(1)); col = int(mline.group(2))
            if current_basename:
                violations.append((current_basename, ln, col))
            else:
                # look back to find a filename context
                found = False
                for back in range(1,6):
                    if i-back < 0: break
                    prev = lines[i-back]
                    if '/' in prev and KNOWN_SOURCE_EXTS_RE.search(prev):
                        current_basename = os.path.basename(prev.strip())
                        violations.append((current_basename, ln, col))
                        found = True; break
                if not found:
                    violations.append(("<unknown_file>", ln, col))
    seen = set(); dedup = []
    for v in violations:
        if v not in seen:
            dedup.append(v); seen.add(v)
    return dedup, reported_total

def parse_results_json_path(path):
    out = set()
    if not os.path.exists(path):
        return out
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        # parse error -> treat as empty result
        print(f"[WARN] failed to parse results file {path}: {e}", file=sys.stderr)
        return out
    files = data.get("files", []) if isinstance(data, dict) else []
    for fe in files:
        fp = fe.get("file", "")
        basename = os.path.basename(fp)
        for m in fe.get("messages", []) or []:
            line = m.get("line"); col = m.get("column")
            if line is None or col is None:
                continue
            try:
                out.add((basename, int(line), int(col)))
            except Exception:
                continue
    return out

# --- runner/worker ---------------------------------------------------------

def run_style_instance(style_script, org, repo, pr, clang_config, workdir_prefix, instance_entry, max_command_time=None):
    """
    Runs style_reviewcpp for one entry and returns a result dict.
    instance_entry: the original entry dict from swe file
    """
    instance_id = instance_entry.get("instance_id")
    base_commit = instance_entry.get("base_commit", "")
    patch = instance_entry.get("patch", "") or ""
    raw_ps = instance_entry.get("problem_statement", "") or ""
    repo_field = instance_entry.get("repo", "")
    # Parse repo into org/repo if needed (but we accept org/repo passed separately)
    out_filename = f"results_{instance_id}.json"
    workdir = f"{workdir_prefix}_{instance_id}"

    os.makedirs(workdir, exist_ok=True)

    repo_url = f"https://github.com/{org}/{repo}.git"
    cmd = [
        sys.executable, style_script,
        "--repo-url", repo_url,
        "--pr", str(pr),
        "--clang-tidy-config", clang_config,
        "--work-dir", workdir,
        "--out", out_filename,
        "--instance-id", instance_id
    ]

    start_ts = datetime.utcnow().isoformat() + "Z"
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    end_ts = datetime.utcnow().isoformat() + "Z"

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    # parse problem_statement and results
    parsed_list, reported = parse_problem_statement(raw_ps)
    parsed_set = set(parsed_list)
    parsed_basenames = {p[0] for p in parsed_list if p[0] != "<unknown_file>"}
    parsed_total = len(parsed_list)

    results_set = parse_results_json_path(out_filename)
    results_total = len(results_set)

    # missing = parsed_set - results_set (unknown_file entries treated as missing)
    missing_set = set()
    for v in parsed_set:
        if v[0] == "<unknown_file>":
            missing_set.add(v); continue
        if v not in results_set:
            missing_set.add(v)
    missing_count_raw = len(missing_set)

    # additional: results_set - parsed_set, but filter by basename presence in parsed_basenames
    additional_all = {r for r in results_set if r not in parsed_set}
    additional_filtered = {r for r in additional_all if r[0] in parsed_basenames}
    additional_count = len(additional_filtered)

    updated_missing_count = max(0, missing_count_raw - additional_count)

    result = {
        "instance_id": instance_id,
        "base_commit": base_commit,
        "reported_total_in_problem_statement": reported if reported is not None else "",
        "parsed_total_from_problem_statement": parsed_total,
        "messages_found_in_results_json": results_total,
        "missing_count_raw": missing_count_raw,
        "additional_count_filtered_by_basename": additional_count,
        "updated_missing_count": updated_missing_count,
        # debug/tracing:
        "stdout": stdout[:4000],
        "stderr": stderr[:4000],
        "start_ts": start_ts,
        "end_ts": end_ts,
        "missing_list_sample": list(missing_set)[:100],
        "additional_list_sample": list(additional_filtered)[:100],
        "parsed_list_sample": parsed_list[:200],
        "results_list_sample": list(results_set)[:200],
        # path to produced results file
        "results_file": out_filename
    }
    return result

# --- main / orchestrator ---------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Parallel compare sweagent problem_statement vs style_review outputs.")
    p.add_argument("--org", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--pr", required=True, type=int)
    p.add_argument("--swe-file", default="sweagent_clang-tidy_results.json")
    p.add_argument("--style-script", default="multiswebench_local/multi_swe_bench/harness/CPP_style_review/style_reviewcpp.py")
    p.add_argument("--clang-config", default="multiswebench_local/multi_swe_bench/harness/CPP_style_review/.clang-tidy")
    p.add_argument("--workdir-prefix", default="tempcpp1")
    p.add_argument("--csv-out", default=None)
    p.add_argument("--max-workers", type=int, default=None, help="Max parallel workers (default: min(8, (cpu_count or 4) * 2))")
    p.add_argument("--write-diffs", default=None, help="Optional directory to write per-instance JSON diffs (parsed, results, missing, additional)")
    return p.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.swe_file):
        print(f"ERROR: swe file not found: {args.swe_file}", file=sys.stderr); sys.exit(2)
    if not os.path.exists(args.style_script):
        print(f"ERROR: style script not found: {args.style_script}", file=sys.stderr); sys.exit(2)
    if not os.path.exists(args.clang_config):
        print(f"ERROR: clang config not found: {args.clang_config}", file=sys.stderr); sys.exit(2)

    with open(args.swe_file, "r", encoding="utf-8") as f:
        swe = json.load(f)

    target_repo = f"{args.org}/{args.repo}"
    entries = [e for e in swe if e.get("repo") == target_repo and int(e.get("pull_number", -1)) == int(args.pr)]
    if not entries:
        print(f"No matching entries for {target_repo} PR {args.pr} in {args.swe_file}."); sys.exit(0)
    print(f"Found {len(entries)} entries for {target_repo} PR {args.pr}.\n")

    # max workers default
    if args.max_workers is None:
        cpu = os.cpu_count() or 4
        max_workers = min(8, cpu * 2)
    else:
        max_workers = max(1, args.max_workers)
    print(f"Running up to {max_workers} parallel workers.")

    # prepare diffs dir if requested
    if args.write_diffs:
        os.makedirs(args.write_diffs, exist_ok=True)

    results = []
    start_all = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_entry = {}
        for entry in entries:
            # submit each instance to the thread pool
            fut = ex.submit(run_style_instance,
                            args.style_script, args.org, args.repo, args.pr, args.clang_config,
                            args.workdir_prefix, entry)
            future_to_entry[fut] = entry.get("instance_id")

        # stream results as they complete
        for fut in as_completed(future_to_entry):
            iid = future_to_entry[fut]
            try:
                res = fut.result()
            except Exception as e:
                print(f"[ERROR] instance {iid} raised exception: {e}", file=sys.stderr)
                # create a failure row
                res = {
                    "instance_id": iid,
                    "base_commit": "",
                    "reported_total_in_problem_statement": "",
                    "parsed_total_from_problem_statement": 0,
                    "messages_found_in_results_json": 0,
                    "missing_count_raw": 0,
                    "additional_count_filtered_by_basename": 0,
                    "updated_missing_count": 0,
                    "stdout": "",
                    "stderr": f"Exception: {e}",
                    "results_file": ""
                }
            # print concise log line
            print("\n" + "="*60)
            print(f"Instance: {res['instance_id']}  parsed={res['parsed_total_from_problem_statement']}  results={res['messages_found_in_results_json']}  missing_raw={res['missing_count_raw']}  additional_filtered={res['additional_count_filtered_by_basename']}  updated_missing={res['updated_missing_count']}")
            # print truncated stdout/stderr for quick debugging
            if res.get("stdout"):
                print(" stdout (trunc):")
                print(res["stdout"][:1000])
            if res.get("stderr"):
                print(" stderr (trunc):")
                print(res["stderr"][:1000])
            print("="*60 + "\n")

            # optionally write per-instance diff JSON
            if args.write_diffs:
                diffd = {
                    "instance_id": res["instance_id"],
                    "reported_total_in_problem_statement": res["reported_total_in_problem_statement"],
                    "parsed_list_sample": res.get("parsed_list_sample", []),
                    "results_list_sample": res.get("results_list_sample", []),
                    "missing_list_sample": res.get("missing_list_sample", []),
                    "additional_list_sample": res.get("additional_list_sample", []),
                    "stdout": res.get("stdout",""),
                    "stderr": res.get("stderr",""),
                    "results_file": res.get("results_file",""),
                    "start_ts": res.get("start_ts",""),
                    "end_ts": res.get("end_ts",""),
                }
                fn = os.path.join(args.write_diffs, f"diff_{res['instance_id']}.json")
                try:
                    with open(fn, "w", encoding="utf-8") as df:
                        json.dump(diffd, df, indent=2)
                except Exception as e:
                    print(f"[WARN] failed to write diff for {res['instance_id']}: {e}", file=sys.stderr)

            results.append(res)

    elapsed = time.time() - start_all
    print(f"All done in {elapsed:.1f}s. Collected {len(results)} results.")

    # write summary CSV
    csv_path = args.csv_out or f"clang_tidy_summary_{args.org}_{args.repo}_pr{args.pr}.csv"
    fieldnames = [
        "org","repo","pull_number","instance_id","base_commit",
        "reported_total_in_problem_statement","parsed_total_from_problem_statement",
        "messages_found_in_results_json","missing_count_raw","additional_count_filtered_by_basename","updated_missing_count",
        "start_ts","end_ts","results_file"
    ]
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as cf:
            w = csv.DictWriter(cf, fieldnames=fieldnames)
            w.writeheader()
            for r in results:
                w.writerow({
                    "org": args.org,
                    "repo": args.repo,
                    "pull_number": args.pr,
                    "instance_id": r.get("instance_id",""),
                    "base_commit": r.get("base_commit",""),
                    "reported_total_in_problem_statement": r.get("reported_total_in_problem_statement",""),
                    "parsed_total_from_problem_statement": r.get("parsed_total_from_problem_statement",0),
                    "messages_found_in_results_json": r.get("messages_found_in_results_json",0),
                    "missing_count_raw": r.get("missing_count_raw",0),
                    "additional_count_filtered_by_basename": r.get("additional_count_filtered_by_basename",0),
                    "updated_missing_count": r.get("updated_missing_count",0),
                    "start_ts": r.get("start_ts",""),
                    "end_ts": r.get("end_ts",""),
                    "results_file": r.get("results_file","")
                })
        print(f"Wrote CSV summary to: {csv_path}")
    except Exception as e:
        print(f"[ERROR] failed to write CSV {csv_path}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
