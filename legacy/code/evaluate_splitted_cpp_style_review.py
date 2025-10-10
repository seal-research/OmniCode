#!/usr/bin/env python3
"""
compare_clang_tidy_runs_final.py

Usage example:
  python compare_clang_tidy_runs_final.py \
    --org apache --repo dubbo --pr 10638 \
    --swe-file sweagent_clang-tidy_results.json \
    --style-script multiswebench_local/multi_swe_bench/harness/CPP_style_review/style_reviewcpp.py \
    --clang-config multiswebench_local/multi_swe_bench/harness/CPP_style_review/.clang-tidy

What changed:
 - Additional errors (present in results1.json but not in problem_statement)
   are counted ONLY if their filename basename (after last '/') matches
   a basename present in the parsed problem_statement for that instance.
 - Final CSV includes raw missing, additional (filtered by basename), and updated missing.
"""
import argparse
import json
import os
import re
import csv
import shlex
import subprocess
import sys

def parse_args():
    p = argparse.ArgumentParser(description="Compare sweagent problem_statement vs style_review output per instance_id.")
    p.add_argument("--org", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--pr", required=True, type=int, help="Pull request number")
    p.add_argument("--swe-file", default="sweagent_clang-tidy_results.json")
    p.add_argument("--style-script", default="multiswebench_local/multi_swe_bench/harness/CPP_style_review/style_reviewcpp.py", help="Path to style_reviewcpp.py")
    p.add_argument("--clang-config", default="multiswebench_local/multi_swe_bench/harness/CPP_style_review/.clang-tidy", help="Path to .clang-tidy")
    p.add_argument("--workdir-prefix", default="tempcpp", help="Prefix for per-run workdir")
    p.add_argument("--csv-out", default=None, help="CSV output path (defaults to clang_tidy_summary_<org>_<repo>_pr<pr>.csv)")
    return p.parse_args()

def normalize_problem_statement_text(ps):
    """Convert literal \\n sequences to real newlines and normalize CRLF."""
    if ps is None:
        return ""
    if "\\n" in ps and ps.count("\\n") >= ps.count("\n"):
        ps = ps.replace("\\r\\n", "\\n").replace("\\n", "\n")
    ps = ps.replace("\r\n", "\n").replace("\r", "\n")
    return ps

def parse_problem_statement(ps_text):
    """
    Parse problem_statement text and return:
      - violations: list of (basename, line:int, column:int)
      - reported_total: int or None
    """
    ps = normalize_problem_statement_text(ps_text)
    violations = []
    reported_total = None

    # reported totals
    m1 = re.search(r'Total\s+Unique\s+Violations\s*:\s*(\d+)', ps, flags=re.IGNORECASE)
    if not m1:
        m1 = re.search(r'Violations\s*:\s*(\d+)', ps, flags=re.IGNORECASE)
    if m1:
        try:
            reported_total = int(m1.group(1))
        except Exception:
            reported_total = None

    # File: blocks
    file_block_re = re.compile(r'^\s*File:\s*(?P<path>.+?)\s*\n(?P<body>.*?)(?=(?:^\s*File:\s*)|\Z)', flags=re.MULTILINE | re.DOTALL)
    found_blocks = list(file_block_re.finditer(ps))

    if found_blocks:
        for b in found_blocks:
            fullpath = b.group("path").strip()
            basename = os.path.basename(fullpath)
            body = b.group("body")
            for lm in re.finditer(r'Line\s+(\d+)\s*,\s*Column\s+(\d+)', body, flags=re.IGNORECASE):
                try:
                    line = int(lm.group(1))
                    col = int(lm.group(2))
                    violations.append((basename, line, col))
                except Exception:
                    continue
        # dedupe preserving order
        seen = set()
        dedup = []
        for v in violations:
            if v not in seen:
                dedup.append(v)
                seen.add(v)
        return dedup, reported_total

    # fallback scanning
    current_basename = None
    lines = ps.splitlines()
    for i, raw in enumerate(lines):
        line = raw.rstrip()
        if '/' in line and re.search(r'\.(c|cc|cpp|cxx|h|hpp|hh|ipp|inl|m|mm)\b', line, flags=re.IGNORECASE):
            current_basename = os.path.basename(line.strip())
            continue

        mfile = re.match(r'^\s*File\s*:\s*(.+)$', line, flags=re.IGNORECASE)
        if mfile:
            candidate = mfile.group(1).strip()
            current_basename = os.path.basename(candidate)
            continue

        mline = re.search(r'Line\s+(\d+)\s*,\s*Column\s+(\d+)', line, flags=re.IGNORECASE)
        if mline:
            line_n = int(mline.group(1))
            col_n = int(mline.group(2))
            if current_basename:
                violations.append((current_basename, line_n, col_n))
            else:
                found = False
                for back in range(1,6):
                    if i-back < 0:
                        break
                    prev = lines[i-back]
                    if '/' in prev and re.search(r'\.(c|cc|cpp|cxx|h|hpp|hh|ipp|inl|m|mm)\b', prev, flags=re.IGNORECASE):
                        current_basename = os.path.basename(prev.strip())
                        violations.append((current_basename, line_n, col_n))
                        found = True
                        break
                if not found:
                    violations.append(("<unknown_file>", line_n, col_n))

    # dedupe preserving order
    seen = set()
    dedup = []
    for v in violations:
        if v not in seen:
            dedup.append(v)
            seen.add(v)
    return dedup, reported_total

def parse_results_json(path):
    """Return set of (basename,line,col) from results1.json messages."""
    if not os.path.exists(path):
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)
    except Exception as e:
        print(f"ERROR: failed to parse {path}: {e}", file=sys.stderr)
        return set()
    out = set()
    files = j.get("files", []) if isinstance(j, dict) else []
    for fe in files:
        file_path = fe.get("file", "")
        basename = os.path.basename(file_path)
        for m in fe.get("messages", []) or []:
            line = m.get("line"); col = m.get("column")
            if line is None or col is None:
                continue
            try:
                out.add((basename, int(line), int(col)))
            except Exception:
                continue
    return out

def run_style_script(style_script, org, repo, pr, clang_config, workdir, instance_id,swefile):
    repo_url = f"https://github.com/{org}/{repo}.git"
    cmd = [
        sys.executable, style_script,
        "--repo-url", repo_url,
        "--pr", str(pr),
        "--clang-tidy-config", clang_config,
        "--work-dir", workdir,
        "--out", "results1.json",
        "--instance-id", instance_id,
        "--swe-results",swefile
    ]
    print("Running:", " ".join(shlex.quote(x) for x in cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.returncode, proc.stdout, proc.stderr

def main():
    args = parse_args()
    swe_path = args.swe_file
    if not os.path.exists(swe_path):
        print(f"ERROR: swe file not found: {swe_path}", file=sys.stderr); sys.exit(2)
    if not os.path.exists(args.style_script):
        print(f"ERROR: style script not found: {args.style_script}", file=sys.stderr); sys.exit(2)
    if not os.path.exists(args.clang_config):
        print(f"ERROR: clang config not found: {args.clang_config}", file=sys.stderr); sys.exit(2)

    with open(swe_path, "r", encoding="utf-8") as f:
        swe = json.load(f)

    target_repo = f"{args.org}/{args.repo}"
    entries = [e for e in swe if e.get("repo") == target_repo and int(e.get("pull_number", -1)) == int(args.pr)]
    if not entries:
        print(f"No matching entries for {target_repo} PR {args.pr} in {swe_path}.")
        sys.exit(0)
    print(f"Found {len(entries)} entries for {target_repo} PR {args.pr}.\n")

    csv_rows = []
    csv_path = args.csv_out or f"clang_tidy_summary_{args.org}_{args.repo}_pr{args.pr}.csv"

    for idx, entry in enumerate(entries, start=1):
        instance_id = entry.get("instance_id")
        base_commit = entry.get("base_commit", "")
        patch = entry.get("patch", "")
        raw_ps = entry.get("problem_statement", "") or ""

        print("="*80)
        print(f"[{idx}/{len(entries)}] instance_id={instance_id}")
        print(f"base_commit={base_commit}")
        preview = raw_ps.replace("\n", "\\n")[:500]
        print("problem_statement preview (escaped):", preview)

        # parse problem_statement
        parsed_list, reported = parse_problem_statement(raw_ps)
        parsed_set = set(parsed_list)
        total_parsed = len(parsed_list)
        # collect basenames seen in problem_statement (exclude unknown)
        parsed_basenames = {p[0] for p in parsed_list if p[0] != "<unknown_file>"}

        print("reported_total (from problem_statement):", reported if reported is not None else "N/A")
        print("parsed_total (from problem_statement):", total_parsed)
        if parsed_list:
            print("parsed violations (sample up to 20):")
            for v in parsed_list[:20]:
                print("  ", v)

        # run style script
        workdir = f"{args.workdir_prefix}_{instance_id}"
        os.makedirs(workdir, exist_ok=True)
        rc, out, err = run_style_script(args.style_script, args.org, args.repo, args.pr, args.clang_config, workdir, instance_id,args.swe_file)
        print(f"style script exit code={rc}, stdout len={len(out)}, stderr len={len(err)}")
        if out.strip():
            print("stdout (truncated):")
            print(out[:800])
        if err.strip():
            print("stderr (truncated):")
            print(err[:800])

        # parse results1.json
        results_json = "results1.json"
        results_set = parse_results_json(results_json)
        print("messages found in results1.json:", len(results_set))

        # missing: parsed_set - results_set (unknown_file entries treated as missing)
        missing_set = set()
        for v in parsed_set:
            if v[0] == "<unknown_file>":
                missing_set.add(v)
                continue
            if v not in results_set:
                missing_set.add(v)
        missing_count_raw = len(missing_set)

        # additional: results_set - parsed_set, but only count those with basename present in parsed_basenames
        additional_set_all = {r for r in results_set if r not in parsed_set}
        additional_set_filtered = {r for r in additional_set_all if r[0] in parsed_basenames}
        additional_count = len(additional_set_filtered)

        # updated missing = max(0, missing_raw - additional_count)
        updated_missing_count = max(0, missing_count_raw - additional_count)

        print(f"missing_count_raw: {missing_count_raw}")
        print(f"additional_count (filtered by parsed basenames): {additional_count}")
        print(f"updated_missing_count (max(0, missing_raw - additional)): {updated_missing_count}")

        if missing_set:
            print("missing entries (up to 50):")
            for m in list(missing_set)[:50]:
                print("  ", m)
        if additional_set_filtered:
            print("additional entries (filtered, up to 50):")
            for a in list(additional_set_filtered)[:50]:
                print("  ", a)

        # archive results1.json
        if os.path.exists(results_json):
            dest = f"results_{instance_id}.json"
            try:
                if os.path.exists(dest):
                    os.remove(dest)
                os.rename(results_json, dest)
                print(f"archived {results_json} -> {dest}")
            except Exception as e:
                print("WARN: failed to archive results1.json:", e, file=sys.stderr)

        csv_rows.append({
            "org": args.org,
            "repo": args.repo,
            "pull_number": args.pr,
            "instance_id": instance_id,
            "base_commit": base_commit,
            "reported_total_in_problem_statement": reported if reported is not None else "",
            "parsed_total_from_problem_statement": total_parsed,
            "messages_found_in_results_json": len(results_set),
            "missing_count_raw": missing_count_raw,
            "additional_count_filtered_by_basename": additional_count,
            "updated_missing_count": updated_missing_count
        })

    # write CSV
    fieldnames = [
        "org","repo","pull_number","instance_id","base_commit",
        "reported_total_in_problem_statement","parsed_total_from_problem_statement",
        "messages_found_in_results_json","missing_count_raw","additional_count_filtered_by_basename","updated_missing_count"
    ]
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as cf:
            w = csv.DictWriter(cf, fieldnames=fieldnames)
            w.writeheader()
            for r in csv_rows:
                w.writerow(r)
        print("\nWrote CSV summary to:", csv_path)
    except Exception as e:
        print("ERROR: failed to write CSV:", e, file=sys.stderr)

    print("Done.")

if __name__ == "__main__":
    main()
