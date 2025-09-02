#!/usr/bin/env python3
"""
Convert Style Errors to SWE Agent Input Format (batched per ~10 violations)

This script converts style error files from the style review output into
the format expected by sweagent_regular.py. Instead of one instance per file,
each instance contains at least 10 violations (or the remaining ones),
but all violations from a file must stay in the same instance.

On repeated runs, new instances are appended to the output JSON file
(which is expected to contain a top-level JSON array).
"""

import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Optional


def load_errors_from_jsonl(jsonl_path: Path, target_label: str) -> List[Dict]:
    """Extract error records for a specific PR label from a JSONL file."""
    if not jsonl_path.exists():
        print(f"File not found: {jsonl_path}")
        return []
    with open(jsonl_path, 'r') as f:
        for line in f:
            try:
                obj = json.loads(line)
                if obj.get("label") == target_label:
                    print(f" Found style errors for: {target_label}")
                    return obj.get("files", [])
            except json.JSONDecodeError:
                continue
    print(f"PR {target_label} not found in {jsonl_path}")
    return []


def extract_violations_from_file(file_report: Dict) -> List[str]:
    """Extract violation messages from a file report."""
    violations = []
    for message in file_report.get("messages", []):
        line = message.get("line", 0)
        column = message.get("column", 0)
        msg = message.get("message", "")
        source = message.get("source", "")
        violation = f"Line {line}, Column {column}: {msg} [{source}]"
        violations.append(violation)
    return violations


def generate_problem_statement_for_batch(
    file_reports: List[Dict],
    original_map: Dict,
    style_tool: str
) -> str:
    """Generate a problem statement for a batch of files."""
    parts = []
    total_violations = 0
    for file_report in file_reports:
        file_path = file_report.get("file", "").replace("/workspace/repo/", "")
        orig = original_map.get(file_report.get("file", ""), {})
        orig_score = orig.get("score", 10.0)
        orig_errors = orig.get("error_count", 0)
        patched_score = file_report.get("score", 10.0)
        patched_errors = file_report.get("error_count", 0)
        violations = extract_violations_from_file(file_report)
        total_violations += len(violations)

        if patched_errors == 0:
            parts.append(f"No {style_tool.upper()} violations found in {file_path}.")
            continue

        section = f"""File: {file_path}
Score: {patched_score}/10.0 (was {orig_score}/10.0)
Violations: {patched_errors} (was {orig_errors})

"""
        for v in violations[:20]:
            section += f"  {v}\n"
        if len(violations) > 20:
            section += f"\n  ... and {len(violations) - 20} more violations\n"

        parts.append(section)

    joined = "\n\n".join(parts)
    summary = f"""
Summary for this batch:
- Files: {len(file_reports)}
- Total Violations: {total_violations}

Please fix all the above {style_tool.upper()} violations while maintaining original functionality.
Focus on the most critical issues first and ensure the code follows Java best practices.
"""
    return f"Fix the following {style_tool.upper()} style violations in this batch:\n\n{joined}\n{summary}"


def create_sweagent_instance(
    org: str,
    repo: str,
    pr_number: int,
    base_commit: str,
    problem_statement: str,
    instance_idx: int
) -> Dict:
    """Create a SWE agent instance in the expected format, numbered sequentially."""
    instance_id = f"{org}__{repo}_{pr_number}_{instance_idx}"
    return {
        "repo": org + "/" + repo,
        "pull_number": pr_number,
        "instance_id": instance_id,
        "issue_numbers": [],
        "base_commit": base_commit,
        "patch": "",
        "test_patch": "",
        "problem_statement": problem_statement,
        "hints_text": "",
        "created_at": "",
        "version": "",
        "PASS_TO_PASS": "",
        "FAIL_TO_PASS": "",
        "bad_patches": []
    }


def load_original_dataset_info(
    org: str, repo: str, pr_number: int,
    dataset_path: str = "data/multiswebench_data/mswebench_instances.json"
) -> Optional[Dict]:
    """Load original dataset information for the specific PR."""
    if not Path(dataset_path).exists():
        print(f" Dataset not found: {dataset_path}")
        return None
    try:
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        target_repo = f"{org}/{repo}"
        for instance in dataset:
            if instance.get("repo") == target_repo and instance.get("pull_number") == pr_number:
                print(f" Found dataset info for: {target_repo}:{pr_number}")
                return instance
        print(f" Instance {target_repo}:{pr_number} not found in dataset")
        return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Convert Style Errors to SWE Agent Input (batched per ~10 violations, append mode)")
    parser.add_argument("--org", required=True, help="Organization name (e.g., apache)")
    parser.add_argument("--repo", required=True, help="Repository name (e.g., dubbo)")
    parser.add_argument("--pr_number", type=int, required=True, help="Pull request number")
    parser.add_argument("--style_tool", required=True, choices=["checkstyle", "pmd"], help="Style tool used")
    parser.add_argument("--output", required=True, help="Output file path (JSON array)")
    parser.add_argument("--dataset_path", default="data/multiswebench_data/mswebench_instances.json", help="Path to original dataset")
    parser.add_argument("--base_commit", help="Base commit hash (if not in dataset)")
    args = parser.parse_args()

    print(f"\n  Converting Style Errors to SWE Agent Input (batched per ~10 violations, append mode)")
    print("=" * 60)

    target_label = f"{args.org}/{args.repo}:pr-{args.pr_number}"
    jsonl_filename = f"unique_results_{args.style_tool}.jsonl"
    jsonl_path = Path(jsonl_filename)

    file_reports = load_errors_from_jsonl(jsonl_path, target_label)
    if not file_reports:
        print(" No violations found or failed to load style errors.")
        sys.exit(1)

    dataset_info = load_original_dataset_info(args.org, args.repo, args.pr_number, args.dataset_path)
    base_commit = args.base_commit or (dataset_info.get("base_commit") if dataset_info else "main")

    # Filter files with violations
    files_with_errors = [r for r in file_reports if r.get("messages")]
    if not files_with_errors:
        print(" No files with errors found in the PR.")
        sys.exit(1)

    print(f" Found {len(files_with_errors)} file(s) with errors. Grouping into batches of >=10 violations...")

    # Load existing output JSON if present
    existing_instances = []
    output_path = Path(args.output)
    if output_path.exists():
        try:
            with open(output_path, "r") as f:
                existing_instances = json.load(f)
            if not isinstance(existing_instances, list):
                print(" Output file does not contain a JSON array. Exiting.")
                sys.exit(1)
            print(f" Loaded {len(existing_instances)} existing instances from {args.output}")
        except Exception as e:
            print(f" Failed to load existing output file: {e}")
            sys.exit(1)

    # figure next index (continue numbering)
    instance_idx = len(existing_instances) + 1
    instances = []
    batch = []
    batch_violation_count = 0
    original_map = {r.get("file"): r for r in file_reports}

    for fr in files_with_errors:
        violations_count = len(fr.get("messages", []))
        if batch and batch_violation_count + violations_count >= 10:
            ps = generate_problem_statement_for_batch(batch, original_map, args.style_tool)
            instances.append(create_sweagent_instance(args.org, args.repo, args.pr_number, base_commit, ps, instance_idx))
            print(f"  Created instance {instance_idx} with {batch_violation_count} violations ({len(batch)} file(s))")
            instance_idx += 1
            batch = []
            batch_violation_count = 0
        batch.append(fr)
        batch_violation_count += violations_count

    if batch:
        ps = generate_problem_statement_for_batch(batch, original_map, args.style_tool)
        instances.append(create_sweagent_instance(args.org, args.repo, args.pr_number, base_commit, ps, instance_idx))
        print(f"  Created instance {instance_idx} with {batch_violation_count} violations ({len(batch)} file(s))")

    all_instances = existing_instances + instances

    print(f"\n Saving {len(all_instances)} total instance(s) to {args.output}...")
    try:
        with open(args.output, 'w') as f:
            json.dump(all_instances, f, indent=2)
        print(" Successfully saved (appended mode).")
    except Exception as e:
        print(f" Failed to save: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
