#!/usr/bin/env python3
"""
Convert Pylint Results (from pr_pylint_review.py) to SWE Agent Input Format.

- Input: results.json (single JSON object with "label", "files", "overview")
- Groups violations either by file (default, --batch_size=0) or into batches of ~N violations
- Outputs a JSON array of SWE-agent instances
"""

import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict


def extract_violations_from_file(file_report: Dict) -> List[str]:
    """Convert pylint messages into readable strings."""
    violations = []
    for message in file_report.get("messages", []):
        line = message.get("line", 0)
        column = message.get("column", 0)
        msg = message.get("message", "")
        symbol = message.get("symbol", "")
        violation = f"Line {line}, Column {column}: {msg} [{symbol}]"
        violations.append(violation)
    return violations


def generate_problem_statement_for_batch(
    file_reports: List[Dict], style_tool: str = "pylint"
) -> str:
    """Generate a natural-language problem statement for a batch of files."""
    parts = []
    total_violations = 0

    for file_report in file_reports:
        file_path = file_report.get("file", "")
        score = file_report.get("score", 10.0)
        errors = file_report.get("error_count", 0)
        violations = extract_violations_from_file(file_report)
        total_violations += len(violations)

        if errors == 0 and not violations:
            continue

        section = f"""File: {file_path}
Score: {score}/10.0
Violations: {len(violations)}

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
"""
    return f"Fix the following {style_tool.upper()} style violations in this batch:\n\n{joined}\n{summary}"


def create_sweagent_instance(
    org: str,
    repo: str,
    pr_number: int,
    problem_statement: str,
    violations: list,
    instance_idx: int
) -> Dict:
    """Create a SWE agent instance dict."""
    instance_id = f"{org}__{repo}-{pr_number}"
    review_id = f"{org}__{repo}-{pr_number}_{instance_idx}"
    style_review_summary = {
        "total_files": len(violations), 
        "total_messages": 0,
        "files": {}
    }
    total_messages = 0
    for file in violations: 
        style_review_summary["files"][file["file"]] = {}
        style_review_summary["files"][file["file"]]["message_count"] = len(file["messages"])
        style_review_summary["files"][file["file"]]["messages"] = file["messages"]
        total_messages += len(file["messages"])
    style_review_summary["total_messages"] = total_messages
    with open("../codearena_instances_python.json", 'r') as f:
        data = json.load(f)
    for i in data: 
        if i["instance_id"] == instance_id:
            instance = i
    instance["original_instance_id"] = instance_id
    instance["instance_id"] = review_id
    instance["problem_statement"] = problem_statement
    instance["hints_text"] = None
    instance["bad_patches"] = []
    instance["style_review"] = style_review_summary
    return instance


def main():
    parser = argparse.ArgumentParser(description="Convert Pylint results.json into SWE Agent instances")
    parser.add_argument("--instance-id", required=True, help="org__repo-pr-number (e.g. astropy__astropy-14508)")
    parser.add_argument("--results", default="results.json", help="Path to results.json from pr_pylint_review.py")
    parser.add_argument("--output", required=True, help="Path to save SWE Agent instances (JSON array)")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="Approx number of violations per instance (0 = group by file)")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        sys.exit(1)

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    file_reports = results.get("files", [])
    if not file_reports:
        print("No files with violations found.")
        sys.exit(1)

    # Load existing output
    output_path = Path(args.output)
    existing_instances = []
    if output_path.exists():
        with open(output_path, "r") as f:
            existing_instances = json.load(f)
        if not isinstance(existing_instances, list):
            print("Output file must be a JSON array.")
            sys.exit(1)

    instance_idx = len(existing_instances) + 1
    instances = []

    org = args.instance_id.split('__')[0]
    repo = '-'.join(args.instance_id.split('__')[-1].split('-')[:-1])
    number = args.instance_id.split('-')[-1]

    if args.batch_size == 0:
        # Group by file: one instance per file
        for fr in file_reports:
            if not fr.get("messages"):
                continue
            ps = generate_problem_statement_for_batch([fr], style_tool="pylint")
            instances.append(create_sweagent_instance(org, repo, number, ps, [fr], instance_idx))
            print(f"Created instance {instance_idx} for file {fr.get('file')}")
            instance_idx += 1
    else:
        # Group into batches of ~N violations
        batch = []
        batch_violation_count = 0
        for fr in file_reports:
            violations_count = len(fr.get("messages", []))
            if not violations_count:
                continue

            if batch and batch_violation_count + violations_count >= args.batch_size:
                ps = generate_problem_statement_for_batch(batch, style_tool="pylint")
                instances.append(
                    create_sweagent_instance(org, repo, number, ps, batch, instance_idx)
                )
                print(f"Created instance {instance_idx} with {batch_violation_count} violations ({len(batch)} file(s))")
                instance_idx += 1
                batch = []
                batch_violation_count = 0
            
            batch.append(fr)
            batch_violation_count += violations_count

        if batch:
            ps = generate_problem_statement_for_batch(batch, style_tool="pylint")
            instances.append(
                create_sweagent_instance(org, repo, number, ps, batch, instance_idx)
            )
            print(f"Created instance {instance_idx} with {batch_violation_count} violations ({len(batch)} file(s))")

    all_instances = existing_instances + instances
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_instances, f, indent=2)

    print(f"Saved {len(all_instances)} instance(s) to {args.output}")


if __name__ == "__main__":
    main()
