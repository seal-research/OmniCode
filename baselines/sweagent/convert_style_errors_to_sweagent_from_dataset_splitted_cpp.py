#!/usr/bin/env python3
"""
Convert Style Errors to SWE Agent Input (batched per ~10 unique violations)

This updated script supports two input formats for `results.json`:

1. A single JSON object with fields:
   {
     "label": "org/repo:pr-123",
     "files": [ ... ]
   }

2. A JSONL file (each line is a JSON object) as before.

On repeated runs, new instances are appended to the output JSON file
(which is expected to contain a top-level JSON array).

Change: only unique errors per file are considered. Uniqueness is determined
by (source, message, line, column). THIS VERSION: only `.cpp` files are considered.
It also prints the average unique style violations per generated instance.
"""

import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple


def load_errors_from_results(json_path: Path, target_label: str) -> List[Dict[str, Any]]:
    """Load errors from either a single JSON file (object or list) or JSONL.

    Returns the `files` array for the matching label, or an empty list if not found.
    """
    if not json_path.exists():
        print(f"File not found: {json_path}")
        return []

    # Try to parse as a normal JSON file first
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        # If it's a dict with label/files
        if isinstance(data, dict):
            if data.get("label") == target_label:
                print(f" Found style errors for: {target_label} (single JSON object)")
                return data.get("files", [])
            # If the file itself is a top-level map of labels to file-lists (less likely), try to find
            if "files" in data and "label" not in data:
                # Not expected, but attempt to treat as files array
                print(f" Detected top-level 'files' in JSON without label; returning files.")
                return data.get("files", [])
            # Maybe the JSON is a list of objects
        if isinstance(data, list):
            for obj in data:
                if isinstance(obj, dict) and obj.get("label") == target_label:
                    print(f" Found style errors for: {target_label} (in JSON list)")
                    return obj.get("files", [])

    except json.JSONDecodeError:
        # Fallthrough: perhaps it's JSONL
        pass
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        # Fallthrough to JSONL attempt

    # Try JSONL fallback: each line is a JSON object
    try:
        with open(json_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if obj.get("label") == target_label:
                        print(f" Found style errors for: {target_label} (JSONL)")
                        return obj.get("files", [])
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Failed JSONL fallback parse: {e}")

    print(f"PR {target_label} not found in {json_path}")
    return []


def _message_identity_tuple(msg_obj: Dict[str, Any]) -> Tuple[Any, Any, Any, Any]:
    """Return a tuple that identifies a message uniquely based on source, message, line, column."""
    return (
        msg_obj.get("source"),
        msg_obj.get("message"),
        msg_obj.get("line"),
        msg_obj.get("column"),
    )


def get_unique_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return list of messages with duplicates removed (preserving first occurrence order).

    Uniqueness determined by (source, message, line, column).
    """
    seen = set()
    unique = []
    for m in messages:
        key = _message_identity_tuple(m)
        if key not in seen:
            seen.add(key)
            unique.append(m)
    return unique


def extract_violations_from_file(file_report: Dict[str, Any]) -> List[str]:
    """Extract unique violation messages from a file report as human-readable strings."""
    violations: List[str] = []
    messages = file_report.get("messages", []) or []
    unique_msgs = get_unique_messages(messages)
    for message in unique_msgs:
        line = message.get("line", 0)
        column = message.get("column", 0)
        msg = message.get("message", "")
        source = message.get("source", "")
        mtype = message.get("type")
        prefix = f"({mtype}) " if mtype else ""
        violation = f"Line {line}, Column {column}: {prefix}{msg} [{source}]"
        violations.append(violation)
    return violations


def generate_problem_statement_for_batch(
    file_reports: List[Dict[str, Any]],
    original_map: Dict[str, Dict[str, Any]],
    style_tool: str,
) -> str:
    """Generate a problem statement for a batch of files, using unique violations."""
    parts: List[str] = []
    total_violations = 0
    for file_report in file_reports:
        file_path_raw = file_report.get("file", "")
        # Normalize file path shown to user by stripping common prefixes
        file_path = file_path_raw.replace("/workspace/repo/", "").replace("/tmp/", "")
        orig = original_map.get(file_report.get("file", ""), {})
        # Determine original error_count by deduping original messages if available
        orig_messages = orig.get("messages", []) or []
        orig_errors = len(get_unique_messages(orig_messages)) if orig_messages else orig.get("error_count", 0)
        orig_score = orig.get("score", 10.0)

        # For patched (current) report, compute unique messages
        patched_messages = file_report.get("messages", []) or []
        unique_patched_messages = get_unique_messages(patched_messages)
        patched_errors = len(unique_patched_messages)
        patched_score = file_report.get("score", 10.0)

        total_violations += patched_errors

        if patched_errors == 0:
            parts.append(f"No {style_tool.upper()} violations found in {file_path}.")
            continue

        section = f"""File: {file_path}
Score: {patched_score}/10.0 (was {orig_score}/10.0)
Violations: {patched_errors} (was {orig_errors})

"""
        # show up to 20 unique violations
        for v in (extract_violations_from_file(file_report)[:20]):
            section += f"  {v}\n"
        if patched_errors > 20:
            section += f"\n  ... and {patched_errors - 20} more unique violations\n"

        parts.append(section)

    joined = "\n\n".join(parts)
    summary = f"""
Summary for this batch:
- Files: {len(file_reports)}
- Total Unique Violations: {total_violations}

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
    instance_idx: int,
    unique_violations: int = 0,
) -> Dict[str, Any]:
    """Create a SWE agent instance in the expected format, numbered sequentially.

    Added field:
      - unique_violations: integer count of unique violations included in this instance
    """
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
        "bad_patches": [],
        "unique_violations": unique_violations,
    }


def load_original_dataset_info(
    org: str,
    repo: str,
    pr_number: int,
    dataset_path: str = "data/multiswebench_data/mswebench_instances.json",
) -> Optional[Dict[str, Any]]:
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


def is_cpp_file_path(path_str: str) -> bool:
    """Return True if the file path corresponds to a .cpp file (case-insensitive)."""
    if not path_str:
        return False
    return str(path_str).lower().endswith(".cpp")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert Style Errors to SWE Agent Input (batched per ~10 unique violations, append mode)"
        )
    )
    parser.add_argument("--org", required=True, help="Organization name (e.g., apache)")
    parser.add_argument("--repo", required=True, help="Repository name (e.g., dubbo)")
    parser.add_argument("--pr_number", type=int, required=True, help="Pull request number")
    parser.add_argument(
        "--style_tool", required=True, choices=["checkstyle", "pmd", "clang-tidy", "other"], help="Style tool used"
    )
    parser.add_argument("--output", default="cpp_style_errors.json", help="Output file path (JSON array)")
    parser.add_argument(
        "--dataset_path",
        default="data/multiswebench_data/mswebench_instances.json",
        help="Path to original dataset",
    )
    parser.add_argument("--base_commit", help="Base commit hash (if not in dataset)")
    parser.add_argument("--results", default="results.json", help="Path to results JSON/JSONL file")
    args = parser.parse_args()

    print(f"\n  Converting Style Errors to SWE Agent Input (batched per ~10 unique violations, append mode)")
    print("=" * 60)

    target_label = f"{args.org}/{args.repo}:pr-{args.pr_number}"
    jsonl_path = Path(args.results)

    file_reports = load_errors_from_results(jsonl_path, target_label)
    if not file_reports:
        print(" No violations found or failed to load style errors.")
        sys.exit(1)

    # --- FILTER: Keep only .cpp files ---
    cpp_file_reports = [r for r in file_reports if is_cpp_file_path(r.get("file", ""))]
    if not cpp_file_reports:
        print(" No .cpp files found in the PR results. Exiting.")
        sys.exit(1)

    print(f" Filtered to {len(cpp_file_reports)} .cpp file(s) from the results (non-.cpp files ignored).")
    # --------------------------------------

    dataset_info = load_original_dataset_info(args.org, args.repo, args.pr_number, args.dataset_path)
    base_commit = args.base_commit or (dataset_info.get("base_commit") if dataset_info else "main")

    # Filter files with unique violations (only among .cpp files)
    files_with_errors = []
    for r in cpp_file_reports:
        msgs = r.get("messages", []) or []
        if len(get_unique_messages(msgs)) > 0:
            files_with_errors.append(r)

    if not files_with_errors:
        print(" No .cpp files with errors found in the PR (after deduplication).")
        sys.exit(1)

    print(f" Found {len(files_with_errors)} .cpp file(s) with unique errors. Grouping into batches of >=10 unique violations...")

    # Load existing output JSON if present
    existing_instances: List[Dict[str, Any]] = []
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
    instances: List[Dict[str, Any]] = []
    batch: List[Dict[str, Any]] = []
    batch_violation_count = 0
    # Map original files by their path (use only the filtered .cpp file reports)
    original_map = {r.get("file"): r for r in cpp_file_reports}

    for fr in files_with_errors:
        unique_msgs = get_unique_messages(fr.get("messages", []) or [])
        violations_count = len(unique_msgs)
        # If adding this file would exceed the threshold, flush current batch first
        if batch and (batch_violation_count + violations_count) >= 10:
            ps = generate_problem_statement_for_batch(batch, original_map, args.style_tool)
            inst = create_sweagent_instance(args.org, args.repo, args.pr_number, base_commit, ps, instance_idx, batch_violation_count)
            instances.append(inst)
            print(f"  Created instance {instance_idx} with {batch_violation_count} unique violations ({len(batch)} file(s))")
            instance_idx += 1
            batch = []
            batch_violation_count = 0

        batch.append(fr)
        batch_violation_count += violations_count

    # Flush remaining batch
    if batch:
        ps = generate_problem_statement_for_batch(batch, original_map, args.style_tool)
        inst = create_sweagent_instance(args.org, args.repo, args.pr_number, base_commit, ps, instance_idx, batch_violation_count)
        instances.append(inst)
        print(f"  Created instance {instance_idx} with {batch_violation_count} unique violations ({len(batch)} file(s))")

    all_instances = existing_instances + instances

    # Compute and show averages
    def safe_avg(seq: List[int]) -> float:
        return float(sum(seq)) / len(seq) if seq else 0.0

    new_counts = [i.get("unique_violations", 0) for i in instances]
    avg_new = safe_avg(new_counts)

    all_counts = [i.get("unique_violations") for i in all_instances if isinstance(i.get("unique_violations", None), int)]
    avg_all = safe_avg(all_counts)

    print("\nSummary of violations per instance:")
    if instances:
        print(f" - Instances created in this run: {len(instances)}")
        print(f" - Total unique violations in newly created instances: {sum(new_counts)}")
        print(f" - Average unique violations per newly created instance: {avg_new:.2f}")
    else:
        print(" - No instances were created in this run.")

    if all_counts:
        print(f" - Instances in output with 'unique_violations' field: {len(all_counts)}")
        print(f" - Average unique violations per instance (across output file): {avg_all:.2f}")
    else:
        print(" - No existing instances in output contained 'unique_violations' field to compute overall average.")

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
