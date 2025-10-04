#!/usr/bin/env python3
import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Optional


def load_errors_from_jsonl(jsonl_path: Path, target_label: str) -> List[Dict]:
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
    violations = []
    for message in file_report.get("messages", []):
        line = message.get("line", 0)
        column = message.get("column", 0)
        msg = message.get("message", "")
        source = message.get("source", "")
        violation = f"Line {line}, Column {column}: {msg} [{source}]"
        violations.append(violation)
    return violations


def extract_modified_files_from_patch(patch_text: str) -> List[str]:
    """Extract a list of modified file paths from a unified diff/patch string."""
    modified_files = []
    for line in patch_text.splitlines():
        if line.startswith("+++ b/"):
            filepath = line.replace("+++ b/", "").strip()
            modified_files.append(filepath)
    print(modified_files)
    return modified_files


def filter_patch_by_files(patch_text: str, modified_files: List[str]) -> str:
    """Extract only the patch hunks that correspond to the modified files."""
    filtered_lines = []
    lines = patch_text.splitlines()
    i = 0
    keep = False
    while i < len(lines):
        line = lines[i]
        if line.startswith("diff --git"):
            # Look ahead to see what file is being patched
            j = i + 1
            while j < len(lines) and not lines[j].startswith("+++ b/"):
                j += 1
            if j < len(lines):
                file_path = lines[j].replace("+++ b/", "").strip()
                keep = file_path in modified_files
            else:
                keep = False

        if keep:
            filtered_lines.append(line)

        i += 1

    return "\n".join(filtered_lines)


def generate_problem_statement(original_errors: List[Dict], patched_errors: List[Dict], style_tool: str, modified_files: List[str]) -> str:
    original_map = {error["file"]: error for error in original_errors}
    patched_map = {error["file"]: error for error in patched_errors}

    filtered_patched_map = {f: patched_map[f] for f in patched_map if f.replace("/workspace/repo/", "") in modified_files}

    problematic_files = []
    for file_path, patched_report in filtered_patched_map.items():
        if patched_report["error_count"] > 0:
            original_report = original_map.get(file_path, {"error_count": 0, "score": 10.0})
            problematic_files.append({
                "file": file_path,
                "original_score": original_report.get("score", 10.0),
                "patched_score": patched_report.get("score", 10.0),
                "original_errors": original_report.get("error_count", 0),
                "patched_errors": patched_report["error_count"],
                "violations": extract_violations_from_file(patched_report)
            })

    if not problematic_files:
        return f"No {style_tool.upper()} violations found in modified files."

    problem_statement = f"""Fix the following {style_tool.upper()} style violations in the modified code:\n\n"""
    total_violations = 0

    for file_info in problematic_files[:10]:
        total_violations += file_info["patched_errors"]
        display_path = file_info["file"].replace("/workspace/repo/", "")
        problem_statement += f"File: {display_path}\n"
        problem_statement += f"Score: {file_info['patched_score']}/10.0 (was {file_info['original_score']}/10.0)\n"
        problem_statement += f"Violations: {file_info['patched_errors']} (was {file_info['original_errors']})\n\n"

        for i, violation in enumerate(file_info["violations"][:5]):
            problem_statement += f"  {violation}\n"

        if len(file_info["violations"]) > 5:
            problem_statement += f"  ... and {len(file_info['violations']) - 5} more violations\n"

        problem_statement += "\n"

    total_original_errors = sum(original_map.get(f["file"], {}).get("error_count", 0) for f in problematic_files)
    total_patched_errors = sum(f["patched_errors"] for f in problematic_files)

    if total_original_errors > 0:
        improvement = total_original_errors - total_patched_errors
        improvement_text = f"Improved by {improvement} violations" if improvement > 0 else f"Still has {total_patched_errors} violations"
    else:
        improvement_text = f"Has {total_patched_errors} violations"

    problem_statement += f"""
Summary:
- Total modified files with violations: {len(problematic_files)}
- Total violations: {total_violations}
- {improvement_text}

Please fix all the violations while maintaining the original functionality of the code.
Focus on the most critical issues first and ensure the code follows Java best practices.
"""
    return problem_statement


def create_sweagent_instance(org: str, repo: str, pr_number: int, base_commit: str, problem_statement: str, original_patch: str = "") -> Dict:
    instance_id = f"{org}/{repo}:{pr_number}"
    return {
        "instance_id": instance_id,
        "org": org,
        "repo": repo,
        "number": pr_number,
        "base_commit": base_commit,
        "problem_statement": problem_statement,
        "patch": "",
        "mode": "stylereview"
    }


def load_original_dataset_info(org: str, repo: str, pr_number: int, dataset_path: str = "data/multiswebench_data/mswebench_instances.json") -> Optional[Dict]:
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
    parser = argparse.ArgumentParser(description="Convert Style Errors to SWE Agent Input")
    parser.add_argument("--org", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--pr_number", type=int, required=True)
    parser.add_argument("--style_tool", required=True, choices=["checkstyle", "pmd"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset_path", default="data/multiswebench_data/mswebench_instances.json")
    parser.add_argument("--base_commit", help="Base commit hash (if not in dataset)")
    args = parser.parse_args()

    print(f"\n  Converting Style Errors to SWE Agent Input")
    print("=" * 60)
    print(f"Org        : {args.org}")
    print(f"Repo       : {args.repo}")
    print(f"PR Number  : {args.pr_number}")
    print(f"Style Tool : {args.style_tool}")
    print(f"Output     : {args.output}")
    print("=" * 60)

    target_label = f"{args.org}/{args.repo}:pr-{args.pr_number}"
    jsonl_filename = f"original_results_{args.style_tool}.jsonl"
    jsonl_path = Path(jsonl_filename)

    print(f"\n Loading style errors from {jsonl_path}...")
    original_errors = load_errors_from_jsonl(jsonl_path, target_label)
    patched_errors = original_errors

    if not original_errors:
        print(" No violations found or failed to load style errors.")
        sys.exit(1)

    print("\n Loading dataset info for base commit...")
    dataset_info = load_original_dataset_info(args.org, args.repo, args.pr_number, args.dataset_path)
    base_commit = args.base_commit or (dataset_info.get("base_commit") if dataset_info else "main")
    patch_text = dataset_info.get("patch", "") if dataset_info else ""

    print("\n Extracting modified files from patch...")
    modified_files = extract_modified_files_from_patch(patch_text)

    print("\n Filtering patch to only modified files...")
    filtered_patch = filter_patch_by_files(patch_text, modified_files)

    print("\n Generating problem statement...")
    problem_statement = generate_problem_statement(original_errors, patched_errors, args.style_tool, modified_files)

    print("\n Creating SWE-agent instance...")
    sweagent_instance = create_sweagent_instance(
        org=args.org,
        repo=args.repo,
        pr_number=args.pr_number,
        base_commit=base_commit,
        problem_statement=problem_statement,
        original_patch=filtered_patch
    )

    print(f"\n Saving instance to {args.output}...")
    try:
        with open(args.output, 'w') as f:
            json.dump([sweagent_instance], f, indent=2)
        print(" Successfully saved.")
    except Exception as e:
        print(f" Failed to save: {e}")
        sys.exit(1)

    print("\n Sample run command:")
    print("=" * 60)
    output_dir = f"sweagent_{args.style_tool}_{args.org}_{args.repo}_{args.pr_number}_results"
    print(f"""python sweagent_regular.py \\
  -i {args.output} \\
  -o {output_dir} \\
  --style_tool {args.style_tool} \\
  --model_name "gemini/gemini-2.5-flash-preview-04-17" \\
  --api_key [your_api_key]""")
    print("=" * 60)
    print("\n🎉 Done!")


if __name__ == "__main__":
    main()

