#!/usr/bin/env python3
"""
Convert Style Errors to SWE Agent Input Format
This script converts style error files from the style review output into the format
expected by jsweagent_regular.py.
Usage:
    python convert_style_errors_to_sweagent.py \
        --org apache \
        --repo dubbo \
        --pr_number 10638 \
        --style_tool checkstyle \
        --output sweagent_input.json
"""

import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Optional

def load_style_errors(file_path: Path) -> List[Dict]:
    """Load style errors from a JSON file."""
    if not file_path.exists():
        print(f"Style errors file not found: {file_path}")
        return []

    try:
        with open(file_path, 'r') as f:
            errors = json.load(f)
        print(f"Loaded style errors: {file_path}")
        print(f"   Files with violations: {len([f for f in errors if f['error_count'] > 0])}")
        return errors
    except Exception as e:
        print(f"Error loading style errors: {e}")
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

def generate_problem_statement(original_errors: List[Dict], patched_errors: List[Dict], style_tool: str) -> str:
    """Generate a problem statement from style violations."""

    # Create a mapping of file paths to their error reports
    original_map = {error["file"]: error for error in original_errors}
    patched_map = {error["file"]: error for error in patched_errors}

    # Find files that still have violations after patching
    problematic_files = []

    for file_path, patched_report in patched_map.items():
        if patched_report["error_count"] > 0:
            original_report = original_map.get(file_path, {"error_count": 0, "score": 10.0})

            problematic_files.append({
                "file": file_path,
                "original_score": original_report["score"],
                "patched_score": patched_report["score"],
                "original_errors": original_report["error_count"],
                "patched_errors": patched_report["error_count"],
                "violations": extract_violations_from_file(patched_report)
            })

    if not problematic_files:
        return f"No {style_tool.upper()} violations found. The code appears to be style-compliant."

    # Build problem statement
    problem_statement = f"""Fix the following {style_tool.upper()} style violations in the codebase:
"""

    total_violations = 0
    # Limit to first 10 files to keep problem statement manageable
    for file_info in problematic_files[:10]:
        total_violations += file_info["patched_errors"]

        # Clean up file path for display
        display_path = file_info["file"].replace("/workspace/repo/", "")

        problem_statement += f"File: {display_path}\n"
        problem_statement += f"Score: {file_info['patched_score']}/10.0 (was {file_info['original_score']}/10.0)\n"
        problem_statement += f"Violations: {file_info['patched_errors']} (was {file_info['original_errors']})\n\n"

        # Show violations (limit to first 5 to keep problem statement manageable)
        for i, violation in enumerate(file_info["violations"][:5]):
            problem_statement += f"  {violation}\n"

        if len(file_info["violations"]) > 5:
            problem_statement += f"  ... and {len(file_info['violations']) - 5} more violations\n"

        problem_statement += "\n"

    # Calculate overall improvement
    total_original_errors = sum(original_map.get(f["file"], {}).get("error_count", 0) for f in problematic_files)
    total_patched_errors = sum(f["patched_errors"] for f in problematic_files)

    if total_original_errors > 0:
        improvement = total_original_errors - total_patched_errors
        improvement_text = f"Improved by {improvement} violations" if improvement > 0 else f"Still has {total_patched_errors} violations"
    else:
        improvement_text = f"Has {total_patched_errors} violations"

    problem_statement += f"""
Summary:
- Total files with violations: {len(problematic_files)}
- Total violations: {total_violations}
- {improvement_text}
Please fix all the violations while maintaining the original functionality of the code.
Focus on the most critical issues first and ensure the code follows Java best practices.
"""

    return problem_statement

def create_sweagent_instance(
    org: str,
    repo: str,
    pr_number: int,
    base_commit: str,
    problem_statement: str,
    original_patch: str = ""
) -> Dict:
    """Create a SWE agent instance in the expected format."""

    instance_id = f"{org}/{repo}:{pr_number}"

    return {
        "instance_id": instance_id,
        "org": org,
        "repo": repo,
        "number": pr_number,
        "base_commit": base_commit,
        "problem_statement": problem_statement,
        "patch": original_patch,
        "mode": "stylereview"
    }

def load_original_dataset_info(org: str, repo: str, pr_number: int, dataset_path: str = "data/multiswebench_data/mswebench_instances.json") -> Optional[Dict]:
    """Load original dataset information for the specific PR."""

    if not Path(dataset_path).exists():
        print(f"⚠️  Dataset not found: {dataset_path}")
        return None

    try:
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)

        target_repo = f"{org}/{repo}"
        for instance in dataset:
            if (instance.get("repo") == target_repo and 
                instance.get("pull_number") == pr_number):
                print(f"Found instance in dataset: {target_repo}:{pr_number}")
                return instance

        print(f"Instance {target_repo}:{pr_number} not found in dataset")
        return None

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def main():
    """Main conversion workflow."""

    parser = argparse.ArgumentParser(description="Convert Style Errors to SWE Agent Input")
    parser.add_argument("--org", required=True, help="Organization name (e.g., apache)")
    parser.add_argument("--repo", required=True, help="Repository name (e.g., dubbo)")
    parser.add_argument("--pr_number", type=int, required=True, help="Pull request number")
    parser.add_argument("--style_tool", required=True, choices=["checkstyle", "pmd"], help="Style tool used")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument("--dataset_path", default="data/multiswebench_data/mswebench_instances.json", help="Path to original dataset")
    parser.add_argument("--base_commit", help="Base commit hash (if not in dataset)")

    args = parser.parse_args()

    print("Converting Style Errors to SWE Agent Input")
    print("=" * 50)
    print(f"Organization: {args.org}")
    print(f"Repository: {args.repo}")
    print(f"PR Number: {args.pr_number}")
    print(f"Style Tool: {args.style_tool}")
    print(f"Output: {args.output}")
    print()

    # Construct file paths
    base_path = Path("./data/java_style_review") / args.org / args.repo / "style_review" / f"style-review-{args.pr_number}"
    original_errors_path = base_path / "original_style_errors.json"
    patched_errors_path = base_path / "patched_style_errors.json"

    print(f"Looking for style error files:")
    print(f"   Original: {original_errors_path}")
    print(f"   Patched: {patched_errors_path}")
    print()

    # Load style errors
    print("Loading style error files...")
    original_errors = load_style_errors(original_errors_path)
    patched_errors = load_style_errors(patched_errors_path)

    if not original_errors and not patched_errors:
        print("No style error files found. Make sure the style review has been run.")
        sys.exit(1)

    # Load original dataset information
    print("\nLoading original dataset information...")
    dataset_info = load_original_dataset_info(args.org, args.repo, args.pr_number, args.dataset_path)
    # Get base commit
    base_commit = args.base_commit
    if not base_commit and dataset_info:
        base_commit = dataset_info.get("base_commit", "")
    if not base_commit:
        base_commit = "main"  # fallback

    # Get original patch if available
    original_patch = ""
    if dataset_info:
        original_patch = dataset_info.get("patch", "")

    # Generate problem statement
    print("\n Generating problem statement...")
    problem_statement = generate_problem_statement(original_errors, patched_errors, args.style_tool)

    print(f"Generated problem statement ({len(problem_statement)} characters)")

    # Create SWE agent instance
    print("\n🔧 Creating SWE agent instance...")
    sweagent_instance = create_sweagent_instance(
        org=args.org,
        repo=args.repo,
        pr_number=args.pr_number,
        base_commit=base_commit,
        problem_statement=problem_statement,
        original_patch=original_patch
    )

    # Save to output file
    print(f"\nSaving to {args.output}...")
    try:
        with open(args.output, 'w') as f:
            json.dump([sweagent_instance], f, indent=2)

        print(f"Successfully saved SWE agent input: {args.output}")
        print(f"   Instance ID: {sweagent_instance['instance_id']}")
        print(f"   Base commit: {sweagent_instance['base_commit']}")
        print(f"   Problem statement length: {len(sweagent_instance['problem_statement'])} characters")

    except Exception as e:
        print(f" Error saving output file: {e}")
        sys.exit(1)

    # Generate command to run SWE agent
    print(f"\n🚀To run the SWE agent, use this command:")
    print("=" * 60)
    output_dir = f"sweagent_{args.style_tool}_{args.org}_{args.repo}_{args.pr_number}_results"
    command = f"""python sweagent_regular.py \\
  -i {args.output} \\
  -o {output_dir} \\
  --style_tool {args.style_tool} \\
  --model_name \"gemini/gemini-2.5-flash-preview-04-17\" \\
  --api_key [your_api_key]"""
    print(command)
    print("=" * 60)

    print("\n🎉 Conversion completed successfully!")

if __name__ == "__main__":
    main() 