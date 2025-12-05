"""
This script analyzes .traj files in a given folder and categorizes them based on the taxonomy below: 

Failure Modes
│
├── Patch Generated
│   │
│   ├── Auto-Submitted Patch
│   │      → Reason: Premature Submission caused by tool call errors
│   │      → Category: Premature Submit
|   |
|   ├── Failed to Build after Applying Patch
│   │      → Reason 1: Patch Apply Failed
│   │      → Reason 2: Env Build Failed after Patch Applied
│   │      → Category: Invalid Patch
│   │
│   ├── Patch in Wrong File
|   |      → Reason: Localization Error
|   |      → Category: Failed Localization
│   │
│   └── Patch in Correct File
│       │
│       └── Instance Unresolved
│              → Reason: Tests Failed
│              → Category: Incorrect Fix
|
└── No Patch Generated:
    │
    ├── Premature Submission
    │      → Reason: Submitted empty patch after tool call errors 
    │      → Category: Empty Patch
    │
    └── Otherwise
           → Reason 1: Max Cost Reached
           → Reason 2: Consecutive Empty Tool Calls
           → Category: No Patch Generated

Example Command: 
python trajectory_analysis.py --traj_dir baselines/logs --ref ref_results.json 

Arguments:
--traj_dir: Path to directory containing sweagent run logs
--ref: Path to file that contains resolved/unresolved results
--output: Output file path

"""

import json
from pathlib import Path
from typing import Dict, Any
import re


def load_traj(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except:
        return {}


def detect_patch_generated(traj: dict) -> bool:
    info = traj.get("info", None)
    if "submission" in info.keys(): 
        if info["submission"]:
            return True
    return False


def extract_modified_files(diff_text: str) -> set[str]:
    """
    Extract file paths modified in a unified diff.
    Returns a set of file paths.
    """
    if not diff_text:
        return set()

    # Matches "diff --git a/path/to/file.py b/path/to/file.py"
    pattern = r"diff --git a/(.*?) b/"
    return set(re.findall(pattern, diff_text))


def detect_correct_file(traj: dict, gold_diff: str) -> bool:
    """
    Determine if the model patch touches the same files as the gold patch.
    Returns True if overlaps exist or if both modify the same set of files.
    """
    # Extract model submission patch
    info = traj.get("info", {})
    model_diff = info.get("submission", "")

    # Extract file sets
    gold_files = extract_modified_files(gold_diff)
    model_files = extract_modified_files(model_diff)

    if not model_files:
        return False  # model touched nothing

    # Perfect match
    if sorted(gold_files) == sorted(model_files):
        return True

    # Partial overlap is also considered correct
    if gold_files.issubset(model_files) or model_files.issubset(gold_files):
        return True

    return False


def detect_auto_submit(traj: dict) -> bool:
    for step in traj.get("trajectory", []):
        s = step.get("state", {})
        if s.get("auto_submit") is True:
            return True
        if isinstance(step.get("action"), str) and "auto" in step["action"].lower():
            return True
    return False


def detect_submit_without_patch(traj: dict) -> bool:
    for step in traj.get("trajectory", []):
        if isinstance(step.get("action"), str) and "submit" in step["action"].lower():
            return True
    return False


def analyze_unresolved_traj(traj: dict, gold_diff: str) -> str:
    """
    Only called for unresolved instances.
    Applies standard “unresolved” breakdown logic.
    """
    patch_generated = detect_patch_generated(traj)
    correct_file = detect_correct_file(traj, gold_diff)
    auto_submit = detect_auto_submit(traj)
    submit_without_patch = detect_submit_without_patch(traj)

    if patch_generated:
        if auto_submit:
            return "premature_submit"
        elif not correct_file:
            return "failed_localization"
        else:
            return "incorrect_fix"

    else:
        if submit_without_patch:
            return "empty_patch"
        else:
            return "no_patch_generated"


def categorize_with_ref(ref_json_path: str, traj_folder: str):
    if ref_json_path is None:
        all_ids = set()
        resolved_ids = set()
        unresolved_ids = set()
        patch_apply_fail_ids = set()
    else:
        ref = json.load(open(ref_json_path))
        all_ids = set(ref.get("submitted_ids", []))
        resolved_ids = set(ref.get("resolved_ids", []))
        unresolved_ids = set(ref.get("unresolved_ids", []))
        patch_apply_fail_ids = set(ref.get("patch_apply_fail_ids", []))

    traj_folder = Path(traj_folder)
    results = {}

    # NEW: total patches counted
    total_parsed = 0
    with open("gold_patches.json") as f:
        data = json.load(f)

    for traj_path in traj_folder.rglob("*.traj"):
        total_parsed += 1  # Count every file

        instance_id = traj_path.stem

        # Skip resolved
        # if instance_id not in all_ids:
        #     continue
        if instance_id in resolved_ids:
            continue

        traj = load_traj(traj_path)
        patch_generated = detect_patch_generated(traj)
        if instance_id in patch_apply_fail_ids:
            reason = "invalid_patch"
        elif instance_id in unresolved_ids:
            reason = analyze_unresolved_traj(traj, data[instance_id])
        else:
            # Not resolved & not unresolved
            if patch_generated:
                if detect_auto_submit(traj):
                    reason = "failed_tool_call_submission"
                else:
                    reason = "invalid_patch"
            else:
                if detect_submit_without_patch(traj):
                    reason = "empty_patch"
                else:
                    reason = "no_patch_generated"

        results.setdefault(reason, []).append(instance_id)

    return results, total_parsed


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_dir", default="baselines/sweagent/logs")
    parser.add_argument("--ref", default="results_ref.json")
    parser.add_argument("--output", default="failure_modes.json")
    args = parser.parse_args()

    result, total = categorize_with_ref(args.ref, args.traj_dir)

    print("\n=== Categorization (Skipping Resolved) ===")
    output_results = result.copy()
    for k, v in result.items():
        print(f"\n{k.upper()} ({len(v)})")
        output_results[k.upper()+"_COUNT"] = len(v)
        for inst in v:
            print(" -", inst)
    output_results["total"] = total
    if args.output:
        with open(args.output, "w") as f:
            json.dump(output_results, f, indent=2)
        print("\nSaved to:", args.output)
