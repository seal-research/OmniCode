# utility script for result aggregation for java/cpp runs

import json 
import os

aggregated_data = {
    "total_instances": 0,
    "submitted_instances": 0,
    "completed_instances": 0,
    "incomplete_instances": 0,
    "resolved_instances": 0,
    "unresolved_instances": 0,
    "empty_patch_instances": 0,
    "error_instances": 0, 
    "submitted_ids": [],
    "completed_ids": [],
    "incomplete_ids": [],
    "resolved_ids": [],
    "unresolved_ids": [],
    "empty_patch_ids": [],
    "error_ids": []
}

file_path = "multiswebench_runs/BugFixing/output"
reports = []
for foldername, _, filenames in os.walk(file_path):
    for filename in filenames:
        if filename == "final_report.json":
            report = os.path.join(foldername, filename)
            reports.append(report)

reports.sort()
for report in reports:
    with open(report, 'r') as r:
        data = json.load(r)
        aggregated_data["total_instances"] += data.get("total_instances")
        aggregated_data["submitted_instances"] += data.get("submitted_instances")
        aggregated_data["completed_instances"] += data.get("completed_instances")
        aggregated_data["incomplete_instances"] += data.get("incomplete_instances")
        aggregated_data["resolved_instances"] += data.get("resolved_instances")
        aggregated_data["unresolved_instances"] += data.get("unresolved_instances")
        aggregated_data["empty_patch_instances"] += data.get("empty_patch_instances")
        aggregated_data["error_instances"] += data.get("error_instances")
        aggregated_data["submitted_ids"].extend(data.get("submitted_ids", []))
        aggregated_data["completed_ids"].extend(data.get("completed_ids", []))
        aggregated_data["incomplete_ids"].extend(data.get("incomplete_ids", []))
        aggregated_data["resolved_ids"].extend(data.get("resolved_ids", []))
        aggregated_data["unresolved_ids"].extend(data.get("unresolved_ids", []))
        aggregated_data["empty_patch_ids"].extend(data.get("empty_patch_ids", []))
        aggregated_data["error_ids"].extend(data.get("error_ids", []))
    print(f"{report.split('/')[-2]}: {data.get('total_instances')} total instances, "
        f"{data.get('submitted_instances')} submitted, "
        f"{data.get('completed_instances')} completed, "
        f"{data.get('incomplete_instances')} incomplete, "
        f"{data.get('resolved_instances')} resolved, "
        f"{data.get('unresolved_instances')} unresolved, "
        f"{data.get('empty_patch_instances')} empty patches, "
        f"{data.get('error_instances')} errors")

# Calculate resolve rate
resolve_rate = (
    aggregated_data["resolved_instances"] / aggregated_data["total_instances"]
    if aggregated_data["total_instances"] > 0 else 0
)
summary = "Summary: "
for key, value in aggregated_data.items():
    if isinstance(value, list):
        continue
    summary += f"{key.replace('_', ' ').capitalize()}: {value}, "
print(summary)
print(f"Resolve Rate: {resolve_rate:.2%}")

# Write output to MSWEBugFixing_results.json
output = aggregated_data.copy()
output["resolve_rate"] = resolve_rate
with open("bf_results.json", "w") as f:
    json.dump(output, f, indent=2)
