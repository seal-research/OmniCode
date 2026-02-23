# utility script for result aggregation for python runs

import json 
import os

aggregated_data = {
    "total_instances": 0,
    "completed_instances": 0,
    "incomplete_instances": 0,
    "resolved_instances": 0,
    "unresolved_instances": 0,
    "empty_patch_instances": 0,
    "patch_apply_fail_instances": 0,
    "error_instances": 0, 
    "submitted_ids": [],
    "completed_ids": [],
    "incomplete_ids": [],
    "resolved_ids": [],
    "unresolved_ids": [],
    "empty_patch_ids": [],
    "patch_apply_fail_ids": [],
    "error_ids": []
}

file_path = "logs/"
total = 0
reports = []
for foldername, _, filenames in os.walk(file_path):
    if foldername.startswith('logs/gold'): continue
    if foldername.endswith('_report'):
        id = foldername.split('_report')[0].split('/')[-1]
        aggregated_data["submitted_ids"].append(id)
    for filename in filenames:
        if filename == "report.json":
            report = os.path.join(foldername, filename)
            reports.append(report)
reports.sort()
for report in reports:
    with open(report, 'r') as r:
        data = json.load(r)
        for id in data.keys():
            if data[id]["patch_successfully_applied"]:
                if id not in aggregated_data["completed_ids"]: 
                    aggregated_data["completed_ids"].append(id)
                if data[id]["resolved"]:
                    if id not in aggregated_data["resolved_ids"]:
                        aggregated_data["resolved_ids"].append(id)
                else:
                    if id not in aggregated_data["unresolved_ids"]:
                        aggregated_data["unresolved_ids"].append(id)
            else: 
                if id not in aggregated_data["incomplete_ids"]:
                    aggregated_data["incomplete_ids"].append(id)
                if data[id]["patch_exists"]:
                    if id not in aggregated_data["error_ids"]:
                        aggregated_data["error_ids"].append(id)
                else: 
                    if id not in aggregated_data["emply_patch_ids"]:
                        aggregated_data["empty_patch_ids"].append(id)
aggregated_data["total_instances"] = len(aggregated_data["submitted_ids"])
aggregated_data["completed_instances"] = len(aggregated_data["completed_ids"])
aggregated_data["incomplete_instances"] = len(aggregated_data["incomplete_ids"])
aggregated_data["resolved_instances"] = len(aggregated_data["resolved_ids"])
aggregated_data["unresolved_instances"] = len(aggregated_data["unresolved_ids"])
aggregated_data["error_instances"] = len(aggregated_data["error_ids"])
aggregated_data["empty_patch_instances"] = len(aggregated_data["empty_patch_ids"])
patch_apply_fail = []
for id in aggregated_data["submitted_ids"]: 
    if id not in aggregated_data["completed_ids"]:
        aggregated_data["patch_apply_fail_ids"].append(id)
aggregated_data["patch_apply_fail_instances"] = len(aggregated_data["patch_apply_fail_ids"])

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

with open("completed_ids.txt", "w") as f:
    for id in aggregated_data["completed_ids"]: 
        f.write(id)
        f.write('\n')
