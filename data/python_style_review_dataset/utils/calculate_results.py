import json

file = "experiments/aider_gemini.json"

with open(file, 'r') as f:
    data = json.load(f)

metrics = {
    "total_instances": 0, 
    "score": 0.0,
    "fix_rate": 0.0, 
    "error_ratio": 0.0,
    "score2": 0.0
}



for instance in data:
    metrics["total_instances"] += 1
    metrics["score"] += max((instance["solved_errors"] - instance["newly_created_errors"])/instance["dataset_total_errors"], 0)
    metrics["fix_rate"] += instance["solved_errors"]/instance["dataset_total_errors"]
    metrics["error_ratio"] += (instance["solved_errors"] - instance["dataset_total_errors"] + instance["newly_created_errors"])/instance["dataset_total_errors"]
    metrics["score2"] += instance["solved_errors"]/(instance["dataset_total_errors"] + instance["newly_created_errors"])

metrics["score"] /= metrics["total_instances"]
metrics["fix_rate"] /= metrics["total_instances"]
metrics["error_ratio"] /= metrics["total_instances"]
metrics["score2"] /= metrics["total_instances"]

print(metrics)