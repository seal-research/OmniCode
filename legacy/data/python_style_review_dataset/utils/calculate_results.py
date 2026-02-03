import json
import numpy as np
from scipy import stats


def ci(m, n): 
    return 1.96 * np.sqrt(m * (1 - m) / n)

def calculate_score(file, test_run_file):
    with open(file, 'r') as f:
        data = json.load(f)

    with open(test_run_file, 'r') as f:
        test_results = json.load(f)

    resolved = test_results["resolved_ids"]

    metrics = {
        "total_instances": 0, 
        "score": 0.0,
        "fix_rate": 0.0, 
        "error_ratio": 0.0,
        "score2": 0.0
    }

    def mean_confidence_interval(data: list[float], confidence: float = 0.95) -> list[float, float, float, float]:
        arr = np.array(data, dtype=float)
        mean = arr.mean()
        sem = stats.sem(arr)
        h = sem * stats.t.ppf((1 + confidence) / 2., len(arr) - 1)
        return mean, h

    scores = []
    for instance in data:
        metrics["total_instances"] += 1
        if instance["instance_id"] in resolved:
            scores.append(max((instance["solved_errors"] - instance["newly_created_errors"])/instance["dataset_total_errors"], 0))
            metrics["score"] += max((instance["solved_errors"] - instance["newly_created_errors"])/instance["dataset_total_errors"], 0)
        else: 
            scores.append(0)
        metrics["fix_rate"] += instance["solved_errors"]/instance["dataset_total_errors"]
        metrics["error_ratio"] += (instance["dataset_total_errors"] - instance["solved_errors"] + instance["newly_created_errors"])/instance["dataset_total_errors"]
        metrics["score2"] += instance["solved_errors"]/(instance["dataset_total_errors"] + instance["newly_created_errors"])

    metrics["score"] /= metrics["total_instances"]
    metrics["fix_rate"] /= metrics["total_instances"]
    metrics["error_ratio"] /= metrics["total_instances"]
    metrics["score2"] /= metrics["total_instances"]
    metrics["confidence"] = mean_confidence_interval(scores)

    print(metrics)
    return metrics


if __name__ == "__main__":
    test_prefix = "experiments/test_run_results/"
    file_prefix = "experiments/"
    test_runs = ["sweagent_gemini.json", "sweagent_deepseek.json", "sweagent_gpt5.json", "sweagent_qwen3.json", "aider_gemini.json"]
    metrics = {}
    for run in test_runs: 
        metrics[run.split('.')[0]] = calculate_score(file_prefix + run, test_prefix + run)
    with open("style_review_scores.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)