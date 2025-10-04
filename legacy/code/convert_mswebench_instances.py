import os
import json
import sys

def process_dataset_file(dataset_path):
    instances = []
    with open(dataset_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            instance = {}
            # Fill fields from dataset entry
            org = entry.get("org", "")
            repo = entry.get("repo", "")
            pr_number = entry.get("number", "")
            problem_statement = [f'{issue["title"]}\n{issue["body"]}' for issue in entry.get("resolved_issues", [])]
            f2p = list(entry.get("f2p_tests").keys())
            f2p.extend(list(entry.get("s2p_tests").keys())) # add skip-to-pass tests
            f2p.extend(list(entry.get("n2p_tests").keys())) # add none-to-pass tests
            instance["repo"] = f"{org}/{repo}" if org and repo else ""
            instance["pull_number"] = pr_number
            instance["instance_id"] = f"{org}__{repo}_{pr_number}" if org and repo and pr_number else ""
            instance["issue_numbers"] = [issue["number"] for issue in entry.get("resolved_issues", [])]
            instance["base_commit"] = entry.get("base", {}).get("sha", "")
            instance["patch"] = entry.get("fix_patch", "")
            instance["test_patch"] = entry.get("test_patch", "")
            instance["problem_statement"] = problem_statement[0]
            instance["hints_text"] = f'{entry.get("title", "")}\n{entry.get("body", "")}'
            instance["created_at"] = ""
            instance["version"] = ""
            instance["PASS_TO_PASS"] = list(entry.get("p2p_tests").keys())
            instance["FAIL_TO_PASS"] = f2p
            if entry.get("bad_patches"): 
                instance["bad_patches"] = entry.get("bad_patches")
            else: 
                instance["bad_patches"] = []
            instances.append(instance)
    return instances

def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <repo__org_dataset.jsonl>")
        sys.exit(1)
    dataset_path = sys.argv[1]
    if not os.path.isfile(dataset_path):
        if not os.path.isdir(dataset_path):
            print(f"{dataset_path} does not exist")
            sys.exit(1)
        print(f"{dataset_path} is a directory, processing all JSONL files in it")
        files = [f for f in os.listdir(dataset_path) if f.endswith("_dataset.jsonl")]
        if not files:
            print(f"No dataset files found in {dataset_path}")
            sys.exit(1)
        files.sort()
        dataset_files = [os.path.join(dataset_path, f) for f in files]
        instances = []
        for dataset_file in dataset_files:
            print(f"Processing dataset file: {dataset_file}")
            instances.extend(process_dataset_file(dataset_file))
    else:
        print(f"Processing single dataset file: {dataset_path}")
        if not dataset_path.endswith("_dataset.jsonl"):
            print(f"Expected dataset file to end with '_dataset.jsonl', got {dataset_path}")
            sys.exit(1)
        if not os.path.isfile(dataset_path):
            print(f"Dataset file does not exist: {dataset_path}")
            sys.exit(1)
        instances = process_dataset_file(dataset_path)
    # out_path = dataset_path.replace("_dataset.jsonl", "_instances.json")
    # with open(out_path, "w") as out_f:
    #     json.dump(instances, out_f, indent=2)
    # print(f"Wrote {len(instances)} instances to {out_path}")
    # codearena_instance_path = "data/codearena_instances_java.json"
    codearena_instance_path = "data/codearena_instances_java.json"
    # Load existing instances if file exists
    if os.path.isfile(codearena_instance_path):
        with open(codearena_instance_path, "r") as f:
            try:
                existing_instances = json.load(f)
            except Exception:
                existing_instances = []
    else:
        existing_instances = []
    # Build a dict for fast lookup
    id_to_instance = {inst.get("instance_id", ""): inst for inst in existing_instances}
    # Update or add new instances
    for inst in instances:
        id_to_instance[inst["instance_id"]] = inst
    # Write back to file
    with open(codearena_instance_path, "w") as out_f:
        json.dump(list(id_to_instance.values()), out_f, indent=2)
    print(f"Wrote {len(instances)} new/updated instances to {codearena_instance_path}")

if __name__ == "__main__":
    main()