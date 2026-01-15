import json

def mswebench_id_to_codearena_id(original_id_file, outfile):
    gold_instances = []
    with open(original_id_file, 'r') as f: 
        for line in f:
            instance = line.strip()
            org = instance.split('/')[0]
            repo = instance.split('/')[1].split(':')[0]
            pr = instance.split('/')[1].split(':')[1]
            gold_instances.append(f"{org}__{repo}_{pr}")

    with open(outfile, 'w') as f:
        for instance in gold_instances:
            f.write(f"{instance}\n")

def filter_dataset_for_gold_instances(gold_instances_file, outfile):
    gold_instances = []
    with open(gold_instances_file, 'r') as f:
        for line in f:
            gold_instances.append(line.strip())
    filtered_dataset = []
    with open("mswebench_instances.json", 'r') as f:
        data = json.load(f)
        for instance in data: 
            if instance.get("instance_id") in gold_instances:
                filtered_dataset.append(instance)

    print("filtered java dataset contains: ", len(filtered_dataset), "instances")

    with open(outfile, 'w') as f:
        json.dump(filtered_dataset, f, indent=2)

if __name__ == "__main__":
    mswebench_id_to_codearena_id("mswebench_java_gold_resolved_original.txt", "mswebench_java_gold_resolved.txt")
    filter_dataset_for_gold_instances("mswebench_java_gold_resolved.txt", "codearena_instances_java.json")