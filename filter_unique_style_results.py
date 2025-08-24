import json

input_file = "original_results_pmd.jsonl"
output_file = "unique_results_pmd.jsonl"

seen_repos = set()   # tracks org/repo processed
with open(input_file, "r") as fin, open(output_file, "w") as fout:
    for line in fin:
        record = json.loads(line)

        # Parse org/repo:pr-id
        label = record["label"]  # e.g., elastic/logstash:pr-17021
        repo_id, pr_id = label.split(":")
        
        # Only keep the first PR for a repo_id
        if repo_id in seen_repos:
            continue
        seen_repos.add(repo_id)

        # Deduplicate errors within this PR
        unique_sources = set()
        for f in record.get("files", []):
            unique_msgs = []
            for msg in f.get("messages", []):
                # Dedup by "source" field
                if msg["source"] not in unique_sources:
                    unique_sources.add(msg["source"])
                    unique_msgs.append(msg)
            f["messages"] = unique_msgs  # replace with deduped list

        # Write updated record
        fout.write(json.dumps(record) + "\n")

print(f"✅ Unique results written to {output_file}")
