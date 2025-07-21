import os
import json

instance_id = ""
diffs_dir = f"baselines/badpatchllm/logs/gemini_outputs/{instance_id}/resolved"
output_jsonl = f"baselines/badpatchllm/logs/gemini_outputs/predictions_test.jsonl"

with open(output_jsonl, "w") as outfile:
    for fname in sorted(os.listdir(diffs_dir)):
        if fname.endswith(".diff"):
            with open(os.path.join(diffs_dir, fname), "r") as f:
                patch = f.read()
            json.dump({"instance_id": instance_id, "patch": patch}, outfile)
            outfile.write("\n")

print(f "Created {output_jsonl}")

