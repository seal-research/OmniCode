import json

# Paths
INPUT = "baselines/badpatchllm/logs/gemini_outputs/modified_dataset.json"
OUTPUT = "baselines/badpatchllm/logs/gemini_outputs/testgen_ready.jsonl"

# Model name
MODEL_NAME = "gemini-2.5-flash"

# Read modified dataset
with open(INPUT, "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert to JSONL with model_patch + model_name_or_path
with open(OUTPUT, "w", encoding="utf-8") as f_out:
    for instance in data:
        instance_id = instance.get("instance_id")
        bad_patches = instance.get("bad_patches", [])

        # Prefer patch from badpatchllm > gemini
        selected = None
        for bp in bad_patches:
            if isinstance(bp, dict) and bp.get("source") in {"badpatchllm", "gemini"}:
                selected = bp.get("patch")
                break

        if not selected:
            print(f"Skipping {instance_id}: No valid bad patch found.")
            continue

        instance["model_patch"] = selected
        instance["model_name_or_path"] = MODEL_NAME

        f_out.write(json.dumps(instance) + "\n")

print(f" Output written to {OUTPUT}")
