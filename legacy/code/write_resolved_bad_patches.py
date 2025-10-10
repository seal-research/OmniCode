import os
import json

# Path to the input JSON
input_path = "data/codearena_instances.json"
# Base directory where resolved patches should be written
base_output_dir = "baselines/badpatchllm/logs/gemini_outputs"

# Load all instances
with open(input_path, "r") as f:
    instances = json.load(f)

count_written = 0
for instance in instances:
    instance_id = instance.get("instance_id")
    bad_patches = instance.get("bad_patches", [])

    if not bad_patches or not instance_id:
        continue

    resolved_dir = os.path.join(base_output_dir, instance_id, "resolved")
    os.makedirs(resolved_dir, exist_ok=True)

    for patch in bad_patches:
        idx = patch["idx"]
        diff = patch["patch"]

        patch_path = os.path.join(resolved_dir, f"patch_{idx}.diff")
        # Inside the for patch in bad_patches loop
        patch["source"] = "gemini"  # Override source

        # Then write patch file as before

        with open(patch_path, "w") as f:
            f.write(diff)
        count_written += 1

print(f"Wrote {count_written} bad patch files to disk.")
