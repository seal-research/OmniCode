import json
from pathlib import Path

# Paths
existing_path = Path("data/codearena_instances.json")
new_path = Path("baselines/badpatchllm/logs/gemini_outputs/modified_dataset.json")

# Load both JSONs
with existing_path.open("r", encoding="utf-8") as f:
    existing_instances = json.load(f)

with new_path.open("r", encoding="utf-8") as f:
    new_instances = json.load(f)

# Build a mapping for fast lookup
existing_map = {inst["instance_id"]: inst for inst in existing_instances}
new_map = {inst["instance_id"]: inst for inst in new_instances}

# Merge reviews
for instance_id, new_inst in new_map.items():
    if "reviews" not in new_inst or not new_inst["reviews"]:
        continue  # skip if no new reviews

    if instance_id in existing_map:
        existing_reviews = existing_map[instance_id].get("reviews", [])
        existing_map[instance_id]["reviews"] = list(set(existing_reviews + new_inst["reviews"]))
    else:
        # If not present in original, append as a new instance
        existing_instances.append(new_inst)

# Save back to original file
with existing_path.open("w", encoding="utf-8") as f:
    json.dump(existing_instances, f, indent=2)

print("Reviews successfully appended.")
