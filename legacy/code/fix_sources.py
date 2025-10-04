import json

json_path = "data/codearena_instances.json"

with open(json_path, "r") as f:
    data = json.load(f)

for instance in data:
    if instance.get("instance_id") == "astropy__astropy-13033":
        for bp in instance.get("bad_patches", []):
            if bp.get("source") == "agentless":
                bp["source"] = "gemini"

with open(json_path, "w") as f:
    json.dump(data, f, indent=2)

print("✅ Patched all 'agentless' bad_patches to 'gemini' for astropy__astropy-13033")
