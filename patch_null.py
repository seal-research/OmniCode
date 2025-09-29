import json

# Input/output file
file_path = "sweagent_clang-tidy_results_deepseek.json"

# Load JSON
with open(file_path, "r") as f:
    data = json.load(f)

# Replace empty string in "patch" with None
for entry in data:
    if "patch" in entry :
        entry["patch"] = ""

# Save back
with open(file_path, "w") as f:
    json.dump(data, f, indent=2)

print("✅ Updated sweagent_pmd_results.json (empty patch → null)")
