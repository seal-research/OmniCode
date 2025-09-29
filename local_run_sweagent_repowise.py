#!/usr/bin/env python3
import json, os, sys, subprocess

USAGE = "Usage: {} <org> <repo> <pr> [--dry-run]".format(sys.argv[0])

if len(sys.argv) < 4:
    print("ERROR: missing required arguments.\n" + USAGE)
    sys.exit(1)

org = sys.argv[1]
repo = sys.argv[2]
pr = sys.argv[3]
dry_run = "--dry-run" in sys.argv

# keep defaults exactly as original
tasks_file = "sweagent_clang-tidy_results.json"
instances_file = "sweagent_clang-tidy_results.json"

api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    print("ERROR: OPENROUTER_API_KEY not set in environment.")
    sys.exit(1)

with open(instances_file, "r") as f:
    data = json.load(f)

for entry in data:
    instance = entry.get("instance_id")
    if not instance:
        continue

    # only run when all three substrings are present in instance_id
    if not (org in instance and repo in instance and pr in instance):
        continue

    cmd = [
        "python", "baselines/sweagent/sweagent_regular.py",
        "--input_tasks", tasks_file,
        "--api_key", api_key,
        "--output_dir", f"QW_{instance}",
        "--use_apptainer", "False",
        "--instance_ids", instance,
        "--mode", "stylereview-cpp-clangtidy",
        "--model_name", "openrouter/qwen/qwen3-32b"
    ]
    if dry_run:
        print("[DRY RUN] " + " ".join(cmd))
    else:
        print("Running:", instance)
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            print(f"Warning: command for {instance} exited with {proc.returncode}, continuing.", file=sys.stderr)
