#!/usr/bin/env python3
import json, os, sys, subprocess

tasks_file = sys.argv[1] if len(sys.argv) > 1 else "sweagent_pmd_results.json"
instances_file = sys.argv[2] if len(sys.argv) > 2 else "sweagent_pmd_results.json"
dry_run = "--dry-run" in sys.argv

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
    cmd = [
        "python", "baselines/sweagent/sweagent_regular.py",
        "--input_tasks", tasks_file,
        "--api_key", api_key,
        "--output_dir", f"DS_{instance}",
        "--use_apptainer", "False",
        "--instance_ids", instance,
        "--mode", "stylereview-java-pmd",
        "--model_name", "openrouter/deepseek/deepseek-chat-v3.1"
    ]
    if dry_run:
        print("[DRY RUN] " + " ".join(cmd))
    else:
        print("Running:", instance)
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            print(f"Warning: command for {instance} exited with {proc.returncode}, continuing.", file=sys.stderr)
