#!/usr/bin/env python3
"""
generate_and_submit_slurm.py

Generates SLURM job scripts (one per instance_id in sweagent_pmd_results.json)
and submits them with `sbatch` (non-blocking).

Example:
  python generate_and_submit_slurm.py \
    --swejson /home/dd732/OmniCode/sweagent_pmd_results.json \
    --omnidir /home/dd732/OmniCode \
    --conda-env /home/dd732/myenv \
    --model openrouter/openai/gpt-5-mini

By default: reads /home/dd732/OmniCode/sweagent_pmd_results.json and writes scripts to ./slurm_jobs
"""
import argparse
import json
import os
import subprocess
from pathlib import Path
import shlex

def sanitize_filename(s: str) -> str:
    # safe filename: replace slashes and spaces
    return s.replace('/', '__').replace(' ', '_')

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=unicorn_{sanitized}
#SBATCH --output=unicorn_{sanitized}_%j.out
#SBATCH --error=unicorn_{sanitized}_%j.err
#SBATCH --time=13:00:00
#SBATCH --nodes=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --partition=dutta

# --- Setup environment ---
export PATH=/share/apps/software/anaconda3/bin:$PATH
export PATH=/share/apps/software/apptainer/apptainer-1.4.0/bin:$PATH
export PATH="$PATH:/share/dutta/software/LLVM-21.1.1-Linux-X64/bin"
# Load conda
source /share/apps/software/anaconda3/etc/profile.d/conda.sh

# Activate environment
conda activate {conda_env}

{api_export_line}

# Navigate to working directory
cd {omnidir} || exit 1

# Run the target Python script with the filled placeholders
python baselines/sweagent/sweagent_regular.py \\
  --input_tasks {input_tasks} \\
  --api_key $OPENROUTER_API_KEY \\
  --output_dir GPT_{sanitized} \\
  --use_apptainer True \\
  --instance_ids {instance_id_quoted} \\
  --mode stylereview-java-pmd \\
  --g2 True \\
  --model_name {model_name}
"""

def main():
    p = argparse.ArgumentParser(description="Generate + submit SLURM scripts for sweagent instances")
    p.add_argument('--swejson', default='/home/dd732/OmniCode/sweagent_pmd_results.json',
                   help='Path to sweagent_pmd_results.json')
    p.add_argument('--omnidir', default='/home/dd732/OmniCode',
                   help='Path to OmniCode repo (cd into this in the job)')
    p.add_argument('--conda-env', default='/home/dd732/myenv',
                   help='Conda environment to activate in the job')
    p.add_argument('--outdir', default='slurm_jobs', help='Directory to write generated SLURM scripts')
    p.add_argument('--model', default='openrouter/openai/gpt-5-mini', help='Model name for --model_name')
    p.add_argument('--dry-run', action='store_true', help='Only generate scripts; do not call sbatch')
    p.add_argument('--no-embed-key', action='store_true',
                   help="Do NOT embed OPENROUTER_API_KEY into job scripts even if set in current env")
    args = p.parse_args()

    swejson_path = Path(args.swejson)
    if not swejson_path.exists():
        print(f"ERROR: sweagent JSON not found at: {swejson_path}")
        return

    try:
        items = json.loads(swejson_path.read_text(encoding='utf-8'))
    except Exception as e:
        print(f"ERROR: failed to read/parse JSON: {e}")
        return

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    api_key_value = os.environ.get('OPENROUTER_API_KEY')
    if api_key_value and not args.no_embed_key:
        # WARNING: embedding the key will place it in plaintext in the generated scripts.
        api_export_line = f'export OPENROUTER_API_KEY="{api_key_value}"'
        print("NOTE: OPENROUTER_API_KEY is present in environment and WILL be embedded into generated scripts.")
        print("If you don't want that, re-run with --no-embed-key or unset OPENROUTER_API_KEY.")
    else:
        sample_key = api_key_value if api_key_value else "sk-or-<redacted>"
        api_export_line = f'# export OPENROUTER_API_KEY="{sample_key}"  # uncomment or set in job environment as needed'

    input_tasks_abs = str(swejson_path.resolve())

    submitted = []
    for obj in items:
        instance_id = obj.get('instance_id')
        if not instance_id:
            print("Skipping entry with no instance_id.")
            continue

        sanitized = sanitize_filename(instance_id)
        instance_id_quoted = shlex.quote(instance_id)

        content = SLURM_TEMPLATE.format(
            sanitized=sanitized,
            conda_env=args.conda_env,
            api_export_line=api_export_line,
            omnidir=args.omnidir,
            input_tasks=input_tasks_abs,
            instance_id_quoted=instance_id_quoted,
            model_name=args.model
        )

        slurm_path = outdir / f'unicorn_{sanitized}.slurm'
        slurm_path.write_text(content, encoding='utf-8')
        slurm_path.chmod(0o750)
        print(f"Generated {slurm_path}")

        if args.dry_run:
            print(f"Dry-run: not submitting {slurm_path}")
            continue

        # Submit job
        try:
            res = subprocess.run(['sbatch', str(slurm_path)], capture_output=True, text=True)
        except FileNotFoundError:
            print("ERROR: `sbatch` not found. Are you on the cluster head node? Aborting submissions.")
            break

        if res.returncode == 0:
            print(f"Submitted {slurm_path}: {res.stdout.strip()}")
            submitted.append((slurm_path, res.stdout.strip()))
        else:
            print(f"Failed to submit {slurm_path} (rc={res.returncode})")
            if res.stdout:
                print("sbatch stdout:", res.stdout.strip())
            if res.stderr:
                print("sbatch stderr:", res.stderr.strip())

    print("\nSummary:")
    print(f"  Total instances in JSON: {len(items)}")
    print(f"  Jobs submitted: {len(submitted)}")
    if api_key_value and not args.no_embed_key:
        print("WARNING: OPENROUTER_API_KEY was embedded into generated scripts (plaintext).")
    print("Done.")

if __name__ == '__main__':
    main()
