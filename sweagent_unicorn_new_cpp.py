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
    --model openrouter/openai/gpt-5-mini \
    --org apache --repo dubbo --pr 10638

If --org/--repo/--pr are provided, only instances matching those will be processed.
"""
import argparse
import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Optional, Tuple

def sanitize_filename(s: str) -> str:
    return s.replace('/', '__').replace(' ', '_')

# NOTE: keep memory large (you set 256G previously)
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
  --output_dir baselines/sweagent/logs/sweagent_outputs/GPT_{sanitized} \\
  --use_apptainer True \\
  --instance_ids {instance_id_quoted} \\
  --mode stylereview-cpp-clangtidy \\
  --g2 True \\
  --model_name {model_name}
"""

# Parse org/repo/pr from either JSON fields or the instance_id
INSTANCE_ID_RE = re.compile(r'(?P<org>[^_]+)__(?P<repo>[^_]+)_(?P<pr>\d+)_.*')

def parse_from_item(item: dict) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """
    Return (org, repo, pr) if available, else (None, None, None).
    Uses item['repo'] (format 'org/repo') and item['pull_number'] when available,
    otherwise attempts to parse item['instance_id'] using the expected pattern.
    """
    org = repo = None
    pr = None

    # Try JSON fields first
    repo_field = item.get('repo')  # e.g. "apache/dubbo"
    if isinstance(repo_field, str) and '/' in repo_field:
        try:
            org, repo = repo_field.split('/', 1)
        except Exception:
            org = None
            repo = None

    pull_number = item.get('pull_number')
    if pull_number is not None:
        try:
            pr = int(pull_number)
        except Exception:
            pr = None

    # If any missing, try parsing instance_id
    if (org is None or repo is None or pr is None) and item.get('instance_id'):
        m = INSTANCE_ID_RE.match(item['instance_id'])
        if m:
            org = org or m.group('org')
            repo = repo or m.group('repo')
            if pr is None:
                try:
                    pr = int(m.group('pr'))
                except Exception:
                    pr = None

    return org, repo, pr

def matches_filters(item: dict, org_f: Optional[str], repo_f: Optional[str], pr_f: Optional[int]) -> bool:
    if org_f is None and repo_f is None and pr_f is None:
        return True  # no filtering requested
    org, repo, pr = parse_from_item(item)
    if org_f is not None:
        if org is None or org.lower() != org_f.lower():
            return False
    if repo_f is not None:
        # allow repo filter to be specified either as 'repo' or 'org/repo'
        if '/' in repo_f:
            repo_candidate = repo_f.split('/', 1)[1]
        else:
            repo_candidate = repo_f
        if repo is None or repo.lower() != repo_candidate.lower():
            return False
    if pr_f is not None:
        if pr is None or pr != pr_f:
            return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Generate + submit SLURM scripts for sweagent instances")
    parser.add_argument('--swejson', default='/home/dd732/OmniCode/sweagent_clang-tidy_results.json',
                        help='Path to sweagent_pmd_results.json')
    parser.add_argument('--omnidir', default='/home/dd732/OmniCode',
                        help='Path to OmniCode repo (cd into this in the job)')
    parser.add_argument('--conda-env', default='/home/dd732/myenv',
                        help='Conda environment to activate in the job')
    parser.add_argument('--outdir', default='slurm_jobs', help='Directory to write generated SLURM scripts')
    parser.add_argument('--model', default='openrouter/openai/gpt-5-mini', help='Model name for --model_name')
    parser.add_argument('--dry-run', action='store_true', help='Only generate scripts; do not call sbatch')
    parser.add_argument('--no-embed-key', action='store_true',
                        help="Do NOT embed OPENROUTER_API_KEY into job scripts even if set in current env")
    # filters
    parser.add_argument('--org', type=str, help='Filter by org (e.g. apache)')
    parser.add_argument('--repo', type=str, help='Filter by repo (e.g. dubbo or apache/dubbo)')
    parser.add_argument('--pr', type=int, help='Filter by pull request number (e.g. 10638)')
    args = parser.parse_args()

    swejson_path = Path(args.swejson)
    if not swejson_path.exists():
        print(f"ERROR: sweagent JSON not found at: {swejson_path}")
        return

    try:
        items = json.loads(swejson_path.read_text(encoding='utf-8'))
    except Exception as e:
        print(f"ERROR: failed to read/parse JSON: {e}")
        return

    # Apply filters
    filtered = [it for it in items if matches_filters(it, args.org, args.repo, args.pr)]
    print(f"Total instances in JSON: {len(items)}. Matched by filter: {len(filtered)}.")
    if len(filtered) == 0:
        print("No instances match the provided filters. Exiting.")
        return

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    api_key_value = os.environ.get('OPENROUTER_API_KEY')
    if api_key_value and not args.no_embed_key:
        api_export_line = f'export OPENROUTER_API_KEY="{api_key_value}"'
        print("NOTE: OPENROUTER_API_KEY is present in environment and WILL be embedded into generated scripts.")
        print("If you don't want that, re-run with --no-embed-key or unset OPENROUTER_API_KEY.")
    else:
        sample_key = api_key_value if api_key_value else "sk-or-<redacted>"
        api_export_line = f'# export OPENROUTER_API_KEY="{sample_key}"  # uncomment or set in job environment as needed'

    input_tasks_abs = str(swejson_path.resolve())

    submitted = []
    for obj in filtered:
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
    print(f"  Matched instances processed: {len(filtered)}")
    print(f"  Jobs submitted: {len(submitted)}")
    if api_key_value and not args.no_embed_key:
        print("WARNING: OPENROUTER_API_KEY was embedded into generated scripts (plaintext).")
    print("Done.")

if __name__ == '__main__':
    main()
