#!/usr/bin/env python3
"""
Iteratively run sweagent_regular.py for a range of instance IDs.

Usage:
    python run_sweagent_iterative.py
"""

import subprocess

def main():
    # Take user inputs
    org = input("Enter organization (org): ").strip()
    repo = input("Enter repository name (repo): ").strip()
    pr = input("Enter PR number (pr): ").strip()
    start = int(input("Enter start number: ").strip())
    end = int(input("Enter end number: ").strip())

    for number in range(start, end + 1):
        instance_id = f"{org}__{repo}_{pr}_{number}"
        output_dir = f"sweagent_pmd_{org}_{repo}_{pr}_results"
        
        cmd = [
            "python",
            "baselines/sweagent/sweagent_regular.py",
            "--input_tasks", "sweagent_input.json",
            "--api_key", "$GEMINI_API_KEY",
            "--output_dir", output_dir,
            "--use_apptainer", "True",
            "--instance_ids", instance_id,
            "--mode", "stylereview-java-pmd"
        ]

        print(f"Running command for instance_id={instance_id} ...")
        try:
            subprocess.run(" ".join(cmd), shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Command failed for instance_id={instance_id}: {e}")

if __name__ == "__main__":
    main()
