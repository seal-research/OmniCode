#!/usr/bin/env python3
"""
Run sweagent_regular.py in parallel for a range of instance IDs.

Usage:
    python run_sweagent_parallel.py
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import os
import sys
from pathlib import Path
from tqdm import tqdm

# === Configuration ===
ORG = "simdjson"
REPO = "simdjson"
PR = 2178
START = 139
END = 147

# Cap concurrency sensibly; change MAX_WORKERS if you want more/less parallelism
CPU = os.cpu_count() or 1
MAX_WORKERS = min(32, CPU * 2)

# Path to script to run (relative to this script)
SWEAGENT_SCRIPT = "baselines/sweagent/sweagent_regular.py"

# Input args constant
INPUT_TASKS = "cpp_style_errors.json"
USE_APPTAINER = "True"
MODE = "stylereview-cpp-clangtidy"

# Directory to store run-specific logs (and where sweagent will write results)
OUTPUT_DIR_BASE = f"sweagent_clang-tidy_{ORG}_{REPO}_{PR}_results"

# Optional: timeout for each subprocess (seconds). Set to None to wait indefinitely.
SUBPROCESS_TIMEOUT = None


def build_cmd(instance_id: str, api_key: str):
    """Return a list command for subprocess.run (no shell)."""
    return [
        sys.executable,
        SWEAGENT_SCRIPT,
        "--input_tasks", INPUT_TASKS,
        "--api_key", api_key,
        "--output_dir", OUTPUT_DIR_BASE,
        "--use_apptainer", USE_APPTAINER,
        "--instance_ids", instance_id,
        "--mode", MODE,
    ]


def run_instance(instance_id: str, api_key: str):
    """Run one instance and capture output. Returns a dict with status and logs."""
    cmd = build_cmd(instance_id, api_key)
    # Ensure logs directory exists
    Path(OUTPUT_DIR_BASE).mkdir(parents=True, exist_ok=True)

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=os.environ,
            check=False,  # we'll handle exit codes ourselves
            timeout=SUBPROCESS_TIMEOUT,
        )
        result = {
            "instance_id": instance_id,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except subprocess.TimeoutExpired as e:
        result = {
            "instance_id": instance_id,
            "returncode": -1,
            "stdout": getattr(e, "stdout", "") or "",
            "stderr": (getattr(e, "stderr", "") or "") + f"\nTIMEOUT after {SUBPROCESS_TIMEOUT}s",
        }
    except Exception as e:
        result = {
            "instance_id": instance_id,
            "returncode": -2,
            "stdout": "",
            "stderr": f"Exception running command: {e}",
        }

    # write per-instance log (stdout+stderr)
    log_file = Path(OUTPUT_DIR_BASE) / f"{instance_id}.log"
    try:
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"COMMAND: {' '.join(cmd)}\n\n")
            f.write("=== STDOUT ===\n")
            f.write(result["stdout"] or "")
            f.write("\n\n=== STDERR ===\n")
            f.write(result["stderr"] or "")
            f.write(f"\n\n=== RETURN CODE: {result['returncode']} ===\n")
    except Exception as e:
        # If log write fails, attach that info to stderr
        result["stderr"] += f"\n\nFailed to write log file {log_file}: {e}"

    return result


def main():
    # read API key from environment
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("Warning: GEMINI_API_KEY not set in environment. Passing empty key to command.")

    instance_ids = [f"{ORG}__{REPO}_{PR}_{i}" for i in range(START, END + 1)]
    print(f"Running {len(instance_ids)} instances with up to {MAX_WORKERS} parallel workers...")

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(run_instance, iid, api_key): iid for iid in instance_ids}

        for fut in tqdm(as_completed(futures), total=len(futures), desc="instances"):
            iid = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {
                    "instance_id": iid,
                    "returncode": -3,
                    "stdout": "",
                    "stderr": f"Executor exception: {e}",
                }
            results.append(res)
            # print short status inline (you can remove or adjust this)
            if res["returncode"] == 0:
                tqdm.write(f"[OK]    {res['instance_id']}")
            else:
                tqdm.write(f"[FAIL]  {res['instance_id']} (code {res['returncode']}) - see log: {OUTPUT_DIR_BASE}/{res['instance_id']}.log")

    # summary
    ok = sum(1 for r in results if r["returncode"] == 0)
    fail = len(results) - ok
    print("\nSummary:")
    print(f"  Total runs : {len(results)}")
    print(f"  Succeeded  : {ok}")
    print(f"  Failed     : {fail}")
    if fail:
        print(f"  See logs in: {OUTPUT_DIR_BASE} (one log per instance)")

if __name__ == "__main__":
    main()
