from __future__ import annotations

import docker
import json
import resource
import traceback
import os
import shutil
import sys
import time
import subprocess
import select


from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
from subprocess import Popen, PIPE, TimeoutExpired


from CodeArena_grading import get_eval_report_test_generation, get_fail_to_fail
from CodeArena_test_spec import make_test_spec, TestSpec
from swebench.harness.utils import str2bool
from utils import load_swebench_dataset, load_CodeArena_prediction_dataset, update_test_spec_with_specific_test_names
from run_evaluation_GenTests import get_dataset_from_preds

from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
    INSTANCE_IMAGE_BUILD_DIR,
    KEY_INSTANCE_ID,
    RUN_EVALUATION_LOG_DIR,
)

def get_gold_predictions(dataset_name: str, instance_ids: list, split: str):
    """
    Get ground truth tests and their corresponding patches from the FAIL_TO_PASS section.
    
    Args:
        dataset_name (str): Name of the dataset
        instance_ids (list): List of instance IDs to process
        split (str): Dataset split to use
        
    Returns:
        list: List of dictionaries containing instance IDs, patches, and failing test information
    """
    dataset = load_swebench_dataset(dataset_name, split)
    results = []
    
    for datum in dataset:
        if datum[KEY_INSTANCE_ID] not in instance_ids:
            continue
        
        # loading gold prediction results assumes direct employment of swe-bench-verified
        result = {
            KEY_INSTANCE_ID: datum[KEY_INSTANCE_ID],
            "repo": datum["repo"],
            "base_commit": datum["base_commit"],
            "gold_patch": datum["patch"],
            "candidate_test_patch": datum["test_patch"], # gold test patch
            "version": datum["version"],
            "model_name_or_path": "gold"
        }
        
        # Add bad patches if they exist
        if "bad_patches" in datum:
            result["bad_patches"] = datum["bad_patches"]
        elif "bad_patch" in datum:
            result["bad_patches"] = [datum["bad_patch"]]
            
        results.append(result)
    
    return results

def load_mswebench_dataset(instance_ids: list, 
                           predictions: dict,
                           dataset_base_path: str = "./multiswebench_local/mswebench_dataset",
                           mswebench_dataset_path: str = "./multiswebench_local/data/datasets"):
    dataset_files = []
    for root, _, files in os.walk(dataset_base_path):
        for file in files:
            if file.endswith("_dataset.jsonl"):
                dataset_files.append(os.path.join(root, file))
    
    if not dataset_files:
        print("Error: No dataset files found in", dataset_base_path)
        return 
    mswebench_dataset_file = os.path.join(mswebench_dataset_path, "dataset.jsonl")
    
    print(f"Writing MSWE-Bench dataset to {mswebench_dataset_file}...")
    with open(mswebench_dataset_file, "w", encoding="utf-8") as f_mswedataset:
        for instance_id in instance_ids:
            for dataset_file in dataset_files:
                with open(dataset_file, "r", encoding="utf-8") as f_dataset:
                    for line in f_dataset:
                        if not line.strip():
                            continue
                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError as e:
                            print(f"Error parsing JSON from {dataset_file}: {e}")
                            continue
                        item_instance_id = item.get("org") + "__" + item.get("repo") + "_" + str(item.get("number")) 
                        # if item.get("instance_id") == instance_id:
                        if item_instance_id == instance_id:
                            # use the candidate test patch for this instance in 
                            item["test_patch"] = predictions[instance_id]["candidate_test_patch"]
                            print(f"Writing item to {mswebench_dataset_file}...")
                            f_mswedataset.write(json.dumps(item) + "\n")
                            break

def create_multiswebench_config(predictions, 
                                dataset_path, 
                                max_workers, 
                                force_rebuild, 
                                run_id, 
                                timeout, 
                                bad_patch_index = -1, 
                                phase="all",
                                use_apptainer: bool = False,
                                data_dir: Path = Path("./multiswebench_local/data")):
    """Set up configuration for Multi-SWE-Bench evaluation."""    
    patch_file_sub_path = Path("patches") / f"{run_id}_patches.jsonl"
    patch_file = data_dir / patch_file_sub_path
    print(f"Writing {len(predictions)} patches to {patch_file}...")
    with open(patch_file, 'w', encoding='utf-8') as f:
        for instance in predictions:
            item = predictions[instance]
            org = instance.split("__")[0]
            repo = instance.split("_")[-2]
            # org, repo = item.get("repo").split("/")
            number = instance.split("_")[-1]
            if bad_patch_index != -1:
                bad_patch = item.get("bad_patches", [])
                if bad_patch_index < len(bad_patch):
                    patch_data = {
                        "org": org,
                        "repo": repo,
                        "number": number,
                        "fix_patch": bad_patch[bad_patch_index]["patch"],
                    }
                    f.write(json.dumps(patch_data) + "\n")
            else:
                patch_data = {
                    "org": org,
                    "repo": repo,
                    "number": number,
                    "fix_patch": item.get("gold_patch", "")
                }
                f.write(json.dumps(patch_data) + "\n")
                    
    # Determine mode based on phase
    mode = {"image": "image", "instance": "instance_only"}.get(phase, "evaluation")
    
    # Create config
    print(f"Creating configuration for phase: {phase}...")
    config = {
        "mode": mode,
        "workdir": str(data_dir / "workdir"),
        "patch_files": [str(patch_file)],
        "dataset_files": [str(data_dir / "datasets/dataset.jsonl")],
        "force_build": force_rebuild,
        "output_dir": str(data_dir / "output"),
        "specifics": [],
        "skips": [],
        "repo_dir": str(data_dir / "repos"),
        "need_clone": True,
        "global_env": [],
        "clear_env": True,
        "stop_on_error": False,
        "max_workers": max_workers,
        "max_workers_build_image": max(1, max_workers // 2),
        "max_workers_run_instance": max(1, max_workers // 2),
        "log_dir": str(data_dir / "logs"),
        "log_level": "INFO",
        "log_to_console": True,
        "use_apptainer": use_apptainer,
    }
    
    config_file = data_dir / f"{run_id}_{phase}_config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to {config_file}")
    return str(config_file)
    
def run_with_timeout(cmd, timeout_seconds=1800):
    """Run a command with timeout and real-time output streaming."""
    print(f"Running command with {timeout_seconds}s timeout: {' '.join(cmd)}")
    
    try:
        # Start process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Set up streaming
        start_time = time.time()
        stdout_lines, stderr_lines = [], []
        outputs = [process.stdout, process.stderr]
        
        # Monitor process
        while process.poll() is None:
            # Check for timeout
            if(timeout_seconds>0 and time.time() - start_time > timeout_seconds):
                print(f"Process timed out after {timeout_seconds} seconds!")
                process.kill()
                return None, "Timeout exceeded", 1
            
            # Read output
            readable, _, _ = select.select(outputs, [], [], 1.0)
            for stream in readable:
                line = stream.readline()
                if line:
                    if stream == process.stdout:
                        stdout_lines.append(line)
                        print(f"STDOUT: {line.strip()}")
                    else:
                        stderr_lines.append(line)
                        is_error = any(level in line for level in ["ERROR", "CRITICAL", "FATAL"])
                        print(f"{'STDERR' if is_error else 'LOG'}: {line.strip()}")
        
        # Read any remaining output
        for line in process.stdout:
            stdout_lines.append(line)
            print(f"STDOUT: {line.strip()}")
        for line in process.stderr:
            stderr_lines.append(line)
            is_error = any(level in line for level in ["ERROR", "CRITICAL", "FATAL"])
            print(f"{'STDERR' if is_error else 'LOG'}: {line.strip()}")
        
        # Return results
        stdout = "".join(stdout_lines)
        stderr = "".join(stderr_lines)
        
        if process.returncode != 0:
            print(f"Command failed with exit code {process.returncode}")
        
        return stdout, stderr, process.returncode
        
    except Exception as e:
        print(f"Error running command: {e}")
        return None, str(e), 1

def run_multiswebench_phase(config_file, phase="all", timeout=1800):
    """Run a specific phase of Multi-SWE-Bench evaluation."""
    script_path = "./multiswebench_local/multi_swe_bench/harness/run_evaluation.py"

    # Validate inputs
    if not os.path.exists(script_path):
        print(f"Script not found: {script_path}")
        return None

    if not config_file:
        print("Error: No config file provided")
        return None

    # Run the subprocess
    cmd = [sys.executable, script_path, "--config", config_file]
    stdout, stderr, returncode = run_with_timeout(cmd, timeout)

    if returncode != 0:
        print(f"Command failed with code {returncode}")
        return None

    # Process results
    try:
        with open(config_file, 'r') as f:
            config_data = json.load(f)

        final_report_path = Path(config_data["output_dir"]) / "final_report.json"
        if final_report_path.exists():
            with open(final_report_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading report: {e}")

    return None

def run_instance(
        instances: list,
        config_file: str, 
        timeout: int,
        predictions_path,
        tag: str = 'singleton',
        runid: str = ''):
    with tqdm(total=len(instances), smoothing=0) as pbar:
        # Run evaluation
        if config_file and os.path.exists(config_file):
            print(f"Config file created at: {config_file}")
            report = run_multiswebench_phase(config_file, 'all', timeout)
            if report:
                print("Multi-SWE-Bench BugFixing evaluation completed!")
                print(f"Total instances: {report.get('total_instances', 0)}")
                print(f"Resolved instances: {report.get('resolved_instances', 0)}")
                print(f"Unresolved instances: {report.get('unresolved_instances', 0)}")
                return report
            else:
                print("Multi-SWE-Bench BugFixing evaluation failed to produce a report")
        else:
            print("Failed to create config file, cannot run evaluation")
    
    return None

def run_instances(
        predictions: dict,
        instances: list,
        cache_level: str,
        clean: bool,
        force_rebuild: bool,
        max_workers: int,
        run_id: str,
        timeout: int,
        predictions_path,
        dataset_path: str = "./multiswebench_local/data/datasets/dataset.jsonl",
        use_apptainer: bool = False,
        instance_ids: list = None
    ):
    """
    Run all instances for the given predictions in parallel.

    Args:
        predictions (dict): Predictions dict generated by the model
        instances (list): List of instances
        cache_level (str): Cache level
        clean (bool): Clean images above cache level
        force_rebuild (bool): Force rebuild images
        max_workers (int): Maximum number of workers
        run_id (str): Run ID
        timeout (int): Timeout for running tests
    """
    # run instances in parallel
    print(f"Running {len(instances)} instances...")    
    print(f"Running instances with gold patch...")
    
    success_dict = {}
    instance_dict = {}
    for instance in instances:
        instance_dict[instance["instance_id"]] = instance
    base_path = Path(f"multiswebench_runs/TestGeneration/{run_id}/{Path(predictions_path).stem}/gold/")
    clean_directories(base_path)
    load_mswebench_dataset(instance_ids, predictions, mswebench_dataset_path=f"{base_path}/data/datasets/") # if using local jsonl, this is just the same as gold predictions but with original labels
    config_file = create_multiswebench_config(
        predictions=instance_dict, 
        dataset_path=dataset_path, 
        max_workers=max_workers, 
        force_rebuild=force_rebuild, 
        run_id=run_id, 
        timeout=timeout,
        use_apptainer=use_apptainer,
        data_dir=Path(f"multiswebench_runs/TestGeneration/{run_id}/{Path(predictions_path).stem}/gold/data/")
    )
    report = run_instance(
        instances=instance_dict,
        config_file=config_file,
        timeout=timeout,
        tag="gold",
        runid=run_id,
        predictions_path=predictions_path
    )   
    if report:
        success_dict["gold_successes"] = report.get("resolved_ids", [])
        success_dict["gold_failures"] = report.get("unresolved_ids", [])

    print(f"Running instances with bad patches...")
    max_bad_patches = max([len(instance_dict[instance_id].get("bad_patches", [])) for instance_id in instance_dict])
    for i in range(max_bad_patches):
        print(f"Running instances with bad patch index: {i}")
        base_path = Path(f"multiswebench_runs/TestGeneration/{run_id}/{Path(predictions_path).stem}/bad_patch_{i}/")
        clean_directories(base_path)
        load_mswebench_dataset(instance_ids, predictions, mswebench_dataset_path=f"{base_path}/data/datasets/") # if using local jsonl, this is just the same as gold predictions but with original labels
        config_file = create_multiswebench_config(
            predictions=instance_dict, 
            dataset_path=dataset_path, 
            max_workers=max_workers, 
            force_rebuild=force_rebuild, 
            run_id=run_id, 
            timeout=timeout, 
            bad_patch_index=i,
            use_apptainer=use_apptainer,
            data_dir=Path(f"multiswebench_runs/TestGeneration/{run_id}/{Path(predictions_path).stem}/bad_patch_{i}/data/")
        )
        report = run_instance(
            instances=instance_dict,
            config_file=config_file,
            timeout=timeout,
            tag=f"bad_patch_{i}",
            runid=run_id,
            predictions_path=predictions_path
        )
        if report:
            existing_success = success_dict.get(f"bad_patch_successes", [])
            existing_failures = success_dict.get(f"bad_patch_failures", [])
            resolved_ids = [f"bad_patch_{i}_of_{id}" for id in report.get("resolved_ids", [])]
            existing_failures.extend(resolved_ids)
            error_ids = report.get("error_ids", [])
            unresolved_ids = [f"bad_patch_{i}_of_{id}" for id in report.get("unresolved_ids", []) if id not in error_ids]
            existing_success.extend(unresolved_ids)
            success_dict[f"bad_patch_successes"] = existing_success
            success_dict[f"bad_patch_failures"] = existing_failures
    
    report = {"bad_patches_results": 
              {"EXPECTED_FAIL": 
               {"success": success_dict.get("bad_patch_failures", []),
                "failure": success_dict.get("bad_patch_successes", [])}},
                "gold_tests_status": {
                    "EXPECTED_PASS": {
                        "success": success_dict.get("gold_successes", []),
                        "failure": success_dict.get("gold_failures", [])
                    }
                }}
    print("Report:", report)
    print("Saving final report")
    with open(f"multiswebench_runs/TestGeneration/{run_id}/{Path(predictions_path).stem}/report.json", "w") as f:
        json.dump(report, f, indent=2)

    

def clean_directories(base_path):
    data_dir = Path(f"{base_path}/data")
    dirs_to_remove = [
        "workdir",
        "logs",
        "output"
    ]
    
    for dir_path in dirs_to_remove:
        path = os.path.join(data_dir, dir_path)
        if os.path.exists(path):
            print(f"Removing {path}")
            try:
                shutil.rmtree(path)
            except Exception as e:
                print(f"Error removing {path}: {e}")
        
        os.makedirs(path, exist_ok=True)

    # Create directories
    print("Creating directory structure...")
    os.makedirs(data_dir, exist_ok=True)
    for subdir in ["workdir", "logs", "output", "patches", "datasets", "repos"]:
        os.makedirs(data_dir / subdir, exist_ok=True)
    
    print("Directory cleaning completed")

def main(
        dataset_name: str,
        split: str,
        instance_ids: list,
        predictions_path: str,
        max_workers: int,
        force_rebuild: bool,
        cache_level: str,
        clean: bool,
        open_file_limit: int,
        run_id: str,
        timeout: int,
        use_apptainer: bool = False,
    ):
    """
    Run evaluation harness for the given dataset and predictions.
    """
    # set open file limit
    assert len(run_id) > 0, "Run ID must be provided"
    resource.setrlimit(resource.RLIMIT_NOFILE, (open_file_limit, open_file_limit))
    base_path = f"multiswebench_runs/TestGeneration/{run_id}/{Path(predictions_path).stem}"

    # load predictions as map of instance_id to prediction
    if predictions_path == 'gold':
        print("Using gold predictions - ignoring predictions_path")
        predictions = get_gold_predictions(dataset_name, instance_ids, split) # Gold Prediction should correspond to ground truth test (PASS TO FAIL)
    else:
        if predictions_path.endswith(".json"):
            with open(predictions_path, "r") as f:
                predictions = json.load(f)
        elif predictions_path.endswith(".jsonl"):
            with open(predictions_path, "r") as f:
                predictions = [json.loads(line) for line in f]
        else:
            raise ValueError("Predictions path must be \"gold\", .json, or .jsonl")
        for pred in predictions:
            pred["candidate_test_patch"] = pred.get("model_patch", "")
    
    predictions = {pred[KEY_INSTANCE_ID]: pred for pred in predictions}

    # get dataset from predictions
    if(not predictions_path == 'gold'):
        dataset = get_dataset_from_preds(
            dataset_name, 
            split, 
            instance_ids, 
            run_id=run_id, 
            generated_tests_path=predictions_path, 
            codearena_instances=dataset_name # necessary because of current function structure
        )
    else:
        dataset = get_gold_predictions(dataset_name, instance_ids, split)

    if not dataset:
        print("No instances to run.")
    else:
        # dataset => instances in function
        run_instances(predictions, dataset, cache_level, clean, force_rebuild, max_workers, run_id, timeout, predictions_path=predictions_path, use_apptainer=use_apptainer, dataset_path=f"{base_path}data/datasets/dataset.jsonl", instance_ids=instance_ids)





if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", default="princeton-nlp/SWE-bench_Verified", type=str, help="Name of dataset or path to JSON file.")
    parser.add_argument("--split", type=str, default="test", help="Split of the dataset")
    parser.add_argument("--instance_ids", nargs="+", type=str, help="Instance IDs to run (space separated)")
    parser.add_argument("--predictions_path", type=str, help="Path to predictions file - if 'gold', uses gold predictions", required=True)
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of workers (should be <= 75%% of CPU cores)")
    parser.add_argument("--open_file_limit", type=int, default=4096, help="Open file limit")
    parser.add_argument(
        "--timeout", type=int, default=1_800, help="Timeout (in seconds) for running tests for each instance"
        )
    parser.add_argument(
        "--force_rebuild", type=str2bool, default=False, help="Force rebuild of all images"
    )
    parser.add_argument(
        "--cache_level",
        type=str,
        choices=["none", "base", "env", "instance"],
        help="Cache level - remove images above this level",
        default="env",
    )
    # if clean is true then we remove all images that are above the cache level
    # if clean is false, we only remove images above the cache level if they don't already exist
    parser.add_argument(
        "--clean", type=str2bool, default=False, help="Clean images above cache level"
    )
    parser.add_argument("--run_id", type=str, required=True, help="Run ID - identifies the run")
    args = parser.parse_args()

    main(**vars(args))