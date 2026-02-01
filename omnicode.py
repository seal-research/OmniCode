import argparse
from pathlib import Path
import importlib
import json
import os, shutil
import subprocess
import sys
import glob
import time
import select

# # from run_evaluation_GenTests import main as GenTestMain
# # from runevaluation_StyleReview import main as PythonStyleReviewMain
# # from mswebench_run_evaluation_GenTests import main as MSWEGenTestMain
# # imports and monkey patches swebench
# from monkeypatched_swebench import swebench
from swebench.harness.utils import str2bool
from swebench.harness.run_evaluation import main as RegularEval
# from OmniCode_grading import test_passed_prefix_match, test_failed_prefix_match

from src import GenTestMain, PythonStyleReviewMain, MSWEGenTestMain, swebench, test_passed_prefix_match, test_failed_prefix_match

CUR_DIR = Path(__file__).parent
REPO_DATA_PATH = CUR_DIR / "data/omnicode_repo_data.py"
REPO_DATA = eval(REPO_DATA_PATH.read_text())

def execute_command(func, **kwargs):
    """Wrapper to execute a function safely."""
    func(**kwargs)

def generate_gold_patch_predictions(dataset_files, instance_ids=None, max_instances=0):
    """Generate predictions from gold patches in the dataset files."""
    predictions = []
    added_count = 0

    # Convert to infinite if no limit specified
    max_instances = float('inf') if max_instances <= 0 else max_instances

    # Print instance IDs we're looking for
    if instance_ids:
        print(f"Filtering for specific instances: {instance_ids}")

        # Extract repos from instance_ids to filter dataset files
        target_repos = set()
        for instance_id in instance_ids:
            if ":" in instance_id:
                repo_part = instance_id.split(":")[0]
                target_repos.add(repo_part.replace("/", "__"))
            else:
                repo_part = instance_id.replace("__", "/").split("_")[0]
                target_repos.add(repo_part.replace("/", "__"))

        # Filter dataset files to only include relevant repos
        filtered_files = []
        for dataset_file in dataset_files:
            for repo in target_repos:
                if repo in dataset_file:
                    filtered_files.append(dataset_file)
                    print(f"Including dataset file: {dataset_file}")
                    break
        dataset_files = filtered_files

    if not dataset_files:
        print("No matching dataset files found!")
        return []

    # Process each dataset file
    for dataset_file in dataset_files:
        print(f"Processing dataset file: {dataset_file}")

        # Stop if we've reached max instances
        if added_count >= max_instances:
            break

        with open(dataset_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip() or added_count >= max_instances:
                    continue

                try:
                    item = json.loads(line)
                    item_id = f"{item['org']}/{item['repo']}:{item['number']}"
                    # Also support underscore format for compatibility
                    item_id_underscore = f"{item['org']}__{item['repo']}_{item['number']}"

                    # Skip if not in requested instances (check both formats)
                    if instance_ids and item_id not in instance_ids and item_id_underscore not in instance_ids:
                        continue

                    # Create prediction entry using underscore format to match existing predictions
                    pred = {
                        "instance_id": item_id_underscore,  # Use underscore format like existing predictions
                        "model_name_or_path": "gold_patch",
                        "full_output": None,
                        "model_patch": {
                            "model_name_or_path": "gold",
                            "instance_id": item_id_underscore,
                            "model_patch": item.get('fix_patch', '')
                        },
                        # Also include legacy fields for compatibility
                        "id": item_id,
                        "org": item['org'],
                        "repo": item['repo'],
                        "number": item['number'],
                        "patch": item.get('fix_patch', '')
                    }

                    if pred["patch"]:
                        predictions.append(pred)
                        added_count += 1
                        print(f"  Added prediction for {pred['id']} ({added_count}/{max_instances})")

                        # Stop if we've found all requested instances
                        if instance_ids and len(predictions) == len(instance_ids):
                            print("Found all requested instances.")
                            return predictions

                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON from {dataset_file}: {e}")

    print(f"Total predictions added: {len(predictions)}")
    return predictions

def setup_multiswebench_config(
    predictions,
    max_workers,
    force_rebuild,
    run_id,
    timeout,
    phase="all",
    use_apptainer=False,
    g2=False
):
    """Set up configuration for Multi-SWE-Bench evaluation."""
    if g2:
        user_name = os.getenv("USER")
        data_dir = Path(f"/scratch/{user_name}/multiswebench_runs/BugFixing")
    else:
        data_dir = Path("multiswebench_runs/BugFixing")

    # Create directories
    print("Creating directory structure...")
    os.makedirs(data_dir, exist_ok=True)

    # Create all necessary directories with proper f-string formatting
    subdirs = [
        f"workdir/{run_id}",
        f"logs/{run_id}",
        f"output/{run_id}",
        f"patches",
        f"repos/{run_id}",
        "configs"
    ]

    for subdir in subdirs:
        dir_path = data_dir / subdir
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

    for pred in predictions:
        if not pred.get("org") or not pred.get("repo"):
            instance_info = pred["instance_id"].split("_")
            pred["org"] = instance_info[0]
            pred["repo"] = instance_info[2]
            pred["number"] = instance_info[-1]
            if "model_patch" in pred and "model_patch" in pred["model_patch"]:
                pred["patch"] = pred["model_patch"]["model_patch"]
            elif "model_patch" in pred and isinstance(pred["model_patch"], str):
                pred["patch"] = pred["model_patch"]
            else:
                raise RuntimeError("No patch found in data")

    # Get unique repos for image building
    for pred in predictions:
        if not pred.get("org") or not pred.get("repo"):
            instance_info = pred["instance_id"].split("_")
            pred["org"] = instance_info[0]
            pred["repo"] = instance_info[2]
            pred["number"] = instance_info[-1]
            pred["patch"] = pred.get("model_patch", "")
    unique_repos = {f"{pred['org']}/{pred['repo']}" for pred in predictions}
    print(f"Will build images for these repos: {unique_repos}")

    # Create patches file
    patch_file = data_dir / "patches" / f"{run_id}_patches.jsonl"
    print(f"Writing {len(predictions)} patches to {patch_file}...")
    with open(patch_file, 'w', encoding='utf-8') as f:
        for pred in predictions:
            patch_data = {
                "org": pred.get("org"),
                "repo": pred.get("repo"),
                "number": pred.get("number"),
                "fix_patch": pred.get("patch", "")
            }
            f.write(json.dumps(patch_data) + "\n")

    # Find dataset files matching our repos
    dataset_base_path = "./multiswebench_local/mswebench_dataset"
    dataset_files = []

    print(f"Finding relevant dataset files...")
    for root, _, files in os.walk(dataset_base_path):
        for file in files:
            if file.endswith("_dataset.jsonl"):
                for repo in unique_repos:
                    org_repo = repo.replace('/', '__')
                    if org_repo in file:
                        dataset_files.append(os.path.join(root, file))
                        print(f"  Found dataset: {os.path.join(root, file)}")
                        break

    # Determine mode based on phase
    mode = {"image": "image", "instance": "instance_only"}.get(phase, "evaluation")

    # Create config
    print(f"Creating configuration for phase: {phase}...")
    config = {
        "mode": mode,
        "workdir": str(data_dir / "workdir" / f"{run_id}"),
        "patch_files": [str(patch_file)],
        "dataset_files": dataset_files,
        "force_build": force_rebuild,
        "output_dir": str(data_dir / "output" / f"{run_id}"),  # Make this run-specific
        "specifics": [],
        "skips": [],
        "repo_dir": str(data_dir / "repos" / f"{run_id}"),
        "need_clone": True,
        "global_env": [],
        "clear_env": True,
        "stop_on_error": False,
        "max_workers": max_workers,
        "max_workers_build_image": max(1, max_workers // 2),
        "max_workers_run_instance": max(1, max_workers // 2),
        "log_dir": str(data_dir / "logs" / f"{run_id}"),
        "log_level": "DEBUG",
        "log_to_console": True,
        "use_apptainer": use_apptainer, 
        "g2": g2,
    }

    config_file = data_dir / "configs" / f"{run_id}_{phase}_config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    print(f"Configuration saved to {config_file}")
    return str(config_file)

def run_with_timeout(cmd, timeout_seconds=10000):
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

def run_multiswebench_phase(config_file, phase="all", timeout=10000):
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

def clean_docker_images(image_prefix, use_apptainer=False):
    """Clean up Docker/Apptainer images with the specified prefix."""
    if use_apptainer:
        # For apptainer, we don't need to clean up images the same way
        print(f"Skipping image cleanup for apptainer mode with prefix: {image_prefix}")
        return
        
    try:
        print(f"Cleaning up Docker images with prefix: {image_prefix}")
        cmd = ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}", f"{image_prefix}*"]
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, text=True)

        images = [img for img in result.stdout.strip().split('\n') if img]

        if images:
            print(f"Found {len(images)} images to remove")
            for img in images:
                print(f"Removing image: {img}")
                subprocess.run(["docker", "rmi", "-f", img], check=False)
        else:
            print("No images found to clean up")
    except Exception as e:
        print(f"Error cleaning Docker images: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run Omnicode Benchmarks")

    # Common parameters
    parser.add_argument("--dataset_name", default="data/omnicode_instances_python.json",
                        help="Name of the dataset")
    parser.add_argument("--predictions_path", nargs="+", required=True,
                        help="Paths to predictions files. Use 'gold' for gold patches.")
    parser.add_argument("--max_workers", type=int, default=1,
                        help="Number of maximum workers to use")
    parser.add_argument("--run_id", required=True,
                        help="Run ID for the evaluation; please ensure run_id is unique while running parallel instances.")
    parser.add_argument("--instance_ids", nargs="*",
                        help="Optional instance IDs")
    parser.add_argument("--open_file_limit", type=int, default=4096,
                        help="Maximum number of open files")
    parser.add_argument("--timeout", type=int, default=10000,
                        help="Timeout for individual evaluations in seconds")
    parser.add_argument("--force_rebuild", type=str2bool, default=False,
                        help="Force rebuild of all images")
    parser.add_argument("--cache_level", type=str,
                        choices=["none", "base", "env", "instance"],
                        default="env", help="Cache level - remove images above this level")
    parser.add_argument("--clean", type=str2bool, default=False,
                        help="Clean images above cache level")
    parser.add_argument("--max_instances", type=int, default=0,
                        help="Maximum number of instances to process (0 for unlimited)")
    parser.add_argument("--mswe_phase", choices=["all", "image", "instance"],
                        default="all", help="Which phase of Multi-SWE-Bench to run")
    parser.add_argument("--list_instances", action="store_true",
                        help="Just list available instances without running evaluation")

    # Benchmark flags
    parser.add_argument("--BugFixing", action="store_true",
                        help="Run the regular BugFixing benchmark")
    parser.add_argument("--TestGeneration", action="store_true",
                        help="Run the TestGeneration benchmark")
    parser.add_argument("--CodeReview", action="store_true",
                        help="Run the CodeReview benchmark")
    parser.add_argument("--CodeMigration", action="store_true",
                        help="Run the CodeMigration benchmark")
    parser.add_argument("--MSWEBugFixing", action="store_true",
                        help="Run the Multi-SWE-Bench BugFixing benchmark")
    parser.add_argument("--MSWETestGeneration", action="store_true",
                        help="Run the Multi-SWE-Bench TestGeneration benchmark")
    parser.add_argument("--StyleReview", action="store_true",
                        help="Run the StyleReview benchmark")
    parser.add_argument("--PythonStyleReview", action="store_true",
                        help="Run the PythonStyleReview benchmark")

    # Style review specific parameters
    parser.add_argument("--min_score", type=float, default=None,
                        help="Minimum acceptable style score (0-10) for StyleReview")
    parser.add_argument("--review_type",type=str,default="checkstyle",help="Please enter checkstyle or pmd")
    parser.add_argument("--max_severity", type=str,
                        choices=['convention', 'warning', 'error'], default=None,
                        help="Maximum acceptable severity level for StyleReview")
    parser.add_argument("--language", type=str, choices=['auto', 'python', 'java'],
                        default='auto', help="Language for StyleReview, auto for automatic detection")

    parser.add_argument("--use_apptainer", type=str2bool, default=False, help="run with docker or apptainer")
    parser.add_argument("--g2", type=str2bool, default=False, help="run on g2 cluster")

    args = parser.parse_args()

    local = False if args.g2 else True

    # Collect active flags
    active_flags = []
    for flag in ["BugFixing", "TestGeneration", "CodeReview", "CodeMigration",
                "MSWEBugFixing", "MSWETestGeneration", "StyleReview"]:
        if getattr(args, flag):
            active_flags.append(flag)

    # Ensure at least one flag is provided
    if not active_flags:
        print("Error: You must specify at least one benchmark flag.")
        return

    # Ensure predictions paths match active flags
    if len(args.predictions_path) != len(active_flags):
        print(f"Error: You provided {len(args.predictions_path)} predictions path(s), "
              f"but {len(active_flags)} mode flag(s) are active. These must match.")
        return

    # Map predictions paths to flags
    predictions_map = dict(zip(active_flags, args.predictions_path))

    # Special case: just list instances
    if args.list_instances and "MSWEBugFixing" in active_flags:
        print("Listing available instances...")
        dataset_base_path = "./multiswebench_local/mswebench_dataset"
        dataset_files = []
        for root, _, files in os.walk(dataset_base_path):
            for file in files:
                if file.endswith("_dataset.jsonl"):
                    dataset_files.append(os.path.join(root, file))

        generate_gold_patch_predictions(dataset_files, max_instances=0)
        return

    # Update constants for Omni tasks
    omnicode_flags = ["BugFixing", "TestGeneration", "CodeReview", "CodeMigration"]
    if any(flag in active_flags for flag in omnicode_flags):
        for instance_repo in REPO_DATA:
            swebench.versioning.constants.MAP_REPO_TO_VERSION_PATHS[instance_repo] = REPO_DATA[instance_repo]["MAP_REPO_TO_VERSION_PATHS"]
            swebench.versioning.constants.MAP_REPO_TO_VERSION_PATTERNS[instance_repo] = REPO_DATA[instance_repo]["MAP_REPO_TO_VERSION_PATTERNS"]
            swebench.harness.constants.MAP_REPO_VERSION_TO_SPECS[instance_repo] = REPO_DATA[instance_repo]["MAP_REPO_VERSION_TO_SPECS"]

            # from swebench.harness.log_parsers import parse_log_pytest, parse_log_pytest_options, parse_log_pytest_v2
            from swebench.harness.log_parsers.python import parse_log_pytest, parse_log_pytest_options, parse_log_pytest_v2
            if "MAP_REPO_TO_PARSER" in REPO_DATA[instance_repo]:
                repo_log_parser = eval(REPO_DATA[instance_repo]["MAP_REPO_TO_PARSER"])
            else:
                repo_log_parser = parse_log_pytest
            swebench.harness.log_parsers.MAP_REPO_TO_PARSER[instance_repo] = repo_log_parser
        importlib.reload(swebench)

    # Execute tasks based on flags
    if "BugFixing" in active_flags:
        print("Executing BugFixing...")
        execute_command(
            RegularEval,
            dataset_name=args.dataset_name,
            split="test",
            instance_ids=args.instance_ids,
            predictions_path=predictions_map["BugFixing"],
            max_workers=args.max_workers,
            force_rebuild=args.force_rebuild,
            cache_level=args.cache_level,
            clean=args.clean,
            open_file_limit=args.open_file_limit,
            run_id=args.run_id,
            timeout=args.timeout,
            # namespace=,
            # rewrite_reports=,
            # modal=,
            # instance_image_tag=,
            # report_dir=,
            use_apptainer=args.use_apptainer,
            local=local
        )

    if "TestGeneration" in active_flags:
        print("Executing TestGeneration...")
        execute_command(
            GenTestMain,
            dataset_name=args.dataset_name,
            split="test",
            instance_ids=args.instance_ids,
            predictions_path=predictions_map["TestGeneration"],
            max_workers=args.max_workers,
            force_rebuild=args.force_rebuild,
            cache_level=args.cache_level,
            clean=args.clean,
            open_file_limit=args.open_file_limit,
            run_id=args.run_id,
            timeout=args.timeout,
            use_apptainer=args.use_apptainer,
        )

    if "CodeReview" in active_flags:
        print("Executing CodeReview...")
        execute_command(
            RegularEval,
            dataset_name=args.dataset_name,
            split="test",
            instance_ids=args.instance_ids,
            predictions_path=predictions_map["CodeReview"],
            max_workers=args.max_workers,
            force_rebuild=args.force_rebuild,
            cache_level=args.cache_level,
            clean=args.clean,
            open_file_limit=args.open_file_limit,
            run_id=args.run_id,
            timeout=args.timeout,
            # namespace=,
            # rewrite_reports=,
            # modal=,
            # instance_image_tag=,
            # report_dir=,
            use_apptainer=args.use_apptainer,
            local=local
        )

    if "PythonStyleReview" in active_flags:
        print("Executing Python Style Review...")
        execute_command(
            RegularEval,
            dataset_name="data/omnicode_style_instances_python.json",
            split="test",
            instance_ids=args.instance_ids,
            predictions_path=predictions_map["PythonStyleReview"],
            max_workers=args.max_workers,
            force_rebuild=args.force_rebuild,
            cache_level=args.cache_level,
            clean=args.clean,
            open_file_limit=args.open_file_limit,
            run_id=args.run_id,
            timeout=args.timeout,
            # namespace=,
            # rewrite_reports=,
            # modal=,
            # instance_image_tag=,
            # report_dir=,
            use_apptainer=args.use_apptainer,
            local=local
        )

    if "CodeMigration" in active_flags:
        print("Executing CodeMigration...")
        print("Code Migration is not yet supported!")

    if "MSWEBugFixing" in active_flags:
        print("Executing Multi-SWE-Bench BugFixing...")

        # Create image prefix
        mswe_image_prefix = f"mswebench_{args.run_id}"

        # Process predictions
        if predictions_map["MSWEBugFixing"] == "gold":
            print("Using gold patches from the dataset...")
            dataset_base_path = "./multiswebench_local/mswebench_dataset"
            dataset_files = []
            for root, _, files in os.walk(dataset_base_path):
                for file in files:
                    if file.endswith("_dataset.jsonl"):
                        dataset_files.append(os.path.join(root, file))

            if not dataset_files:
                print("Error: No dataset files found in", dataset_base_path)
                return

            predictions = generate_gold_patch_predictions(
                dataset_files,
                args.instance_ids,
                args.max_instances
            )

            if not predictions:
                print("Error: No valid predictions could be generated from gold patches")
                return

            print(f"Generated {len(predictions)} predictions from gold patches")
        else:
            try:
                with open(predictions_map["MSWEBugFixing"], 'r') as f:
                    if predictions_map["MSWEBugFixing"].endswith(".jsonl"):
                        predictions = []
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                predictions.append(json.loads(line))
                            except json.JSONDecodeError:
                                print(f"Warning: Skipping corrupt line: {line}")
                    else:
                        predictions = json.load(f)
                
                if args.instance_ids is not None:
                    # Filter predictions for requested instances
                    filtered_predictions = [p for p in predictions if p['instance_id'] in args.instance_ids]
                    
                    # Check if any requested instances are missing from predictions
                    found_instance_ids = {p['instance_id'] for p in filtered_predictions}
                    missing_instance_ids = [iid for iid in args.instance_ids if iid not in found_instance_ids]
                    
                    if missing_instance_ids:
                        print(f"Missing predictions for instances: {missing_instance_ids}")
                        print("Generating gold patch predictions for missing instances...")
                        
                        # Load dataset files to generate gold predictions for missing instances
                        dataset_base_path = "./multiswebench_local/mswebench_dataset"
                        dataset_files = []
                        for root, _, files in os.walk(dataset_base_path):
                            for file in files:
                                if file.endswith("_dataset.jsonl"):
                                    dataset_files.append(os.path.join(root, file))
                        
                        if dataset_files:
                            missing_predictions = generate_gold_patch_predictions(
                                dataset_files,
                                missing_instance_ids,
                                0  # No limit for missing instances
                            )
                            print(f"Generated {len(missing_predictions)} gold patch predictions for missing instances")
                            filtered_predictions.extend(missing_predictions)
                        else:
                            print("Warning: No dataset files found to generate missing predictions")
                    
                    predictions = filtered_predictions
                if args.max_instances > 0 and len(predictions) > args.max_instances:
                    print(f"Limiting to {args.max_instances} instances out of {len(predictions)}")
                    predictions = predictions[:args.max_instances]
            except Exception as e:
                print(f"Error loading predictions file: {e}")
                try:
                    predictions = []
                    with open(predictions_map["MSWEBugFixing"], 'r') as f:
                        for line in f:
                            predictions.append(json.loads(line))
                except Exception as e:
                    print(f"Error loading fallback predictions file: {e}")
                    return        
        # Clean existing images if needed
        if args.force_rebuild:
            clean_docker_images(mswe_image_prefix, args.use_apptainer)

        print(f"Loaded {len(predictions)} predictions for Multi-SWE-Bench BugFixing")
        print(f"type of predictions: {type(predictions)}")
        if not isinstance(predictions, list):
            predictions = [predictions]  # Ensure it's a list
        
        if len(predictions) == 0:
            print(f"Found no predictions!")
            return 

        # Set up config
        config_file = setup_multiswebench_config(
            predictions=predictions,
            max_workers=args.max_workers,
            force_rebuild=args.force_rebuild,
            run_id=args.run_id,
            timeout=args.timeout,
            phase=args.mswe_phase,
            use_apptainer=args.use_apptainer,
            g2=args.g2
        )

        # Run evaluation
        if config_file and os.path.exists(config_file):
            print(f"Config file created at: {config_file}")
            report = run_multiswebench_phase(config_file, args.mswe_phase, args.timeout)

            if report:
                print("Multi-SWE-Bench BugFixing evaluation completed!")
                print(f"Total instances: {report.get('total_instances', 0)}")
                print(f"Resolved instances: {report.get('resolved_instances', 0)}")
                print(f"Unresolved instances: {report.get('unresolved_instances', 0)}")
            else:
                print("Multi-SWE-Bench BugFixing evaluation failed to produce a report")
        else:
            print("Failed to create config file, cannot run evaluation")

        if args.g2:
            instance_id = args.instance_ids[0]
            org = instance_id.split("__")[0]
            repo = instance_id.split("__")[1].split("_")[0]
            user_name = os.getenv("USER")
            shutil.copytree(f"/scratch/{user_name}/multiswebench_runs/BugFixing/workdir/{args.run_id}/{org}/{repo}/evals", f"/share/dutta/{user_name}/multiswebench_runs/BugFixing/workdir/{args.run_id}", dirs_exist_ok=True)
            shutil.copytree(f"/scratch/{user_name}/multiswebench_runs/BugFixing/logs/{args.run_id}", f"/share/dutta/{user_name}/multiswebench_runs/BugFixing/logs/{args.run_id}", dirs_exist_ok=True)
            shutil.copytree(f"/scratch/{user_name}/multiswebench_runs/BugFixing/output/{args.run_id}", f"/share/dutta/{user_name}/multiswebench_runs/BugFixing/output/{args.run_id}", dirs_exist_ok=True)
            shutil.rmtree(f"/scratch/{user_name}/multiswebench_runs/BugFixing/logs/{args.run_id}")
            shutil.rmtree(f"/scratch/{user_name}/multiswebench_runs/BugFixing/output/{args.run_id}")
            shutil.rmtree(f"/scratch/{user_name}/multiswebench_runs/BugFixing/workdir/{args.run_id}")

    if "MSWETestGeneration" in active_flags:
        print("Executing Multi-SWE-Bench TestGeneration...")
        execute_command(
            MSWEGenTestMain,
            dataset_name=args.dataset_name,
            split="test",
            instance_ids=args.instance_ids,
            predictions_path=predictions_map["MSWETestGeneration"],
            max_workers=args.max_workers,
            force_rebuild=args.force_rebuild,
            cache_level=args.cache_level,
            clean=args.clean,
            open_file_limit=args.open_file_limit,
            run_id=args.run_id,
            timeout=args.timeout,
            use_apptainer=args.use_apptainer,
        )



    if "StyleReview" in active_flags:
        print("Executing StyleReview...")
        language = args.language

        # Determine language if auto-detection is selected
        if language == 'auto':
            # Simple auto-detection logic based on file extensions
            if args.instance_ids and len(args.instance_ids) > 0:
                # For now just defaulting to Java if instance_ids are specified
                language = 'java'
            else:
                language = 'python'  # Default to Python if unsure
            print(f"Auto-detected language: {language}")

        if language == 'java':
            print("Using Java StyleReview...")

            # Use the exact same data loading approach as MSWEBugFixing
            # Create image prefix consistent with MSWEBugFixing
            mswe_image_prefix = f"mswebench_{args.run_id}"

            # Process predictions exactly like MSWEBugFixing
            if predictions_map["StyleReview"] == "gold":
                print("Using gold patches from the dataset...")
                dataset_base_path = "./multiswebench_local/mswebench_dataset"
                dataset_files = []
                for root, _, files in os.walk(dataset_base_path):
                    for file in files:
                        if file.endswith("_dataset.jsonl"):
                            dataset_files.append(os.path.join(root, file))

                if not dataset_files:
                    print("Error: No dataset files found in", dataset_base_path)
                    return

                predictions = generate_gold_patch_predictions(
                    dataset_files,
                    args.instance_ids,
                    args.max_instances
                )

                if not predictions:
                    print("Error: No valid predictions could be generated from gold patches")
                    return

                print(f"Generated {len(predictions)} predictions from gold patches")
            else:
                # Load predictions from file
                try:
                    with open(predictions_map["StyleReview"], 'r') as f:
                        predictions = json.load(f)

                    if args.max_instances > 0 and len(predictions) > args.max_instances:
                        print(f"Limiting to {args.max_instances} instances out of {len(predictions)}")
                        predictions = predictions[:args.max_instances]
                except Exception as e:
                    print(f"Error loading predictions file: {e}")
                    return

            # Here's a major change: instead of using a fixed dataset name,
            # use the dataset files from the same location as MSWEBugFixing
            # which we already know works with these predictions

            # Find dataset files matching our repos
            dataset_base_path = "./multiswebench_local/mswebench_dataset"
            dataset_files = []

            # Get unique repos for finding datasets
            unique_repos = {f"{pred['org']}/{pred['repo']}" for pred in predictions}

            print(f"Finding relevant dataset files...")
            for root, _, files in os.walk(dataset_base_path):
                for file in files:
                    if file.endswith("_dataset.jsonl"):
                        for repo in unique_repos:
                            org_repo = repo.replace('/', '__')
                            if org_repo in file:
                                dataset_files.append(os.path.join(root, file))
                                print(f"  Found dataset: {os.path.join(root, file)}")
                                break

            if not dataset_files:
                print("Error: No matching dataset files found")
                return

            # Join dataset files with commas for command line argument
            dataset_files_arg = ",".join(dataset_files)

            # Directly call Java style review script with the predictions
            print("Running Java StyleReview directly...")

            # Convert predictions to a temporary file path
            temp_predictions_path = f"temp_{args.run_id}_java_style_predictions.json"
            with open(temp_predictions_path, 'w') as f:
                json.dump(predictions, f)

            try:
                # Use the exact path provided
                if args.review_type=='pmd':
                    script_path="multiswebench_local/multi_swe_bench/harness/style_review/run_java_style_review_pmd.py"
                else:
                    script_path = "multiswebench_local/multi_swe_bench/harness/style_review/run_java_style_review.py"

                if not os.path.exists(script_path):
                    print(f"Error: Java style review script not found at: {script_path}")
                    return

                print(f"Found Java style review script at: {script_path}")

                # Build command
                cmd = [
                    sys.executable,
                    script_path,
                    "--dataset_name", dataset_files_arg,  # Use the found dataset files
                    "--split", "test",
                    "--predictions_path", temp_predictions_path,
                    "--max_workers", str(args.max_workers),
                    "--force_rebuild", str(args.force_rebuild),
                    "--cache_level", args.cache_level,
                    "--clean", str(args.clean),
                    "--open_file_limit", str(args.open_file_limit),
                    "--run_id", args.run_id,
                    "--timeout", str(args.timeout)                   
                ]

                # Add optional arguments if they're set
                if args.min_score is not None:
                    cmd.extend(["--min_score", str(args.min_score)])
                if args.max_severity is not None:
                    cmd.extend(["--max_severity", args.max_severity])
                if args.instance_ids:
                    cmd.extend(["--instance_ids"] + args.instance_ids)
                
                # Run the command
                print(f"Executing command: {' '.join(cmd)}")
                result = run_with_timeout(cmd, args.timeout)
                if result and result[2] == 0:
                    print("Java StyleReview completed successfully")
                else:
                    print("Java StyleReview failed")
            finally:
                # Clean up temporary file
                if os.path.exists(temp_predictions_path):
                    os.remove(temp_predictions_path)
        else:
            print("Using Python StyleReview...")
            execute_command(
                PythonStyleReviewMain,
                dataset_name=args.dataset_name,
                split="test",
                instance_ids=args.instance_ids,
                predictions_path=predictions_map["StyleReview"],
                max_workers=args.max_workers,
                force_rebuild=args.force_rebuild,
                cache_level=args.cache_level,
                clean=args.clean,
                open_file_limit=args.open_file_limit,
                run_id=args.run_id,
                timeout=args.timeout,
                min_score=args.min_score,
                max_severity=args.max_severity,
                use_apptainer=args.use_apptainer,
            )

if __name__ == "__main__":
    main()
