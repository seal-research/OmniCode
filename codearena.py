import argparse
from pathlib import Path
import importlib
import json
import os
import subprocess
import sys
import glob
import time
import signal

from run_evaluation_GenTests import main as GenTestMain
import swebench
from swebench.harness.utils import str2bool
from swebench.harness.run_evaluation import main as RegularEval

CUR_DIR = Path(__file__).parent
REPO_DATA_PATH = CUR_DIR / "data/codearena_repo_data.py"
REPO_DATA = eval(REPO_DATA_PATH.read_text())

def execute_command(func, **kwargs):
    """Wrapper to execute a function safely and catch errors."""
    func(**kwargs)

def generate_gold_patch_predictions(dataset_files, instance_ids=None, max_instances=0):
    """Generate predictions from gold patches in the dataset files."""
    predictions = []
    
    # Keep track of how many predictions we've added
    added_count = 0
    
    # Early check for max_instances
    if max_instances <= 0:
        max_instances = float('inf')  # No limit
    
    # Print what instances we're looking for
    if instance_ids:
        print(f"Filtering for specific instances: {instance_ids}")
        
        # Extract repos from instance_ids to filter dataset files
        target_repos = set()
        for instance_id in instance_ids:
            if ":" in instance_id:
                repo_part = instance_id.split(":")[0]  # Extract org/repo part
                target_repos.add(repo_part.replace("/", "__"))  # Convert to format in filenames
        
        print(f"Looking only in dataset files for repos: {target_repos}")
        
        # Filter dataset files to only include relevant repos
        filtered_dataset_files = []
        for dataset_file in dataset_files:
            for repo in target_repos:
                if repo in dataset_file:
                    filtered_dataset_files.append(dataset_file)
                    print(f"Including dataset file: {dataset_file}")
                    break
        
        # Replace the original list with the filtered one
        dataset_files = filtered_dataset_files
    
    if not dataset_files:
        print("No matching dataset files found for the requested instances!")
        return predictions
    
    for dataset_file in dataset_files:
        print(f"Processing dataset file: {dataset_file}")
        
        # If we've already reached max_instances, break early
        if added_count >= max_instances:
            print(f"Reached maximum of {max_instances} instances, stopping.")
            break
        
        with open(dataset_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                
                # If we've already reached max_instances, break early
                if added_count >= max_instances:
                    break
                
                try:
                    item = json.loads(line)
                    # Create ID for filtering
                    item_id = f"{item['org']}/{item['repo']}:{item['number']}"
                    
                    # Filter by instance_ids if provided
                    if instance_ids and item_id not in instance_ids:
                        continue
                    
                    # Create prediction entry from dataset item
                    pred = {
                        "id": item_id,
                        "org": item['org'],
                        "repo": item['repo'],
                        "number": item['number'],
                        "patch": item.get('fix_patch', '')
                    }
                    
                    if pred["patch"]:  # Only include entries with non-empty patches
                        predictions.append(pred)
                        added_count += 1
                        print(f"  Added prediction for {pred['id']} ({added_count}/{max_instances})")
                        
                        # If we've found all requested instances, we can stop
                        if instance_ids and len(predictions) == len(instance_ids):
                            print("Found all requested instances, stopping search.")
                            return predictions
                        
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON from {dataset_file} at line {i+1}: {e}")
                    continue
    
    print(f"Total predictions added: {len(predictions)}")
    return predictions

def setup_multiswebench_config(predictions, max_workers, force_rebuild, run_id, timeout, phase="all"):
    """Set up configuration for Multi-SWE-Bench evaluation."""
    data_dir = Path("data/multiswebench")
    
    # Create necessary directories
    print("Creating directory structure...")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(data_dir / "workdir", exist_ok=True)
    os.makedirs(data_dir / "logs", exist_ok=True)
    os.makedirs(data_dir / "output", exist_ok=True)
    os.makedirs(data_dir / "patches", exist_ok=True)
    os.makedirs(data_dir / "datasets", exist_ok=True)
    os.makedirs(data_dir / "repos", exist_ok=True)
    
    # Extract unique repo/org pairs from the predictions to know which images to build
    unique_repos = set()
    for pred in predictions:
        unique_repos.add(f"{pred['org']}/{pred['repo']}")
    
    print(f"Will only build images for these repos: {unique_repos}")
    
    # Create a patch file from the predictions
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
    
    # Find appropriate dataset files in the specified location
    dataset_base_path = "./multiswebench/mswebench_dataset"
    dataset_files = []
    
    # Only include dataset files that correspond to our repos
    print(f"Finding relevant dataset files in {dataset_base_path}...")
    for root, _, files in os.walk(dataset_base_path):
        for file in files:
            if file.endswith("_dataset.jsonl"):
                # Check if this dataset file is for one of our repos
                for repo in unique_repos:
                    # Extract org/repo from filename
                    org_repo = repo.replace('/', '__')
                    if org_repo in file:
                        dataset_files.append(os.path.join(root, file))
                        print(f"  Found relevant dataset: {os.path.join(root, file)}")
                        break
    
    # Determine mode based on phase
    if phase == "image":
        mode = "image"
    elif phase == "instance":
        mode = "instance_only"
    else:
        mode = "evaluation"
    
    # Create config with the correct parameter names
    print(f"Creating configuration for phase: {phase}...")
    config = {
        "mode": mode,
        "workdir": str(data_dir / "workdir"),
        "patch_files": [str(patch_file)],
        "dataset_files": dataset_files,
        "force_build": force_rebuild,
        "output_dir": str(data_dir / "output"),
        "specifics": [],  # We filter by instance IDs in the prediction generation
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
        "log_level": "DEBUG",  # Use DEBUG for more verbose output
        "log_to_console": True
    }
    
    config_file = data_dir / f"{run_id}_{phase}_config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to {config_file}")
    return str(config_file)  # Return as string, not Path object

def run_with_timeout(cmd, timeout_seconds=1800):
    """Run a command with timeout and proper signal handling."""
    print(f"Running command with {timeout_seconds}s timeout: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Monitor process with timeout
        start_time = time.time()
        stdout_lines = []
        stderr_lines = []
        
        # Set up polling to read output
        import select
        outputs = [process.stdout, process.stderr]
        while process.poll() is None:
            # Check if we've exceeded timeout
            if time.time() - start_time > timeout_seconds:
                print(f"Process timed out after {timeout_seconds} seconds!")
                process.kill()
                return None, "Timeout exceeded", 1
            
            # Read any available output
            readable, _, _ = select.select(outputs, [], [], 1.0)
            for stream in readable:
                line = stream.readline()
                if line:
                    if stream == process.stdout:
                        stdout_lines.append(line)
                        print(f"STDOUT: {line.strip()}")
                    else:
                        stderr_lines.append(line)
                        # Check if this is an actual error or just a log message
                        if "ERROR" in line or "CRITICAL" in line or "FATAL" in line:
                            print(f"STDERR: {line.strip()}")
                        else:
                            print(f"LOG: {line.strip()}")
        
        # Read any remaining output
        for line in process.stdout:
            stdout_lines.append(line)
            print(f"STDOUT: {line.strip()}")
        for line in process.stderr:
            stderr_lines.append(line)
            if "ERROR" in line or "CRITICAL" in line or "FATAL" in line:
                print(f"STDERR: {line.strip()}")
            else:
                print(f"LOG: {line.strip()}")
        
        stdout = "".join(stdout_lines)
        stderr = "".join(stderr_lines)
        
        if process.returncode != 0:
            print(f"Command failed with exit code {process.returncode}")
            return stdout, stderr, process.returncode
        
        return stdout, stderr, 0
        
    except Exception as e:
        print(f"Error running command: {e}")
        return None, str(e), 1

def run_multiswebench_phase(config_file, phase="all", timeout=1800):
    """Run a specific phase of Multi-SWE-Bench evaluation."""
    # Determine which script and arguments to use
    script_path = "./multiswebench/multi_swe_bench/harness/run_evaluation.py"
    
    # Check if the script exists
    if not os.path.exists(script_path):
        print(f"Script not found: {script_path}")
        return None
    
    # Check if config_file is None
    if config_file is None:
        print("Error: config_file is None, cannot run evaluation")
        return None
        
    # Ensure config_file is a string path
    config_file_path = str(config_file)
    
    # Run with timeout
    print(f"Running command with full config file path: {config_file_path}")
    cmd = [sys.executable, script_path, "--config", config_file_path]
    stdout, stderr, returncode = run_with_timeout(cmd, timeout)
    
    if returncode != 0:
        print(f"Command failed with code {returncode}")
        print(f"stderr: {stderr}")
        return None
    
    # Return evaluation results if available
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

def clean_docker_images(image_prefix):
    """Clean up Docker images with the specified prefix."""
    try:
        print(f"Cleaning up Docker images with prefix: {image_prefix}")
        cmd = ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}", f"{image_prefix}*"]
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, text=True)
        
        images = result.stdout.strip().split('\n')
        images = [img for img in images if img]  # Filter out empty lines
        
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
    parser = argparse.ArgumentParser(description="Run CodeArena Benchmarks")

    # Add arguments for common parameters
    parser.add_argument("--dataset_name", default="data/codearena_instances.json", help="Name of the dataset")
    parser.add_argument(
        "--predictions_path",
        nargs="+",
        required=True,
        help="Paths to predictions files (.json or .jsonl). Use 'gold' to use gold patches.",
    )
    parser.add_argument(
        "--max_workers", type=int, default=1, help="Number of maximum workers to use"
    )
    parser.add_argument("--run_id", required=True, help="Run ID for the evaluation")
    parser.add_argument(
        "--instance_ids",
        nargs="*",
        help="Optional instance IDs",
    )
    parser.add_argument(
        "--open_file_limit",
        type=int,
        default=4096,
        help="Maximum number of open files",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1_800,
        help="Timeout for individual evaluations in seconds",
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
    parser.add_argument(
        "--clean", type=str2bool, default=False, help="Clean images above cache level"
    )
    # Add an option to limit the number of instances for testing
    parser.add_argument(
        "--max_instances", type=int, default=0, 
        help="Maximum number of instances to process (0 for unlimited)"
    )
    # Add a phase option for MSWE
    parser.add_argument(
        "--mswe_phase", 
        choices=["all", "image", "instance"],
        default="all",
        help="Which phase of Multi-SWE-Bench to run"
    )

    # Add flags for selecting the benchmark
    parser.add_argument(
        "--BugFixing", action="store_true", help="Run the regular BugFixing benchmark"
    )
    parser.add_argument(
        "--TestGeneration", action="store_true", help="Run the TestGeneration benchmark"
    )
    parser.add_argument(
        "--CodeReview", action="store_true", help="Run the CodeReview benchmark"
    )
    parser.add_argument(
        "--CodeMigration", action="store_true", help="Run the CodeMigration benchmark"
    )
    # Multi-SWE-Bench flags - you can add specific ones for different task types
    parser.add_argument(
        "--MSWEBugFixing", action="store_true", help="Run the Multi-SWE-Bench BugFixing benchmark"
    )
    parser.add_argument(
        "--MSWETestGeneration", action="store_true", help="Run the Multi-SWE-Bench TestGeneration benchmark"
    )

    args = parser.parse_args()

    # Collect the active flags
    active_flags = []
    if args.BugFixing:
        active_flags.append("BugFixing")
    if args.TestGeneration:
        active_flags.append("TestGeneration")
    if args.CodeReview:
        active_flags.append("CodeReview")
    if args.CodeMigration:
        active_flags.append("CodeMigration")
    if args.MSWEBugFixing:
        active_flags.append("MSWEBugFixing")
    if args.MSWETestGeneration:
        active_flags.append("MSWETestGeneration")

    # Ensure at least one flag is provided
    if not active_flags:
        print(
            "Error: You must specify at least one benchmark flag (--BugFixing, --TestGeneration, --CodeReview, "
            "--CodeMigration, --MSWEBugFixing, or --MSWETestGeneration)."
        )
        return

    # Ensure the number of predictions paths matches the number of active flags
    if len(args.predictions_path) != len(active_flags):
        print(
            f"Error: You provided {len(args.predictions_path)} predictions path(s), "
            f"but {len(active_flags)} mode flag(s) are active. These numbers must match."
        )
        return

    # Map the predictions paths to the flags
    predictions_map = dict(zip(active_flags, args.predictions_path))

    # Update constants in swebench for the original CodeArena tasks
    if any(flag in active_flags for flag in ["BugFixing", "TestGeneration", "CodeReview", "CodeMigration"]):
        for instance_repo in REPO_DATA:
            swebench.versioning.constants.MAP_REPO_TO_VERSION_PATHS[instance_repo] = REPO_DATA[instance_repo]["MAP_REPO_TO_VERSION_PATHS"]
            swebench.versioning.constants.MAP_REPO_TO_VERSION_PATTERNS[instance_repo] = REPO_DATA[instance_repo]["MAP_REPO_TO_VERSION_PATTERNS"]
            swebench.harness.constants.MAP_REPO_VERSION_TO_SPECS[instance_repo] = REPO_DATA[instance_repo]["MAP_REPO_VERSION_TO_SPECS"]
    
            from swebench.harness.log_parsers import parse_log_pytest, parse_log_pytest_options, parse_log_pytest_v2
            if "MAP_REPO_TO_PARSER" in REPO_DATA[instance_repo]:
                repo_log_parser = eval(REPO_DATA[instance_repo]["MAP_REPO_TO_PARSER"])
            else:
                repo_log_parser = parse_log_pytest
            swebench.harness.log_parsers.MAP_REPO_TO_PARSER[instance_repo] = repo_log_parser
        
        importlib.reload(swebench)

    # Handle BugFixing
    if "BugFixing" in active_flags:
        print("Executing BugFixing...")
        execute_command(
            RegularEval,
            dataset_name=args.dataset_name,
            split="test",  # Assuming split is always 'test', can be parameterized
            instance_ids=args.instance_ids,
            predictions_path=predictions_map["BugFixing"],
            max_workers=args.max_workers,
            force_rebuild=args.force_rebuild,
            cache_level=args.cache_level,
            clean=args.clean,
            open_file_limit=args.open_file_limit,
            run_id=args.run_id,
            timeout=args.timeout
        )

    # Handle TestGeneration
    if "TestGeneration" in active_flags:
        print("Executing TestGeneration...")
        execute_command(
            GenTestMain,
            dataset_name=args.dataset_name,
            split="test",  # Assuming split is always 'test', can be parameterized
            instance_ids=args.instance_ids,
            predictions_path=predictions_map["TestGeneration"],
            max_workers=args.max_workers,
            force_rebuild=args.force_rebuild,
            cache_level=args.cache_level,
            clean=args.clean,
            open_file_limit=args.open_file_limit,
            run_id=args.run_id,
            timeout=args.timeout,
        )

    # Handle CodeReview
    if "CodeReview" in active_flags:
        print("Executing CodeReview...")
        execute_command(
            RegularEval,
            dataset_name=args.dataset_name,
            split="test",  # Assuming split is always 'test', can be parameterized
            instance_ids=args.instance_ids,
            predictions_path=predictions_map["CodeReview"],
            max_workers=args.max_workers,
            force_rebuild=args.force_rebuild,
            cache_level=args.cache_level,
            clean=args.clean,
            open_file_limit=args.open_file_limit,
            run_id=args.run_id,
            timeout=args.timeout
        )

    # Handle CodeMigration
    if "CodeMigration" in active_flags:
        print("Executing CodeMigration...")
        print(f"Code Migration is not yet supported!")

    # Handle MSWEBugFixing
    if "MSWEBugFixing" in active_flags:
        print("Executing Multi-SWE-Bench BugFixing...")
        
        # Create a unique image prefix for this run to avoid conflicts with CodeArena
        mswe_image_prefix = f"mswebench_{args.run_id}"
        
        # Check if using gold patches
        if predictions_map["MSWEBugFixing"] == "gold":
            print("Using gold patches from the dataset...")
            # Find dataset files
            dataset_base_path = "./multiswebench/mswebench_dataset"
            dataset_files = []
            for root, _, files in os.walk(dataset_base_path):
                for file in files:
                    if file.endswith("_dataset.jsonl"):
                        dataset_files.append(os.path.join(root, file))
            
            if not dataset_files:
                print("Error: No dataset files found in", dataset_base_path)
                return
                
            # Generate predictions from gold patches
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
            # Use provided predictions file
            try:
                with open(predictions_map["MSWEBugFixing"], 'r') as f:
                    predictions = json.load(f)
                    
                # Limit the number of instances if requested
                if args.max_instances > 0 and len(predictions) > args.max_instances:
                    print(f"Limiting to {args.max_instances} instances out of {len(predictions)}")
                    predictions = predictions[:args.max_instances]
            except Exception as e:
                print(f"Error loading predictions file: {e}")
                return
        
        # Clean up any existing images with this prefix if force_rebuild is True
        if args.force_rebuild:
            clean_docker_images(mswe_image_prefix)
        
        # Set up config for the specified phase
        config_file = setup_multiswebench_config(
            predictions=predictions,
            max_workers=args.max_workers,
            force_rebuild=args.force_rebuild,
            run_id=args.run_id,
            timeout=args.timeout,
            phase=args.mswe_phase
        )
        
        # Make sure the config file was created successfully
        if config_file and os.path.exists(config_file):
            print(f"Config file created successfully at: {config_file}")
            
            # Run just the specified phase
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

    # Handle MSWETestGeneration
    if "MSWETestGeneration" in active_flags:
        print("Executing Multi-SWE-Bench TestGeneration...")
        # Similar pattern for test generation...
        if predictions_map["MSWETestGeneration"] == "gold":
            print("Using gold test cases from the dataset is not supported for TestGeneration")
            return
        else:
            print("Multi-SWE-Bench TestGeneration is not yet implemented")


if __name__ == "__main__":
    main()