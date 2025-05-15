from pathlib import Path
import json
import logging
import tempfile
import subprocess
import os
from typing import Optional, Tuple

from datasets import load_dataset
from tqdm import tqdm
import pandas as pd

CUR_DIR = Path(__file__).parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def run_aider_single(
    instance: dict,
    model_name: str,
    api_key: str,
    output_dir: Path,
    model_provider: str,
    mode: str = "bugfixing",
    thinking_budget: int | None = None,
) -> Tuple[Optional[str], dict]:
    """
    Run aider on a single instance.
    
    Args:
        instance: Dictionary containing instance information
        model_name: Name of the model to use
        api_key: API key for the model
        output_dir: Directory to store outputs
        mode: Mode to run in (currently unused but kept for compatibility)
        thinking_budget: Thinking budget for the model (currently unused but kept for compatibility)
        
    Returns:
        Tuple of (error message if any, output dictionary)
    """
    # Create a temporary directory for this instance
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Clone the repository
        repo_url = f"https://github.com/{instance['repo']}"
        logger.info(f"Cloning repository: {repo_url}")
        subprocess.run(["git", "clone", repo_url, temp_path], check=True)
        
        # Checkout the specific commit
        logger.info(f"Checking out commit: {instance['base_commit']}")
        subprocess.run(["git", "checkout", instance['base_commit']], cwd=temp_path, check=True)
        
        # Write the problem statement to a file
        problem_file = temp_path / "problem.txt"
        problem_file.write_text(instance['problem_statement'])
        logger.info(f"Problem statement written to: {problem_file}")
        
        # Run aider
        try:
            # Set up environment variables for aider
            env = {
                f"{model_provider}_API_KEY": api_key,
                f"{model_provider}_MODEL": model_name,
                **os.environ
            }
            
            # Run aider with the problem statement
            logger.info("Starting Aider process...")
            logger.info(f"Using model: {model_name}")
            
            # First try to run aider with --version to check if it's installed correctly
            try:
                version_check = subprocess.run(
                    ["aider", "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                logger.info(f"Aider version: {version_check.stdout.strip()}")
            except Exception as e:
                logger.error(f"Error checking aider version: {str(e)}")
            
            # Run the actual aider command
            # logger.info(problem_file.read_text())
            result = subprocess.run(
                ["aider", "--message-file", str(problem_file), "--model", model_name, "--no-auto-commits", "--no-gitignore", "--no-pretty", "--yes"],
                cwd=temp_path,
                capture_output=True,
                text=True,
                env=env,
                check=True,
                timeout=300  # 5 minutes timeout for testing
            )
            logger.info("Aider process completed")
            logger.info(f"Aider stdout: {result.stdout[:500]}...")  # Log first 500 chars of output
            if result.stderr:
                logger.warning(f"Aider stderr: {result.stderr}")
            
            # Get the git diff after aider's changes
            diff_result = subprocess.run(
                ["git", "diff"],
                cwd=temp_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Create instance directory structure
            instance_dir = output_dir / instance['instance_id']
            instance_dir.mkdir(parents=True, exist_ok=True)
            
            # Save the patch in SWE-bench format
            patch_file = instance_dir / "fix.patch"
            patch_file.write_text(diff_result.stdout)
            
            # Save the full output for reference
            output = {
                "instance_id": instance['instance_id'],
                "model_name": model_name,
                "full_output": result.stdout,
                "model_patch": diff_result.stdout  # The actual git patch
            }
            
            # Save the output metadata
            output_file = instance_dir / f"{instance['instance_id']}.pred"
            output_file.write_text(json.dumps(output))
            
            return None, output
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Error running aider: {e.stderr if e.stderr else e.stdout}"
            logger.error(error_msg)
            return error_msg, {}
        except subprocess.TimeoutExpired as e:
            error_msg = f"Aider process timed out after {e.timeout} seconds"
            logger.error(error_msg)
            if e.stdout:
                logger.error(f"Partial stdout: {e.stdout[:500]}...")
            if e.stderr:
                logger.error(f"Partial stderr: {e.stderr}")
            return error_msg, {}
        except Exception as e:
            error_msg = f"Unexpected error running aider: {str(e)}"
            logger.error(error_msg)
            return error_msg, {}

def main(
    input_tasks_path: Path,
    output_dir_path: Path,
    model_name: str,
    api_key: str,
    model_provider: str,
    instance_ids: list[str] | None = None,
    mode: str = "bugfixing",
    thinking_budget: int | None = None,
):
    """
    Main function to run aider on multiple instances.
    
    Args:
        input_tasks_path: Path to input tasks file
        output_dir_path: Path to output directory
        model_name: Name of the model to use
        api_key: API key for the model
        instance_ids: Optional list of instance IDs to process
        mode: Mode to run in (currently unused but kept for compatibility)
        thinking_budget: Thinking budget for the model (currently unused but kept for compatibility)
    """
    # Load the dataset
    if input_tasks_path.exists():
        if input_tasks_path.suffix.endswith("json"):
            dataset = json.loads(input_tasks_path.read_text())
        elif input_tasks_path.suffix.endswith("jsonl"):
            dataset = [json.loads(i) for i in input_tasks_path.read_text().splitlines()]
        elif input_tasks_path.suffix.endswith("csv"):
            dataset = pd.read_csv(input_tasks_path).to_dict('records')
        else:
            raise RuntimeError(f"Data type ({input_tasks_path.suffix}) not supported")
    else:
        dataset = load_dataset(str(input_tasks_path))
        if isinstance(dataset, dict):
            dataset = dataset['test']

    if not (isinstance(dataset, list) and all(isinstance(d, dict) for d in dataset)):
        raise RuntimeError("Data follows incorrect format")

    # Filter by instance IDs if provided
    if instance_ids is not None:
        dataset = [d for d in dataset if d["instance_id"] in instance_ids]

    # Track completed instances
    existing_ids = set()
    output_file_path = output_dir_path / "all_preds.jsonl"
    
    if output_file_path.exists():
        with open(output_file_path) as f:
            for line in f:
                data = json.loads(line)
                instance_id = data["instance_id"]
                existing_ids.add(instance_id)
    
    logger.info(f"Read {len(existing_ids)} already completed ids from {output_file_path}")

    # Process each instance
    with open(output_file_path, "a+") as f:
        for datum in tqdm(dataset, desc=f"Inference for {model_name}"):
            instance_id = datum["instance_id"]
            if instance_id in existing_ids:
                continue
                
            error, output = run_aider_single(
                datum,
                model_name=model_name,
                api_key=api_key,
                output_dir=output_dir_path,
                mode=mode,
                thinking_budget=thinking_budget,
                model_provider=model_provider,
            )
            
            if error:
                logger.error(f"Error processing instance {instance_id}: {error}")
                continue
                
            print(json.dumps(output), file=f, flush=True)

if __name__ == '__main__':
    import os
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_tasks", type=str, required=True)
    parser.add_argument("--instance_ids", type=str, required=False, default=None)
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    parser.add_argument("-m", "--model_name", type=str, default="gemini/gemini-2.5-pro-preview-05-06")
    parser.add_argument("-k", "--api_key", type=str, required=True)
    parser.add_argument("--mode", type=str, default="bugfixing", choices=["bugfixing", "testgen", "bugfixing-java", "testgen-java", "stylereview"])
    parser.add_argument("--thinking_budget", type=int, default=0)
    parser.add_argument("--model_provider", type=str, default="gemini")
    args = parser.parse_args()

    main(
        input_tasks_path=Path(args.input_tasks),
        output_dir_path=Path(args.output_dir),
        model_name=args.model_name,
        instance_ids=args.instance_ids.split(",") if args.instance_ids else None,
        api_key=args.api_key,
        mode=args.mode,
        thinking_budget=args.thinking_budget,
        model_provider=args.model_provider.upper(),
    ) 