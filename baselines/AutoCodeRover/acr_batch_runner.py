#!/usr/bin/env python3
"""
Batch runner for AutoCodeRover that processes all modes on all instances.
This script preserves all existing functionality while adding batch processing capabilities.
Suitable for cluster environments with robust error handling and progress tracking.
"""

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Add the auto-code-rover directory to the path
sys.path.insert(0, str(Path(__file__).parent / "auto-code-rover"))

from acr_runner import load_tasks, run_single

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


def load_config(config_file: Path) -> Dict:
    """
    Load configuration from JSON file.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        log.info(f"Loaded configuration from {config_file}")
        return config
    except Exception as e:
        log.error(f"Failed to load configuration from {config_file}: {e}")
        raise


def create_mode_directories(base_output_dir: Path, modes: List[str]) -> Dict[str, Path]:
    """
    Create directories for each mode.
    
    Args:
        base_output_dir: Base output directory
        modes: List of modes to create directories for
        
    Returns:
        Dictionary mapping mode names to their directories
    """
    mode_dirs = {}
    for mode in modes:
        mode_dir = base_output_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        mode_dirs[mode] = mode_dir
        log.info(f"Created directory for mode '{mode}': {mode_dir}")
    return mode_dirs


def save_patch_to_file(patch_content: Optional[Dict], instance_id: str, mode_dir: Path) -> bool:
    """
    Save patch content to a file named after the instance.
    
    Args:
        patch_content: Patch content from ACR (can be None if failed)
        instance_id: Instance ID for naming the file
        mode_dir: Directory to save the patch in
        
    Returns:
        True if patch was saved successfully, False otherwise
    """
    if patch_content is None:
        log.warning(f"No patch content for instance {instance_id}")
        return False
    
    patch_file = mode_dir / f"{instance_id}.patch"
    
    try:
        # Save the patch content as JSON for consistency
        with patch_file.open("w", encoding="utf-8") as f:
            json.dump(patch_content, f, indent=2, ensure_ascii=False)
        log.info(f"Saved patch for {instance_id} to {patch_file}")
        return True
    except Exception as e:
        log.error(f"Failed to save patch for {instance_id}: {e}")
        return False


def save_intermediate_results(results: List[Dict], mode_dir: Path, mode: str) -> None:
    """
    Save intermediate results to a JSONL file.
    
    Args:
        results: List of result dictionaries
        mode_dir: Directory to save results in
        mode: Mode name for the filename
    """
    results_file = mode_dir / f"{mode}_results.jsonl"
    try:
        with results_file.open("w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        log.info(f"Saved {len(results)} intermediate results to {results_file}")
    except Exception as e:
        log.error(f"Failed to save intermediate results: {e}")


def save_progress(progress: Dict, output_dir: Path) -> None:
    """
    Save progress information to a file.
    
    Args:
        progress: Progress dictionary
        output_dir: Directory to save progress in
    """
    progress_file = output_dir / "batch_progress.json"
    try:
        with progress_file.open("w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2, ensure_ascii=False)
        log.info(f"Saved progress to {progress_file}")
    except Exception as e:
        log.error(f"Failed to save progress: {e}")


def run_batch_processing(
    input_tasks: Path,
    output_dir: Path,
    model_name: str,
    acr_root: Path,
    modes: List[str],
    instance_ids: Optional[str] = None,
    style_feedback_file: Optional[Path] = None,
    save_intermediate: bool = True,
    intermediate_save_interval: int = 10,
    max_retries: int = 3,
    timeout_seconds: int = 3600
) -> None:
    """
    Run batch processing for all modes on all instances.
    
    Args:
        input_tasks: Path to input tasks file
        output_dir: Base output directory
        model_name: Model name to use
        acr_root: ACR root directory
        modes: List of modes to run
        instance_ids: Optional comma-separated list of instance IDs to process
        style_feedback_file: Optional style feedback file for stylereview mode
        save_intermediate: Whether to save intermediate results
        intermediate_save_interval: How often to save intermediate results
        max_retries: Maximum number of retries for failed tasks
        timeout_seconds: Timeout for individual task processing
    """
    start_time = time.time()
    log.info(f"Starting batch processing with model: {model_name}")
    log.info(f"Processing modes: {modes}")
    log.info(f"Output directory: {output_dir}")
    log.info(f"Timeout per task: {timeout_seconds} seconds")
    
    # Load tasks
    tasks = load_tasks(input_tasks)
    log.info(f"Loaded {len(tasks)} tasks from {input_tasks}")
    
    # Filter by instance IDs if specified
    if instance_ids:
        subset = set(instance_ids.split(","))
        tasks = [t for t in tasks if t["instance_id"] in subset]
        log.info(f"Filtered to {len(tasks)} tasks based on instance IDs")
    
    # Create mode directories
    mode_dirs = create_mode_directories(output_dir, modes)
    
    # Initialize progress tracking
    progress = {
        "start_time": start_time,
        "total_tasks": len(tasks),
        "total_modes": len(modes),
        "completed_tasks": 0,
        "failed_tasks": 0,
        "mode_progress": {}
    }
    
    # Process each mode
    for mode_idx, mode in enumerate(modes):
        log.info(f"Processing mode {mode_idx + 1}/{len(modes)}: {mode}")
        mode_dir = mode_dirs[mode]
        results = []
        mode_start_time = time.time()
        
        progress["mode_progress"][mode] = {
            "start_time": mode_start_time,
            "completed": 0,
            "failed": 0,
            "total": len(tasks)
        }
        
        # Process each task for this mode
        for i, task in enumerate(tasks):
            instance_id = task["instance_id"]
            task_start_time = time.time()
            
            log.info(f"Processing {instance_id} ({i+1}/{len(tasks)}) in mode {mode}")
            
            # Get style feedback for stylereview mode
            style_feedback = None
            if mode == "stylereview":
                if style_feedback_file:
                    style_feedback = style_feedback_file.read_text(encoding="utf-8")
                elif "style_review" in task:
                    style_feedback = task["style_review"]
                else:
                    log.warning(f"No style feedback found for {instance_id} in stylereview mode")
            
            # Run ACR with retries
            patch_result = None
            success = False
            error_msg = None
            
            for attempt in range(max_retries):
                try:
                    # Set a timeout for the task
                    import signal
                    
                    def timeout_handler(signum, frame):
                        raise TimeoutError(f"Task {instance_id} timed out after {timeout_seconds} seconds")
                    
                    # Set the timeout signal
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(timeout_seconds)
                    
                    try:
                        patch_result = run_single(
                            task, 
                            model_name, 
                            mode_dir, 
                            acr_root, 
                            mode, 
                            style_feedback, 
                            agentic=True  # Always use agentic mode as required
                        )
                        # Check if run_single returned a valid result (not None)
                        success = patch_result is not None
                        if success:
                            break
                        else:
                            error_msg = "run_single returned None (ACR failed)"
                            log.warning(f"Attempt {attempt + 1} failed for {instance_id}: {error_msg}")
                    finally:
                        # Cancel the alarm
                        signal.alarm(0)
                        
                except TimeoutError as e:
                    error_msg = f"Timeout after {timeout_seconds} seconds"
                    log.warning(f"Attempt {attempt + 1} failed for {instance_id}: {error_msg}")
                except Exception as e:
                    error_msg = str(e)
                    log.warning(f"Attempt {attempt + 1} failed for {instance_id}: {error_msg}")
                
                if attempt < max_retries - 1:
                    log.info(f"Retrying {instance_id} (attempt {attempt + 2}/{max_retries})")
                    time.sleep(2 ** attempt)  # Exponential backoff
            
            # Save patch to file
            patch_saved = save_patch_to_file(patch_result, instance_id, mode_dir)
            
            # Log the result
            if success:
                log.info(f"✅ Task {instance_id} completed successfully")
            else:
                log.warning(f"❌ Task {instance_id} failed: {error_msg}")
            
            # Record result
            result = {
                "instance_id": instance_id,
                "mode": mode,
                "model_name": model_name,
                "patch_saved": patch_saved,
                "patch_content": patch_result,
                "success": success,
                "processing_time": time.time() - task_start_time,
                "attempts": attempt + 1 if not success else 1,
                "error": error_msg if not success else None
            }
            results.append(result)
            
            # Update progress
            if success:
                progress["mode_progress"][mode]["completed"] += 1
                progress["completed_tasks"] += 1
            else:
                progress["mode_progress"][mode]["failed"] += 1
                progress["failed_tasks"] += 1
            
            # Save intermediate results periodically
            if save_intermediate and (i + 1) % intermediate_save_interval == 0:
                save_intermediate_results(results, mode_dir, mode)
                save_progress(progress, output_dir)
        
        # Save final results for this mode
        save_intermediate_results(results, mode_dir, mode)
        
        # Update final mode progress
        progress["mode_progress"][mode]["end_time"] = time.time()
        progress["mode_progress"][mode]["total_time"] = time.time() - mode_start_time
        
        # Print summary for this mode
        successful = sum(1 for r in results if r["success"])
        log.info(f"Mode {mode} completed: {successful}/{len(results)} successful")
    
    # Save final progress
    progress["end_time"] = time.time()
    progress["total_time"] = time.time() - start_time
    save_progress(progress, output_dir)
    
    # Print final summary
    total_successful = progress["completed_tasks"]
    total_failed = progress["failed_tasks"]
    total_time = progress["total_time"]
    
    log.info("=" * 60)
    log.info("BATCH PROCESSING COMPLETED!")
    log.info("=" * 60)
    log.info(f"Total time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
    log.info(f"Total tasks processed: {total_successful + total_failed}")
    log.info(f"Successful: {total_successful}")
    log.info(f"Failed: {total_failed}")
    log.info(f"Success rate: {total_successful/(total_successful + total_failed)*100:.1f}%")
    log.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Batch runner for AutoCodeRover that processes all modes on all instances"
    )
    parser.add_argument(
        "-i", "--input-tasks", 
        type=Path, 
        help="Path to input tasks file (JSON, JSONL, or CSV)"
    )
    parser.add_argument(
        "-o", "--output-dir", 
        type=Path, 
        help="Base output directory for all results"
    )
    parser.add_argument(
        "-m", "--model-name", 
        help="Model name to use (e.g., openrouter/meta-llama/llama-4-scout)"
    )
    parser.add_argument(
        "--acr-root",
        type=lambda p: Path(p).expanduser().resolve(),
        default=Path(__file__).parent.resolve(),
        help="Directory that contains AutoCodeRover's app/ folder"
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["bugfixing", "testgen", "stylereview", "codereview"],
        help="Modes to run (default: all modes)"
    )
    parser.add_argument(
        "--instance-ids", 
        help="Comma-separated subset of task IDs to process"
    )
    parser.add_argument(
        "--style-feedback", 
        type=Path, 
        help="File containing style feedback for stylereview mode"
    )
    parser.add_argument(
        "--no-intermediate-save", 
        action="store_true",
        help="Disable saving intermediate results"
    )
    parser.add_argument(
        "--intermediate-save-interval", 
        type=int, 
        help="How often to save intermediate results"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "batch_config.json",
        help="Configuration file path"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        help="Maximum number of retries for failed tasks"
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        help="Timeout for individual task processing in seconds"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.input_tasks:
        config["input_tasks"] = str(args.input_tasks)
    if args.output_dir:
        config["output_dir"] = str(args.output_dir)
    if args.model_name:
        config["model_name"] = args.model_name
    if args.modes:
        config["modes"] = args.modes
    if args.intermediate_save_interval:
        config["intermediate_save_interval"] = args.intermediate_save_interval
    if args.max_retries:
        config["max_retries"] = args.max_retries
    if args.timeout_seconds:
        config["timeout_seconds"] = args.timeout_seconds
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, config.get("log_level", "INFO")))
    
    # Validate inputs
    input_tasks = Path(config["input_tasks"])
    if not input_tasks.exists():
        log.error(f"Input tasks file does not exist: {input_tasks}")
        sys.exit(1)
    
    style_feedback_file = None
    if args.style_feedback:
        style_feedback_file = args.style_feedback
        if not style_feedback_file.exists():
            log.error(f"Style feedback file does not exist: {style_feedback_file}")
            sys.exit(1)
    
    # Create output directory
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run batch processing
    run_batch_processing(
        input_tasks=input_tasks,
        output_dir=output_dir,
        model_name=config["model_name"],
        acr_root=Path(config["acr_root"]),
        modes=config["modes"],
        instance_ids=args.instance_ids,
        style_feedback_file=style_feedback_file,
        save_intermediate=not args.no_intermediate_save and config.get("save_intermediate", True),
        intermediate_save_interval=config.get("intermediate_save_interval", 10),
        max_retries=config.get("max_retries", 3),
        timeout_seconds=config.get("timeout_seconds", 3600)
    )


if __name__ == "__main__":
    main() 