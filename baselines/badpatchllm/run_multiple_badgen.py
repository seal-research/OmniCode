import json
import subprocess
import sys
import concurrent.futures
import logging
from pathlib import Path
from datetime import datetime
import os

gemini_key = os.environ.get("GEMINI_API_KEY", None)

# Function to run the bash script for a given instance_id
def run_script(instance_id):
    logging.info(f"Starting process for instance ID: {instance_id}")
    result = subprocess.run(
        [
            "python",
            "baselines/badpatchllm/generate_bad.py",
            "--instance_id",
            instance_id,
            "-o",
            "bad_patch_results",
            "-k",
            gemini_key,
            "-n",
            "3",
            "--run_id",
            "bpg",
            "--max_workers",
            "4",
        ],
        capture_output=True,
        text=True
    )
    return result.returncode, instance_id, result.stdout, result.stderr

if __name__ == '__main__':
    # Check if enough arguments are provided
    if len(sys.argv) < 3:
        print("Usage: python script.py <data_path> <max_workers>")
        sys.exit(1)
    
    data_path = sys.argv[1]
    max_workers = int(sys.argv[2])
    
    # Set up logging
    log_filename = f"logs/process_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    Path(log_filename).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Load data from file
    logging.info(f"Loading data from {data_path}")
    instance_ids = set([i.strip() for i in Path(data_path).read_text().splitlines()])
    
    logging.info(f"Found {len(instance_ids)} instances")
    
    # Check if we should use single process mode
    if max_workers == 1:
        logging.info("Using single process mode")
        # Process each instance sequentially
        for instance_id in instance_ids:
            try:
                returncode, instance_id, stdout, stderr = run_script(instance_id)
                if returncode == 0:
                    logging.info(f"Successfully processed instance ID: {instance_id}")
                    # Log stdout to file if needed
                    if stdout.strip():
                        logging.debug(f"Output for {instance_id}:\n{stdout}")
                else:
                    logging.error(f"Failed to process instance ID: {instance_id}")
                    logging.error(f"Error for {instance_id}: {stderr}")
            except Exception as e:
                logging.exception(f"Exception while processing task: {e}")
    else:
        # Use ProcessPoolExecutor for parallel processing
        logging.info(f"Starting parallel execution with max_workers={max_workers}")
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks and gather futures
            futures = [executor.submit(run_script, instance_id) for instance_id in instance_ids]
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    returncode, instance_id, stdout, stderr = future.result()
                    if returncode == 0:
                        logging.info(f"Successfully processed instance ID: {instance_id}")
                        # Log stdout to file if needed
                        if stdout.strip():
                            logging.debug(f"Output for {instance_id}:\n{stdout}")
                    else:
                        logging.error(f"Failed to process instance ID: {instance_id}")
                        logging.error(f"Error for {instance_id}: {stderr}")
                except Exception as e:
                    logging.exception(f"Exception while processing task: {e}")
    
    logging.info("All processing completed")
    print(f"Processing completed. See log file: {log_filename}")