import os
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple
import logging

from google.cloud import compute_v1
from google.api_core.exceptions import GoogleAPIError

from utils import list_to_bash_array, check_vm_exists, get_vm_status, reset_vm, wait_for_operation, start_vm, get_command, create_image_wrapped, delete_vm

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", None)
DOCKER_PAT = os.environ.get("DOCKER_PAT", None)

# Retry configuration
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 5  # seconds
MAX_RETRY_DELAY = 60  # seconds

VERTEXAI_LOCATION = "us-east5"

def exponential_backoff(retry_number):
    """Calculate delay with exponential backoff and jitter."""
    delay = min(MAX_RETRY_DELAY, INITIAL_RETRY_DELAY * (2 ** retry_number))
    # Add jitter to avoid thundering herd problem
    jitter = random.uniform(0, 0.3 * delay)
    return delay + jitter


def process_single_vm(
    project_id: str,
    zone: str,
    job_prefix: str,
    vm_name: str,
    vm_instance_ids: List[str],
    startup_script: str,
    machine_type: str,
    disk_size_gb: int,
    disk_image: str,
    data_bucket: str,
    overwrite: bool
) -> Tuple[str, bool]:
    """Process a single VM with retry logic."""
    instance_client = compute_v1.InstancesClient()

    for retry in range(MAX_RETRIES):
        try:
            # Check if VM already exists
            vm_exists = check_vm_exists(instance_client, project_id, zone, vm_name)

            if vm_exists:
                if overwrite:
                    # Delete the VM and create a new one
                    logger.info(f"VM {vm_name} exists and overwrite is set to True.")
                    delete_result = delete_vm(instance_client, project_id, zone, vm_name)
                    if not delete_result:
                        logger.error(f"Failed to delete VM: {vm_name}. Retrying...")
                        raise Exception(f"Failed to delete VM: {vm_name}")
                    # VM was deleted, so we'll create a new one
                    vm_exists = False
                else:
                    # Get the VM status
                    status = get_vm_status(instance_client, project_id, zone, vm_name)

                    # Reset the VM with the new startup script
                    reset_operation = reset_vm(instance_client, project_id, zone, vm_name, startup_script, vm_instance_ids)
                    wait_result = wait_for_operation(reset_operation, project_id, zone)

                    if not wait_result:
                        logger.error(f"Failed to update metadata for VM: {vm_name}. Retrying...")
                        raise Exception(f"Failed to update metadata for VM: {vm_name}")

                    if status == "TERMINATED":
                        # Start the VM if it's stopped
                        start_operation = start_vm(instance_client, project_id, zone, vm_name)
                        wait_result = wait_for_operation(start_operation, project_id, zone)

                        if wait_result:
                            logger.info(f"Successfully started existing VM: {vm_name}")
                            return vm_name, True
                        else:
                            logger.error(f"Failed to start VM: {vm_name}. Retrying...")
                            raise Exception(f"Failed to start VM: {vm_name}")
                    elif status == "RUNNING":
                        logger.info(f"VM {vm_name} is already running. Metadata updated with new instance IDs.")
                        return vm_name, True
                    else:
                        logger.warning(f"VM {vm_name} exists but is in state {status}. No action taken.")
                        return vm_name, False

            if not vm_exists:
                # Define the VM configuration
                instance = compute_v1.Instance()
                instance.name = vm_name
                instance.machine_type = f"zones/{zone}/machineTypes/{machine_type}"

                # Define the disk configuration
                disk = compute_v1.AttachedDisk()
                disk.auto_delete = True
                disk.boot = True

                initialize_params = compute_v1.AttachedDiskInitializeParams()
                initialize_params.disk_size_gb = disk_size_gb
                initialize_params.source_image = disk_image
                disk.initialize_params = initialize_params
                instance.disks = [disk]

                # Define the network configuration
                network_interface = compute_v1.NetworkInterface()
                network_interface.network = "global/networks/default"

                access_config = compute_v1.AccessConfig()
                access_config.name = "External NAT"
                access_config.type_ = "ONE_TO_ONE_NAT"
                network_interface.access_configs = [access_config]

                instance.network_interfaces = [network_interface]

                # Set the service account and scopes
                service_account = compute_v1.ServiceAccount()
                service_account.email = "default"
                service_account.scopes = [
                        "https://www.googleapis.com/auth/devstorage.read_write",
                        "https://www.googleapis.com/auth/logging.write",
                        "https://www.googleapis.com/auth/monitoring.write",
                        "https://www.googleapis.com/auth/compute",  # Add compute scope for VM self-deletion
                        "https://www.googleapis.com/auth/cloud-platform",  # Add cloud platform scope for Gemini API
                    ]
                instance.service_accounts = [service_account]

                # Set up as spot/preemptible instance
                scheduling = compute_v1.Scheduling()
                scheduling.provisioning_model = "SPOT"
                instance.scheduling = scheduling

                # Create metadata with startup script
                metadata = compute_v1.Metadata()
                item = compute_v1.Items()
                item.key = "startup-script"
                item.value = startup_script
                metadata.items = [item]
                instance.metadata = metadata

                # Create the VM
                logger.info(f"Creating new VM {vm_name} to process {len(vm_instance_ids)} instances...")
                operation = instance_client.insert(
                    project=project_id,
                    zone=zone,
                    instance_resource=instance
                )

                wait_result = wait_for_operation(operation, project_id, zone)
                if wait_result:
                    logger.info(f"Successfully created VM: {vm_name}")
                    return vm_name, True
                else:
                    logger.error(f"Failed to create VM: {vm_name}. Retrying...")
                    raise Exception(f"Failed to create VM: {vm_name}")

            # If we get here, something unexpected happened
            return vm_name, False

        except Exception as e:
            if retry < MAX_RETRIES - 1:
                delay = exponential_backoff(retry)
                logger.warning(f"Attempt {retry+1} failed for VM {vm_name}: {str(e)}. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"All {MAX_RETRIES} attempts failed for VM {vm_name}: {str(e)}")
                return vm_name, False

    # This should not be reached, but just in case
    return vm_name, False


def create_distributed_compute_vms(
    project_id: str,
    zone: str,
    job_prefix: str,
    instance_ids: list[str],
    command: str,
    base_vm_name: str,
    key: str,
    machine_type: str = "e2-standard-4",
    disk_size_gb: int = 100,
    num_vms: int = 20,
    disk_image: str | None = None,
    data_bucket: str = "your-data-bucket",
    overwrite: bool = False,
    vm_num_offset: int = 0,
    max_workers: int = 10,  # Control parallelism
    indices_to_run: list[int] | None = None,
    vertex_setup: str = ""
):
    """Create or reuse multiple Compute Engine VMs in parallel to distribute the processing."""

    # Calculate how many instances each VM should handle
    total_instances = len(instance_ids)
    instances_per_vm = total_instances // num_vms
    remainder = total_instances % num_vms

    # Prepare tasks for parallel execution
    tasks = []

    for vm_index in range(num_vms):

        if indices_to_run is not None and vm_index not in indices_to_run:
            continue

        # Calculate the start and end indices for this VM
        # Distribute any remainder instances to the first few VMs
        start_index = vm_index * instances_per_vm + min(vm_index, remainder)
        extra = 1 if vm_index < remainder else 0
        end_index  = start_index + instances_per_vm + extra

        # Get the subset of instance IDs for this VM
        vm_instance_ids = instance_ids[start_index:end_index]

        # Create a unique name for this VM
        vm_name = f"{base_vm_name}-{vm_num_offset + vm_index}"

        startup_script = f"""#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status
# Set up logging
MAIN_LOGFILE="/home/ays57/seds/codearena/batch_processing.log"
exec > >(tee -a "$MAIN_LOGFILE") 2>&1
echo "$(date): Script started"
echo "Running as user: $(whoami)"
echo "Home directory: $HOME"
su - ays57 << 'EOSU'
export GEMINI_API_KEY="{key}"
export GITHUB_TOKEN="{GITHUB_TOKEN}"
export DOCKER_PAT="{DOCKER_PAT}"
# for aider
export PATH="/home/ays57/.local/bin:$PATH"
{vertex_setup}
echo "Now running as $(whoami) with home directory $HOME"
# Explicitly set PATH to include common conda locations
export PATH="$HOME/miniconda3/bin:$HOME/anaconda3/bin:/opt/conda/bin:$PATH"
echo "PATH is set to: $PATH"
# Check if required commands exist
command -v gcloud >/dev/null 2>&1 || {{ echo "ERROR: gcloud is not installed"; exit 1; }}
# command -v conda >/dev/null 2>&1 || {{ echo "ERROR: conda is not installed"; exit 1; }}
conda --version
# Set trap for cleanup
cleanup() {{
  echo "$(date): Script interrupted, cleaning up..."
  # Any cleanup actions here
}}
trap cleanup EXIT INT TERM
export BUCKET_NAME="{data_bucket}"
instances=({list_to_bash_array(vm_instance_ids)})
# Check if instances array is empty
if [ ${{#instances[@]}} -eq 0 ]; then
  echo "ERROR: No instance IDs provided."
  exit 1
fi
# Verify only the bucket access, not the job_prefix directory
gcloud storage ls gs://${{BUCKET_NAME}}/ >/dev/null 2>&1 || {{
  echo "ERROR: Cannot access destination bucket gs://${{BUCKET_NAME}}/"
  exit 1
}}
cd /home/ays57/seds/codearena || {{ echo "ERROR: Directory not found"; exit 1; }}
# Ensure conda environment activation works in non-interactive shells
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate seds || {{ echo "ERROR: Failed to activate conda environment 'seds'"; exit 1; }}
# Loop through the assigned instances
for INSTANCE_ID in "${{instances[@]}}"; do
  echo "$(date): Processing instance $INSTANCE_ID"

  # Clean up logs directory if it exists
  [ -d "logs" ] && rm -rf logs
  mkdir logs
  # Set up instance-specific logging
  INSTANCE_LOGFILE="logs/instance_${{INSTANCE_ID}}.log"

  # Run your processing script with timeout
  echo "$(date): Running python script for instance $INSTANCE_ID" | tee -a "$INSTANCE_LOGFILE"
  {command} 2>&1 | tee -a "$INSTANCE_LOGFILE"

  PYTHON_EXIT_CODE=${{PIPESTATUS[0]}}
  if [ $PYTHON_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Processing failed for instance $INSTANCE_ID with exit code $PYTHON_EXIT_CODE" | tee -a "$INSTANCE_LOGFILE"
  fi

  # Upload results
  echo "$(date): Uploading results for instance $INSTANCE_ID" | tee -a "$INSTANCE_LOGFILE"
  gcloud storage cp -r logs/ gs://${{BUCKET_NAME}}/{job_prefix}/$INSTANCE_ID 2>&1

  if [ $? -eq 0 ]; then
    echo "$(date): Completed instance $INSTANCE_ID" | tee -a "$INSTANCE_LOGFILE"
  else
    echo "ERROR: Upload failed for instance $INSTANCE_ID" | tee -a "$INSTANCE_LOGFILE"
  fi
done
echo "$(date): All processing completed"
EOSU

echo "$(date): Shutting down VM"
poweroff
"""

        # Add task for this VM
        tasks.append((
            project_id,
            zone,
            job_prefix,
            vm_name,
            vm_instance_ids,
            startup_script,
            machine_type,
            disk_size_gb,
            disk_image,
            data_bucket,
            overwrite,
        ))

    # Execute tasks in parallel
    successful_vms = []
    failed_vms = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_vm = {
            executor.submit(process_single_vm, *task): task[3]  # vm_name is at index 3
            for task in tasks
        }

        # Process results as they complete
        for future in as_completed(future_to_vm):
            vm_name = future_to_vm[future]
            try:
                result_name, success = future.result()
                if success:
                    successful_vms.append(result_name)
                    logger.info(f"VM {result_name} successfully processed")
                else:
                    failed_vms.append(result_name)
                    logger.warning(f"VM {result_name} processing failed")
            except Exception as e:
                failed_vms.append(vm_name)
                logger.error(f"Exception for VM {vm_name}: {str(e)}")

    # Log summary
    logger.info(f"VM Creation Summary: {len(successful_vms)} successful, {len(failed_vms)} failed")
    if failed_vms:
        logger.warning(f"Failed VMs: {', '.join(failed_vms)}")

    return successful_vms


VERTEXT_SETUP_DICT = {
5: """export VERTEXAI_PROJECT="gemini-5-459920"
export VERTEXAI_LOCATION="us-east5"
export GOOGLE_APPLICATION_CREDENTIALS="../gemini5.json" """,
4: """export VERTEXAI_PROJECT="gemini-4-459920"
export VERTEXAI_LOCATION="us-east5"
export GOOGLE_APPLICATION_CREDENTIALS="../gemini4.json" """,
3: """export VERTEXAI_PROJECT="gemini-3-459920"
export VERTEXAI_LOCATION="us-east5"
export GOOGLE_APPLICATION_CREDENTIALS="../gemini3.json" """,
}


if __name__ == "__main__":
    import sys
    from pathlib import Path
    import argparse
    import random

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Create distributed batch jobs for processing instances")
    parser.add_argument("instances_path", help="Path to file containing instance IDs")
    parser.add_argument("job_type", help="Type of job to run (e.g., 'sanity', 'bp-gen')")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the image from the base VM")
    parser.add_argument("--overwrite", action="store_true", help="If specified, delete existing VMs with the same name before creating new ones")
    parser.add_argument("--dummy", action="store_true", help="Run on 4 instances with 4 VMs")
    parser.add_argument("--vm_num_offset", type=int, required=False, default=0)
    parser.add_argument("--num_vms", type=int, default=20, required=False)
    parser.add_argument("--randomise", action="store_true", help="randomise sequence of instances being processed")
    parser.add_argument("--max_parallel", type=int, default=10, help="Maximum number of VMs to create in parallel")
    parser.add_argument("--zone", type=str, default=None, help="zone in which to create VMs")
    parser.add_argument("--base_zone", type=str, required=True, help="base vm zone")
    parser.add_argument("--base_vm", type=str, required=True, help="Base VM name, e.g. sedsbase")
    parser.add_argument("--vm_idx_to_run", type=str, help="comma seperate list if ints indicating vms to run", default=None)
    parser.add_argument("--vertex_setup", type=int, default=None)
    parser.add_argument("--key", "-k", type=str, default=None)

    args = parser.parse_args()

    # Parse arguments
    job_type = args.job_type
    rebuild = args.rebuild
    overwrite = rebuild or args.overwrite

    # Hardcode the base VM name
    base_vm_name = args.base_vm
    base_zone = args.base_zone
    project_id = "gen-lang-client-0511233871"
    zone = args.zone
    if zone is None:
        zone = base_zone

    key = args.key
    if key is None:
        raise RuntimeError(f"Could not fine key")

    # Define image family for this job type
    command = get_command(job_type)

    if command is None:
        raise RuntimeError(f"No command found for job type: {job_type}")

    if args.instances_path == "dummy":
        instances_list = ["sympy__sympy-23950", "pydata__xarray-4356", "ytdl-org__youtube-dl-32725", "celery__celery-8486"]
    else:
        instances_list = Path(args.instances_path).read_text().splitlines()

    if args.randomise:
        random.shuffle(instances_list)

    vertex_setup = VERTEXT_SETUP_DICT[args.vertex_setup] if args.vertex_setup is not None else ""

    indices_to_run = [int(s) for s in args.vm_idx_to_run.split(',')] if args.vm_idx_to_run is not None else None

    try:
        disk_image = create_image_wrapped(
            rebuild=rebuild,
            project_id=project_id,
            zone=base_zone,
            source_instance=base_vm_name,
        )

        vms = create_distributed_compute_vms(
            project_id=project_id,
            zone=zone,
            job_prefix=f"sweb-{job_type}",
            instance_ids=instances_list,
            command=command,
            base_vm_name=base_vm_name,
            key=key,
            machine_type="e2-standard-4",
            disk_size_gb=100,
            num_vms=len(instances_list) if args.instances_path == "dummy" else args.num_vms,
            disk_image=disk_image,
            data_bucket="seds-store",
            overwrite=overwrite,
            vm_num_offset=args.vm_num_offset,
            max_workers=args.max_parallel,
            indices_to_run=indices_to_run,
            vertex_setup=vertex_setup,
        )

        logger.info(f"Successfully managed {len(vms)} VMs to process instances")

    except Exception as e:
        logger.error(f"Error creating image or submitting jobs: {e}")
        sys.exit(1)
