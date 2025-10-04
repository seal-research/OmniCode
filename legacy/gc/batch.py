import argparse

from utils import list_to_bash_array, create_image_wrapped, get_command
from google.cloud import batch_v1


def create_distributed_batch_jobs(
    project_id: str,
    region: str,
    job_prefix: str,
    instance_ids: list[str],
    command: str,
    machine_type: str = "e2-standard-4",
    disk_size_gb: int = 100,
    num_jobs: int = 20,
    disk_image: str = None,
    data_bucket: str = "your-data-bucket",
):
    """Create multiple Cloud Batch jobs to distribute the processing of all instances."""
    client = batch_v1.BatchServiceClient()

    # Calculate how many instances each job should handle
    total_instances = len(instance_ids)
    instances_per_job = total_instances // num_jobs
    remainder = total_instances % num_jobs
    
    jobs = []
    
    for job_index in range(num_jobs):
        # Calculate the start and end indices for this job
        start_index = job_index * instances_per_job
        end_index = start_index + instances_per_job
        
        # Distribute any remainder instances to the first few jobs
        if job_index < remainder:
            end_index += 1
        
        # Get the subset of instance IDs for this job
        job_instance_ids = instance_ids[start_index:end_index]
        
        # Skip creating a job if there are no instances to process
        if not job_instance_ids:
            continue
        
        # Define the VM resources
        resources = batch_v1.ComputeResource()
        resources.cpu_milli = 4000  # 4 vCPUs
        resources.memory_mib = 16384  # 16 GB RAM
        resources.boot_disk_mib = disk_size_gb * 1024
        
        # Define the job script that will process all assigned instances
        runnable = batch_v1.Runnable()
        runnable.script = batch_v1.Runnable.Script()
        runnable.script.text = f"""#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status
# Set up logging
MAIN_LOGFILE="/home/ays57/seds/codearena/batch_processing.log"
exec > >(tee -a "$MAIN_LOGFILE") 2>&1
echo "$(date): Script started"
echo "Running as user: $(whoami)"
echo "Home directory: $HOME"
su - ays57 << 'EOSU'
echo "Now running as $(whoami) with home directory $HOME"
# Explicitly set PATH to include common conda locations
export PATH="$HOME/miniconda3/bin:$HOME/anaconda3/bin:/opt/conda/bin:$PATH"
echo "PATH is set to: $PATH"
# Check if required commands exist
command -v gcloud >/dev/null 2>&1 || {{ echo "ERROR: gcloud is not installed"; exit 1; }}
conda --version
# Set trap for cleanup
cleanup() {{
  echo "$(date): Script interrupted, cleaning up..."
  # Any cleanup actions here
}}
trap cleanup EXIT INT TERM

# Define instance IDs for this specific job
instances=({list_to_bash_array(job_instance_ids)})

# Check if instances array is empty
if [ ${{#instances[@]}} -eq 0 ]; then
  echo "ERROR: No instance IDs provided."
  exit 1
fi

# Verify only the bucket access, not the job_prefix directory
gcloud storage ls gs://{data_bucket}/ >/dev/null 2>&1 || {{ 
  echo "ERROR: Cannot access destination bucket gs://{data_bucket}/"
  exit 1
}}

cd /home/ays57/seds/codearena || {{ echo "ERROR: Directory not found"; exit 1; }}
# Ensure conda environment activation works in non-interactive shells
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate seds || {{ echo "ERROR: Failed to activate conda environment 'seds'"; exit 1; }}

# Log the instances this job will process
echo "$(date): Job {job_prefix}-{job_index} will process ${{#instances[@]}} instances: ${{instances[@]}}"

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
  {command}
  
  PYTHON_EXIT_CODE=${{PIPESTATUS[0]}}
  if [ $PYTHON_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Processing failed for instance $INSTANCE_ID with exit code $PYTHON_EXIT_CODE" | tee -a "$INSTANCE_LOGFILE"
  fi
  
  # Upload results
  echo "$(date): Uploading results for instance $INSTANCE_ID" | tee -a "$INSTANCE_LOGFILE"
  gcloud storage cp -r logs/ gs://{data_bucket}/{job_prefix}/$INSTANCE_ID 2>&1
  
  if [ $? -eq 0 ]; then
    echo "$(date): Completed instance $INSTANCE_ID" | tee -a "$INSTANCE_LOGFILE"
  else
    echo "ERROR: Upload failed for instance $INSTANCE_ID" | tee -a "$INSTANCE_LOGFILE"
  fi
done

echo "$(date): All processing completed"
EOSU

echo "$(date): Job completed, shutting down VM"
"""
        
        task = batch_v1.TaskSpec()
        task.runnables = [runnable]
        task.compute_resource = resources
        
        # Define task group - we only need 1 task per job
        # since the script will loop through all assigned instances
        group = batch_v1.TaskGroup()
        group.task_count = 1
        group.task_spec = task
        
        # Create the job with machine image specification
        job = batch_v1.Job()
        job.task_groups = [group]
        
        # Create the allocation policy
        allocation_policy = batch_v1.AllocationPolicy()
        
        # Create an instance policy
        instance_policy = batch_v1.AllocationPolicy.InstancePolicy()
        instance_policy.machine_type = machine_type
        
        # Set boot disk with the image if specified
        if disk_image:
            boot_disk = batch_v1.AllocationPolicy.Disk()
            boot_disk.image = disk_image
            instance_policy.boot_disk = boot_disk
        
        # Create an InstancePolicyOrTemplate and set the policy
        instance_policy_or_template = batch_v1.AllocationPolicy.InstancePolicyOrTemplate()
        instance_policy_or_template.policy = instance_policy
        
        # Add the instance policy to the allocation policy's instances list
        allocation_policy.instances = [instance_policy_or_template]
        
        # Set the allocation policy on the job
        job.allocation_policy = allocation_policy

        # Set a unique job name for each job
        job_name = f"{job_prefix}-{job_index}"
        parent = f"projects/{project_id}/locations/{region}"
        
        # Submit the job
        response = client.create_job(parent=parent, job=job, job_id=job_name)
        print(f"Created job: {job_name} to process instances {start_index} to {end_index}")
        jobs.append(response)

    return jobs


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Create distributed batch jobs for processing instances")
    parser.add_argument("instances_path", help="Path to file containing instance IDs")
    parser.add_argument("job_type", help="Type of job to run (e.g., 'sanity', 'bp-gen')")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the image from the base VM")
    
    args = parser.parse_args()
    
    # Parse arguments
    instances_path = Path(args.instances_path)
    job_type = args.job_type
    rebuild = args.rebuild
    
    # Hardcode the base VM name
    base_vm_name = "sedsbase-20250502-030654"
    project_id = "rv-project-457202"
    zone = "us-central1-a"
    
    # Define image family for this job type
    
    command = get_command(job_type)
 
    instances_list = instances_path.read_text().splitlines()
    # instances_list = ["scrapy__scrapy-6608", "statsmodels__statsmodels-9395", "ytdl-org__youtube-dl-32725", "camel-ai__camel-1478"]
    
    try:

        disk_image = create_image_wrapped(
            rebuild=rebuild,
            project_id=project_id,
            zone=zone,
            source_instance=base_vm_name,
        )
        
        # Create and submit the batch jobs
        jobs = create_distributed_batch_jobs(
            project_id=project_id,
            region="us-central1",
            job_prefix=f"sweb-{job_type}",
            instance_ids=instances_list,
            command=command,
            machine_type="e2-standard-4",
            disk_size_gb=100,
            num_jobs=20,
            disk_image=disk_image,
            data_bucket="seds-store"
        )
        
        print(f"Successfully submitted {len(jobs)} jobs to instances")
        
    except Exception as e:
        print(f"Error creating image or submitting jobs: {e}")
        sys.exit(1)

