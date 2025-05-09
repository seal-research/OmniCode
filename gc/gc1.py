from google.cloud import batch_v1
import json


def list_to_bash_array(data):
    """
    Converts a Python list to a string representation of a Bash array.

    Args:
        data: A Python list.

    Returns:
        A string representing the list as a Bash array.
    """
    return " ".join(map(json.dumps, data))

def create_distributed_batch_jobs(
    project_id: str,
    region: str,
    job_prefix: str,
    instance_ids: list[str],
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
        end_index = start_index + instances_per_job - 1
        
        # Distribute any remainder instances to the first few jobs
        if job_index < remainder:
            end_index += 1
            start_index += job_index
            end_index += job_index
        else:
            start_index += remainder
            end_index += remainder
        
        # Calculate how many instances this job will process
        job_instance_count = end_index - start_index + 1
        
        # Define the VM resources
        resources = batch_v1.ComputeResource()
        resources.cpu_milli = 4000  # 4 vCPUs
        resources.memory_mib = 16384  # 16 GB RAM
        resources.boot_disk_mib = disk_size_gb * 1024
        
        # Define the job script with instance range awareness
        runnable = batch_v1.Runnable()
        runnable.script = batch_v1.Runnable.Script()
        runnable.script.text = f"""
        #!/bin/bash
        
        instances={list_to_bash_array(instance_ids)}

        cd /home/ays57/seds/codearena
        conda activate seds

        # Loop through the assigned instance range
        for INSTANCE_ID in $instances; do
            echo "Processing instance $INSTANCE_ID"
            
            # Download your data from bucket
            gsutil cp gs://{data_bucket}/instances/$INSTANCE_ID.data /tmp/input.data
            
            # Run your processing script
            python codearena.py --BugFixing --predictions_path gold --run_id test --instance_ids $INSTANCE_ID
            
            # Upload results back to bucket
            gsutil cp -r logs/ gs://{data_bucket}/{job_prefix}/$INSTANCE_ID
            
            echo "Completed instance $INSTANCE_ID"
        done
        """
        
        task = batch_v1.TaskSpec()
        task.runnables = [runnable]
        task.compute_resource = resources
        
        # Define task group - for this approach we only need 1 task per job
        # since the script will loop through all assigned instances
        group = batch_v1.TaskGroup()
        group.task_count = 1  # One task that processes multiple instances
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

    # import sys
    # from pathlib import Path

    # if len(sys.argv) < 3:
    #     print(f"Usage: python {sys.argv[1]} <script_path> <job_name>")
    #     sys.exit(1)


    # script_path = Path(sys.argv[2])
    # job_name = sys.argv[3]

    # if not script_path.exists():
    #     print(f"Could not find script at {script_path}")
    #     sys.exit(1)

    # script = script_path.read_text()

    jobs = create_distributed_batch_jobs(
        project_id="rv-project-457202",
        region="us-central1",
        job_prefix="sweb-sanity",
        instance_ids=["scrapy__scrapy-6608", "statsmodels__statsmodels-9395", "ytdl-org__youtube-dl-32725", "camel-ai__camel-1478"],
        machine_type="e2-standard-4",
        disk_size_gb=50,
        num_jobs=2,
        disk_image="projects/rv-project-457202/global/images/sedsbase-image",
        data_bucket="sedsstore"
    )
    
    print(f"Successfully submitted {len(jobs)} jobs to process 1000 instances")