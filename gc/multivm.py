from google.cloud import compute_v1

from utils import list_to_bash_array, check_vm_exists, get_vm_status, reset_vm, wait_for_operation, start_vm, get_command, create_image_wrapped, delete_vm

def create_distributed_compute_vms(
    project_id: str,
    zone: str,
    job_prefix: str,
    instance_ids: list[str],
    command: str,
    machine_type: str = "e2-standard-4",
    disk_size_gb: int = 100,
    num_vms: int = 20,
    disk_image: str = None,
    data_bucket: str = "your-data-bucket",
    overwrite: bool = False
):
    """Create or reuse multiple Compute Engine VMs to distribute the processing of all instances."""
    
    # The function implementation remains the same until the VM existence check
    
    # Calculate how many instances each VM should handle
    total_instances = len(instance_ids)
    instances_per_vm = total_instances // num_vms
    remainder = total_instances % num_vms
    
    instance_client = compute_v1.InstancesClient()
    created_or_reused_vms = []
    
    for vm_index in range(num_vms):
        # Calculate the start and end indices for this VM
        start_index = vm_index * instances_per_vm
        end_index = start_index + instances_per_vm
        
        # Distribute any remainder instances to the first few VMs
        if vm_index < remainder:
            end_index += 1
        
        # Get the subset of instance IDs for this VM
        vm_instance_ids = instance_ids[start_index:end_index]
        
        # Create a unique name for this VM
        vm_name = f"seds-vm-{vm_index}"

        startup_script = f"""#!/bin/bash
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
# command -v conda >/dev/null 2>&1 || {{ echo "ERROR: conda is not installed"; exit 1; }}
conda --version
# Set trap for cleanup
cleanup() {{
  echo "$(date): Script interrupted, cleaning up..."
  # Any cleanup actions here
}}
trap cleanup EXIT INT TERM
instances=({list_to_bash_array(vm_instance_ids)})
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
  gcloud storage cp -r logs/ gs://{data_bucket}/{job_prefix}/$INSTANCE_ID 2>&1
  
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
        
        # Check if VM already exists
        vm_exists = check_vm_exists(instance_client, project_id, zone, vm_name)
        
        if vm_exists:

            if overwrite:
                # Delete the VM and create a new one
                print(f"VM {vm_name} exists and overwrite is set to True.")
                delete_result = delete_vm(instance_client, project_id, zone, vm_name)
                if not delete_result:
                    print(f"Failed to delete VM: {vm_name}. Skipping.")
                    continue
                # VM was deleted, so we'll create a new one
                vm_exists = False

            else:
                # Get the VM status
                status = get_vm_status(instance_client, project_id, zone, vm_name)
                
                # Reset the VM with the new startup script
                reset_operation = reset_vm(instance_client, project_id, zone, vm_name, startup_script, vm_instance_ids)
                wait_result = wait_for_operation(reset_operation, project_id, zone)
                
                if not wait_result:
                    print(f"Failed to update metadata for VM: {vm_name}")
                    continue
                
                if status == "TERMINATED":
                    # Start the VM if it's stopped
                    start_operation = start_vm(instance_client, project_id, zone, vm_name)
                    wait_result = wait_for_operation(start_operation, project_id, zone)
                    
                    if wait_result:
                        print(f"Successfully started existing VM: {vm_name}")
                        created_or_reused_vms.append(vm_name)
                    else:
                        print(f"Failed to start VM: {vm_name}")
                elif status == "RUNNING":
                    print(f"VM {vm_name} is already running. Metadata updated with new instance IDs.")
                    created_or_reused_vms.append(vm_name)
                else:
                    print(f"VM {vm_name} exists but is in state {status}. No action taken.")
        
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
            initialize_params.source_image = disk_image  # Use your custom image
            disk.initialize_params = initialize_params
            instance.disks = [disk]
            
            # Define the network configuration
            network_interface = compute_v1.NetworkInterface()
            network_interface.network = "global/networks/default"
            
            # Add an access config to enable external IP
            access_config = compute_v1.AccessConfig()
            access_config.name = "External NAT"
            access_config.type_ = "ONE_TO_ONE_NAT"
            network_interface.access_configs = [access_config]
            
            instance.network_interfaces = [network_interface]
            
            # Set the service account and scopes
            service_account = compute_v1.ServiceAccount()
            service_account.email = "default"  # Use the default service account
            service_account.scopes = [
                "https://www.googleapis.com/auth/devstorage.read_write",  # Access to GCS
                "https://www.googleapis.com/auth/logging.write",
                "https://www.googleapis.com/auth/monitoring.write",
            ]
            instance.service_accounts = [service_account]
            
            # Create metadata with startup script
            metadata = compute_v1.Metadata()
            item = compute_v1.Items()
            item.key = "startup-script"
            item.value = startup_script
            metadata.items = [item]
            instance.metadata = metadata
            
            # Create the VM
            print(f"Creating new VM {vm_name} to process {len(vm_instance_ids)} instances...")
            operation = instance_client.insert(
                project=project_id,
                zone=zone,
                instance_resource=instance
            )
            
            # Pass the zone to the wait_for_operation function
            wait_result = wait_for_operation(operation, project_id, zone)
            if wait_result:
                print(f"Successfully created VM: {vm_name}")
                created_or_reused_vms.append(vm_name)
            else:
                print(f"Failed to create VM: {vm_name}")
    
    return created_or_reused_vms

if __name__ == "__main__":
    import sys
    from pathlib import Path
    import argparse
    
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Create distributed batch jobs for processing instances")
    parser.add_argument("instances_path", help="Path to file containing instance IDs")
    parser.add_argument("job_type", help="Type of job to run (e.g., 'sanity', 'bp-gen')")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the image from the base VM")
    parser.add_argument("--overwrite", action="store_true", help="If specified, delete existing VMs with the same name before creating new ones")
    
    args = parser.parse_args()
    
    # Parse arguments
    instances_path = Path(args.instances_path)
    job_type = args.job_type
    rebuild = args.rebuild
    overwrite = rebuild or args.overwrite
    
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

        vms = create_distributed_compute_vms(
            project_id=project_id,
            zone=zone,
            job_prefix=f"sweb-{job_type}",
            instance_ids=instances_list,
            command=command,
            machine_type="e2-standard-4",
            disk_size_gb=100,
            num_vms=20,
            disk_image=disk_image,
            data_bucket="sedsstore",
            overwrite=overwrite,
        )
        
        print(f"Successfully managed {len(vms)} VMs to process instances")
        
    except Exception as e:
        print(f"Error creating image or submitting jobs: {e}")
        sys.exit(1)

