from google.cloud import compute_v1
import json
import time
import base64
from google.api_core.exceptions import NotFound


def list_to_bash_array(data):
    """
    Converts a Python list to a string representation of a Bash array.

    Args:
        data: A Python list.

    Returns:
        A string representing the list as a Bash array.
    """
    return " ".join(map(json.dumps, data))


def wait_for_operation(operation, project_id, zone):
    """Wait for a compute engine zone operation to complete."""
    print(f"Waiting for operation {operation.name} to complete...")
    
    # Create a zone operations client
    operation_client = compute_v1.ZoneOperationsClient()
    
    while operation.status != compute_v1.Operation.Status.DONE:
        time.sleep(5)
        operation = operation_client.get(
            project=project_id, 
            zone=zone,
            operation=operation.name
        )
    
    if operation.error:
        print(f"Error during operation: {operation.error}")
        return False
    return True


def check_vm_exists(instance_client, project_id, zone, vm_name):
    """Check if a VM with the given name exists."""
    try:
        instance_client.get(project=project_id, zone=zone, instance=vm_name)
        return True
    except NotFound:
        return False


def get_vm_status(instance_client, project_id, zone, vm_name):
    """Get the status of a VM."""
    instance = instance_client.get(project=project_id, zone=zone, instance=vm_name)
    return instance.status


def start_vm(instance_client, project_id, zone, vm_name):
    """Start a stopped VM."""
    print(f"Starting existing VM {vm_name}...")
    operation = instance_client.start(project=project_id, zone=zone, instance=vm_name)
    return operation


def reset_vm(instance_client, project_id, zone, vm_name, startup_script, instance_ids):
    """Reset a VM by updating its metadata with new startup script and instance IDs."""
    print(f"Updating metadata for VM {vm_name}...")
    
    # Get the current instance
    instance = instance_client.get(project=project_id, zone=zone, instance=vm_name)
    
    # Create a new metadata object with the updated startup script
    metadata = compute_v1.Metadata()
    
    # Copy existing metadata items if they exist
    existing_items = []
    if instance.metadata and instance.metadata.items:
        for item in instance.metadata.items:
            if item.key != "startup-script":
                existing_items.append(item)
    
    # Add the new startup script item
    script_item = compute_v1.Items()
    script_item.key = "startup-script"
    script_item.value = startup_script
    existing_items.append(script_item)
    
    # Set all items
    metadata.items = existing_items
    
    # Set the fingerprint to ensure we're updating the latest version
    if instance.metadata:
        metadata.fingerprint = instance.metadata.fingerprint
    
    # Update the metadata
    operation = instance_client.set_metadata(
        project=project_id,
        zone=zone,
        instance=vm_name,
        metadata_resource=metadata
    )
    
    return operation


VM_NAME_PREFIX = "sweb-sanity"

def create_distributed_compute_vms(
    project_id: str,
    region: str,
    zone: str,
    job_prefix: str,
    instance_ids: list[str],
    machine_type: str = "e2-standard-4",
    disk_size_gb: int = 100,
    num_vms: int = 20,
    disk_image: str = None,
    data_bucket: str = "your-data-bucket",
    reuse_existing_vms: bool = True
):
    """Create or reuse multiple Compute Engine VMs to distribute the processing of all instances."""
    
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
        vm_name = f"{VM_NAME_PREFIX}-vm-{vm_index}"
        
        # Create the startup script with instance processing
        startup_script = f"""#!/bin/bash

# Set up instance IDs to process
instances=({list_to_bash_array(vm_instance_ids)})

cd /home/ays57/seds/codearena
conda activate seds

# Loop through the assigned instances
for INSTANCE_ID in "${{instances[@]}}"; do
    echo "Processing instance $INSTANCE_ID"
    
    # Download your data from bucket
    gsutil cp gs://{data_bucket}/instances/$INSTANCE_ID.data /tmp/input.data
    
    # Run your processing script
    python codearena.py --BugFixing --predictions_path gold --run_id test --instance_ids $INSTANCE_ID
    
    # Upload results back to bucket
    gsutil cp -r logs/ gs://{data_bucket}/{job_prefix}/$INSTANCE_ID
    
    echo "Completed instance $INSTANCE_ID"
done

# Optional: Shut down VM when done to avoid ongoing charges
poweroff
"""
        
        # Check if VM already exists
        vm_exists = check_vm_exists(instance_client, project_id, zone, vm_name)
        
        if vm_exists and reuse_existing_vms:
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
        else:
            # Create a new VM
            if vm_exists:
                print(f"VM {vm_name} already exists but reuse_existing_vms is set to False. Skipping.")
                continue
                
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

    instances_path = Path(sys.argv[1])
    instances_list = instances_path.read_text().splitlines()


    vms = create_distributed_compute_vms(
        project_id="rv-project-457202",
        region="us-central1",
        zone="us-central1-a",
        job_prefix="sweb-sanity",
        instance_ids=instances_list,
        machine_type="e2-standard-4",
        disk_size_gb=100,
        num_vms=20,
        disk_image="projects/rv-project-457202/global/images/sedsbase-image",
        data_bucket="sedsstore",
        reuse_existing_vms=True  # Set to False if you want to skip existing VMs
    )
    
    print(f"Successfully managed {len(vms)} VMs to process instances")