import os
import time
import datetime
import random
import json
from typing import List, Dict, Any, Optional
import logging

from google.cloud import compute_v1
from google.cloud import storage

from utils import get_command

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", None)
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", None)

# Retry configuration
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 5  # seconds
MAX_RETRY_DELAY = 60  # seconds


def exponential_backoff(retry_number):
    """Calculate delay with exponential backoff and jitter."""
    delay = min(MAX_RETRY_DELAY, INITIAL_RETRY_DELAY * (2 ** retry_number))
    # Add jitter to avoid thundering herd problem
    jitter = random.uniform(0, 0.3 * delay)
    return delay + jitter


def wait_for_operation(operation, project_id, zone=None):
    """Wait for a Google Cloud operation to complete."""
    if not operation:
        return False

    if zone:
        # Zone-specific operation
        client = compute_v1.ZoneOperationsClient()
        request = compute_v1.GetZoneOperationRequest(
            operation=operation.name,
            project=project_id,
            zone=zone
        )
        method = client.get
    else:
        # Global operation
        client = compute_v1.GlobalOperationsClient()
        request = compute_v1.GetGlobalOperationRequest(
            operation=operation.name,
            project=project_id
        )
        method = client.get

    while True:
        op = method(request)
        if op.status == compute_v1.Operation.Status.DONE:
            if op.error:
                logger.error(f"Operation failed: {op.error.message}")
                return False
            return True
        time.sleep(1)


def wait_for_global_operation(operation, project_id):
    """Wait for a global operation to complete."""
    client = compute_v1.GlobalOperationsClient()
    
    while True:
        operation_result = client.get(
            project=project_id,
            operation=operation.name
        )
        
        if operation_result.status == compute_v1.Operation.Status.DONE:
            if operation_result.error:
                logger.error(f"Operation failed: {operation_result.error.message}")
                return False
            return True
            
        time.sleep(1)


def create_snapshot_from_instance(
    project_id: str,
    source_instance: str,
    source_zone: str,
) -> str:
    """Create a snapshot from the boot disk of the source instance."""
    for retry in range(MAX_RETRIES):
        try:
            # First, get the instance details to find the boot disk
            instance_client = compute_v1.InstancesClient()
            instance = instance_client.get(
                project=project_id,
                zone=source_zone,
                instance=source_instance
            )
            
            # Find the boot disk
            boot_disk = None
            for disk in instance.disks:
                if disk.boot:
                    boot_disk = disk
                    break
            
            if not boot_disk:
                raise Exception(f"No boot disk found for instance {source_instance}")
            
            # Extract the disk name from the disk source URL
            disk_name = boot_disk.source.split('/')[-1]
            
            # Create a timestamp for the snapshot name
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            snapshot_name = f"{source_instance}-snap-{timestamp}"
            
            # Create the snapshot
            disks_client = compute_v1.DisksClient()
            snapshot_request = compute_v1.Snapshot()
            snapshot_request.name = snapshot_name
            
            logger.info(f"Creating snapshot {snapshot_name} from disk {disk_name}")
            operation = disks_client.create_snapshot(
                project=project_id,
                zone=source_zone,
                disk=disk_name,
                snapshot_resource=snapshot_request
            )
            
            # Wait for the snapshot to complete
            if wait_for_operation(operation, project_id, source_zone):
                logger.info(f"Successfully created snapshot: {snapshot_name}")
                snapshot_url = f"projects/{project_id}/global/snapshots/{snapshot_name}"
                return snapshot_url
            else:
                raise Exception(f"Failed to create snapshot: {snapshot_name}")
            
        except Exception as e:
            if retry < MAX_RETRIES - 1:
                delay = exponential_backoff(retry)
                logger.warning(f"Attempt {retry+1} failed to create snapshot: {str(e)}. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"All {MAX_RETRIES} attempts failed to create snapshot: {str(e)}")
                raise
    
    # This should never be reached due to the exception above
    raise Exception(f"Failed to create snapshot after {MAX_RETRIES} attempts")


def create_instance_template(
    project_id: str,
    template_name: str, 
    boot_snapshot_url: str,
    machine_type: str = "e2-standard-4",
    disk_size_gb: int = 100
) -> str:
    """
    Create an instance template using a snapshot as boot disk.
    Returns the template's self link if successful.
    """
    for retry in range(MAX_RETRIES):
        try:
            template_client = compute_v1.InstanceTemplatesClient()
            
            # Check if template already exists
            try:
                existing_template = template_client.get(
                    project=project_id,
                    instance_template=template_name
                )
                logger.info(f"Template {template_name} already exists, using existing template")
                return existing_template.self_link
            except Exception:
                # Template doesn't exist, continue with creation
                pass
            
            # Create a new template with explicit properties
            template = compute_v1.InstanceTemplate()
            template.name = template_name
            
            # Set properties
            properties = compute_v1.InstanceProperties()
            properties.machine_type = machine_type
            
            # Set up boot disk from snapshot
            disk = compute_v1.AttachedDisk()
            disk.auto_delete = True
            disk.boot = True
            
            initialize_params = compute_v1.AttachedDiskInitializeParams()
            initialize_params.disk_size_gb = disk_size_gb
            initialize_params.source_snapshot = boot_snapshot_url
            disk.initialize_params = initialize_params
            
            properties.disks = [disk]
            
            # Set up network
            network_interface = compute_v1.NetworkInterface()
            network_interface.network = "global/networks/default"
            access_config = compute_v1.AccessConfig()
            access_config.name = "External NAT"
            access_config.type_ = "ONE_TO_ONE_NAT"
            network_interface.access_configs = [access_config]
            properties.network_interfaces = [network_interface]
            
            # Set up service account
            service_account = compute_v1.ServiceAccount()
            service_account.email = "default"
            service_account.scopes = [
                "https://www.googleapis.com/auth/devstorage.read_write",
                "https://www.googleapis.com/auth/logging.write",
                "https://www.googleapis.com/auth/monitoring.write",
            ]
            properties.service_accounts = [service_account]
            
            # Set up as spot/preemptible instance
            properties.scheduling = compute_v1.Scheduling()
            properties.scheduling.provisioning_model = "SPOT"
            
            template.properties = properties
            
            # Create the template
            logger.info(f"Creating template {template_name} from snapshot")
            operation = template_client.insert(
                project=project_id,
                instance_template_resource=template
            )
            
            # Wait for operation to complete
            result = wait_for_global_operation(operation, project_id)
            
            if result:
                # Get the created template to return its self_link
                created_template = template_client.get(
                    project=project_id,
                    instance_template=template_name
                )
                logger.info(f"Successfully created template: {template_name}")
                return created_template.self_link
            else:
                raise Exception(f"Failed to create template: {template_name}")
                
        except Exception as e:
            if retry < MAX_RETRIES - 1:
                delay = exponential_backoff(retry)
                logger.warning(f"Attempt {retry+1} failed for template {template_name}: {str(e)}. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"All {MAX_RETRIES} attempts failed for template {template_name}: {str(e)}")
                raise
    
    # This should never be reached due to the exception above
    raise Exception(f"Failed to create template after {MAX_RETRIES} attempts")


def create_instance_group_manager(
    project_id: str,
    zone: str,
    mig_name: str,
    template_url: str,
    base_instance_name: str,
    target_size: int
) -> str:
    """Create a Managed Instance Group."""
    for retry in range(MAX_RETRIES):
        try:
            mig_client = compute_v1.InstanceGroupManagersClient()
            
            # Check if MIG already exists
            try:
                existing_mig = mig_client.get(
                    project=project_id,
                    zone=zone,
                    instance_group_manager=mig_name
                )
                logger.info(f"MIG {mig_name} already exists, resizing to {target_size} instances")
                
                # Resize the existing MIG
                resize_request = compute_v1.ResizeInstanceGroupManagerRequest(
                    project=project_id,
                    zone=zone,
                    instance_group_manager=mig_name,
                    size=target_size
                )
                operation = mig_client.resize(resize_request)
                
                if wait_for_operation(operation, project_id, zone):
                    logger.info(f"Successfully resized MIG {mig_name} to {target_size} instances")
                    return existing_mig.self_link
                else:
                    raise Exception(f"Failed to resize MIG {mig_name}")
                    
            except Exception as e:
                if "not found" not in str(e).lower():
                    raise
                # MIG doesn't exist, continue with creation
                pass
            
            # Create a new MIG
            mig = compute_v1.InstanceGroupManager()
            mig.name = mig_name
            mig.instance_template = template_url
            mig.base_instance_name = base_instance_name
            mig.target_size = target_size
            
            # Create the MIG
            logger.info(f"Creating MIG {mig_name} with {target_size} instances")
            operation = mig_client.insert(
                project=project_id,
                zone=zone,
                instance_group_manager_resource=mig
            )
            
            # Wait for operation to complete
            if wait_for_operation(operation, project_id, zone):
                # Get the created MIG to return its self_link
                created_mig = mig_client.get(
                    project=project_id,
                    zone=zone,
                    instance_group_manager=mig_name
                )
                logger.info(f"Successfully created MIG: {mig_name}")
                return created_mig.self_link
            else:
                raise Exception(f"Failed to create MIG: {mig_name}")
                
        except Exception as e:
            if retry < MAX_RETRIES - 1:
                delay = exponential_backoff(retry)
                logger.warning(f"Attempt {retry+1} failed for MIG {mig_name}: {str(e)}. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"All {MAX_RETRIES} attempts failed for MIG {mig_name}: {str(e)}")
                raise
    
    # This should never be reached due to the exception above
    raise Exception(f"Failed to create MIG after {MAX_RETRIES} attempts")


def distribute_tasks_to_instances(
    instance_ids: List[str],
    num_vms: int,
    indices_to_run: Optional[List[int]] = None
) -> Dict[str, List[str]]:
    """Distribute tasks to VM instances."""
    # Calculate how many instances each VM should handle
    total_instances = len(instance_ids)
    instances_per_vm = total_instances // num_vms
    remainder = total_instances % num_vms
    
    # Prepare task distribution
    task_distribution = {}
    
    for vm_index in range(num_vms):
        # Skip if not in indices_to_run
        if indices_to_run is not None and vm_index not in indices_to_run:
            continue
            
        # Calculate the start and end indices for this VM
        # Distribute any remainder instances to the first few VMs
        start_index = vm_index * instances_per_vm + min(vm_index, remainder)
        extra = 1 if vm_index < remainder else 0
        end_index = start_index + instances_per_vm + extra
        
        # Get the subset of instance IDs for this VM
        vm_instance_ids = instance_ids[start_index:end_index]
        
        # Add to task distribution
        task_distribution[str(vm_index)] = vm_instance_ids
    
    return task_distribution


def upload_task_distribution(
    project_id: str,
    bucket_name: str,
    job_prefix: str,
    task_distribution: Dict[str, List[str]]
) -> str:
    """Upload task distribution to GCS."""
    # Initialize storage client
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    
    # Create blob path
    blob_path = f"{job_prefix}/task_distribution.json"
    blob = bucket.blob(blob_path)
    
    # Upload task distribution as JSON
    blob.upload_from_string(json.dumps(task_distribution))
    
    return f"gs://{bucket_name}/{blob_path}"


def create_instance_group_metadata(
    project_id: str,
    zone: str,
    mig_name: str,
    job_prefix: str,
    command: str,
    data_bucket: str
) -> bool:
    """Set metadata including startup script for all instances in the MIG."""
    for retry in range(MAX_RETRIES):
        try:
            igm_client = compute_v1.InstanceGroupManagersClient()
            
            # Create the startup script
            startup_script = f"""#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status
# Set up logging
MAIN_LOGFILE="/home/ays57/seds/codearena/batch_processing.log"
exec > >(tee -a "$MAIN_LOGFILE") 2>&1
echo "$(date): Script started"
echo "Running as user: $(whoami)"
echo "Home directory: $HOME"
su - ays57 << 'EOSU'
export GEMINI_API_KEY="{GEMINI_API_KEY}"
export GITHUB_TOKEN="{GITHUB_TOKEN}"
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
export BUCKET_NAME="{data_bucket}"

# Get VM instance information
VM_NAME=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/name" -H "Metadata-Flavor: Google")
VM_INDEX=$(echo $VM_NAME | sed 's/.*-\\([0-9]\\+\\)$/\\1/')
echo "Running as VM $VM_NAME with index $VM_INDEX"

# Download the task distribution file
TASK_FILE="/tmp/task_distribution.json"
gsutil cp gs://${{BUCKET_NAME}}/{job_prefix}/task_distribution.json $TASK_FILE
if [ ! -f "$TASK_FILE" ]; then
  echo "ERROR: Could not find task distribution file"
  exit 1
fi

# Get the assigned instances for this VM
MY_INSTANCES=$(python3 -c "import json; f=open('$TASK_FILE'); data=json.load(f); print(' '.join(data.get('$VM_INDEX', [])))" 2>/dev/null)
if [ -z "$MY_INSTANCES" ]; then
  echo "No instances assigned to this VM. Exiting."
  exit 0
fi

# Convert to array
read -ra instances <<< "$MY_INSTANCES"

# Check if instances array is empty
if [ ${{#instances[@]}} -eq 0 ]; then
  echo "ERROR: No instance IDs provided."
  exit 1
fi

# Verify bucket access
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
            
            # Create a new template with the startup script
            # This creates a separate template that doesn't reference a source instance
            # but instead sets all properties manually
            new_template_name = f"{mig_name}-template"
            
            # Check if the new template already exists
            template_client = compute_v1.InstanceTemplatesClient()
            try:
                existing_template = template_client.get(
                    project=project_id,
                    instance_template=new_template_name
                )
                logger.info(f"Template {new_template_name} already exists, using existing template")
                
                # Update the MIG to use the existing template
                mig_client = compute_v1.InstanceGroupManagersClient()
                update_template_request = compute_v1.InstanceGroupManagersSetInstanceTemplateRequest(
                    instance_template=existing_template.self_link
                )
                operation = mig_client.set_instance_template(
                    project=project_id,
                    zone=zone,
                    instance_group_manager=mig_name,
                    instance_group_managers_set_instance_template_request_resource=update_template_request
                )
                
                if wait_for_operation(operation, project_id, zone):
                    logger.info(f"Successfully updated MIG {mig_name} to use template {new_template_name}")
                    return True
                    
                raise Exception(f"Failed to update MIG {mig_name} to use template {new_template_name}")
                
            except Exception as e:
                if "not found" not in str(e).lower():
                    raise
                # Template doesn't exist, continue with creation
                pass
            
            # Get original MIG configuration to copy settings
            mig = igm_client.get(
                project=project_id,
                zone=zone,
                instance_group_manager=mig_name
            )
            
            # Get the original template to copy its settings
            original_template_name = mig.instance_template.split('/')[-1]
            original_template = template_client.get(
                project=project_id,
                instance_template=original_template_name
            )
            
            # Create a new template with the same settings but add the startup script
            new_template = compute_v1.InstanceTemplate()
            new_template.name = new_template_name
            
            # Create new properties
            new_properties = compute_v1.InstanceProperties()
            
            # Copy machine type and other settings from original template if available
            # Otherwise, set reasonable defaults
            if hasattr(original_template, 'properties') and original_template.properties:
                # Copy machine type
                if hasattr(original_template.properties, 'machine_type') and original_template.properties.machine_type:
                    new_properties.machine_type = original_template.properties.machine_type
                else:
                    new_properties.machine_type = "e2-standard-4"
                
                # Copy disks
                if hasattr(original_template.properties, 'disks') and original_template.properties.disks:
                    new_properties.disks = original_template.properties.disks
                
                # Copy network interfaces
                if hasattr(original_template.properties, 'network_interfaces') and original_template.properties.network_interfaces:
                    new_properties.network_interfaces = original_template.properties.network_interfaces
                
                # Copy service accounts
                if hasattr(original_template.properties, 'service_accounts') and original_template.properties.service_accounts:
                    new_properties.service_accounts = original_template.properties.service_accounts
                
                # Copy tags
                if hasattr(original_template.properties, 'tags') and original_template.properties.tags:
                    new_properties.tags = original_template.properties.tags
            else:
                # Set up reasonable defaults if we couldn't get the original properties
                
                # Set machine type
                new_properties.machine_type = "e2-standard-4"
                
                # Set up disk
                disk = compute_v1.AttachedDisk()
                disk.auto_delete = True
                disk.boot = True
                initialize_params = compute_v1.AttachedDiskInitializeParams()
                initialize_params.disk_size_gb = 100
                initialize_params.source_image = "projects/debian-cloud/global/images/family/debian-11"
                disk.initialize_params = initialize_params
                new_properties.disks = [disk]
                
                # Set up network
                network_interface = compute_v1.NetworkInterface()
                network_interface.network = "global/networks/default"
                access_config = compute_v1.AccessConfig()
                access_config.name = "External NAT"
                access_config.type_ = "ONE_TO_ONE_NAT"
                network_interface.access_configs = [access_config]
                new_properties.network_interfaces = [network_interface]
                
                # Set up service account
                service_account = compute_v1.ServiceAccount()
                service_account.email = "default"
                service_account.scopes = [
                    "https://www.googleapis.com/auth/devstorage.read_write",
                    "https://www.googleapis.com/auth/logging.write",
                    "https://www.googleapis.com/auth/monitoring.write",
                ]
                new_properties.service_accounts = [service_account]
            
            # Always set up as spot/preemptible instance
            new_properties.scheduling = compute_v1.Scheduling()
            new_properties.scheduling.provisioning_model = "SPOT"
            
            # Add startup script to metadata
            metadata = compute_v1.Metadata()
            startup_item = compute_v1.Items()
            startup_item.key = "startup-script"
            startup_item.value = startup_script
            metadata.items = [startup_item]
            new_properties.metadata = metadata
            
            # Assign the new properties to the template
            new_template.properties = new_properties
            
            # Create the new template
            logger.info(f"Creating new template {new_template_name} with startup script")
            operation = template_client.insert(
                project=project_id,
                instance_template_resource=new_template
            )
            
            # Wait for template creation
            if not wait_for_global_operation(operation, project_id):
                raise Exception(f"Failed to create template {new_template_name}")
                
            # Get the created template
            created_template = template_client.get(
                project=project_id,
                instance_template=new_template_name
            )
            
            # Update the MIG to use the new template
            mig_client = compute_v1.InstanceGroupManagersClient()
            update_template_request = compute_v1.InstanceGroupManagersSetInstanceTemplateRequest(
                instance_template=created_template.self_link
            )
            operation = mig_client.set_instance_template(
                project=project_id,
                zone=zone,
                instance_group_manager=mig_name,
                instance_group_managers_set_instance_template_request_resource=update_template_request
            )
            
            if wait_for_operation(operation, project_id, zone):
                logger.info(f"Successfully updated MIG {mig_name} to use template {new_template_name}")
                return True
                
            raise Exception(f"Failed to update MIG {mig_name} to use template {new_template_name}")
            
        except Exception as e:
            if retry < MAX_RETRIES - 1:
                delay = exponential_backoff(retry)
                logger.warning(f"Attempt {retry+1} failed to set metadata for MIG {mig_name}: {str(e)}. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"All {MAX_RETRIES} attempts failed to set metadata for MIG {mig_name}: {str(e)}")
                raise
    
    # This should never be reached due to the exception above
    raise Exception(f"Failed to set metadata after {MAX_RETRIES} attempts")


def manage_mig_instances(
    project_id: str,
    zone: str,
    mig_name: str,
    wait_for_stability: bool = True,
    timeout_seconds: int = 300
) -> bool:
    """Wait for all instances in the MIG to be created and become stable."""
    igm_client = compute_v1.InstanceGroupManagersClient()
    
    if not wait_for_stability:
        return True
        
    start_time = time.time()
    
    while True:
        # Check if we've exceeded the timeout
        if time.time() - start_time > timeout_seconds:
            logger.warning(f"Timeout reached waiting for MIG {mig_name} to stabilize")
            return False
            
        # Get the current state of the MIG
        mig = igm_client.get(
            project=project_id,
            zone=zone,
            instance_group_manager=mig_name
        )
        
        # Check if all instances are created and stable
        if mig.current_actions.none == mig.target_size:
            logger.info(f"All {mig.target_size} instances in MIG {mig_name} are stable")
            return True
            
        # Log the current state
        logger.info(f"MIG {mig_name} status: {mig.target_size} instances requested, "
                   f"{mig.current_actions.creating} creating, "
                   f"{mig.current_actions.deleting} deleting, "
                   f"{mig.current_actions.recreating} recreating, "
                   f"{mig.current_actions.refreshing} refreshing, "
                   f"{mig.current_actions.restarting} restarting")
                   
        # Wait before checking again
        time.sleep(10)


def create_distributed_compute_mig(
    project_id: str,
    zone: str,
    job_prefix: str,
    instance_ids: List[str],
    command: str,
    base_vm_name: str,
    source_zone: str = None,
    num_vms: int = 20,
    data_bucket: str = "your-data-bucket",
    indices_to_run: Optional[List[int]] = None,
):
    """Create a Managed Instance Group to distribute compute tasks."""
    try:
        # Use source_zone if provided, otherwise default to target zone
        actual_source_zone = source_zone if source_zone else zone
        
        # Step 1: Create a snapshot from the source VM
        snapshot_url = create_snapshot_from_instance(
            project_id=project_id,
            source_instance=base_vm_name,
            source_zone=actual_source_zone
        )
        logger.info(f"Created snapshot from instance: {base_vm_name}")
        
        # Step 2: Create an instance template using the snapshot
        template_name = f"{base_vm_name}-template-{job_prefix.replace('/', '-')}"
        template_url = create_instance_template(
            project_id=project_id,
            template_name=template_name,
            boot_snapshot_url=snapshot_url,
            machine_type="e2-standard-4",
            disk_size_gb=100
        )
        logger.info(f"Created instance template: {template_name}")
        
        # Step 3: Create task distribution
        # Calculate how many VMs to actually create
        effective_num_vms = len(indices_to_run) if indices_to_run is not None else num_vms
        task_distribution = distribute_tasks_to_instances(
            instance_ids=instance_ids,
            num_vms=num_vms,
            indices_to_run=indices_to_run
        )
        logger.info(f"Distributed {len(instance_ids)} tasks across {effective_num_vms} VMs")
        
        # Step 4: Upload task distribution to GCS
        task_file_path = upload_task_distribution(
            project_id=project_id,
            bucket_name=data_bucket,
            job_prefix=job_prefix,
            task_distribution=task_distribution
        )
        logger.info(f"Uploaded task distribution to {task_file_path}")
        
        # Step 5: Create or update the MIG
        mig_name = f"{base_vm_name}-mig-{job_prefix.replace('/', '-')}"
        mig_url = create_instance_group_manager(
            project_id=project_id,
            zone=zone,
            mig_name=mig_name,
            template_url=template_url,
            base_instance_name=f"{base_vm_name}",
            target_size=effective_num_vms
        )
        logger.info(f"Created MIG: {mig_name}")
        
        # Step 6: Set metadata (including startup script) for the MIG
        metadata_result = create_instance_group_metadata(
            project_id=project_id,
            zone=zone,
            mig_name=mig_name,
            job_prefix=job_prefix,
            command=command,
            data_bucket=data_bucket
        )
        logger.info(f"Set metadata for MIG: {mig_name}")
        
        # Step 7: Wait for instances to be created and become stable
        stability_result = manage_mig_instances(
            project_id=project_id,
            zone=zone,
            mig_name=mig_name
        )
        
        if stability_result:
            logger.info(f"All instances in MIG {mig_name} are up and running")
        else:
            logger.warning(f"Some instances in MIG {mig_name} may not be fully ready")
        
        return mig_name
        
    except Exception as e:
        logger.error(f"Error creating MIG for distributed compute: {str(e)}")
        raise


if __name__ == "__main__":
    import sys
    from pathlib import Path
    import argparse
    import random
    import os

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Create distributed batch jobs for processing instances using MIGs")
    parser.add_argument("instances_path", help="Path to file containing instance IDs")
    parser.add_argument("job_type", help="Type of job to run (e.g., 'sanity', 'bp-gen')")
    parser.add_argument("--dummy", action="store_true", help="Run on 4 instances with 4 VMs")
    parser.add_argument("--vm_num_offset", type=int, required=False, default=0)
    parser.add_argument("--num_vms", type=int, default=20, required=False)
    parser.add_argument("--randomise", action="store_true", help="randomise sequence of instances being processed")
    parser.add_argument("--max_parallel", type=int, default=10, help="Maximum number of VMs to create in parallel")
    parser.add_argument("--zone", type=str, help="Zone in which to create job VMs (defaults to auto-detection)")
    parser.add_argument("--source_zone", type=str, help="Zone where the base VM is located (defaults to --zone or auto-detection)")
    parser.add_argument("--base_vm", type=str, required=True, help="Base VM name, e.g. sedsbase")
    parser.add_argument("--vm_idx_to_run", type=str, help="comma seperate list if ints indicating vms to run", default=None)

    args = parser.parse_args()

    job_type = args.job_type
    zone = args.zone
    source_zone = args.source_zone
    base_vm_name = args.base_vm
    project_id = "gen-lang-client-0511233871"
    
    logger.info(f"Using target zone: {zone}")
    logger.info(f"Using source zone: {source_zone}")
    logger.info(f"Project ID: {project_id}")
    logger.info(f"Base VM: {base_vm_name}")

    command = get_command(job_type)

    if command is None:
        raise RuntimeError(f"No command found for job type: {job_type}")

    if args.instances_path == "dummy":
        instances_list = ["sympy__sympy-23950", "pydata__xarray-4356", "ytdl-org__youtube-dl-32725", "celery__celery-8486"]
    else:
        instances_list = Path(args.instances_path).read_text().splitlines()

    if args.randomise:
        random.shuffle(instances_list)

    indices_to_run = [int(s) for s in args.vm_idx_to_run.split(',')] if args.vm_idx_to_run is not None else None

    # Create MIG with distributed tasks
    mig_name = create_distributed_compute_mig(
        project_id=project_id,
        zone=zone,
        job_prefix=f"sweb-{job_type}",
        instance_ids=instances_list,
        command=command,
        base_vm_name=base_vm_name,
        source_zone=source_zone,
        num_vms=len(instances_list) if args.instances_path == "dummy" else args.num_vms,
        data_bucket="seds-store",
        indices_to_run=indices_to_run,
    )

    logger.info(f"Successfully created/updated MIG {mig_name} to process instances")
    logger.info(f"Tasks are distributed across instances in the MIG")
    logger.info(f"Output will be stored in gs://seds-store/sweb-{job_type}/<instance_id>")