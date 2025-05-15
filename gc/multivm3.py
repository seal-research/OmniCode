import os
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Any, Optional
import logging
import threading

from google.cloud import compute_v1

# Set up GCP region options
DEFAULT_REGIONS = [
    "us-central1", "us-east1", "us-east4", "us-west1", "us-west2", "us-west3", "us-west4",
    "europe-central2", "europe-north1", "europe-west1", "europe-west2", "europe-west3", 
    "europe-west4", "europe-west6", "europe-west8", "europe-west9", "europe-southwest1",
    "asia-east1", "asia-east2", "asia-northeast1", "asia-northeast2", "asia-northeast3",
    "asia-south1", "asia-south2", "asia-southeast1", "asia-southeast2", 
    "australia-southeast1", "australia-southeast2"
]

VERTEXAI_LOCATION = "us-east5"

from utils import list_to_bash_array, check_vm_exists, get_vm_status, reset_vm, wait_for_operation, start_vm, get_command, delete_vm

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", None)
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", None)
DOCKER_PAT = os.environ.get("DOCKER_PAT", None)

# Retry configuration
MAX_RETRIES = 10
INITIAL_RETRY_DELAY = 10  # seconds
MAX_RETRY_DELAY = 60  # seconds

snapshot_semaphore = threading.Semaphore(1)

def get_all_available_regions(project_id: str) -> List[str]:
    """
    Get a list of all available compute regions in the GCP project.
    
    Returns:
        List of region names (e.g., ['us-central1', 'us-east1', ...])
    """
    regions_client = compute_v1.RegionsClient()
    
    try:
        # List all regions available in the project
        request = compute_v1.ListRegionsRequest(project=project_id)
        regions_list = regions_client.list(request=request)
        
        available_regions = []
        for region in regions_list:
            # Only include regions that are UP and available
            if region.status == "UP":
                available_regions.append(region.name)
        
        if not available_regions:
            logger.warning("No available regions found via API. Using default region list.")
            return DEFAULT_REGIONS
            
        logger.info(f"Found {len(available_regions)} available regions: {', '.join(available_regions)}")
        return available_regions
    
    except Exception as e:
        logger.error(f"Error getting available regions: {str(e)}")
        # Return a default list of common regions as fallback
        logger.warning(f"Using default list of regions due to error")
        return DEFAULT_REGIONS



def get_available_zones_for_region(project_id: str, region: str) -> Tuple[str, List[str]]:
    """
    Get a list of all available zones for a specific region.
    Returns a tuple of (region_name, list_of_zones).
    """
    zones_client = compute_v1.ZonesClient()
    
    try:
        # List all zones available in the project
        request = compute_v1.ListZonesRequest(project=project_id)
        zones_list = zones_client.list(request=request)
        
        available_zones = []
        for zone in zones_list:
            # Only include zones that are UP and available, and in the specified region
            if zone.status == "UP" and zone.name.startswith(f"{region}-"):
                available_zones.append(zone.name)
        
        if not available_zones:
            logger.warning(f"No available zones found in region {region}. Using default fallback.")
            # Provide a fallback - try common suffixes but skip 'a' which might be problematic
            available_zones = [f"{region}-b", f"{region}-c"]
            
        logger.info(f"Found {len(available_zones)} available zones in {region}: {', '.join(available_zones)}")
        return (region, available_zones)
    
    except Exception as e:
        logger.error(f"Error getting available zones for region {region}: {str(e)}")
        # Return a default fallback, skipping 'a' which might be problematic
        return (region, [f"{region}-b", f"{region}-c"])  # Try common zone suffixes as fallback


def get_all_regions_available_zones(project_id: str, regions: List[str], max_workers: int = 20) -> Dict[str, List[str]]:
    """
    Parallelize zone availability checks across all regions.
    
    Args:
        project_id: GCP project ID
        regions: List of regions to check for available zones
        max_workers: Maximum number of parallel threads
        
    Returns:
        Dict mapping regions to their available zones
    """
    logger.info(f"Checking zone availability for {len(regions)} regions in parallel...")
    start_time = time.time()
    
    region_to_zones = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all regions for parallel processing
        future_to_region = {
            executor.submit(get_available_zones_for_region, project_id, region): region
            for region in regions
        }
        
        # Process results as they complete
        for future in as_completed(future_to_region):
            region = future_to_region[future]
            try:
                result_region, zones = future.result()
                region_to_zones[result_region] = zones
            except Exception as e:
                logger.error(f"Exception getting zones for region {region}: {str(e)}")
                # Default fallback zones if something goes wrong
                region_to_zones[region] = [f"{region}-b", f"{region}-c"]
    
    duration = time.time() - start_time
    logger.info(f"Completed zone availability check for {len(regions)} regions in {duration:.2f} seconds")
    
    return region_to_zones
def wait_for_global_operation(project_id, operation_name):
    """Wait for a global operation to complete."""
    operation_client = compute_v1.GlobalOperationsClient()

    while True:
        result = operation_client.get(
            project=project_id,
            operation=operation_name
        )

        if result.status == compute_v1.Operation.Status.DONE:
            if result.error:
                logger.error(f"Operation {operation_name} failed: {result.error}")
                return False
            return True

        time.sleep(1)  # Wait before checking again


def check_snapshot_exists(project_id, snapshot_name):
    """Check if a snapshot with the given name exists."""
    snapshot_client = compute_v1.SnapshotsClient()
    try:
        snapshot_client.get(project=project_id, snapshot=snapshot_name)
        return True
    except Exception:
        return False


def delete_snapshot(project_id, snapshot_name):
    """Delete a snapshot."""
    snapshot_client = compute_v1.SnapshotsClient()
    try:
        operation = snapshot_client.delete(project=project_id, snapshot=snapshot_name)
        return wait_for_global_operation(project_id, operation.name)
    except Exception as e:
        logger.error(f"Failed to delete snapshot {snapshot_name}: {str(e)}")
        return False


def create_disk_snapshot(project_id, zone, instance_name, snapshot_name, force_update=False):
    """Create a snapshot of a running VM's boot disk. If force_update is True,
    delete the snapshot if it already exists."""

    # Check if snapshot already exists
    if check_snapshot_exists(project_id, snapshot_name):
        if force_update:
            logger.info(f"Snapshot {snapshot_name} already exists. Deleting it before creating a new one.")
            if not delete_snapshot(project_id, snapshot_name):
                logger.error(f"Failed to delete existing snapshot {snapshot_name}")
                return None
        else:
            logger.info(f"Using existing snapshot {snapshot_name}")
            return f"projects/{project_id}/global/snapshots/{snapshot_name}"

    # Get the disk name from the instance
    instance_client = compute_v1.InstancesClient()
    instance = instance_client.get(project=project_id, zone=zone, instance=instance_name)
    boot_disk_name = instance.disks[0].source.split('/')[-1]

    logger.info(f"Creating snapshot {snapshot_name} from disk {boot_disk_name}")

    # Create a snapshot
    disk_client = compute_v1.DisksClient()
    snapshot_request = compute_v1.Snapshot()
    snapshot_request.name = snapshot_name
    snapshot_request.description = f"Latest snapshot of {boot_disk_name} from {instance_name}"

    try:
        operation = disk_client.create_snapshot(
            project=project_id,
            zone=zone,
            disk=boot_disk_name,
            snapshot_resource=snapshot_request
        )

        wait_result = wait_for_operation(operation, project_id, zone)
        if wait_result:
            logger.info(f"Successfully created snapshot {snapshot_name}")
            # Return the full snapshot path
            return f"projects/{project_id}/global/snapshots/{snapshot_name}"
        else:
            logger.error(f"Failed to create snapshot {snapshot_name}")
            return None
    except Exception as e:
        logger.error(f"Exception creating snapshot: {str(e)}")
        return None


def exponential_backoff(retry_number):
    """Calculate delay with exponential backoff and jitter."""
    delay = min(MAX_RETRY_DELAY, INITIAL_RETRY_DELAY * (2 ** retry_number))
    # Add jitter to avoid thundering herd problem
    jitter = random.uniform(0, 0.3 * delay)
    return delay + jitter


def get_region_quota_usage(project_id: str, region: str) -> Dict[str, Tuple[float, float]]:
    """
    Query the current quota usage for CPU, IP addresses, and disk in a specific region.
    
    Returns:
        Dict mapping quota metrics to (usage, limit) tuples
    """
    service = compute_v1.RegionsClient()
    
    try:
        region_info = service.get(project=project_id, region=region)
        
        # Extract relevant quotas (CPUS, DISKS_TOTAL_GB, IN_USE_ADDRESSES)
        quotas = {}
        for quota in region_info.quotas:
            if quota.metric in ['CPUS', 'DISKS_TOTAL_GB', 'IN_USE_ADDRESSES']:
                quotas[quota.metric] = (quota.usage, quota.limit)
        
        logger.info(f"Quota for region {region}: {quotas}")
        return quotas
        
    except Exception as e:
        logger.error(f"Error retrieving quota for region {region}: {str(e)}")
        # Return default values with high usage (to avoid selecting this region)
        return {
            'CPUS': (1000, 1000),  # Fully used
            'DISKS_TOTAL_GB': (1000, 1000),
            'IN_USE_ADDRESSES': (1000, 1000)
        }


def get_region_capacity_for_vm_type(
    project_id: str,
    region: str,
    zone: str,
    machine_type: str,
    cpus_per_vm: int,
    disk_size_gb: int
) -> Tuple[str, int]:
    """
    Calculate VM capacity for a single region.
    Returns a tuple of (region_name, max_vms) for the given region.
    """
    try:
        # Get quota usage
        quotas = get_region_quota_usage(project_id, region)
        
        # Calculate how many VMs we can create based on CPU quota
        cpu_usage, cpu_limit = quotas.get('CPUS', (0, 0))
        available_cpus = max(0, cpu_limit - cpu_usage)
        
        if cpus_per_vm > 0:
            max_vms_by_cpu = int(available_cpus / cpus_per_vm)
        else:
            max_vms_by_cpu = 0
        
        # Calculate constraint based on disk quota
        disk_usage, disk_limit = quotas.get('DISKS_TOTAL_GB', (0, 0))
        available_disk_gb = max(0, disk_limit - disk_usage)
        max_vms_by_disk = int(available_disk_gb / disk_size_gb)
        
        # Calculate constraint based on IP address quota
        ip_usage, ip_limit = quotas.get('IN_USE_ADDRESSES', (0, 0))
        available_ips = max(0, ip_limit - ip_usage)
        max_vms_by_ip = int(available_ips)
        
        # The limiting factor is the minimum of the three constraints
        max_vms = min(max_vms_by_cpu, max_vms_by_disk, max_vms_by_ip)
        
        logger.info(f"Region {region} can support {max_vms} more VMs" +
                  f" (CPU: {max_vms_by_cpu}, Disk: {max_vms_by_disk}, IP: {max_vms_by_ip})")
        
        return (region, max_vms)
    
    except Exception as e:
        logger.error(f"Error calculating capacity for region {region}: {str(e)}")
        # Return minimal capacity for this region
        return (region, 0)


def calculate_vm_capacity_per_region(
    project_id: str, 
    regions: List[str],
    machine_type: str,
    disk_size_gb: int,
    max_workers: int = 20
) -> Dict[str, int]:
    """
    Calculate how many VMs of the specified type can be created in each region
    based on current quota usage. Uses parallelization for faster quota retrieval.
    
    Returns:
        Dict mapping regions to number of VMs that can be created
    """
    # Get machine type details to know CPU count - do this once upfront
    machine_type_client = compute_v1.MachineTypesClient()
    
    # Use one zone to look up machine type
    test_zone = f"{regions[0]}-a" if '-' in regions[0] else f"{regions[0]}-a"
    
    try:
        machine_info = machine_type_client.get(
            project=project_id,
            zone=test_zone,
            machine_type=machine_type
        )
        cpus_per_vm = machine_info.guest_cpus
        logger.info(f"Machine type {machine_type} has {cpus_per_vm} CPUs")
    except Exception as e:
        logger.error(f"Error getting machine type info: {str(e)}")
        cpus_per_vm = 4  # Default assumption if lookup fails
    
    # Prepare tasks for parallel execution
    tasks = []
    for region in regions:
        # Extract region and zone
        if '-' in region:
            region_name = region
            zone = f"{region}-a"  # Use first zone in the region
        else:
            region_name = region
            zone = f"{region}-a"
        
        tasks.append((
            project_id,
            region_name,
            zone,
            machine_type,
            cpus_per_vm,
            disk_size_gb
        ))
    
    # Execute quota queries in parallel
    logger.info(f"Checking quota for {len(regions)} regions in parallel...")
    start_time = time.time()
    
    region_capacity = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_region = {
            executor.submit(get_region_capacity_for_vm_type, *task): task[1]  # region_name is at index 1
            for task in tasks
        }
        
        # Process results as they complete
        for future in as_completed(future_to_region):
            region = future_to_region[future]
            try:
                region_name, max_vms = future.result()
                region_capacity[region_name] = max_vms
            except Exception as e:
                logger.error(f"Exception for region {region}: {str(e)}")
                region_capacity[region] = 0  # Default to no capacity on error
    
    duration = time.time() - start_time
    logger.info(f"Completed quota check for {len(regions)} regions in {duration:.2f} seconds")
    
    return region_capacity


def process_single_vm(
    project_id: str,
    zone: str,
    job_prefix: str,
    vm_name: str,
    vm_instance_ids: List[str],
    startup_script: str,
    machine_type: str,
    disk_size_gb: int,
    snapshot_source: str,
    data_bucket: str,
    overwrite: bool,
    max_name_attempts: int = 50  # Limit the number of name attempts to prevent infinite loops
) -> Tuple[str, bool, str]:
    """Process a single VM with retry logic and error reporting. If VM with candidate name is running,
    try alternative names instead of modifying the running VM."""
    instance_client = compute_v1.InstancesClient()

    for retry in range(MAX_RETRIES):
        try:
            # Check if VM already exists
            vm_exists = check_vm_exists(instance_client, project_id, zone, vm_name)

            if vm_exists:
                # Get the VM status
                status = get_vm_status(instance_client, project_id, zone, vm_name)
                
                # If VM is running, try a different name with a larger numeric suffix
                if status == "RUNNING":
                    logger.info(f"VM {vm_name} is already running. Trying alternative name with larger index...")
                    
                    # Parse the current name to extract the base part and index
                    parts = vm_name.rsplit('-', 1)
                    if len(parts) == 2 and parts[1].isdigit():
                        base_part = parts[0]
                        current_index = int(parts[1])
                    else:
                        # If the current name doesn't have a numeric suffix, add one
                        base_part = vm_name
                        current_index = 0
                    
                    # Try consecutive indices until we find an available one
                    for i in range(current_index + 1, current_index + max_name_attempts + 1):
                        alternative_name = f"{base_part}-{i}"
                        
                        # Check if the alternative name exists
                        alt_exists = check_vm_exists(instance_client, project_id, zone, alternative_name)
                        if not alt_exists:
                            # Found an available name, use it
                            logger.info(f"Using alternative name {alternative_name} instead of {vm_name}")
                            return process_single_vm(
                                project_id, zone, job_prefix, alternative_name, vm_instance_ids,
                                startup_script, machine_type, disk_size_gb, snapshot_source,
                                data_bucket, overwrite, max_name_attempts - 1  # Decrement to prevent infinite recursion
                            )
                        
                        # If it exists, check if it's running
                        alt_status = get_vm_status(instance_client, project_id, zone, alternative_name)
                        if alt_status != "RUNNING":
                            # Alternative exists but is not running, can use this one
                            logger.info(f"Using alternative name {alternative_name} (exists but not running) instead of {vm_name}")
                            return process_single_vm(
                                project_id, zone, job_prefix, alternative_name, vm_instance_ids,
                                startup_script, machine_type, disk_size_gb, snapshot_source,
                                data_bucket, overwrite, max_name_attempts - 1  # Decrement to prevent infinite recursion
                            )
                    
                    # If we get here, we couldn't find an available alternative name
                    logger.error(f"Could not find an available alternative name for {vm_name} after {max_name_attempts} attempts")
                    return vm_name, False, "NO_AVAILABLE_NAME"
                
                # For non-running VMs, continue with the existing logic
                if overwrite:
                    # Delete the VM and create a new one
                    logger.info(f"VM {vm_name} exists but is not running and overwrite is set to True.")
                    delete_result = delete_vm(instance_client, project_id, zone, vm_name)
                    if not delete_result:
                        logger.error(f"Failed to delete VM: {vm_name}. Retrying...")
                        raise Exception(f"Failed to delete VM: {vm_name}")
                    # VM was deleted, so we'll create a new one
                    vm_exists = False
                else:
                    # For non-running VMs, we can reset and reuse them
                    
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
                            return vm_name, True, "SUCCESS"
                        else:
                            logger.error(f"Failed to start VM: {vm_name}. Retrying...")
                            raise Exception(f"Failed to start VM: {vm_name}")
                    else:
                        logger.warning(f"VM {vm_name} exists but is in state {status}. No action taken.")
                        return vm_name, False, f"INVALID_STATE_{status}"

            if not vm_exists:
                with snapshot_semaphore:
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

                    # Use snapshot source - snapshots are global resources, so this works across regions
                    initialize_params.source_snapshot = snapshot_source

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
                    logger.info(f"Creating new VM {vm_name} in zone {zone} to process {len(vm_instance_ids)} instances...")
                    operation = instance_client.insert(
                        project=project_id,
                        zone=zone,
                        instance_resource=instance
                    )

                    wait_result = wait_for_operation(operation, project_id, zone)
                    if wait_result:
                        logger.info(f"Successfully created VM: {vm_name} in zone {zone}")
                        return vm_name, True, "SUCCESS"
                    else:
                        logger.error(f"Failed to create VM: {vm_name} in zone {zone}. Retrying...")
                        raise Exception(f"Failed to create VM: {vm_name}")

            # If we get here, something unexpected happened
            return vm_name, False, "UNKNOWN_ERROR"

        except Exception as e:
            error_message = str(e).lower()
            # Check if error is related to quota or capacity
            if "quota" in error_message or "capacity" in error_message or "resource" in error_message:
                logger.warning(f"Resource limitation detected in zone {zone}. Error: {str(e)}")
                if retry == MAX_RETRIES - 1:
                    return vm_name, False, "RESOURCE_LIMIT"
            
            if retry < MAX_RETRIES - 1:
                delay = exponential_backoff(retry)
                logger.warning(f"Attempt {retry+1} failed for VM {vm_name} in zone {zone}: {str(e)}. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"All {MAX_RETRIES} attempts failed for VM {vm_name} in zone {zone}: {str(e)}")
                return vm_name, False, "MAX_RETRIES_EXCEEDED"

    # This should not be reached, but just in case
    return vm_name, False, "UNKNOWN_ERROR"


def distribute_vms_by_quota(
    project_id: str,
    regions: List[str],
    num_vms: int,
    machine_type: str,
    disk_size_gb: int,
    min_vms_per_region: int = 1,
    max_vms_per_region: Optional[int] = None,
    max_workers: int = 20
) -> Dict[str, int]:
    """
    Distribute VMs across regions based on available quota.
    
    Args:
        project_id: GCP project ID
        regions: List of regions to distribute VMs across
        num_vms: Total number of VMs to distribute
        machine_type: Machine type for the VMs
        disk_size_gb: Disk size for each VM in GB
        min_vms_per_region: Minimum VMs to allocate to each region (if possible)
        max_vms_per_region: Maximum VMs to allocate to any single region
        max_workers: Maximum number of parallel threads for quota queries
    
    Returns:
        Dict mapping regions to number of VMs to create in each
    """
    # Get capacity for each region
    region_capacity = calculate_vm_capacity_per_region(
        project_id, regions, machine_type, disk_size_gb, max_workers
    )
    
    # Calculate total available capacity
    total_capacity = sum(region_capacity.values())
    
    if total_capacity <= 0:
        logger.warning("No available capacity in any region! Distributing evenly as fallback.")
        return {region: num_vms // len(regions) + (1 if i < num_vms % len(regions) else 0) 
                for i, region in enumerate(regions)}
    
    # Apply max VMs per region if specified
    if max_vms_per_region is not None:
        region_capacity = {
            region: min(capacity, max_vms_per_region) 
            for region, capacity in region_capacity.items()
        }
        # Recalculate total capacity with limits applied
        total_capacity = sum(region_capacity.values())
    
    # Distribute VMs proportionally to available capacity
    vm_distribution = {}
    vms_allocated = 0
    
    # First pass: guarantee minimum VMs per region where possible
    for region, capacity in region_capacity.items():
        if capacity >= min_vms_per_region:
            vm_distribution[region] = min_vms_per_region
            vms_allocated += min_vms_per_region
        else:
            # If region can't support minimum, allocate what it can
            vm_distribution[region] = capacity
            vms_allocated += capacity
    
    # Adjust for remaining capacity after minimums
    remaining_capacity = {
        region: region_capacity[region] - vm_distribution[region]
        for region in regions if region in region_capacity
    }
    adjusted_total = sum(remaining_capacity.values())
    
    # Second pass: allocate remaining VMs proportionally to remaining capacity
    remaining_vms = num_vms - vms_allocated
    
    if remaining_vms > 0 and adjusted_total > 0:
        for region, capacity in remaining_capacity.items():
            # Calculate proportional allocation of remaining VMs
            region_additional = min(
                capacity,  # Don't exceed remaining capacity
                int((capacity / adjusted_total) * remaining_vms)  # Proportional allocation
            )
            
            vm_distribution[region] += region_additional
            vms_allocated += region_additional
    
    # Third pass: allocate any remaining VMs to regions with spare capacity
    remaining_vms = num_vms - vms_allocated
    
    if remaining_vms > 0:
        # Sort regions by remaining capacity after first allocation
        regions_by_remaining_capacity = sorted(
            [(region, region_capacity[region] - vm_distribution[region]) 
             for region in regions if region in region_capacity],
            key=lambda x: x[1],  # Sort by remaining capacity
            reverse=True  # Highest capacity first
        )
        
        # Allocate remaining VMs
        for i in range(remaining_vms):
            if i < len(regions_by_remaining_capacity):
                region, remaining = regions_by_remaining_capacity[i]
                if remaining > 0:
                    vm_distribution[region] += 1
                else:
                    # If no regions have remaining capacity, distribute remaining VMs round-robin
                    fallback_region = regions[i % len(regions)]
                    vm_distribution[fallback_region] = vm_distribution.get(fallback_region, 0) + 1
            else:
                # Wrap around if we have more VMs than regions
                fallback_region = regions[i % len(regions)]
                vm_distribution[fallback_region] = vm_distribution.get(fallback_region, 0) + 1
    
    logger.info(f"VM distribution across regions: {vm_distribution}")
    return vm_distribution


def create_distributed_compute_vms(
    project_id: str,
    base_zone: str,  # Zone where the base VM exists
    worker_regions: List[str],  # Regions where worker VMs will be created
    job_prefix: str,
    instance_ids: List[str],
    command: str,
    base_vm_name: str,
    machine_type: str = "e2-standard-4",
    disk_size_gb: int = 100,
    num_vms: int = 20,
    snapshot_source: str | None = None,
    data_bucket: str = "your-data-bucket",
    overwrite: bool = False,
    vm_num_offset: int = 0,
    max_workers: int = 10,
    indices_to_run: List[int] | None = None,
    max_vms_per_region: Optional[int] = None,
    quota_check_workers: int = 20,
):
    """Create or reuse multiple Compute Engine VMs distributed across regions based on quota availability."""
    
    # Distribute VMs across regions based on quota
    vm_distribution = distribute_vms_by_quota(
        project_id=project_id,
        regions=worker_regions,
        num_vms=num_vms,
        machine_type=machine_type,
        disk_size_gb=disk_size_gb,
        max_vms_per_region=max_vms_per_region,
        max_workers=quota_check_workers
    )

    # Get available zones for all regions in parallel
    region_to_zones = get_all_regions_available_zones(
        project_id=project_id, 
        regions=worker_regions, 
        max_workers=quota_check_workers
    )
    
    # Calculate how many instances each VM should handle
    total_instances = len(instance_ids)
    instances_per_vm = total_instances // num_vms
    remainder = total_instances % num_vms

    # Prepare tasks for parallel execution
    tasks = []
    vm_name_to_zone = {}  # Track which VM is in which zone for error handling
    
    # Track current VM index and instance allocation
    current_vm_index = 0
    current_instance_index = 0
    
    # Distribute VMs across regions according to calculated distribution
    for region, region_vm_count in vm_distribution.items():
        # Skip if no VMs allocated to this region
        if region_vm_count <= 0:
            continue
            
        # Get available zones for this region
        available_zones = region_to_zones.get(region, [f"{region}-a"])
        
        # Distribute VMs across zones in this region (round-robin)
        for i in range(region_vm_count):
            if indices_to_run is not None and current_vm_index not in indices_to_run:
                current_vm_index += 1
                continue
                
            # Choose zone (round-robin within the region)
            zone = available_zones[i % len(available_zones)]
            
            # Calculate instance IDs for this VM
            start_index = current_instance_index
            extra = 1 if current_vm_index < remainder else 0
            instance_count = instances_per_vm + extra
            end_index = start_index + instance_count
            
            # Ensure we don't exceed the available instances
            end_index = min(end_index, total_instances)
            vm_instance_ids = instance_ids[start_index:end_index]
            
            # Create VM name
            vm_name = f"{base_vm_name}-{vm_num_offset + current_vm_index}"
            vm_name_to_zone[vm_name] = zone
            
            # Generate startup script (same as before)
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
export DOCKER_PAT="{DOCKER_PAT}"
export VERTEXAI_LOCATION="{VERTEXAI_LOCATION}"
export VERTEXAI_PROJECT="{project_id}"
echo "Environment variables set including DOCKER PAT"
echo $DOCKER_PAT
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
                snapshot_source,
                data_bucket,
                overwrite,
            ))
            
            # Update indices for next VM
            current_vm_index += 1
            current_instance_index = end_index
            
            # Break if we've allocated all instances
            if current_instance_index >= total_instances:
                break
                
        # Break outer loop if we've allocated all instances
        if current_instance_index >= total_instances:
            break

    # Track VMs by region/zone for reporting
    successful_vms = []
    failed_vms = []
    vm_results_by_region = {region: {"success": 0, "failure": 0} for region in worker_regions}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_vm = {
            executor.submit(process_single_vm, *task): task[3]  # vm_name is at index 3
            for task in tasks
        }

        # Process results as they complete
        for future in as_completed(future_to_vm):
            vm_name = future_to_vm[future]
            zone = vm_name_to_zone.get(vm_name, "unknown-zone")
            region = zone.rsplit('-', 1)[0]  # Extract region from zone
            
            try:
                result_name, success, error_code = future.result()
                if success:
                    successful_vms.append(result_name)
                    logger.info(f"VM {result_name} in {zone} successfully processed")
                    vm_results_by_region[region]["success"] += 1
                else:
                    failed_vms.append((result_name, zone, error_code))
                    logger.warning(f"VM {result_name} in {zone} processing failed: {error_code}")
                    vm_results_by_region[region]["failure"] += 1
            except Exception as e:
                failed_vms.append((vm_name, zone, str(e)))
                logger.error(f"Exception for VM {vm_name} in {zone}: {str(e)}")
                vm_results_by_region[region]["failure"] += 1

    # Log summary by region
    logger.info("VM Creation Summary by Region:")
    for region, results in vm_results_by_region.items():
        logger.info(f"  {region}: {results['success']} successful, {results['failure']} failed")
    
    # Log overall summary
    logger.info(f"Overall VM Creation Summary: {len(successful_vms)} successful, {len(failed_vms)} failed")
    if failed_vms:
        logger.warning(f"Failed VMs: {', '.join([f'{vm} ({zone}: {code})' for vm, zone, code in failed_vms])}")

    return successful_vms


if __name__ == "__main__":
    import sys
    from pathlib import Path
    import argparse
    import random

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Create distributed batch jobs for processing instances across multiple regions")
    parser.add_argument("instances_path", help="Path to file containing instance IDs")
    parser.add_argument("job_type", help="Type of job to run (e.g., 'sanity', 'bp-gen')")
    parser.add_argument("--update-snapshot", action="store_true", help="Create/update the snapshot from the base VM")
    parser.add_argument("--overwrite", action="store_true", help="If specified, delete existing VMs with the same name before creating new ones")
    parser.add_argument("--dummy", action="store_true", help="Run on 4 instances with 4 VMs")
    parser.add_argument("--vm_num_offset", type=int, required=False, default=0)
    parser.add_argument("--num_vms", type=int, default=None, required=False, help="Maximum number of VMs to spin up in total. If not specified, it is equal to number of instances specified")
    parser.add_argument("--randomise", action="store_true", help="randomise sequence of instances being processed")
    parser.add_argument("--max_parallel", type=int, default=5, help="Maximum number of VMs to create in parallel")
    parser.add_argument("--base_zone", type=str, required=True, help="Zone where the base VM is located")
    parser.add_argument("--regions", type=str, default="all",
                       help="Comma-separated list of regions where worker VMs will be created (e.g., 'us-central1,us-east1,us-west1') or 'all' to use all available regions")
    parser.add_argument("--max_vms_per_region", type=int, default=None, 
                       help="Maximum number of VMs to create in any single region (default: no limit)")
    parser.add_argument("--base_vm", type=str, required=True, help="Base VM name, e.g. sedsbase")
    parser.add_argument("--vm_idx_to_run", type=str, help="comma seperate list if ints indicating vms to run", default=None)
    parser.add_argument("--quota_check_workers", type=int, default=10, 
                       help="Maximum number of parallel threads for quota queries (default: 10)")

    args = parser.parse_args()

    # Parse arguments
    job_type = args.job_type
    update_snapshot = args.update_snapshot
    overwrite = args.overwrite

    # Hardcode the base VM name
    base_vm_name = args.base_vm
    project_id = "gen-lang-client-0511233871"
    base_zone = args.base_zone
    
    # Handle 'all' regions option
    if args.regions.lower() == 'all':
        worker_regions = get_all_available_regions(project_id)
        logger.info(f"Using all available regions: {', '.join(worker_regions)}")
    else:
        worker_regions = args.regions.split(',')
        logger.info(f"Using specified regions: {', '.join(worker_regions)}")

    # Define fixed snapshot name
    snapshot_name = f"{base_vm_name}-latest"

    # Define command for this job type
    command = get_command(job_type)

    if command is None:
        raise RuntimeError(f"No command found for job type: {job_type}")

    if args.instances_path == "dummy":
        instances_list = ["sympy__sympy-23950", "pydata__xarray-4356", "ytdl-org__youtube-dl-32725", "celery__celery-8486"]
    else:
        instances_list = Path(args.instances_path).read_text().splitlines()

    if args.randomise:
        random.shuffle(instances_list)

    num_vms = min(args.num_vms, len(instances_list)) if args.num_vms is not None else len(instances_list)
    indices_to_run = [int(s) for s in args.vm_idx_to_run.split(',')] if args.vm_idx_to_run is not None else None

    try:
        # Create or use existing snapshot from the base VM (in its original zone)
        snapshot_source = create_disk_snapshot(
            project_id=project_id,
            zone=base_zone,  # Use base_zone here
            instance_name=base_vm_name,
            snapshot_name=snapshot_name,
            force_update=update_snapshot
        )

        if not snapshot_source:
            raise RuntimeError("Failed to create or get snapshot")

        logger.info(f"Using snapshot {snapshot_source} to create VMs across regions: {worker_regions}")

        vms = create_distributed_compute_vms(
            project_id=project_id,
            base_zone=base_zone,
            worker_regions=worker_regions,  # Pass the list of regions
            job_prefix=f"sweb-{job_type}",
            instance_ids=instances_list,
            command=command,
            base_vm_name=base_vm_name,
            machine_type="e2-standard-4",
            disk_size_gb=100,
            num_vms=len(instances_list) if args.instances_path == "dummy" else num_vms,
            snapshot_source=snapshot_source,
            data_bucket="seds-store",
            overwrite=overwrite,
            vm_num_offset=args.vm_num_offset,
            max_workers=args.max_parallel,
            indices_to_run=indices_to_run,
            max_vms_per_region=args.max_vms_per_region,
            quota_check_workers=args.quota_check_workers
        )

        logger.info(f"Successfully managed {len(vms)} VMs to process instances")

    except Exception as e:
        logger.error(f"Error creating snapshot or submitting jobs: {e}")
        sys.exit(1)