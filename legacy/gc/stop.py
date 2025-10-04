import concurrent.futures
import re
import logging
import time
from datetime import datetime
from typing import List, Tuple, Optional
from google.cloud import compute_v1

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def get_matching_instances(prefix: str, project_id: str = None) -> List[Tuple[str, str]]:
    """Get all instances matching the prefix pattern using the Google Cloud API."""
    logger.info(f"Searching for instances with prefix '{prefix}'")
    start_time = time.time()
    
    # Get project ID if not provided
    if not project_id:
        logger.debug("Project ID not provided, fetching default project")
        # This will use the default project from gcloud config
        client = compute_v1.ProjectsClient()
        project_id = client.get_project(project="me").name.split('/')[-1]
        logger.info(f"Using project ID: {project_id}")
    
    # Create instance client
    instance_client = compute_v1.InstancesClient()
    
    # List all zones
    logger.debug("Fetching list of available zones")
    zone_client = compute_v1.ZonesClient()
    zones = [z.name for z in zone_client.list(project=project_id)]
    logger.debug(f"Found {len(zones)} zones to check")
    
    matching_instances = []
    pattern = re.compile(f"{re.escape(prefix)}[0-9]+")
    
    # Check each zone for matching instances
    for zone in zones:
        logger.debug(f"Checking zone {zone} for matching instances")
        try:
            instances = list(instance_client.list(project=project_id, zone=zone))
            zone_matches = [(instance.name, zone) for instance in instances if pattern.match(instance.name)]
            if zone_matches:
                logger.debug(f"Found {len(zone_matches)} matching instances in zone {zone}")
                matching_instances.extend(zone_matches)
        except Exception as e:
            logger.error(f"Error checking zone {zone}: {e}")
    
    elapsed_time = time.time() - start_time
    logger.info(f"Found total of {len(matching_instances)} matching instances in {elapsed_time:.2f} seconds")
    return matching_instances

def stop_instance(instance_data: Tuple[str, str], project_id: str = None) -> str:
    """Stop a single instance using the Google Cloud API."""
    name, zone = instance_data
    instance_log_prefix = f"[{name}/{zone}]"
    logger.info(f"{instance_log_prefix} Starting instance shutdown")
    start_time = time.time()
    
    # Get project ID if not provided
    if not project_id:
        logger.debug(f"{instance_log_prefix} Project ID not provided, fetching default")
        client = compute_v1.ProjectsClient()
        project_id = client.get_project(project="me").name.split('/')[-1]
        logger.debug(f"{instance_log_prefix} Using project ID: {project_id}")
    
    # Create instance client
    instance_client = compute_v1.InstancesClient()
    
    try:
        # Stop the instance
        logger.debug(f"{instance_log_prefix} Sending stop request")
        operation = instance_client.stop(
            project=project_id,
            zone=zone,
            instance=name
        )
        
        # Wait for the operation to complete
        wait_start = time.time()
        while not operation.status == compute_v1.Operation.Status.DONE:
            if time.time() - wait_start > 300:  # 5 minutes timeout
                logger.warning(f"{instance_log_prefix} Operation taking longer than 5 minutes")
            
            logger.debug(f"{instance_log_prefix} Waiting for stop operation to complete...")
            operation = instance_client.get_operation(
                project=project_id,
                zone=zone,
                operation=operation.name
            )
            time.sleep(2)  # Poll every 2 seconds
        
        if hasattr(operation, 'error') and operation.error:
            logger.error(f"{instance_log_prefix} Operation failed: {operation.error}")
            return f"❌ Failed to stop {name}: {operation.error}"
        
        elapsed_time = time.time() - start_time
        logger.info(f"{instance_log_prefix} Successfully stopped in {elapsed_time:.2f} seconds")
        return f"✓ Stopped {name} in {elapsed_time:.2f}s"
    
    except Exception as e:
        logger.error(f"{instance_log_prefix} Error stopping instance: {str(e)}", exc_info=True)
        return f"❌ Error stopping {name}: {str(e)}"

PROJECT_ID = "gen-lang-client-0511233871"

def stop_all(prefix: str, project_id: str = PROJECT_ID, max_workers: int = 10, verbose: bool = False):
    """Stop all GCP instances with the given prefix."""
    # Set logging level based on verbosity
    if verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    logger.info(f"Starting shutdown of instances with prefix '{prefix}'")
    logger.info(f"Using project ID: {project_id}, max_workers: {max_workers}")
    start_time = time.time()
    
    # Get instances matching the prefix
    instances = get_matching_instances(prefix, project_id)
    
    if not instances:
        logger.warning(f"No instances found matching prefix '{prefix}'")
        return
    
    logger.info(f"Found {len(instances)} instances to stop:")
    for i, (name, zone) in enumerate(instances, 1):
        logger.info(f"  {i}. {name} (zone: {zone})")
    
    # Stop instances in parallel
    success_count = 0
    failed_count = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        logger.info(f"Starting parallel shutdown with {max_workers} workers")
        futures = {
            executor.submit(stop_instance, instance, project_id): instance
            for instance in instances
        }
        
        for future in concurrent.futures.as_completed(futures):
            result = "Unknown result"
            try:
                result = future.result()
                if "✓" in result:
                    success_count += 1
                else:
                    failed_count += 1
                logger.info(result)
            except Exception as e:
                name, zone = futures[future]
                error_msg = f"Unexpected error stopping {name}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                failed_count += 1
    
    total_time = time.time() - start_time
    logger.info(f"Operation summary:")
    logger.info(f"  - Total instances processed: {len(instances)}")
    logger.info(f"  - Successfully stopped: {success_count}")
    logger.info(f"  - Failed to stop: {failed_count}")
    logger.info(f"  - Total execution time: {total_time:.2f} seconds")
    
    if failed_count > 0:
        logger.warning("Some instances failed to stop. Review the logs for details.")

if __name__ == "__main__":
    import fire
    
    # Add a function to enable verbose output through fire
    def main(prefix: str, project_id: str = PROJECT_ID, max_workers: int = 10, verbose: bool = False):
        """
        Stop all GCP instances with the given prefix.
        
        Args:
            prefix: Name prefix to match instances (e.g., 'test-instance-')
            project_id: Google Cloud project ID (default: use from script)
            max_workers: Maximum number of parallel stop operations (default: 10)
            verbose: Enable verbose (DEBUG) logging (default: False)
        """
        stop_all(prefix, project_id, max_workers, verbose)
    
    fire.Fire(main)