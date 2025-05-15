from google.cloud import storage
import re
import asyncio
import concurrent.futures
from typing import List, Tuple
from pathlib import Path

async def analyse(gcs_path: str, all_instances: str, debug: bool = False, max_concurrency: int = 10):
    """
    Analyze reports from Google Cloud Storage using async for improved performance.
    
    Args:
        gcs_path: GCS path in the format "gs://bucket-name/directory"
        all_instances: path to text file containing all instance ids
        debug: Whether to enable debug output
        max_concurrency: Maximum number of concurrent operations
    """

    all_instance_ids = set([i.strip() for i in Path(all_instances).read_text().strip().splitlines()])

    # Parse bucket name and directory prefix from the GCS path
    match = re.match(r"gs://([^/]+)/(.+)", gcs_path)
    if not match:
        raise ValueError("Invalid GCS path. Expected format: gs://bucket-name/directory")
    
    bucket_name, directory_prefix = match.groups()
    if not directory_prefix.endswith('/'):
        directory_prefix += '/'
    
    # Initialize GCS client
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    if debug:
        print(f"Searching in bucket: {bucket_name}")
        print(f"Directory prefix: {directory_prefix}")
    
    # Create thread pool for concurrent GCS operations
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrency)
    
    # List all blobs concurrently
    all_blobs = await asyncio.get_event_loop().run_in_executor(
        executor, lambda: list(bucket.list_blobs(prefix=directory_prefix))
    )
    
    if debug:
        print(f"Found {len(all_blobs)} total objects with prefix {directory_prefix}")
        for blob in all_blobs[:10]:  # Show first 10 for debugging
            print(f"  - {blob.name}")
    
    # Find instance directories
    instance_dirs = await find_instance_directories(bucket, directory_prefix, all_blobs, executor, debug)
    
    print(f"Found {len(instance_dirs)} instance directories")
    if debug and instance_dirs:
        print("Instance IDs found:", instance_dirs)

    # Process all instances concurrently
    results = await process_instances(bucket, directory_prefix, instance_dirs, executor, debug)
    
    present, invalid = results
    failed = list(sorted((all_instance_ids - set(present)) | set(invalid)))
    
    # print(f"{len(present)=}, {len(invalid)=}, {len(failed)=}")
    print('\n'.join(failed))


async def find_instance_directories(bucket, directory_prefix, all_blobs, executor, debug) -> List[str]:
    """Find all instance directories using parallel approaches."""
    instance_dirs = set()
    
    # Approach 1: Extract from actual blob paths
    for blob in all_blobs:
        # Skip the prefix/directory itself
        if blob.name == directory_prefix:
            continue
        
        # Extract relative path from the directory prefix
        relative_path = blob.name[len(directory_prefix):]
        
        # If there's a subdirectory structure, get the first segment as instance ID
        if '/' in relative_path:
            instance_id = relative_path.split('/')[0]
            instance_dirs.add(instance_id)
    
    # Approach 2: Use delimiter to get "directories"
    result = await asyncio.get_event_loop().run_in_executor(
        executor, lambda: bucket.list_blobs(prefix=directory_prefix, delimiter='/')
    )
    
    # This part remains synchronous as it operates on local data
    for prefix in result.prefixes:
        instance_id = prefix.rstrip('/').split('/')[-1]
        instance_dirs.add(instance_id)
        if debug:
            print(f"Found prefix: {prefix} -> instance ID: {instance_id}")
    
    return list(instance_dirs)


async def process_instance(bucket, directory_prefix, instance_id, executor, debug: bool = False) -> Tuple[str, bool]:
    """Process a single instance and return results."""
    instance_prefix = f"{directory_prefix}{instance_id}/"
    
    if debug:
        print(f"Processing instance: {instance_id} with prefix: {instance_prefix}")
    
    # Find all report.json files recursively
    patch_blobs = await asyncio.get_event_loop().run_in_executor(
        executor, 
        lambda: [blob for blob in bucket.list_blobs(prefix=instance_prefix) if blob.name.endswith(f'{instance_id}/patch_1.diff')]
    )

    patch_blobs2 = await asyncio.get_event_loop().run_in_executor(
        executor, 
        lambda: [blob for blob in bucket.list_blobs(prefix=instance_prefix) if blob.name.endswith(f'logs/{instance_id}/patch_1.diff')]
    )
    
    patch_blobs += patch_blobs2
    
    if len(patch_blobs) > 0:
        return instance_id, False
    
    logs_blobs = await asyncio.get_event_loop().run_in_executor(
        executor, 
        lambda: [blob for blob in bucket.list_blobs(prefix=instance_prefix) if blob.name.endswith(f'.log')]
    )

    if len(logs_blobs) == 0:
        return instance_id, True
    
        
    logs_blob = logs_blobs[0]

    # Download and parse the JSON content
    logs_content = await asyncio.get_event_loop().run_in_executor(
        executor, lambda: logs_blob.download_as_text()
    )
    
    if "User location is not supported for the API use" in logs_content:
        return instance_id, True
    
    return instance_id, False


async def process_instances(bucket, directory_prefix, instance_dirs, executor, debug: bool = False) -> Tuple[List[str], List[str]]:
    """Process all instances concurrently and collect results."""
    present, invalid = [], []
    
    # Create tasks for all instances
    tasks = []
    for instance_id in instance_dirs:
        present.append(instance_id)
        tasks.append(process_instance(bucket, directory_prefix, instance_id, executor, debug))
    
    # Run all tasks concurrently and collect results
    results = await asyncio.gather(*tasks)
    
    # Process results
    for instance_id, is_invalid in results:
        if is_invalid:
            invalid.append(instance_id)

    return present, invalid


if __name__ == '__main__':
    import fire
    
    def run_analyse(gcs_path: str, all_instances: str, debug: bool = False, max_concurrency: int = 5):
        """Wrapper to run the async function."""
        asyncio.run(analyse(gcs_path, all_instances, debug, max_concurrency))
    
    fire.Fire(run_analyse)