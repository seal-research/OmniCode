from google.cloud import storage
import json
import re
import asyncio
import concurrent.futures
from typing import List, Tuple, Set, Dict, Optional


async def analyse(gcs_path: str, key: str = "resolved", debug: bool = False, max_concurrency: int = 10):
    """
    Analyze reports from Google Cloud Storage using async for improved performance.
    
    Args:
        gcs_path: GCS path in the format "gs://bucket-name/directory"
        debug: Whether to enable debug output
        max_concurrency: Maximum number of concurrent operations
    """
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
    
    if debug:
        print(f"Found {len(instance_dirs)} instance directories")
    if debug and instance_dirs:
        print("Instance IDs found:", instance_dirs)
    
    # If still no instances, check if the structure is flat (all files at top level)
    if not instance_dirs and all_blobs:
        top_level_reports = [blob for blob in all_blobs if blob.name.endswith('report.json')]
        
        if top_level_reports and debug:
            print("Found report.json files at top level - the structure may be flat")
            for report in top_level_reports:
                print(f"  - {report.name}")
    
    # Process all instances concurrently
    results = await process_instances(bucket, directory_prefix, instance_dirs, executor, key, debug)
    
    skipped, all_instances, present, passed, failed = results
    
    print(f"Skipped: {skipped}")
    print(f"Failed (first 10): {failed[:10]}")
    print(f"{len(all_instances)=}, {len(present)=}, {len(skipped)=}, {len(passed)=}, {len(failed)=}")
    
    # print('\n'.join(sorted(passed)))
    # print('\n'.join(list(sorted(set(all_instances) - set(passed).union(set(failed))))))


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


async def process_instance(bucket, directory_prefix, instance_id, executor, key: str = "resolved", debug: bool = False) -> Tuple[str, bool, bool, bool]:
    """Process a single instance and return results."""
    instance_prefix = f"{directory_prefix}{instance_id}/"
    
    if debug:
        print(f"Processing instance: {instance_id} with prefix: {instance_prefix}")
    
    # Find all report.json files recursively
    report_blobs = await asyncio.get_event_loop().run_in_executor(
        executor, 
        lambda: [blob for blob in bucket.list_blobs(prefix=instance_prefix) if blob.name.endswith('report.json')]
    )
    
    if len(report_blobs) == 0:
        if debug:
            print(f"Info: No report found for {instance_id}, skipping ...")
        return instance_id, False, True, False
    
    report_json_blob = report_blobs[0]
    
    if len(report_blobs) != 1 and debug:
        print(f"Warning: Multiple reports found for {instance_id}, using {report_json_blob.name}")
    
    # Download and parse the JSON content
    report_content = await asyncio.get_event_loop().run_in_executor(
        executor, lambda: report_json_blob.download_as_text()
    )
    
    try:
        report_data = json.loads(report_content)
        
        if instance_id not in report_data:
            if debug:
                print(f"Error: Could not find {instance_id} in {report_json_blob.name}, skipping ...")
            return instance_id, True, True, False
        
        if key not in report_data[instance_id]:
            if debug:
                print(f"Error: Could not find key '{key}' in {report_json_blob.name}, skipping ...")
            return instance_id, True, True, False
        
        passed = report_data[instance_id][key]
        return instance_id, True, False, passed
    
    except json.JSONDecodeError:
        if debug:
            print(f"Error: Invalid JSON in {report_json_blob.name}, skipping ...")
        return instance_id, True, True, False


async def process_instances(bucket, directory_prefix, instance_dirs, executor, key: str = "resolved", debug: bool = False) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    """Process all instances concurrently and collect results."""
    skipped, all_instances, present, passed, failed = [], [], [], [], []
    
    # Create tasks for all instances
    tasks = []
    for instance_id in instance_dirs:
        all_instances.append(instance_id)
        tasks.append(process_instance(bucket, directory_prefix, instance_id, executor, key, debug))
    
    # Run all tasks concurrently and collect results
    results = await asyncio.gather(*tasks)
    
    # Process results
    for instance_id, is_present, is_skipped, is_passed in results:
        if is_present:
            present.append(instance_id)
        
        if is_skipped:
            skipped.append(instance_id)
        elif is_passed:
            passed.append(instance_id)
        else:
            failed.append(instance_id)
    
    return skipped, all_instances, present, passed, failed


if __name__ == '__main__':
    import fire
    
    def run_analyse(gcs_path: str, key: str = "resolved", debug: bool = False, max_concurrency: int = 10):
        """Wrapper to run the async function."""
        asyncio.run(analyse(gcs_path, key, debug, max_concurrency))
    
    fire.Fire(run_analyse)