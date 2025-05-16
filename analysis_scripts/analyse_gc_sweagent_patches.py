from google.cloud import storage
import json
import re
import asyncio
import concurrent.futures
from typing import List, Tuple, Set, Dict, Optional
from datetime import datetime


async def analyse(gcs_path: str, debug: bool = True, max_concurrency: int = 10):
    """
    Analyze all_preds.jsonl files from Google Cloud Storage using async for improved performance.
    
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
    
    # Find instance directories concurrently
    instance_dirs = await find_instance_directories(bucket, directory_prefix, all_blobs, executor, debug)
    
    if debug:
        print(f"Found {len(instance_dirs)} instance directories")
        if instance_dirs:
            print("Instance IDs found:", instance_dirs)
    
    # Process all instances concurrently
    present, generated, not_null = await process_all_instances(
        bucket, directory_prefix, instance_dirs, executor, debug
    )
    
    # print(f"Not Null: {list(sorted(not_null))}")
    # print(f"Null: {list(sorted(set(present) - set(not_null)))}")
    # print(f"{len(present)=}, {len(generated)=}, {len(not_null)=}")
    print('\n'.join(sorted(not_null)))


async def find_instance_directories(bucket, directory_prefix, all_blobs, executor, debug) -> List[str]:
    """Find all instance directories using parallel approaches."""
    instance_dirs = set()
    
    # Extract from actual blob paths (this is fast enough to do synchronously)
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
    
    # Also use delimiter to get "directories" (this needs to be async)
    result = await asyncio.get_event_loop().run_in_executor(
        executor, lambda: bucket.list_blobs(prefix=directory_prefix, delimiter='/')
    )
    
    for prefix in result.prefixes:
        instance_id = prefix.rstrip('/').split('/')[-1]
        instance_dirs.add(instance_id)
        if debug:
            print(f"Found prefix: {prefix} -> instance ID: {instance_id}")
    
    return list(instance_dirs)


async def process_instance(bucket, directory_prefix, instance_id, executor, debug) -> Tuple[bool, bool]:
    """Process a single instance and return (has_generation, has_not_null) tuple."""
    if debug:
        print(f"Processing instance: {instance_id}")
    
    # Check for all_preds.jsonl file
    output_file_path = f"{directory_prefix}{instance_id}/all_preds.jsonl"
    output_file_blob = bucket.blob(output_file_path)
    
    # Check if file exists asynchronously
    exists = await asyncio.get_event_loop().run_in_executor(
        executor, lambda: output_file_blob.exists()
    )
    
    if not exists:
        if debug:
            print(f"No all_preds.jsonl for {instance_id}")
        return False, False

    log_file_path = f"{directory_prefix}{instance_id}/instance_{instance_id}.log" 
    log_file_blob = bucket.blob(log_file_path)
    error_text = "Exit due to unknown error: litellm.RateLimitError: litellm.RateLimitError"
    exists = await asyncio.get_event_loop().run_in_executor(
        executor, lambda: log_file_blob.exists()
    )
    if exists:
        log_content = await asyncio.get_event_loop().run_in_executor(
            executor, lambda: log_file_blob.download_as_text()
        )
        lines = log_content.splitlines()
        start_line, end_line = lines[0], lines[-1]
        # if "Running python script" in start_line and "Uploading results" in end_line:
            # try:
            #     start_time = datetime.strptime(start_line.rsplit(':', 1)[0], "%a %b %d %H:%M:%S %Z %Y")
            #     end_time = datetime.strptime(end_line.rsplit(':', 1)[0], "%a %b %d %H:%M:%S %Z %Y")
            #     print((end_time - start_time).seconds)
            # except Exception as _:
            #     pass

        # if error_text in log_content:
        #     print(instance_id)

    # File exists, download and process it asynchronously
    output_content = await asyncio.get_event_loop().run_in_executor(
        executor, lambda: output_file_blob.download_as_text()
    )
    
    # Get the first line and parse it as JSON
    try:
        first_line = output_content.splitlines()[0]
        output_data = json.loads(first_line)
        
        has_full_output = output_data.get("model_patch", None) is not None and output_data["model_patch"]["model_patch"] is not None
        if has_full_output and debug:
            print(f"Patch found for {instance_id}")
        
        return True, has_full_output
        
    except (IndexError, json.JSONDecodeError) as e:
        if debug:
            print(f"Error processing file for {instance_id}: {str(e)}")
        return True, False


async def process_all_instances(bucket, directory_prefix, instance_dirs, executor, debug) -> Tuple[List[str], List[str], List[str]]:
    """Process all instances concurrently and return results."""
    present = instance_dirs.copy()  # All instance IDs are present
    
    # Create tasks for all instances
    tasks = []
    for instance_id in instance_dirs:
        tasks.append(process_instance(bucket, directory_prefix, instance_id, executor, debug))
    
    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)
    
    # Process results
    generated = []
    not_null = []
    
    for i, (has_generation, has_not_null) in enumerate(results):
        instance_id = instance_dirs[i]
        
        if has_generation:
            generated.append(instance_id)
            
        if has_not_null:
            not_null.append(instance_id)
    
    return present, generated, not_null


if __name__ == '__main__':
    import fire
    
    def run_analyse(gcs_path: str, debug: bool = False, max_concurrency: int = 10):
        """Wrapper to run the async function."""
        asyncio.run(analyse(gcs_path, debug, max_concurrency))
    
    fire.Fire(run_analyse)