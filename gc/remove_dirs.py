from google.cloud import storage
import concurrent.futures
import logging
import sys
import re
from typing import List, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_gcs_path(gcs_path: str) -> Tuple[str, str]:
    """Parse a GCS path in the format gs://bucket-name/path/to/dir into bucket and path."""
    match = re.match(r'gs://([^/]+)(?:/(.*))?', gcs_path)
    if not match:
        logger.error(f"Invalid GCS path format: {gcs_path}. Expected format: gs://bucket-name/path")
        sys.exit(1)
    
    bucket_name = match.group(1)
    base_dir = match.group(2) or ""
    
    return bucket_name, base_dir

def read_directory_list(file_path: str) -> List[str]:
    """Read directory names from a file, one per line."""
    try:
        with open(file_path, 'r') as file:
            return [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        sys.exit(1)

def delete_directory(client: storage.Client, bucket_name: str, base_dir: str, 
                    dir_name: str) -> Tuple[str, bool, str, int]:
    """Delete a directory (objects with a common prefix) from GCS bucket."""
    try:
        bucket = client.bucket(bucket_name)
        
        # Construct the full directory path/prefix
        base_dir = base_dir.rstrip('/') + '/' if base_dir else ''
        full_dir_prefix = f"{base_dir}{dir_name.rstrip('/')}/"
        
        # List all blobs with the directory prefix
        blobs = list(bucket.list_blobs(prefix=full_dir_prefix))
        
        if not blobs:
            return dir_name, False, f"No objects found with prefix {full_dir_prefix}", 0
        
        count = len(blobs)
        logger.info(f"Found {count} objects to delete in {full_dir_prefix}")
        
        # Delete in batches to avoid timeout for large directories
        batch_size = 1000
        for i in range(0, count, batch_size):
            batch = blobs[i:i+batch_size]
            bucket.delete_blobs(batch)
            
        return dir_name, True, f"Deleted {count} objects", count
    except Exception as e:
        logger.error(f"Error deleting {dir_name}: {str(e)}")
        return dir_name, False, str(e), 0

def main(file_path: str, gcs_path: str, max_workers: int = 10):
    """Main function to delete directories from GCS based on names in a local file."""
    # Parse GCS path
    bucket_name, base_dir = parse_gcs_path(gcs_path)
    
    # Read directory names from file
    dir_names = read_directory_list(file_path)
    
    if not dir_names:
        logger.warning(f"No directory names found in {file_path}")
        return
    
    # Initialize GCS client
    try:
        client = storage.Client()
    except Exception as e:
        logger.error(f"Failed to initialize GCS client: {str(e)}")
        logger.error("Make sure you have set up GCP authentication correctly.")
        sys.exit(1)
    
    logger.info(f"Preparing to delete {len(dir_names)} directories from gs://{bucket_name}/{base_dir}")
    
    # Use ThreadPoolExecutor for parallelization - appropriate for I/O bound operations
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit deletion tasks
        future_to_dir = {
            executor.submit(delete_directory, client, bucket_name, base_dir, dir_name): dir_name
            for dir_name in dir_names
        }
        
        # Process results as they complete
        successful = 0
        failed = 0
        total_objects = 0
        
        for future in concurrent.futures.as_completed(future_to_dir):
            dir_name, success, message, count = future.result()
            
            if success:
                logger.info(f"Successfully deleted: {dir_name} - {message}")
                successful += 1
                total_objects += count
            else:
                logger.warning(f"Failed to delete {dir_name}: {message}")
                failed += 1
        
        logger.info(f"Summary: {successful}/{len(dir_names)} directories processed successfully, {failed} failed")
        logger.info(f"Total objects deleted: {total_objects}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Delete directories from Google Cloud Storage")
    parser.add_argument("file_path", help="Path to local file containing directory names")
    parser.add_argument("gcs_path", help="GCS path in format gs://bucket-name/base-dir")
    parser.add_argument("--max-workers", type=int, default=10, 
                        help="Maximum number of concurrent workers (default: 10)")
    parser.add_argument("--verbose", "-v", action="store_true", 
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    main(args.file_path, args.gcs_path, args.max_workers)