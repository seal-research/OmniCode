from pathlib import Path
import shutil
import re
from typing import Union

def clean(results_dir: Union[str, Path]):
    """
    Clean up the directory structure by moving files from 'logs' subdirectories
    to their parent directories and removing other files/directories.
    
    Args:
        results_dir: Path to the results directory, either local or GCS (gs://)
    """
    # Check if the path is a GCS path
    if isinstance(results_dir, str) and results_dir.startswith("gs://"):
        clean_gcs(results_dir)
    else:
        clean_local(results_dir)

def clean_local(results_dir: Union[str, Path]):
    """Clean up a local directory structure."""
    results_dir = Path(results_dir)
    for idir in results_dir.iterdir():
        if not idir.is_dir():
            continue
        ilog_dir = idir / "logs"
        if not ilog_dir.exists():
            continue
        for i in idir.iterdir():
            if i.name != "logs":
                if i.is_dir():
                    shutil.rmtree(i)
                else:
                    i.unlink()
        for item in ilog_dir.iterdir():
            shutil.move(str(item), str(idir / item.name))
    
        ilog_dir.rmdir()

def clean_gcs(gcs_path: str):
    """Clean up a Google Cloud Storage directory structure."""
    from google.cloud import storage
    
    # Extract bucket name and path from gs:// URL
    match = re.match(r'gs://([^/]+)/(.+)', gcs_path)
    if not match:
        raise ValueError(f"Invalid GCS path: {gcs_path}")
    
    bucket_name, base_path = match.groups()
    # Ensure base_path ends with a slash to represent a directory
    if not base_path.endswith('/'):
        base_path += '/'
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    # Helper function to list directories and files
    def list_directory(prefix):
        """List all files and directories directly under a prefix."""
        if not prefix.endswith('/'):
            prefix += '/'
        
        iterator = bucket.list_blobs(prefix=prefix, delimiter='/')
        # Get all blobs directly under this prefix
        blobs = list(iterator)
        # Remove the directory marker itself if it exists
        blobs = [blob for blob in blobs if blob.name != prefix]
        # Get all directory prefixes
        prefixes = list(iterator.prefixes)
        
        return blobs, prefixes
    
    # Helper function to check if a directory exists
    def directory_exists(prefix):
        """Check if a directory exists."""
        if not prefix.endswith('/'):
            prefix += '/'
        
        # List blobs with this prefix to see if any exist
        iterator = bucket.list_blobs(prefix=prefix, delimiter='/')
        blobs = list(iterator)
        prefixes = list(iterator.prefixes)
        
        return bool(blobs) or bool(prefixes)
    
    # Helper function to delete a directory recursively
    def delete_directory(prefix):
        """Delete a directory and all its contents recursively."""
        if not prefix.endswith('/'):
            prefix += '/'
        
        # List all blobs that start with this prefix
        blobs = list(bucket.list_blobs(prefix=prefix))
        
        # Delete each blob
        for blob in blobs:
            blob.delete()
    
    # List all subdirectories in the results directory
    _, subdirs = list_directory(base_path)
    
    for subdir in subdirs:
        # Check if a logs subdirectory exists
        logs_dir = f"{subdir}logs/"
        logs_exists = directory_exists(logs_dir)
        
        if not logs_exists:
            continue
        
        # Get all items in the subdirectory
        files, dirs = list_directory(subdir)
        
        # Delete all files
        for file in files:
            file.delete()
        
        # Delete all directories except logs
        for directory in dirs:
            if directory != logs_dir:
                delete_directory(directory)
        
        # Move all items from logs directory to parent directory
        logs_files, _ = list_directory(logs_dir)
        
        # Move files from logs to parent directory
        for file in logs_files:
            # Get the filename (last part of the path)
            filename = file.name.split('/')[-1]
            
            # Create new destination in parent directory
            destination_name = f"{subdir}{filename}"
            
            # Copy file to new location
            bucket.copy_blob(file, bucket, destination_name)
            
            # Delete original
            file.delete()

if __name__ == '__main__':
    import fire
    fire.Fire(clean)