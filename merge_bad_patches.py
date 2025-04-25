import json
import os
from typing import Dict, List, Any

def load_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file and return a list of dictionaries."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Load a JSON file and return a list of dictionaries."""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json_file(data: List[Dict[str, Any]], file_path: str) -> None:
    """Save data to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def merge_bad_patches(badpatch_path: str, instances_path: str) -> None:
    """Merge bad patches from badpatch.jsonl into codearena_instances.json."""
    # Load the bad patches
    bad_patches = load_jsonl_file(badpatch_path)
    print(f"Loaded {len(bad_patches)} bad patches from {badpatch_path}")
    
    # Load the instances
    instances = load_json_file(instances_path)
    print(f"Loaded {len(instances)} instances from {instances_path}")
    
    # Create mappings for both full list and first patch
    bad_patches_map = {item['instance_id']: item['bad_patches'] for item in bad_patches}
    first_bad_patch_map = {item['instance_id']: item['bad_patches'][0] if item['bad_patches'] else None 
                          for item in bad_patches}
    
    # Update instances with both full list and first patch, and add candidate_test_patch
    updated_count = 0
    for instance in instances:
        instance_id = instance['instance_id']
        if instance_id in bad_patches_map:
            # Add the full list of bad patches
            instance['bad_patches'] = bad_patches_map[instance_id]
            # Add the first bad patch as a separate entry
            instance['bad_patch'] = first_bad_patch_map[instance_id]
            # Add candidate_test_patch as a copy of test_patch
            if 'test_patch' in instance:
                instance['candidate_test_patch'] = instance['test_patch']
            # Add gold_patch as a copy of patch
            if 'patch' in instance:
                instance['gold_patch'] = instance['patch']

            updated_count += 1
    
    # Save the updated instances
    save_json_file(instances, instances_path)
    print(f"Updated {updated_count} instances with bad patches")
    print(f"Added both full list of bad patches and first bad patch to each instance")
    print(f"Added candidate_test_patch as a copy of test_patch")
    print(f"Saved updated instances to {instances_path}")

if __name__ == "__main__":
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct paths relative to the script directory
    badpatch_path = os.path.join(script_dir, "baselines/badpatchllm/logs/gemini_outputs/celery__celery-8489/badpatch.jsonl")
    instances_path = os.path.join(script_dir, "data/codearena_instances.json")
    
    # Check if files exist
    if not os.path.exists(badpatch_path):
        print(f"Error: Bad patch file not found at {badpatch_path}")
        exit(1)
    if not os.path.exists(instances_path):
        print(f"Error: Instances file not found at {instances_path}")
        exit(1)
    
    # Merge the bad patches
    merge_bad_patches(badpatch_path, instances_path) 