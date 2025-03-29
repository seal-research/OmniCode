from pathlib import Path
import json
import glob
import os
from typing import Dict, List
from collections import defaultdict

import fire

def process_instance_data(data: Dict) -> List[Dict]:
    """Process a single instance's data into multiple predictions, one for each patch.
    
    Args:
        data: Dictionary containing instance data with bad_patches
        
    Returns:
        List of dictionaries, one for each patch
    """
    if not ("bad_patches" in data and data["bad_patches"] and len(data["bad_patches"]) > 0):
        return []
        
    predictions = []
    for patch_idx, patch in enumerate(data["bad_patches"], 1):
        predictions.append({
            "instance_id": data["instance_id"],
            "model_name_or_path": data["model_name_or_path"],
            "model_patch": patch,
            "patch_number": patch_idx
        })
    return predictions

def write_jsonl(path: Path, data: List[Dict]):
    """Write data to a JSONL file."""
    path.write_text('\n'.join(json.dumps(d) for d in data))

def clean_badpatch_outputs(
    badpatch_dir: str | Path,
    output_dir: str | Path = None,
):
    """
    Clean and combine badpatch outputs to match codearena's expected format.
    Creates both split files (all_preds_N.jsonl) and combined file (all_preds.jsonl).
    
    Args:
        badpatch_dir: Directory containing gemini_outputs folders with badpatch.jsonl files
        output_dir: Directory to save the cleaned outputs. If None, saves to gemini_outputs/
    """
    badpatch_dir = Path(badpatch_dir)
    if output_dir is None:
        output_dir = badpatch_dir / "gemini_outputs"
    else:
        output_dir = Path(output_dir)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all badpatch.jsonl files
    instance_dirs = glob.glob(str(badpatch_dir / "gemini_outputs" / "*"))
    
    # Store predictions by patch number and combined
    predictions_by_patch = defaultdict(list)
    all_predictions = []
    
    for instance_dir in instance_dirs:
        instance_id = os.path.basename(instance_dir)
        badpatch_file = Path(instance_dir) / "badpatch.jsonl"
        
        if not badpatch_file.exists():
            print(f"Warning: No badpatch.jsonl found for {instance_id}")
            continue
            
        try:
            # Read the file content
            content = badpatch_file.read_text().strip()
            if not content:  # Skip empty files
                print(f"Warning: Empty file for {instance_id}")
                continue
                
            data = json.loads(content)
            instance_predictions = process_instance_data(data)
            
            if not instance_predictions:
                print(f"Warning: No valid patches found for {instance_id}")
                continue
                
            # Add predictions to both collections
            all_predictions.extend(instance_predictions)
            for pred in instance_predictions:
                predictions_by_patch[pred["patch_number"]].append(pred)
                
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in {instance_id}: {e}")
            continue
        except Exception as e:
            print(f"Error processing {instance_id}: {e}")
            continue
    
    # Write split files
    for patch_num in range(1, 6):
        output_path = output_dir / f"all_preds_{patch_num}.jsonl"
        write_jsonl(output_path, predictions_by_patch[patch_num])
        print(f"\nPatch {patch_num} predictions saved to {output_path}")
        print(f"Contains {len(predictions_by_patch[patch_num])} predictions")
    
    # Write combined file
    combined_path = output_dir / "all_preds.jsonl"
    write_jsonl(combined_path, all_predictions)
    
    # Print summary
    instance_count = len(set(p["instance_id"] for p in all_predictions))
    print(f"\nTotal processed:")
    print(f"- {instance_count} unique instances")
    print(f"- {len(all_predictions)} total predictions")
    print(f"- Combined predictions saved to {combined_path}")
    
    # Print patch distribution
    print("\nPatch distribution:")
    for patch_num in sorted(predictions_by_patch.keys()):
        print(f"Patch {patch_num}: {len(predictions_by_patch[patch_num])} predictions")

if __name__ == '__main__':
    fire.Fire(clean_badpatch_outputs) 