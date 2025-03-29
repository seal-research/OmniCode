from pathlib import Path
import json
import glob
import os

import fire

def clean_badpatch_outputs(
    badpatch_dir: str | Path,
    output_path: str | Path = None,
):
    """
    Clean and combine badpatch outputs to match codearena's expected format.
    
    Args:
        badpatch_dir: Directory containing gemini_outputs folders with badpatch.jsonl files
        output_path: Where to save the cleaned output. If None, saves to gemini_outputs/all_preds.jsonl
    """
    badpatch_dir = Path(badpatch_dir)
    if output_path is None:
        output_path = badpatch_dir / "gemini_outputs" / "all_preds.jsonl"
    else:
        output_path = Path(output_path)

    # Find all badpatch.jsonl files
    instance_dirs = glob.glob(str(badpatch_dir / "gemini_outputs" / "*"))
    
    clean_data = []
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
            
            # Check if we have bad patches
            if "bad_patches" in data and data["bad_patches"] and len(data["bad_patches"]) > 0:
                clean_data.append({
                    "instance_id": data["instance_id"],
                    "model_name_or_path": data["model_name_or_path"],
                    "model_patch": data["bad_patches"][0]  # Take first patch
                })
            else:
                print(f"Warning: No valid patches found for {instance_id}")
                
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in {instance_id}: {e}")
            continue
        except Exception as e:
            print(f"Error processing {instance_id}: {e}")
            continue
    
    # Write output in JSONL format
    output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    output_path.write_text('\n'.join(json.dumps(d) for d in clean_data))
    print(f"Cleaned data saved to {output_path}")
    print(f"Processed {len(clean_data)} instances")

if __name__ == '__main__':
    fire.Fire(clean_badpatch_outputs) 