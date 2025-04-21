from pathlib import Path
import json

import fire

def clean_sweagent_outputs(
    sweagent_preds_file_path: str | Path
):  
    sweagent_preds_file_path = Path(sweagent_preds_file_path)
    data = [json.loads(l) for l in sweagent_preds_file_path.read_text().splitlines()]
    clean_data = []
    for d in data:
        if isinstance(d["model_patch"], dict):
            model_patch = d["model_patch"]["model_patch"]
        elif isinstance(d["model_patch"], str):
            model_patch = d["model_patch"] 
        else:
            raise RuntimeError(f"Weird type for model patch in input data at id: {d['instance_id']}")  
        
        clean_data.append(
            {
                "instance_id": d["instance_id"],
                "model_name_or_path": d["model_name_or_path"],
                "model_patch": model_patch,
            }
        )
    sweagent_preds_file_path.write_text('\n'.join([json.dumps(d) for d in clean_data]))

if __name__ == '__main__':
    fire.Fire(clean_sweagent_outputs)