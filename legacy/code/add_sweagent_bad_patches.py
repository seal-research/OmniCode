from pathlib import Path
import json

import fire

def add_sweagent_bad_patches(
    sweagent_preds_file_path: str | Path,
    codearena_data_path: str | Path,
):  
    sweagent_preds_file_path, codearena_data_path = Path(sweagent_preds_file_path), Path(codearena_data_path)
    sweagent_data = [json.loads(l) for l in sweagent_preds_file_path.read_text().splitlines()]
    codearena_data = json.loads(codearena_data_path.read_text())

    for d in sweagent_data:
        for c in codearena_data:
            if d["instance_id"] == c["instance_id"]:
                if "bad_patch" not in c:
                    c["bad_patches"] = [d["model_patch"]]
                else:
                    c["bad_patches"].append(d["model_patch"])

    codearena_data_path.write_text(json.dumps(codearena_data, indent=4))

if __name__ == '__main__':
    fire.Fire(add_sweagent_bad_patches)