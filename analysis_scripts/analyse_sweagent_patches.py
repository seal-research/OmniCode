from pathlib import Path
import json

def analyse(
    results_dir: str | Path,
):
    results_dir = Path(results_dir)
    
    present, generated, not_null = [], [], []

    for instance_dir in results_dir.iterdir():
        instance_id = instance_dir.name
        present.append(instance_id)

        output_file = instance_dir / "all_preds.jsonl"

        if not output_file.exists():
            print(f"No all_preds.jsonl for {instance_id}")
            continue

        generated.append(instance_id)
        output_data = json.loads(output_file.read_text().splitlines()[0])

        if output_data.get("model_patch", None) is not None and output_data["model_patch"]["model_patch"] is not None:
            print(f"Patch found for {instance_id}")
            not_null.append(instance_id)



    print(f"Not Null: {not_null}")
    print(f"{len(present)=}, {len(generated)=}, {len(not_null)=}")


if __name__ == '__main__':

    import fire
    fire.Fire(analyse)