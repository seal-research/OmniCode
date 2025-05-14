import json
from pathlib import Path


def analyse(
    results_dir: str | Path,
):
    results_dir = Path(results_dir)
    

    present, skipped, nulls, founds = [], [], [], []    
    final_all_preds = []
    for predsf in results_dir.rglob("all_preds.jsonl"):
        predsd = json.loads(predsf.read_text().splitlines()[0])

        instance_id = predsd['instance_id']
        present.append(instance_id)

        if "model_patch" not in predsd:
            print(f"No model patch field found for {instance_id}, skipping ...")
            skipped.append(instance_id)
            continue

        if "model_patch" not in predsd["model_patch"]:
            print(f"No model patch field in model patch for {instance_id}, skipping ...")
            skipped.append(instance_id)
            continue

        if predsd["model_patch"]["model_patch"] is None:
            print(f"Model patch for {instance_id} is null, skipping ...")
            nulls.append(instance_id)
            continue

        founds.append(instance_id)
        final_all_preds.append(predsd['model_patch'])
    
    print(f"{len(present)=}, {len(skipped)=}, {len(nulls)=}, {len(founds)=}")

    Path("consolidated_all_preds.jsonl").write_text('\n'.join([json.dumps(i) for i in final_all_preds])) 
    Path("instances_from_consolidated.txt").write_text('\n'.join(sorted(founds)))   

    

if __name__ == '__main__':
    import fire
    fire.Fire(analyse)