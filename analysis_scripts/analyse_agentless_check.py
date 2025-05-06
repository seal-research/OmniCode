from pathlib import Path
import json

def analyse(
    results_dir: str | Path,
):
    results_dir = Path(results_dir)
    
    skipped, present, bp_present, bp_missing = [], [], [], []

    for instance_dir in results_dir.iterdir():
        instance_id = instance_dir.name
        present.append(instance_id)

        patch_dirs = list(instance_dir.glob(f"{instance_id}_*"))

        if len(patch_dirs) == 0:
            print(f"Error: No patches found for {instance_id}, skipping ...")
            skipped.append(instance_id)
            continue

        resolved_dict = {}

        for pdir in patch_dirs:

            report_json_paths = list(pdir.rglob("report.json"))

            if len(report_json_paths) == 0:
                print(f"Info: No report found for {pdir.name}, skipping ...")
                continue

            report_json_path = report_json_paths[0]

            if len(report_json_paths) != 1:
                print(f"Warning: Multiple reports found for {instance_id}, using {report_json_path}")
            
            report_data = json.loads(report_json_path.read_text())

            if instance_id not in report_data:
                print(f"Error: Could not find {instance_id} in {report_json_path}, skipping ...")
                skipped.append(instance_id)
                continue

            if "resolved" not in report_data[instance_id]:
                print(f"Error: Could not find 'resolved' in {report_json_path}, skipping ...")
                skipped.append(instance_id)
                continue

            resolved_dict[pdir.name] = report_data[instance_id]['resolved']
    
        if len(resolved_dict) == 0:
            print(f"Error: No reports found for {instance_id}, skipping ...")
            skipped.append(instance_id)
            continue

        if any(r == False for r in resolved_dict.values()):
            bp_present.append(instance_id)
        else:
            bp_missing.append(instance_id)



    print(f"Skipped: {skipped}")
    print(f"{len(present)=}, {len(skipped)=}, {len(bp_present)=}, {len(bp_missing)=}")

    all_ids = set([i for i in Path("/Users/ays57/Documents/opus/seds/codearena/data/instance_ids.txt").read_text().splitlines()])
    remaining = all_ids - set(present)
    print(f"{len(remaining)=}")

    Path("agentless_check_instances_remaining.txt").write_text(
        '\n'.join(i for i in sorted(remaining))
    )


if __name__ == '__main__':

    import fire
    fire.Fire(analyse)