from pathlib import Path
import json

def analyse(
    results_dir: str | Path,
):
    results_dir = Path(results_dir)
    
    skipped, present, passed, failed = [], [], [], []

    for instance_dir in results_dir.iterdir():
        instance_id = instance_dir.name
        present.append(instance_id)

        report_json_paths = list(instance_dir.rglob("report.json"))

        if len(report_json_paths) == 0:
            print(f"Info: No report found for {instance_dir.name}, skipping ...")
            skipped.append(instance_id)
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

        if report_data[instance_id]['resolved']:
            passed.append(instance_id)
        else:
            failed.append(instance_id)


    print(f"Skipped: {skipped}")
    print(f"{len(present)=}, {len(skipped)=}, {len(passed)=}, {len(failed)=}")


if __name__ == '__main__':

    import fire
    fire.Fire(analyse)