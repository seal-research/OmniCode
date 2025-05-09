from pathlib import Path
import json

def analyse(
    results_dir: str | Path,
    input_data: str | Path,
    output_data: str | Path,
    patches_dir: str | Path
):
    results_dir, patches_dir = Path(results_dir), Path(patches_dir)

    original_data = json.loads(Path(input_data).read_text())
    with_patches_data = []
    
    skipped, present, bp_present, bp_missing = [], [], [], []

    for original_instance in original_data:

        instance_id = original_instance['instance_id']

        with_patches_data.append(original_instance)
        with_patches_data[-1]["bad_patches"] = []
        max_id = 0
        
        instance_dir = results_dir / instance_id

        if not instance_dir.exists():
            print(f"Could not find evaluation for instance: {instance_id}, skipping ...")
            continue

        present.append(instance_id)

        patch_dirs = list(instance_dir.glob(f"{instance_id}_*"))

        if len(patch_dirs) == 0:
            print(f"Error: No patches found for {instance_id}, skipping ...")
            skipped.append(instance_id)
            continue

        resolved_dict = {}


        # if "bad_patches" not in original_instance or len(original_instance["bad_patches"]) == 0:
        #     original_instance["bad_patches"] = []
        #     max_id = 0
        # else:
        #     max_id = max([bp['idx'] for bp in original_instance['bad_patches']])

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

            resolved_status = report_data[instance_id]['resolved']
            resolved_dict[pdir.name] = resolved_status

            if not resolved_status:
                max_id += 1
                patch = json.loads((patches_dir / f"{pdir.name}.jsonl").read_text())["model_patch"]
                with_patches_data[-1]['bad_patches'].append({"idx": max_id, "patch": patch})

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

    Path(output_data).write_text(json.dumps(with_patches_data, indent=2))
    print(f"Wrote instances with patches to {output_data}")

    # all_ids = set([i for i in Path("/Users/ays57/Documents/opus/seds/codearena/data/instance_ids.txt").read_text().splitlines()])
    # remaining = all_ids - set(present)
    # print(f"{len(remaining)=}")

    # Path("agentless_check_instances_remaining.txt").write_text(
    #     '\n'.join(i for i in sorted(remaining))
    # )


if __name__ == '__main__':

    import fire
    fire.Fire(analyse)