from pathlib import Path
import json


def merge_components(
    components_dir: Path | str,
    output_path: Path | str,
    overwrite: bool = False,
):
    components_dir = Path(components_dir)
    output_path = Path(output_path)
    
    if output_path.exists():
        print(f"File already exists at {output_path}, appending to it")
        existing_instances = json.loads(output_path.read_text())
        existing_instance_ids = set(i['instance_id'] for i in existing_instances)
    else:
        existing_instances, existing_instance_ids = [], set()
    
    output = []
    old_instance_ids_to_exclude = set()
    for data_path in components_dir.glob("*_instances.json"):
        print(f"Processing {data_path} ...")
        added_count = 0
        instances_to_add = json.loads(data_path.read_text())
        for instance in instances_to_add:
            if instance['instance_id'] in existing_instance_ids:
                if overwrite:
                    print(f"Instance {instance['instance_id']} being overwritten")
                    output.append(instance)
                    added_count += 1
                    old_instance_ids_to_exclude.add(instance['instance_id'])
                else:
                    print(f"Instance {instance['instance_id']} not being overwritten")
            else:
                output.append(instance)
                added_count += 1

        print(f"Added {added_count}")


    filtered_existing_instances = [i for i in existing_instances if i['instance_id'] not in old_instance_ids_to_exclude]
    output = filtered_existing_instances + output
    
    output_path.write_text(json.dumps(output, indent=2))


if __name__ == '__main__':
    import fire

    fire.Fire(merge_components)
    