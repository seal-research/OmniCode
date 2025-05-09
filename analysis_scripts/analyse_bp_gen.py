from pathlib import Path
import json

def analyse(
    results_dir: str | Path,
):
    results_dir = Path(results_dir)
    
    skipped, present, correct, num_patches = [], [], [], []

    for instance_dir in results_dir.iterdir():
        instance_id = instance_dir.name
        present.append(instance_id)

        bad_patches = list(instance_dir.rglob(r"resolved/patch_*.diff"))
        num_patches.append(len(bad_patches))

        if len(bad_patches) == 0:
            print(f"Error: No bad patches found for {instance_id}, skipping ...")
            skipped.append(instance_id)
            continue

        correct.append(instance_id)



    print(f"Correct: {correct}")
    print(f"{len(present)=}, {len(skipped)=}")


if __name__ == '__main__':

    import fire
    fire.Fire(analyse)