from pathlib import Path
import json
from collections import defaultdict


def get_stats(
    dataset_path: Path | str,
):
    dataset_path = Path(dataset_path)

    data = json.loads(dataset_path.read_text())

    counts = defaultdict(lambda: 0)
    for d in data:
        counts[d['repo']] += 1
    
    for repo, count in counts.items():
        print(f"{repo}\t{count}")

    total_count = sum(counts.values())
    print(f"\ntotal = {total_count}")


if __name__ == '__main__':

    import fire

    fire.Fire(get_stats)