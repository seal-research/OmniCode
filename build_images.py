import docker
import json
from pathlib import Path
import platform

if platform.system() == 'Linux':
    import resource

from monkeypatched_swebench import swebench
from swebench.harness.docker_build import build_env_images

def main(
        data: str,
        instances: str | None = None,
        max_workers: int = 4,
        force_rebuild: bool = False,
        open_file_limit: int = 4096,
        target: str | None = None,
    ):

    instances = instances.split(',') if instances is not None else None

    # set open file limit
    if platform.system() == 'Linux':
        resource.setrlimit(resource.RLIMIT_NOFILE, (open_file_limit, open_file_limit))
    client = docker.from_env()

    # get data
    dataset = json.loads(Path(data).read_text())
    if instances is not None:
        dataset = [d for d in dataset if d['instance_id'] in instances]


    print(f"Building images for {len(dataset)} instances...")
    build_env_images(client, dataset, force_rebuild, max_workers, target)



if __name__ == "__main__":

    import fire

    fire.Fire(main)


