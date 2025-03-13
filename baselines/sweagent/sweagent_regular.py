from pathlib import Path
import json
import logging
import math
import tempfile

from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from jinja2 import Template
import pandas as pd

from sweagent.run.run import main as sweagent_main
from sweagent.run.run_single import RunSingle, RunSingleConfig
from sweagent.environment.swe_env import EnvironmentConfig, DockerDeploymentConfig
from sweagent.environment.repo import GithubRepoConfig
from sweagent.agent.agents import AgentConfig
from sweagent.agent.models import GenericAPIModelConfig
from sweagent.agent.problem_statement import TextProblemStatement, FileProblemStatement

CUR_DIR = Path(__file__).parent
DOTENV_PATH = CUR_DIR / '.env'

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# def run_sweagent_single(
#     instance: dict,
#     model_name: str,
#     output_dir: Path,
# ):
    
#     agent = AgentConfig(
#         model=GenericAPIModelConfig(
#             name=model_name,
#         ),
#     )

#     url = f"https://github.com/{instance['repo']}"

#     env = EnvironmentConfig(
#         deployment=DockerDeploymentConfig(image="python:3.12"),
#         repo=GithubRepoConfig(
#             github_url=url,
#             base_commit=instance['base_commit'],
#         ),
#         post_startup_commands=[],
#     )

    
#     # problem_statement = TextProblemStatement(
#     #     text=PROMPT_TEMPLATE.render(
#     #         issue=instance['problem_statement']
#     #     ),
#     #     id=instance['instance_id'],
#     # )

#     with tempfile.NamedTemporaryFile(delete_on_close=False, mode="w") as fp:
#         fp.write(
#             PROMPT_TEMPLATE.render(
#                 issue=instance['problem_statement']
#             )
#         )
#         fp.close()

        
#         problem_statement = FileProblemStatement(
#             path=Path(fp.name),
#             id=instance['instance_id'],
#         )

#         config = RunSingleConfig(
#             env=env,
#             agent=agent,
#             problem_statement=problem_statement,
#             output_dir=output_dir,
#             env_var_path=DOTENV_PATH,
#         )

#         RunSingle.from_config(config).run()


#     output_file_path = output_dir / problem_statement.id / (problem_statement.id + ".pred") 
#     output = json.loads(output_file_path.read_text())

#     return None, output



def run_sweagent_single(
    instance: dict,
    model_name: str,
    output_dir: Path,
):
    
    url = f"https://github.com/{instance['repo']}"

    with tempfile.NamedTemporaryFile(delete_on_close=False, mode="w") as fp:
        fp.write(instance['problem_statement'])
        fp.close()

        args = [
            "run",
            f"--agent.model.name={model_name}",
            f"--env.repo.github_url={url}",
            f"--env.repo.base_commit={instance['base_commit']}",
            f"--problem_statement.path={str(fp.name)}",
            f"--problem_statement.id={instance['instance_id']}",
            f"--output_dir={output_dir}"
        ]

        sweagent_main(args)
        
    output_file_path = output_dir / instance['instance_id'] / (instance['instance_id']  + ".pred") 
    output = json.loads(output_file_path.read_text())

    return None, output


def main(
    input_tasks_path: Path,
    output_dir_path: Path,
    model_name: str,
    instance_ids: list[str] | None= None,
):
    if input_tasks_path.exists():
        if input_tasks_path.suffix.endswith("json"):
            dataset = json.loads(input_tasks_path.read_text())
        elif input_tasks_path.suffix.endswith("jsonl"):
            dataset = [json.loads(i) for i in input_tasks_path.read_text().splitlines()]
        elif input_tasks_path.suffix.endswith("csv"):
            dataset = pd.read_csv(input_tasks_path).to_dict('records')
        else:
            raise RuntimeError(f"Data type ({input_tasks_path.suffix}) not supported")

    else:
        dataset = load_dataset(str(input_tasks_path))

    if isinstance(dataset, dict):
        dataset = dataset['test']

    if not (isinstance(dataset, list) and all(isinstance(d, dict) for d in dataset)):
        raise RuntimeError(f"Data folllows incorrect format")

    if instance_ids is not None:
        dataset = [d for d in dataset if d["instance_id"] in instance_ids]

    existing_ids = set()
    
    output_dir_path.mkdir(parents=True, exist_ok=True)
    output_file_path = output_dir_path / "all_preds.jsonl"

    if output_file_path.exists():
        with open(output_file_path) as f:
            for line in f:
                data = json.loads(line)
                instance_id = data["instance_id"]
                existing_ids.add(instance_id)
    logger.info(f"Read {len(existing_ids)} already completed ids from {output_file_path}")

    basic_args = {
        "model_name_or_path": model_name,
    }
    
    with open(output_file_path, "a+") as f:
        for datum in tqdm(dataset, desc=f"Inference for {model_name}"):
            instance_id = datum["instance_id"]
            if instance_id in existing_ids:
                continue
            output_dict = {"instance_id": instance_id}
            output_dict.update(basic_args)
            full_output, model_patch = run_sweagent_single(datum, model_name=model_name, output_dir=output_dir_path)
            output_dict["full_output"] = full_output
            output_dict["model_patch"] = model_patch
            print(json.dumps(output_dict), file=f, flush=True)

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_tasks", type=str, required=True)
    parser.add_argument("--instance_ids", type=str, required=False, default=None)
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    parser.add_argument("-m", "--model_name", type=str, default="gpt-4o")
    args = parser.parse_args()

    main(
        input_tasks_path=Path(args.input_tasks),
        output_dir_path=Path(args.output_dir),
        model_name=args.model_name,
        instance_ids=args.instance_ids.split(",") if args.instance_ids else None,
    )

