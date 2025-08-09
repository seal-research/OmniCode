from pathlib import Path
import json
import logging
import tempfile
import base64
import subprocess

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

CONFIG_FILE_MAP = {
    "bugfixing": CUR_DIR / "bugfixing.yaml",
    "testgen": CUR_DIR / "testgen.yaml",
    "bugfixing-java": CUR_DIR / "bugfixing_java.yaml",
    "testgen-java": CUR_DIR / "testgen_java.yaml",
    "stylereview": CUR_DIR / "stylereview.yaml",
    "reviewfix": CUR_DIR / "reviewfix.yaml",
}

# Add style review config map for Java
STYLE_REVIEW_CONFIG_MAP = {
    "checkstyle": CUR_DIR / "jstylereview.yaml",
    "pmd": CUR_DIR / "jstylereview_pmd.yaml",
}


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

class ArgumentTypeError(Exception):
    """An error from trying to convert a command line string to a type."""
    pass

def str2bool(v):
    """
    Minor helper function to convert string to boolean
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError("Boolean value expected.")

def get_reviewfix_faux_problem_statement(instance: dict) -> str:
    bad_patch = [bp for bp in instance['bad_patches'] if bp['source'] == 'badpatchllm'][0]
    problem_statement = instance['problem_statement']
    bad_patch_text = bad_patch['patch']
    review = bad_patch['review']

    faux_str = f"""Consider the following PR description:

      <pr_description>
      {problem_statement}
      </pr_description>

      Additionally, here is a previous patch attempt that failed to resolve this issue.

      <bad_patch>
      {bad_patch_text}
      </bad_patch>

      And here are is a review that attempts to explain why the patch failed:

      <review>
      {review}
      </review>

      Please carefully review the failed patch and its reviews. Use insight from them to **avoid repeating the same mistakes** and to **guide your reasoning** when implementing the fix."""
    return faux_str


def run_sweagent_single(
    instance: dict,
    model_name: str,
    api_key: str | None,
    output_dir: Path,
    mode: str = "bugfixing",
    thinking_budget: int | None = None,
    use_apptainer: bool = False,
):

    url = f"https://github.com/{instance['repo']}"

    if mode not in CONFIG_FILE_MAP:
        raise RuntimeError(f"Unknown mode: {mode}")
    
    if 'java' in mode:
        image = f"omnicodeorg/omnicode:{instance['repo'].replace('/', '_')}_base"
    else:
        image = f"sca63/codearena:{instance['instance_id']}"

    config_file = CONFIG_FILE_MAP[mode]

    with tempfile.NamedTemporaryFile(delete_on_close=False, mode="w") as fp:

        if mode == 'reviewfix':
            # use the problem statement to inject prompt, hacky way to modify prompt easily
            fp.write(get_reviewfix_faux_problem_statement(instance))
        else:
            fp.write(instance['problem_statement'])

        fp.close()

        args = ["run"]

        if config_file is not None:
            args.extend([f"--config",  str(config_file)])

        args += [
            f"--agent.model.name={model_name}",
            f"--agent.model.per_instance_cost_limit=2.0",
            f"--env.repo.github_url={url}",
            f"--env.repo.base_commit={instance['base_commit']}",
            f"--env.deployment.image={image}",
            # override having /testbed be WORKDIR for docker image
            '--env.deployment.docker_args=["-w","/"]',
            f"--problem_statement.path={str(fp.name)}",
            f"--problem_statement.id={instance['instance_id']}",
            f"--output_dir={output_dir}",
        ]

        if api_key is not None:
            args.append(f"--agent.model.api_key={api_key}")

        if mode == 'stylereview':
            # apply gold patch upon starting env, so that agent can modify it based on pylint feedback
            commands = apply_patch_commands(instance["patch"], repo_name=instance["repo"].replace("/", "__"))

            args.append(
                f"--env.post_startup_commands={json.dumps(commands)}",     # note: !r gives Python‑style list
            )


        if thinking_budget is not None:
            if model_name.startswith("gemini"):
                args.append("""--agent.model.completion_kwargs={"thinking":{"type":"enabled","budget_tokens":""" + str(int(thinking_budget)) + """}}""")
            else:
                raise RuntimeError(f"Cannot use thinking budget with non-gemini model: {model_name}")
        
        args.append(f"--use_apptainer={str(use_apptainer).lower()}")

        sweagent_main(args)

    output_file_path = output_dir / instance['instance_id'] / (instance['instance_id']  + ".pred")
    output = json.loads(output_file_path.read_text())

    return None, output

def apply_patch_commands(patch: str, repo_name: str) -> list[str]:
    """
    Return a list of commands that apply the patch to the repo.

    1.  recreate /tmp/patch.diff inside the container
    2.  try git‑apply, fallback to patch -p1 --fuzz
    """
    b64 = base64.b64encode(patch.encode()).decode()
    return [
        # write file atomically
        f"echo '{b64}' | base64 -d > /tmp/patch.diff",
        # cd into repo and apply
        f"""cd /{repo_name} && (
                git apply --allow-empty -v /tmp/patch.diff ||
                patch --batch --fuzz=5 -p1 -i /tmp/patch.diff
            )""",
    ]

def run_style_review_single(
    instance: dict,
    model_name: str,
    api_key: str | None,
    output_dir: Path,
    style_tool: str = "checkstyle",
    thinking_budget: int | None = None,
    timeout: int | None = None,
    use_apptainer: bool = False,
):
    """
    Run style review for a single instance using the specified tool (PMD or Checkstyle).
    """
    if style_tool not in STYLE_REVIEW_CONFIG_MAP:
        raise RuntimeError(f"Unknown style tool: {style_tool}. Must be 'checkstyle' or 'pmd'")

    # Correctly construct repo_path from instance_id
    if ":" in instance.get("instance_id", ""):
        repo_path = instance["instance_id"].split(":")[0]
    else:
        repo_path = instance.get("repo", "")

    repo_path = repo_path.strip()  # Ensure no leading/trailing whitespace
    url = f"https://github.com/{repo_path}"
    config_file = STYLE_REVIEW_CONFIG_MAP[style_tool]

    # Select image and check existence based on container type
    if use_apptainer:
        image = f"omnicodeorg/omnicode:{repo_path.replace('/', '_')}_base.sif"
        # Optionally check for .sif file existence locally
        sif_path = Path.home() / ".apptainer" / "images" / f"{repo_path.replace('/', '_')}_base.sif"
        if not sif_path.exists():
            print(f"\nERROR: Required Apptainer image '{image}' does not exist locally at {sif_path}.")
            print(f"Please build or pull the image before running the agent.")
            raise RuntimeError(f"Apptainer image '{image}' not found locally.")
    else:
        image = f"omnicodeorg/omnicode:{repo_path.replace('/', '_')}_base"
        # Check if the Docker image exists locally
        try:
            result = subprocess.run([
                "docker", "images", "-q", image
            ], capture_output=True, text=True)
            if not result.stdout.strip():
                print(f"\nERROR: Required Docker image '{image}' does not exist locally.")
                print(f"Please build or pull the image before running the agent.")
                print(f"You can try: docker pull {image}")
                raise RuntimeError(f"Docker image '{image}' not found locally.")
        except Exception as e:
            print(f"\nERROR: Failed to check for Docker image '{image}': {e}")
            raise

    with tempfile.NamedTemporaryFile(delete_on_close=False, mode="w") as fp:
        fp.write(instance['problem_statement'])
        fp.close()

        args = ["run"]
        if config_file is not None:
            args.extend([f"--config", str(config_file)])
        args += [
            f"--agent.model.name={model_name}",
            f"--agent.model.per_instance_cost_limit=2.0",
            f"--env.repo.github_url={url}",
            f"--env.repo.base_commit={instance['base_commit']}",
            f"--env.deployment.image={image}",
            f"--problem_statement.path={str(fp.name)}",
            f"--problem_statement.id={instance['instance_id']}",
            f"--output_dir={output_dir}",
        ]
        if api_key is not None:
            args.append(f"--agent.model.api_key={api_key}")
        if thinking_budget is not None:
            if model_name.startswith("gemini"):
                args.append("""--agent.model.completion_kwargs={\"thinking\":{\"type\":\"enabled\",\"budget_tokens\":""" + str(int(thinking_budget)) + """}}""")
            else:
                raise RuntimeError(f"Cannot use thinking budget with non-gemini model: {model_name}")
        # Apply the patch for style review if present
        if "patch" in instance:
            commands = apply_patch_commands(instance["patch"], repo_name=repo_path.replace("/", "__"))
            if commands:
                container_args = ["-w", "/"] + commands
            else:
                container_args = ["-w", "/"]
        else:
            container_args = ["-w", "/"]
        # Pass correct args for apptainer or docker
        if use_apptainer:
            args.append(f"--env.deployment.apptainer_args={json.dumps(container_args)}")
        else:
            args.append(f"--env.deployment.docker_args={json.dumps(container_args)}")
        print("DEBUG: Full args to sweagent_main:", args)
        sweagent_main(args)
    output_file_path = output_dir / instance['instance_id'] / (instance['instance_id'] + ".pred")
    output = json.loads(output_file_path.read_text())
    return None, output

def main(
    input_tasks_path: Path,
    output_dir_path: Path,
    model_name: str,
    api_key: str | None,
    instance_ids: list[str] | None= None,
    mode: str = "bugfixing",
    thinking_budget: int | None = None,
    use_apptainer: bool = False,
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
            full_output, model_patch = run_sweagent_single(datum, model_name=model_name, output_dir=output_dir_path, api_key=api_key, mode=mode, thinking_budget=thinking_budget, use_apptainer=use_apptainer)
            output_dict["full_output"] = full_output
            output_dict["model_patch"] = model_patch
            print(json.dumps(output_dict), file=f, flush=True)
# Update main to support style review via --mode stylereview or --style_tool
from argparse import ArgumentParser
if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("-i", "--input_tasks", type=str, required=True)
    parser.add_argument("--instance_ids", type=str, required=False, default=None)
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    parser.add_argument("-m", "--model_name", type=str, default="gemini/gemini-2.5-flash-preview-04-17")
    parser.add_argument("-k", "--api_key", type=str, default=None)
    parser.add_argument("--mode", type=str, default="bugfixing", choices=["bugfixing", "testgen", "bugfixing-java", "testgen-java", "stylereview", "reviewfix"])
    parser.add_argument("--thinking_budget", type=int, default=0)
    parser.add_argument("--style_tool", type=str, default=None, choices=["checkstyle", "pmd"], help="Style review tool to use (Java)")
    parser.add_argument("--use_apptainer", type=str2bool, default=False, help="run with docker or apptainer")
    args = parser.parse_args()
    
    # Determine if we are in style review mode
    style_review_modes = ["stylereview", "stylereview-java"]
    is_style_review = (args.mode in style_review_modes) or (args.style_tool is not None)

    # Determine style_tool from mode or argument
    style_tool = args.style_tool
    if is_style_review and style_tool is None:
        # Try to infer style_tool from input file or default to checkstyle
        style_tool = "checkstyle"

    if is_style_review:
        # Load dataset
        input_tasks_path = Path(args.input_tasks)
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
            raise RuntimeError(f"Data follows incorrect format")
        if args.instance_ids:
            instance_ids = args.instance_ids.split(",")
            dataset = [d for d in dataset if d["instance_id"] in instance_ids]
        # If mode is in the instance, filter for those with mode stylereview
        dataset = [d for d in dataset if d.get("mode", args.mode) in style_review_modes]
        output_dir_path = Path(args.output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        output_file_path = output_dir_path / f"style_review_{style_tool}_results.jsonl"
        existing_ids = set()
        if output_file_path.exists():
            with open(output_file_path) as f:
                for line in f:
                    data = json.loads(line)
                    instance_id = data["instance_id"]
                    existing_ids.add(instance_id)
        dataset = [d for d in dataset if d["instance_id"] not in existing_ids]
        with open(output_file_path, "a+") as f:
            for datum in tqdm(dataset, desc=f"Style review with {style_tool}"):
                instance_id = datum["instance_id"]
                output_dict = {"instance_id": instance_id, "style_tool": style_tool, "model_name": args.model_name}
                full_output, model_patch = run_style_review_single(datum, model_name=args.model_name, api_key=args.api_key, output_dir=output_dir_path, style_tool=style_tool, thinking_budget=args.thinking_budget)
                output_dict["full_output"] = full_output
                output_dict["model_patch"] = model_patch
                print(json.dumps(output_dict), file=f, flush=True)
    else:
        main(
            input_tasks_path=Path(args.input_tasks),
            output_dir_path=Path(args.output_dir),
            model_name=args.model_name,
            instance_ids=args.instance_ids.split(",") if args.instance_ids else None,
            api_key=args.api_key,
            mode=args.mode,
            # thinking_budget=args.thinking_budget,
            use_apptainer=args.use_apptainer,
        )
