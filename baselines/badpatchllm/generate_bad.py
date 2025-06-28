from pathlib import Path
import json
import logging
import math
import tempfile
import os

from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from jinja2 import Template
import pandas as pd

import google.generativeai as genai
import openai

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))  # Goes from baselines/badpatchllm to project root

from monkeypatched_swebench import swebench
from swebench.harness.run_evaluation import main as RegularEval
from swebench.harness.utils import str2bool

import difflib
import re

from codearena import clean_docker_images, setup_multiswebench_config, run_multiswebench_phase
# from google import genai


# Requests API logic
import os
import requests
from urllib.parse import urlparse
from typing import List, Dict, Optional

GITHUB_API = "https://api.github.com"
token_envs = {k: v for k, v in os.environ.items() if re.match(r"GITHUB.*TOKEN", k)}
TOKEN = next(iter(token_envs.values()), None)

if TOKEN is None:
    raise EnvironmentError("No GitHub token found. Please export a variable like GITHUB_TOKEN or GITHUB_API_TOKEN.")
else:
    print(f"[DEBUG] Using token from env var: {list(token_envs.keys())[0]}")
#TOKEN  = os.getenv("GITHUB_API_TOKEN")


def execute_command(func, **kwargs):
    """Wrapper to execute a function safely and catch errors."""
    func(**kwargs)

def check_patch(
    instance: dict,
    input_dataset: str,
    max_workers: int,
    force_rebuild: bool,
    cache_level: str,
    clean: bool,
    open_file_limit: int,
    run_id,
    timeout: int
):
    print("Executing BugFixing...")

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as temp_file:
        json.dump([instance], temp_file)  # Must be a list of dicts for some evaluation harnesses
        temp_file.flush()

        pred_path = temp_file.name
        print("predictions path name: ", pred_path)
        print("dataset name: ", input_dataset)

        instance_id = instance["instance_id"]

        try:
            execute_command(
                RegularEval,
                dataset_name=input_dataset,
                split="test",
                instance_ids=[instance_id],
                predictions_path=pred_path,
                max_workers=max_workers,
                force_rebuild=force_rebuild,
                cache_level=cache_level,
                clean=clean,
                open_file_limit=open_file_limit,
                run_id=run_id,
                timeout=timeout
            )
        except Exception as e:
            print(e)
            print(f"Evaluation failed for {instance_id}: {e}")
            return False

        report_file = f'logs/run_evaluation/{run_id}/{instance['model_name_or_path']}/{instance_id}/report.json'

        report_file_path = Path(report_file)
        resolved = False
        if report_file_path.exists():

            with open(report_file, 'r') as f:
                report = json.load(f)

            resolved = report[instance_id]['resolved']

    return resolved

def parse_pr_url(pr_url: str):
    parts = urlparse(pr_url).path.strip("/").split("/")
    owner, repo, _, num = parts
    return owner, repo, num

def get_pr_metadata(owner: str, repo: str, pr_number: str, token: str) -> dict:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    url = f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{pr_number}"
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()

def list_pr_files(owner: str, repo: str, pr_number: str, token: str) -> List[str]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    files, page = [], 1
    while True:
        url = f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{pr_number}/files"
        resp = requests.get(url, headers=headers, params={"per_page": 100, "page": page})
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break


        filtered = [
            f["filename"]
            for f in batch
            if not f["filename"].split("/")[-1].startswith("test")
        ]

        files.extend(filtered)
        page += 1
    return files

def fetch_file_at_ref(owner: str, repo: str, path: str, ref: str, token: str) -> Optional[bytes]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3.raw"
    }
    url = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}"
    resp = requests.get(url, headers=headers, params={"ref": ref})
    if resp.status_code == 200:
        return resp.content
    if resp.status_code == 404:
        return None
    resp.raise_for_status()

def get_changed_and_previous_files(pr_url: str, token: str) -> Dict[str, Optional[bytes]]:
    owner, repo, pr_number = parse_pr_url(pr_url)
    print("[DEBUG] TOKEN = ", token)

    pr_meta  = get_pr_metadata(owner, repo, pr_number, token)
    base_sha = pr_meta["base"]["sha"]

    files = list_pr_files(owner, repo, pr_number, token)
    return {
        path: fetch_file_at_ref(owner, repo, path, base_sha, token)
        for path in files
    }

def get_changed_and_current_files(pr_url: str, token: str) -> Dict[str, Optional[bytes]]:
    """
    For each changed file in the PR, fetch its content at the head commit (current state).
    Returns a dict mapping filepath -> bytes (or None if the file is missing).
    """
    owner, repo, pr_number = parse_pr_url(pr_url)
    print("[DEBUG] TOKEN = ", token)

    pr_meta  = get_pr_metadata(owner, repo, pr_number, token)
    head_sha = pr_meta["head"]["sha"]

    files = list_pr_files(owner, repo, pr_number, token)
    return {
        path: fetch_file_at_ref(owner, repo, path, head_sha, token)
        for path in files
    }

def gemini_25(prompt, temperature):

    model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")
    response = model.generate_content(
            prompt,
            generation_config={"temperature": temperature, "top_p": 0.9}
    )

    return response.text

def generate_incorrect_diff(
    prev_files: Dict[str, Optional[bytes]],
    curr_files: Dict[str, Optional[bytes]],
    gemini_api_key: str,
    model_name: str = "gemini-pro",
    temperature: float = 0.8,
    output_diff_path: str = "buggy_changes.diff",
) -> str:
    """
    For each file in curr_files, ask Gemini to inject a subtle functional bug
    into its current content, then diff against the previous content and
    write a unified .diff to output_diff_path.
    """
    genai.configure(api_key=gemini_api_key)
    diffs: List[str] = []

    # prompt = f"""
    # You are given a production-ready source file below. Your task:
    # 1. **Introduce one to three subtle, functional bugs**—for example an off-by-one in a loop, a wrong comparison operator, an incorrect return value in an edge case, a swapped parameter, or a logic inversion.
    # 2. **Do not change any import statements**
    # 3. **Do not break compilation** and **do not introduce any syntax or spelling errors** or make any code-style changes.
    # 4. Preserve formatting and comments; modify only the minimum lines needed to trigger a logical failure under certain inputs.
    # 5. Return **only** the full modified file content, with no explanations or diff markers.

    # --- {path} original content START ---
    # {curr_text}
    # --- {path} original content END ---
    # """

    # regex to identify comment lines with BUG
    # bug_comment_re = re.compile(r'^\s*(#|//|/\*|\*)[^\n]*\bBUG\b', re.IGNORECASE)
    bug_re = re.compile(r'bug', re.IGNORECASE)

    for path, curr_bytes in curr_files.items():
        prev_bytes = prev_files.get(path)

        # if curr_bytes is None or prev_bytes is None:
        #     # skip files added or deleted entirely
        #     continue

        prev_text = prev_bytes.decode("utf-8", errors="ignore") if prev_bytes else ""
        curr_text = curr_bytes.decode("utf-8", errors="ignore") if curr_bytes else ""

        # prompt = f"""
        # Please introduce a functional bug into this source file. Do not change any comments.
        # Make sure the bug would not cause a syntax error.
        # Return only the full modified file content, without any commentary.

        # --- {path} original content start ---
        # {curr_text}
        # --- {path} original content end ---
        # """

        filtered_text = curr_text

        if (path.endswith(".py") or path.endswith('.java')) and curr_text:

            prompt = f"""
            You are given a production-ready source file below. Your task:
            1. **Introduce one to two subtle, functional bugs**— without adding any comments
            2. **Do NOT break compilation** and **do not introduce any syntax or spelling errors** or make any code-style changes.
            3. **Do NOT change any import statements**
            4. Preserve formatting and comments; modify only the minimum lines needed to trigger a logical failure under certain inputs.
            5. Return **only** the full modified file content, with no explanations or diff markers.

            --- {path} original content START ---
            {curr_text}
            --- {path} original content END ---
            """

            modified_text = None

            if(model_name == "gemini-2.5-flash-preview-4-17" and curr_text):
                modified_text = gemini_25(prompt, temperature)

            elif (model_name == "gemini-2.0-flash" and curr_text):
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    prompt,
                    generation_config={"temperature": temperature}
                )

                modified_text = response.text

            if modified_text.startswith("```python"):
                modified_text = modified_text[len("```python"):].lstrip()  # Remove the prefix and leading spaces

            if modified_text.startswith("```java"):
                modified_text = modified_text[len("```java"):].lstrip()  # Remove the prefix and leading spaces

            if modified_text.endswith("```"):
                modified_text = modified_text[:-3].rstrip()  # Remove the suffix and trailing spaces
                # Split into lines for diffing

            filtered_lines: List[str] = []

            # for line in modified_text.splitlines():
            #     if '#' in line:
            #         idx = line.find('#')
            #         comment = line[idx+1:]
            #         if bug_re.search(comment):
            #             # drop the comment, keep only code before '#'
            #             filtered_lines.append(line[:idx].rstrip())
            #         else:
            #             filtered_lines.append(line)
            #     if '//' in line:
            #         idx = line.find('//')
            #         comment = line[idx+2:]
            #         if bug_re.search(comment):
            #             # drop the // comment entirely
            #             filtered_lines.append(line[:idx].rstrip())
            #         else:
            #             filtered_lines.append(line)
            #     else:
            #         filtered_lines.append(line)

            for line in modified_text.splitlines():
                if '#' in line:
                    idx = line.find('#')
                    comment = line[idx + 1:]
                    if bug_re.search(comment):
                        filtered_lines.append(line[:idx].rstrip())
                    else:
                        filtered_lines.append(line)
                elif '//' in line:
                    idx = line.find('//')
                    comment = line[idx + 2:]
                    if bug_re.search(comment):
                        filtered_lines.append(line[:idx].rstrip())
                    else:
                        filtered_lines.append(line)
                else:
                    filtered_lines.append(line)

            filtered_text = "\n".join(filtered_lines)

        prev_lines     = prev_text.splitlines()
        modified_lines = filtered_text.splitlines()

        # Produce a unified diff for this file
        file_diff = difflib.unified_diff(
            prev_lines,
            modified_lines,
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm=""
        )
        diffs.extend(file_diff)


    diff_str = "\n".join(diffs)
    diff_str += "\n"

    return output_diff_path, diff_str


def check_multi_patch(patch, instance_id, item, force_rebuild, run_id, max_workers, timeout):

        mswe_image_prefix = f"mswebench_{run_id}"

        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as temp_file:

            # prediction here

            instance = {}
            instance["id"] = instance_id
            instance["org"] = item["org"]
            instance["repo"] = item["repo"]
            instance["number"] = item["number"]
            instance["patch"] = patch

            json.dump([instance], temp_file)  # Must be a list of dicts for some evaluation harnesses
            temp_file.flush()

            pred_path = temp_file.name
            print("predictions path name: ", pred_path)

            try:
                with open(pred_path, 'r') as f:
                    predictions = json.load(f)

            except Exception as e:
                print(f"Error loading predictions file: {e}")
                return

            # Clean existing images if needed
            if force_rebuild:
                clean_docker_images(mswe_image_prefix)

            # Set up config
            config_file = setup_multiswebench_config(
                predictions=predictions,
                max_workers=max_workers,
                force_rebuild=force_rebuild,
                run_id=run_id,
                timeout=timeout,
                phase="all"
            )

            if config_file and os.path.exists(config_file):
                print(f"Config file created at: {config_file}")
                report = run_multiswebench_phase(config_file, "all", timeout)

                if report:
                    print("Multi-SWE-Bench BugFixing evaluation completed!")
                    print(f"Total instances: {report.get('total_instances', 0)}")
                    print(f"Resolved instances: {report.get('resolved_instances', 0)}")
                    print(f"Unresolved instances: {report.get('unresolved_instances', 0)}")
                else:
                    print("Multi-SWE-Bench BugFixing evaluation failed to produce a report")
            else:
                print("Failed to create config file, cannot run evaluation")

        report_file = f'multiswebench_runs/BugFixing/output/{run_id}/final_report.json'

        report_file_path = Path(report_file)
        resolved = False
        if report_file_path.exists():

            with open(report_file, 'r') as f:
                report = json.load(f)

            resolved = report['resolved_instances']

        return resolved

def main(
    input_tasks_path: Path,
    output_dir_path: Path,
    model_name: str,
    instance_ids: list[str],
    secret_key: str,
    num_patches: int,
    max_workers : int,
    force_rebuild: str2bool,
    cache_level: str,
    clean: str2bool,
    open_file_limit: int,
    run_id,
    timeout: int,
    type: str,
    max_iter: int,
):
    if type == "java":

        instance_id = instance_ids[0] # TODO extend functionality for list of instances

        dataset_base_path = "./multiswebench_local/mswebench_dataset"
        dataset_files = []
        for root, _, files in os.walk(dataset_base_path):
            for file in files:
                if file.endswith("_dataset.jsonl"):
                    dataset_files.append(os.path.join(root, file))

        if not dataset_files:
            print("Error: No dataset files found in", dataset_base_path)
            return

        target_repos = set()

        if ":" in instance_id:
            repo_part = instance_id.split(":")[0]
            target_repos.add(repo_part.replace("/", "__"))

        filtered_files = []
        for dataset_file in dataset_files:
            for repo in target_repos:
                if repo in dataset_file:
                    filtered_files.append(dataset_file)
                    print(f"Including dataset file: {dataset_file}")
                    break
        dataset_files = filtered_files

        if not dataset_files:
            print("No matching dataset files found!")
            return []

        found = False

        for dataset_file in dataset_files:
            print(f"Processing dataset file: {dataset_file}")

            with open(dataset_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue

                    try:
                        item = json.loads(line)
                        item_id = f"{item['org']}/{item['repo']}:{item['number']}"

                        if instance_ids and item_id not in instance_ids:
                            continue

                        if item_id == instance_id:
                            found = True
                            break

                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON from {dataset_file}: {e}")

                if found:
                    break

        pull_number = item['number']

        url = f"https://github.com/{item['org']}/{item['repo']}/pull/{pull_number}"

        curr = get_changed_and_current_files(url, TOKEN)
        prev = get_changed_and_previous_files(url, TOKEN)

        print(f"Generating incorrect diff for instance {instance_id} ...")

        output_d_path = output_dir_path / instance_id
        output_d_path.mkdir(parents=True, exist_ok=True)

        bad_patches = []
        modified_dataset = []

        for i in range(num_patches):

            diff_file, model_patch = generate_incorrect_diff(
                prev_files=prev,
                curr_files=curr,
                gemini_api_key=secret_key,
                model_name=model_name,
                temperature=0.85,
                output_diff_path="java_sanity.diff"
            )

            if model_patch and model_patch not in bad_patches:

                run_id = f'{run_id}_{i}'

                print("RUN_ID: ", run_id)

                resolved = check_multi_patch(model_patch, instance_id,
                item, force_rebuild, run_id, max_workers, force_rebuild)

                if not resolved:
                        bad_patches.append(model_patch)
                        diff_file_path = output_d_path / f"patch_{i}.diff"
                else:
                    diff_file_path = output_d_path / f"resolved/patch_{i}.diff"

                diff_file_path.parent.mkdir(parents=True, exist_ok=True)

                try:
                    with open(diff_file_path, "w") as diff_file:
                        diff_file.write(model_patch)
                except Exception as e:
                    print(f"Error writing .diff file for {instance_id}: {e}")

                if(resolved): sys.exit() # Short Circuiting if patch passes tests...
                if len(bad_patches) == num_patches : break

        modified_datum = {**item, "bad_patches": bad_patches}
        modified_dataset.append(modified_datum)

        json_path = output_dir_path / "modified_dataset.json"

        if json_path.exists():
                existing = json.loads(json_path.read_text())
        else:
            existing = []

        existing.append(modified_datum)

        json_path.write_text(json.dumps(existing, indent=2))

    elif type == "python":
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

        modified_dataset = []

        for datum in tqdm(dataset, desc=f"Inference for {model_name}"):

            instance_id = datum["instance_id"]

            output_d_path = output_dir_path / instance_id
            output_d_path.mkdir(parents=True, exist_ok=True)

            bad_patches = []
            print(f"Processing instance {instance_id}")

            for i in range(1, max_iter + 1):

                # TODO Generate Model Patch Here:

                print(f"Getting patches for instance {instance_id} ...")

                pull_number = datum['instance_id'].split('-')[-1]

                url = f"https://github.com/{datum['repo']}/pull/{pull_number}"
                print("URL: ", url)
                curr = get_changed_and_current_files(url, TOKEN)
                prev = get_changed_and_previous_files(url, TOKEN)

                print(f"Generating incorrect diff for instance {instance_id} ...")

                diff_file, model_patch = generate_incorrect_diff(
                    prev_files=prev,
                    curr_files=curr,
                    gemini_api_key=secret_key,
                    model_name=model_name,
                    temperature=0.85,
                    output_diff_path="my_pr_with_bugs{i}.diff"
                ) # TODO

                if model_patch and model_patch not in bad_patches:

                    #  Run Testing Code Directly Here to Ensure Bad Patches Fail

                    bad_instance = dict()
                    bad_instance['instance_id'] = instance_id
                    bad_instance['model_patch'] = model_patch
                    bad_instance['model_name_or_path'] = model_name

                    print(f"Checking if patch fails tests for instance {instance_id} ...")

                    resolved = check_patch( # BugFixing begins here
                        bad_instance,
                        str(input_tasks_path),
                        max_workers=max_workers,
                        force_rebuild=force_rebuild,
                        cache_level=cache_level,
                        clean=clean,
                        open_file_limit=open_file_limit,
                        run_id=f"{run_id}_{i}",
                        timeout=timeout
                    ) # TODO

                    if not resolved:
                        bad_patches.append(model_patch)
                        diff_file_path = output_d_path / f"patch_{i}.diff"
                    else:
                        diff_file_path = output_d_path / f"resolved/patch_{i}.diff"

                    diff_file_path.parent.mkdir(parents=True, exist_ok=True)

                    try:
                        with open(diff_file_path, "w") as diff_file:
                            diff_file.write(model_patch)
                    except Exception as e:
                        print(f"Error writing .diff file for {instance_id}: {e}")

                    if(resolved): sys.exit() # Short Circuiting if patch passes tests...
                    if len(bad_patches) == num_patches : break

            modified_datum = {**datum, "bad_patches": bad_patches}
            modified_dataset.append(modified_datum)

            json_path = output_dir_path / "modified_dataset.json"

            # (output_dir_path / "modified_dataset.json").write_text(
            #     json.dumps(modified_dataset, indent=2)
            # )

            # 1) Load existing list (or start fresh)
            if json_path.exists():
                existing = json.loads(json_path.read_text())
            else:
                existing = []

            # 2) Append your new record
            existing.append(modified_datum)

            # 3) Write the updated list back (this will preserve old entries)
            json_path.write_text(json.dumps(existing, indent=2))




if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset_name", default="data/codearena_instances.json", help="Name of the dataset")
    parser.add_argument("--instance_ids", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    parser.add_argument("-m", "--model_name", type=str, default="gemini-2.0-flash")
    parser.add_argument("-k", "--api_key", default=None, required=False)
    parser.add_argument("-n", "--num_patches", type=int, required=False, default=5)
    parser.add_argument("--run_id", required=True, help="Run ID for the evaluation")
    parser.add_argument(
        "--max_workers", type=int, required=False, default=1, help="Number of maximum workers to use"
    )
    parser.add_argument(
        "--force_rebuild", type=str2bool, required=False, default=False, help="Force rebuild of all images"
    )
    parser.add_argument(
        "--cache_level",
        type=str,
        required=False,
        choices=["none", "base", "env", "instance"],
        help="Cache level - remove images above this level",
        default="env",
    )
    parser.add_argument(
        "--clean", type=str2bool, required=False, default=False, help="Clean images above cache level"
    )
    parser.add_argument(
        "--open_file_limit",
        type=int,
        required=False,
        default=4096,
        help="Maximum number of open files",
    )
    parser.add_argument(
        "--timeout",
        required=False,
        type=int,
        default=1_800,
        help="Timeout for individual evaluations in seconds",
    )

    parser.add_argument(
        "--type",
        required=False,
        type=str,
        default="python",
        help="Language Type",
    )

    parser.add_argument(
        "--max_iter",
        type=int,
        default=6,
        help="Number of LLM samples to try to generate a bad patch",
    )

    args = parser.parse_args()

    api_key = args.api_key
    if args.api_key is None:
        api_key = os.environ.get("GEMINI_API_KEY", None)

    if api_key is None:
        raise RuntimeError("Gemini API Key not specified")

    main(
        input_tasks_path = Path(args.dataset_name),
        output_dir_path=Path(args.output_dir),
        model_name=args.model_name,
        instance_ids=args.instance_ids.split(",") if args.instance_ids else None,
        secret_key=api_key,
        num_patches=args.num_patches,
        max_workers=args.max_workers,
        force_rebuild=args.force_rebuild,
        cache_level=args.cache_level,
        clean=args.clean,
        open_file_limit=args.open_file_limit,
        run_id=args.run_id,
        timeout=args.timeout,
        type=args.type,
        max_iter=args.max_iter,
    )
