from pathlib import Path
import json
import logging
import math
import tempfile

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

# from google import genai


# Requests API logic
import os
import requests
from urllib.parse import urlparse
from typing import List, Dict, Optional

GITHUB_API = "https://api.github.com"
TOKEN  = os.getenv("GITHUB_TOKEN")


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

    for path, curr_bytes in curr_files.items():
        prev_bytes = prev_files.get(path)
        
        # if curr_bytes is None or prev_bytes is None:
        #     # skip files added or deleted entirely
        #     continue

        prev_text = prev_bytes.decode("utf-8", errors="ignore") if prev_bytes else ""
        curr_text = curr_bytes.decode("utf-8", errors="ignore") if curr_bytes else ""

        prompt = f"""
        You are given a production-ready source file below. Your task:
        1. **Introduce exactly one subtle, functional bug**—for example an off-by-one in a loop, a wrong comparison operator, an incorrect return value in an edge case, a swapped parameter, or a logic inversion.
        2. **Do not break compilation** and **do not introduce any syntax or spelling errors** or code-style changes.
        3. Preserve formatting and comments; modify only the minimum lines needed to trigger a logical failure under certain inputs.
        4. Return **only** the full modified file content, with no explanations or diff markers.

        --- {path} original content START ---
        {curr_text}
        --- {path} original content END ---
        """

        prompt = f"""
        You are given a production-ready source file below. Your task:
        1. **Introduce exactly two subtle, functional bugs**—for example an off-by-one in a loop, a wrong comparison operator, an incorrect return value in an edge case, a swapped parameter, or a logic inversion.
        2. **Do not break compilation** and **do not introduce any syntax or spelling errors** or code-style changes.
        3. Preserve formatting and comments; modify only the minimum lines needed to trigger a logical failure under certain inputs.
        4. Return **only** the full modified file content, with no explanations or diff markers.

        --- {path} original content START ---
        {curr_text}
        --- {path} original content END ---
        """


        # prompt = f"""
        # Please introduce a functional bug into this source file. Do not change any comments. 
        # Make sure the bug would not cause a syntax error.
        # Return only the full modified file content, without any commentary.

        # --- {path} original content start ---
        # {curr_text}
        # --- {path} original content end ---
        # """

        # prompt = f"""
        # Please introduce a functional bug into this source file.
        # Return only the full modified file content, without any commentary.

        # --- {path} original content start ---
        # {curr_text}
        # --- {path} original content end ---
        # """ # In larger files, can make simple syntax errors

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

        if modified_text.endswith("```"):
            modified_text = modified_text[:-3].rstrip()  # Remove the suffix and trailing spaces
            # Split into lines for diffing
        
        prev_lines     = prev_text.splitlines()
        modified_lines = modified_text.splitlines()

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

    return output_diff_path, diff_str

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
    timeout: int
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

    # TODO: Update code to work with list of instances

    output_d_path = output_dir_path / instance_ids[0] # E.g baselines/badpatchllm/logs/gemini_outputs/camel-ai__camel-1469
    
    output_d_path.mkdir(parents=True, exist_ok=True)
    output_file_path = output_d_path /"badpatch.jsonl"

    basic_args = {
        "model_name_or_path": model_name,
    }

    num_patches_ctr = 0

    max_iter = 10

    for datum in tqdm(dataset, desc=f"Inference for {model_name}"):
        instance_id = datum["instance_id"]
        # if instance_id in existing_ids:
        #     continue
        output_dict = {"instance_id": instance_id, "bad_patches": []}
        output_dict.update(basic_args)

        for i in range(1, max_iter + 1):
            
            # TODO Generate Model Patch Here: 


            url = f"https://github.com/{datum['repo']}/pull/{datum['pull_number']}"
            curr = get_changed_and_current_files(url, TOKEN)
            prev = get_changed_and_previous_files(url, TOKEN)

            
            diff_file, model_patch = generate_incorrect_diff(
                prev_files=prev,
                curr_files=curr,
                gemini_api_key=secret_key,
                model_name=model_name,
                temperature=0.85,
                output_diff_path="my_pr_with_bugs{i}.diff"
            )

            # TODO create loop behavior for this

            if model_patch:

                #  Run Testing Code Directly Here to Ensure Bad Patches Fail

                bad_instance = dict()
                bad_instance['instance_id'] = instance_id
                bad_instance['model_patch'] = model_patch
                bad_instance['model_name_or_path'] = model_name
            
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
                )

                if resolved:
                    continue

                num_patches_ctr +=1 


                output_dict["bad_patches"].append(model_patch)

                diff_file_path = output_d_path / f"patch_{i}.diff"
                try:
                    with open(diff_file_path, "w") as diff_file:
                        diff_file.write(model_patch)
                except Exception as e:
                    print(f"Error writing .diff file for {instance_id}: {e}")

                if num_patches_ctr == num_patches : break


        # Update original dataset in-place
        id_to_instance = {inst["instance_id"]: inst for inst in dataset}

        for datum in dataset:
            instance_id = datum["instance_id"]
            if instance_id in id_to_instance:
                id_to_instance[instance_id]["bad_patches"] = output_dict.get("bad_patches", [])

        updated_path = input_tasks_path.parent / f"updated_{input_tasks_path.name}"
        with open(updated_path, "w") as f_out:
            json.dump(list(id_to_instance.values()), f_out, indent=2)




if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset_name", default="data/codearena_instances.json", help="Name of the dataset")
    parser.add_argument("--instance_ids", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    parser.add_argument("-m", "--model_name", type=str, default="gemini-2.0-flash")
    parser.add_argument("-k", "--api_key", type=str, required=True)
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

    args = parser.parse_args()

    main(
        input_tasks_path = Path(args.dataset_name),
        output_dir_path=Path(args.output_dir),
        model_name=args.model_name,
        instance_ids=args.instance_ids.split(",") if args.instance_ids else None,
        secret_key=args.api_key,
        num_patches=args.num_patches,
        max_workers=args.max_workers,
        force_rebuild=args.force_rebuild,
        cache_level=args.cache_level,
        clean=args.clean,
        open_file_limit=args.open_file_limit,
        run_id=args.run_id,
        timeout=args.timeout
    )