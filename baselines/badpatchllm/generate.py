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

CUR_DIR = Path(__file__).parent
DOTENV_PATH = CUR_DIR / '.env'

SECRET_KEY = "" # Insert Personal

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

import random
import re
from typing import Optional


from openai import OpenAI
from typing import Optional

def generate_bad_patch_from_diff(
    diff_text: str,
    api_key: str,
    model_name: str = "gpt-4o",
    temperature: float = 0.8,
    max_tokens: Optional[int] = None,
    output_path: str = "bad_patch.diff",
) -> str:
    """
    Inject functional bugs into a complete .diff using the new openai>=1.0.0 interface.
    """
    # 1) Instantiate the client
    client = OpenAI(api_key=api_key)

    # 2) Build your messages
    system_msg = {
        "role": "system",
        "content": "You are an expert assistant that injects subtle functional bugs into code diffs."
    }
    user_msg = {
        "role": "user",
        "content": (
            "Please introduce a functional bug into the following unified diff.\n"
            "Only output the complete, modified diff (including all '@@' headers, context lines,\n"
            "and properly-prefixed '+' and '-' lines), and nothing else:\n\n"
            f"{diff_text}"
        )
    }

    # 3) Call the chat completion endpoint on your client
    completion = client.chat.completions.create(
        model=model_name,
        messages=[system_msg, user_msg],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # 4) Return the assistant’s diff
    bad_diff = completion.choices[0].message.content

 # 5) Save to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(bad_diff)

    return bad_diff



def run_model_single(
    instance: dict,
    model_name: str,
    secret_key: str,
    min_additions: int = 4,      # minimum number of '+' lines per chunk
    max_addition_length: Optional[int] = None,
):
    full_patch = instance['patch']
    lines = full_patch.splitlines()

    # 1. Identify hunk boundaries
    hunk_indices = [i for i, l in enumerate(lines) if l.startswith('@@ ')]
    hunk_indices.append(len(lines))  # sentinel end

    # 2. Extract hunks as lists of lines
    hunks = []
    for start, end in zip(hunk_indices, hunk_indices[1:]):
        hunks.append(lines[start:end])

    # 3. Helper to count addition lines in a hunk
    def count_adds(hunk):
        return len([l for l in hunk if l.startswith('+') and not l.startswith('+++')])

    # 4. Filter hunks by min and max addition count
    qualifying = [
        hunk for hunk in hunks
        if count_adds(hunk) >= min_additions
           and (max_addition_length is None or count_adds(hunk) <= max_addition_length)
    ]

    # 5. Fallback: if no hunks in the [min…max] range, try any hunk with ≥min additions
    if not qualifying:
        fallback = [h for h in hunks if count_adds(h) >= min_additions]
        chosen_hunk = random.choice(fallback) if fallback else hunks[0]
    else:
        chosen_hunk = random.choice(qualifying)

    # # 3. Filter hunks by addition-line count
    # qualifying = []
    # for hunk in hunks:
    #     adds = [l for l in hunk if l.startswith('+') and not l.startswith('+++')]
    #     if len(adds) >= min_additions:
    #         qualifying.append((hunk, adds))

    # # 4. If no chunk qualifies, fall back to entire patch
    # if not qualifying:
    #     chosen_hunk, additions = hunks[0], [
    #         l for l in lines if l.startswith('+') and not l.startswith('+++')
    #     ]
    # else:
    #     chosen_hunk, additions = random.choice(qualifying)

    # capture both full hunk context and just the added lines
    hunk_block = "\n".join(chosen_hunk)

    # 5. Build prompt around the full hunk, but only ask to modify the +‑lines
    prompt = f"""
        Change the addition lines in the diff below so that you introduce a functional bug.
        Here is the diff you should work with (this includes context, removals, and additions):
        {hunk_block}

        Only output the diff with no other formatting characters.
    """

    # return hunk_block, hunk_block

    # 6. Call Gemini
    genai.configure(api_key=secret_key)
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.9, "top_p": 0.9}
    )

    model_patch = response.text

    if model_patch.startswith("```diff"):
        model_patch = model_patch[len("```diff"):].lstrip()  # Remove the prefix and leading spaces

    if model_patch.endswith("```"):
        model_patch = model_patch[:-3].rstrip()  # Remove the suffix and trailing spaces

    # return model_patch, model_patch

    model_patch = model_patch.splitlines()

    new_hunk = ""

    for idx, line in enumerate(model_patch):
        if line.startswith('@@ '):
            new_hunk = model_patch[idx:]
            break
    else:
        # if no "@@ " found, fall back to everything
        new_hunk = model_patch


    # 8. Reconstruct the full patch, replacing the entire chosen hunk
    new_lines = []
    in_target_hunk = False
    header = chosen_hunk[0] if chosen_hunk and chosen_hunk[0].startswith('@@ ') else None

    for l in lines:
        if l == header:
            # start of the hunk to replace
            in_target_hunk = True
            new_lines.extend(new_hunk)
            continue

        if in_target_hunk:
            # skip all original hunk lines until the next hunk header
            if l.startswith('@@ '):
                in_target_hunk = False
                new_lines.append(l)
            # otherwise just drop this line
            continue

        # outside the target hunk, keep the line
        new_lines.append(l)

    # 9. Return patched text
    return None, "\n".join(new_lines) + "\n"


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

    with open(output_file_path, "a+") as f:
        for datum in tqdm(dataset, desc=f"Inference for {model_name}"):
            instance_id = datum["instance_id"]
            # if instance_id in existing_ids:
            #     continue
            output_dict = {"instance_id": instance_id, "bad_patches": []}
            output_dict.update(basic_args)

            for i in range(1, max_iter + 1):
                full_output, model_patch = run_model_single(datum, model_name=model_name, secret_key=secret_key, max_addition_length=70)
                
                openai_patch = generate_bad_patch_from_diff(datum["patch"],api_key=SECRET_KEY)
                # print("OpenAI Patch: ", openai_patch)

                if model_patch:

                    #  Run Testing Code Directly Here to Ensure Bad Patches Fail

                    bad_instance = dict()
                    bad_instance['instance_id'] = instance_id
                    bad_instance['model_patch'] = model_patch
                    bad_instance['model_name_or_path'] = model_name
                
                    resolved = check_patch(
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

            print(json.dumps(output_dict), file=f, flush=True)

            # Update original dataset in-place
            id_to_instance = {inst["instance_id"]: inst for inst in dataset}

            for datum in dataset:
                instance_id = datum["instance_id"]
                if instance_id in id_to_instance:
                    id_to_instance[instance_id]["bad_patches"] = output_dict.get("bad_patches", [])

            updated_path = input_tasks_path.parent / f"updated_{input_tasks_path.name}"
            with open(updated_path, "w") as f_out:
                json.dump(list(id_to_instance.values()), f_out, indent=2)

    
    # TODO Change Codearena instances.json / -d (from input_tasks_path)


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


# def run_model_single(
#     instance: dict,
#     model_name: str,
#     secret_key: str,
# ):
#     full_patch = instance['patch']
#     lines = full_patch.splitlines()
#     # Extract the first addition line (skip '+++ ' headers)
#     addition_lines = [l for l in lines if l.startswith('+') and not l.startswith('+++')]
#     if not addition_lines:
#         # No additions found, return original patch with newline
#         return None, full_patch + "\n"
#     orig_add = addition_lines[0]

#     # Prompt to modify only this addition line
#     prompt = f"""
#     Change the following diff addition line to introduce a functional bug. Only output the modified addition line.
#     Original addition line:
#     {orig_add}
# """

#     genai.configure(api_key=secret_key)
#     if model_name == "gemini-2.0-flash":
#         model = genai.GenerativeModel(model_name)
#         response = model.generate_content(
#             prompt,
#             generation_config={"temperature": 0.9, "top_p": 0.9}
#         )
#         new_add = response.text.strip()
#         # Remove any markdown backticks
#         new_add = new_add.strip('`').strip()
#         # Ensure it starts with '+'
#         if not new_add.startswith('+'):
#             new_add = '+' + new_add.lstrip('+').strip()

#         # Reconstruct full patch with the new addition
#         new_lines = []
#         replaced = False
#         for l in lines:
#             if l == orig_add and not replaced:
#                 new_lines.append(new_add)
#                 replaced = True
#             else:
#                 new_lines.append(l)
#         model_patch = '\n'.join(new_lines) + '\n'
#         return None, model_patch
#     else:
#         print('Will add support soon!')
#         return None, None



# def run_model_single(
#     instance: dict,
#     model_name: str,
#     secret_key: str,
# ):
#     patch = instance['patch']    

#     prompt = f"""
#     Output a version of the given .diff file that is functionally incorrect. Focus on harder bugs.
#     Only change the "+" lines and do NOT change any other lines.
#     Only output a properly formatted .diff file and nothing else. 
#     Do NOT include triple backticks (```) or any other Markdown formatting. 

#     Diff File:
#     {patch}
#     """

#     prompt = f"""
#     Output a version of the given .diff file that is functionally incorrect. Focus on harder bugs.
#     Do this by changing one addition in the given .diff file and nothing else.

#     Only output a properly formatted .diff file and nothing else. 
#     Do NOT include triple backticks (```) or any other Markdown formatting. 

#     Diff File:
#     {patch}
#     """


#     prompt = f"""
#     Change one addition in the given .diff file and output the new line with its line number.
#     ONLY output the new line and its line number.

#     Diff File:
#     {patch}
#     """

#     prompt = f"""
#     Output a version of the given .diff file that is functionally incorrect.
#     Keep the file identical but only change one addition in the given .diff file and nothing else.

#     Only output this properly formatted .diff file and nothing else. 
#     Do NOT include triple backticks (```) or any other Markdown formatting. 

#     Diff File:
#     {patch}
#     """

#     # Above, testing various prompts

#     # TODO Parse patch, unidiff, can get hunks

#     response = None

#     genai.configure(api_key=secret_key)


#     if(model_name == "gemini-2.0-flash"):
#         model = genai.GenerativeModel(model_name)
#         # response = model.generate_content(prompt)
#         response = model.generate_content(
#             prompt,
#             generation_config={"temperature": 0.9, "top_p": 0.9}  # Adjust values to change responses
#         )
#         model_patch = response.text

        # if model_patch.startswith("```diff"):
        #     model_patch = model_patch[len("```diff"):].lstrip()  # Remove the prefix and leading spaces

        # if model_patch.endswith("```"):
        #     model_patch = model_patch[:-3].rstrip()  # Remove the suffix and trailing spaces
        
#         model_patch += "\n"
    
#         return None, model_patch
#     else:
#         print('Will add support soon!') # TODO: Add other models pending API keys
#         return None, None