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


CUR_DIR = Path(__file__).parent
DOTENV_PATH = CUR_DIR / '.env'

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_model_single(
    instance: dict,
    model_name: str,
    secret_key: str,
):
    patch = instance['patch']    

    prompt = f"""
    Output a version of the given .diff file that is functionally incorrect.
    Only output a properly formatted .diff file and nothing else. 
    Do NOT include triple backticks (```) or any other Markdown formatting. 

    Diff File:
    {patch}
    """

    response = None

    genai.configure(api_key=secret_key)


    if(model_name == "gemini-2.0-flash"):
        model = genai.GenerativeModel(model_name)
        # response = model.generate_content(prompt)
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.9, "top_p": 0.9}  # Adjust values to change responses
        )
        model_patch = response.text

        if model_patch.startswith("```diff"):
            model_patch = model_patch[len("```diff"):].lstrip()  # Remove the prefix and leading spaces

        if model_patch.endswith("```"):
            model_patch = model_patch[:-3].rstrip()  # Remove the suffix and trailing spaces
        
        return None, model_patch
    else:
        print('Will add support soon!') # TODO: Add other models pending API keys
        return None, None

        

def main(
    input_tasks_path: Path,
    output_dir_path: Path,
    model_name: str,
    instance_ids: list[str] | None= None,
    secret_key: str | None=None,
    num_patches: int | None=None,
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

    output_d_path = output_dir_path / instance_ids[0]
    
    output_d_path.mkdir(parents=True, exist_ok=True)
    output_file_path = output_d_path /"badpatch.jsonl"

    basic_args = {
        "model_name_or_path": model_name,
    }

    with open(output_file_path, "a+") as f:
        for datum in tqdm(dataset, desc=f"Inference for {model_name}"):
            instance_id = datum["instance_id"]
            # if instance_id in existing_ids:
            #     continue
            output_dict = {"instance_id": instance_id, "bad_patches": []}
            output_dict.update(basic_args)

            for i in range(1, num_patches + 1):
                full_output, model_patch = run_model_single(datum, model_name=model_name, secret_key=secret_key)
                
                if model_patch:
                    output_dict["bad_patches"].append(model_patch)

                    diff_file_path = output_d_path / f"patch_{i}.diff"
                    try:
                        with open(diff_file_path, "w") as diff_file:
                            diff_file.write(model_patch)
                    except Exception as e:
                        print(f"Error writing .diff file for {instance_id}: {e}")

            print(json.dumps(output_dict), file=f, flush=True)

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_tasks", type=str, required=True)
    parser.add_argument("--instance_ids", type=str, required=False, default=None)
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    parser.add_argument("-m", "--model_name", type=str, default="gemini-2.0-flash")
    parser.add_argument("-k", "--api_key", type=str, required=True)
    parser.add_argument("-n", "--num_patches", type=int, required=False, default=5)

    args = parser.parse_args()

    main(
        input_tasks_path=Path(args.input_tasks),
        output_dir_path=Path(args.output_dir),
        model_name=args.model_name,
        instance_ids=args.instance_ids.split(",") if args.instance_ids else None,
        secret_key=args.api_key,
        num_patches=args.num_patches,
    )