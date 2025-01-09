from pathlib import Path
import json
import logging

from tqdm import tqdm
from datasets import load_dataset, load_from_disk
import openai
from jinja2 import Template
from dotenv import load_dotenv

load_dotenv()

from swebench.inference.make_datasets.utils import extract_diff

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CUR_DIR = Path(__file__).parent

PROMPT_TEMPLATE= Template((CUR_DIR / "baseline_testgen_template.j2").read_text())

SYSTEM_MESSAGE = "You are an expert developer."


def process_instance(
        instance_data,
        model_name: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> tuple[str, str]:

    issue_desc = instance_data['problem_statement']
    prompt = PROMPT_TEMPLATE.render(issue=issue_desc)

    response = openai.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        top_p=top_p,
    )

    completion = response.choices[0].message.content
    patch = extract_diff(completion)

    return completion, patch


def main(
    input_tasks_path: Path,
    output_file_path: Path,
    model_name: str,
):
    if input_tasks_path.exists():
        dataset = load_from_disk(input_tasks_path)
    else:
        dataset = load_dataset(str(input_tasks_path))

    dataset = dataset['test']

    existing_ids = set()
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
            full_output, model_patch = process_instance(datum, model_name=model_name)
            output_dict["full_output"] = full_output
            output_dict["model_patch"] = model_patch
            print(json.dumps(output_dict), file=f, flush=True)

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_tasks", type=str, required=True)
    parser.add_argument("-o", "--output_file", type=str, required=True)
    parser.add_argument("-m", "--model_name", type=str, default="gpt-4o")
    args = parser.parse_args()

    main(
        input_tasks_path=Path(args.input_tasks),
        output_file_path=Path(args.output_file),
        model_name=args.model_name,
    )


