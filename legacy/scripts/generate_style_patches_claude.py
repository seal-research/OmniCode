import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List

import litellm


MODEL_NAME: str = "openrouter/anthropic/claude-3-7-sonnet"


SYSTEM_TEMPLATE: str = (
    "You are a Python code quality expert assistant that can interact with a computer "
    "to analyze and fix code quality issues.\n"
    "You specialize in Pylint-based code analysis and can interpret Pylint violation "
    "reports to provide targeted fixes.\n\n"
    "Your expertise includes:\n"
    "- Python code quality and best practices (PEP 8, PEP 20, etc.)\n"
    "- Pylint rule interpretation and application\n"
    "- Readability, maintainability, and Pythonic design\n"
    "- Error-prone patterns and bug risk prevention\n"
    "- Performance and resource management in Python\n\n"
    "You work with a repository that has been analyzed by Pylint, and your goal is to "
    "resolve all style and quality violations while preserving the original functionality "
    "of the code."
)


INSTANCE_TEMPLATE: str = (
    "You have recently generated a patch to resolve an issue within this repository.\n"
    "Pylint has been run on the modified files and has produced the following feedback:\n\n"
    "{problem_statement}\n\n"
    "Your task is to:\n"
    "1. Analyze the Pylint violations provided in the problem statement\n"
    "2. Understand the specific rules that were violated (e.g., naming conventions, unused imports, complexity issues)\n"
    "3. Apply fixes that resolve these errors while maintaining code functionality\n"
    "4. Ensure your changes follow Python best practices and improve code readability\n"
    "5. Test that your fixes don't introduce new Pylint violations\n"
    "6. Do not introduce any new files to fix the style errors\n\n"
    "Output ONLY a unified diff patch with proper headers and hunks.\n"
    "- Start with: diff --git a/<path> b/<path>\n"
    "- Use '--- a/<path>' and '+++ b/<path>' and '@@' hunks\n"
    "- Do not include any explanations or markdown code fences.\n"
    "- Edit only the files referenced by the problem statement."
)


def load_chosen_instances(instances_txt: Path) -> List[str]:
    instance_ids: List[str] = []
    with open(instances_txt, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if line:
                instance_ids.append(line)
    return instance_ids


def load_subset(subset_json: Path) -> Dict[str, dict]:
    with open(subset_json, "r", encoding="utf-8") as file:
        data = json.load(file)
    return {item["instance_id"]: item for item in data}


def strip_code_fences(text: str) -> str:
    cleaned: str = text.strip()
    cleaned = re.sub(r"^```(?:diff|patch)?\s*", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)
    return cleaned.strip()


def normalize_to_unified_diff(text: str) -> str:
    content: str = strip_code_fences(text)
    if content.startswith("--- a/") or content.startswith("+++ b/"):
        lines = content.splitlines()
        a_path = next((ln[6:] for ln in lines if ln.startswith("--- a/")), None)
        b_path = next((ln[6:] for ln in lines if ln.startswith("+++ b/")), a_path)
        if a_path and b_path:
            content = f"diff --git a/{a_path} b/{b_path}\n{content}"
    return content


def already_written(out_path: Path, instance_id: str) -> bool:
    if not out_path.exists():
        return False
    with open(out_path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("instance_id") == instance_id:
                return True
    return False


def call_llm(problem_statement: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_TEMPLATE},
        {"role": "user", "content": INSTANCE_TEMPLATE.format(problem_statement=problem_statement)},
    ]
    response = litellm.completion(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.2,
        max_tokens=1500,
        top_p=1.0,
        stream=False,
    )
    content = response.choices[0].message.content or ""
    return normalize_to_unified_diff(content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", required=True, help="Path to chosen instances txt")
    parser.add_argument("--dataset", required=True, help="Path to chosen subset JSON")
    parser.add_argument("--out", required=True, help="Output predictions JSONL")
    parser.add_argument(
        "--rate_limit_sleep", type=float, default=0.5, help="Seconds to sleep between LLM calls"
    )
    args = parser.parse_args()

    # OPENROUTER_API_KEY expected to be set in the environment
    if not os.getenv("OPENROUTER_API_KEY"):
        raise SystemExit("Please set OPENROUTER_API_KEY environment variable")

    chosen_ids = load_chosen_instances(Path(args.instances))
    subset = load_subset(Path(args.dataset))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(out_path, "a", encoding="utf-8") as fout:
        for instance_id in chosen_ids:
            if instance_id not in subset:
                print(f"Skip (not in subset): {instance_id}")
                continue

            if already_written(out_path, instance_id):
                print(f"Skip (exists): {instance_id}")
                continue

            problem_statement: str = subset[instance_id].get("problem_statement") or ""
            if not problem_statement.strip():
                print(f"Skip (no problem_statement): {instance_id}")
                continue

            patch: str = ""
            for attempt in range(5):
                try:
                    patch = call_llm(problem_statement)
                    break
                except Exception as error:
                    backoff_seconds = min(60, 2 ** attempt)
                    print(f"Error on {instance_id}: {error} (retry in {backoff_seconds}s)")
                    time.sleep(backoff_seconds)
            else:
                print(f"Failed after retries: {instance_id}")
                continue

            record = {
                "instance_id": instance_id,
                "model_name_or_path": MODEL_NAME,
                "model_patch": patch,
                "full_output": None,
                "patch": patch,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            written += 1
            time.sleep(args.rate_limit_sleep)

    print(f"Wrote {written} predictions to {out_path}")


if __name__ == "__main__":
    main()


