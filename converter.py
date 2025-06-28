import json
import re

def extract_test_statuses(fixed_tests):
    pass_to_pass, fail_to_pass, bad_patches = [], [], []
    for test_name, status in fixed_tests.items():
        run = status.get("run", "")
        fix = status.get("fix", "")
        if run == "PASS" and fix == "PASS":
            pass_to_pass.append(test_name)
        elif run != "PASS" and fix == "PASS":
            fail_to_pass.append(test_name)
        elif fix != "PASS":
            bad_patches.append(test_name)
    return pass_to_pass, fail_to_pass, bad_patches

def clean_and_convert_dataset(input_path, clean_output_path, final_output_path):
    # Step 1: Load and split manually if necessary
    with open(input_path, 'r', encoding='utf-8') as infile:
        raw = infile.read().strip()

    # Attempt to parse as array
    try:
        json_objects = json.loads(raw)
        if isinstance(json_objects, dict):
            json_objects = [json_objects]
    except json.JSONDecodeError:
        # Split using regex between curly braces
        candidates = re.findall(r'\{.*?\}(?=,|\s*$)', raw, flags=re.DOTALL)
        json_objects = []
        for idx, item in enumerate(candidates):
            try:
                json_objects.append(json.loads(item))
            except json.JSONDecodeError as e:
                print(f"⚠️ Skipping invalid object at index {idx}: {e}")

    # Step 2: Write cleaned JSONL
    with open(clean_output_path, 'w', encoding='utf-8') as outfile:
        for obj in json_objects:
            outfile.write(json.dumps(obj, ensure_ascii=False) + '\n')

    # Step 3: Transform to CodeArena format
    with open(clean_output_path, 'r', encoding='utf-8') as infile, \
         open(final_output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line)

            org = data.get("org", "").strip()
            repo = data.get("repo", "").strip()
            number = data.get("number")
            base_commit = data.get("base", {}).get("sha", "")

            fix_patch = data.get("fix_patch", "").strip()
            test_patch = data.get("test_patch", "").strip()
            title = data.get("title", "").strip()
            body = data.get("body", "").strip()
            problem_statement = f"{title}\n{body}".strip()

            fixed_tests = data.get("fixed_tests", {})
            pass_to_pass, fail_to_pass, bad_patches = extract_test_statuses(fixed_tests)

            entry = {
                "repo": f"{org}/{repo}",
                "pull_number": number,
                "instance_id": f"{org}__{repo}_{number}",
                "issue_numbers": [],
                "base_commit": base_commit,
                "patch": fix_patch,
                "test_patch": test_patch,
                "problem_statement": problem_statement,
                "hints_text": "",
                "created_at": "",
                "version": "",
                "PASS_TO_PASS": pass_to_pass,
                "FAIL_TO_PASS": fail_to_pass,
                "bad_patches": bad_patches
            }

            outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')

# 🚀 RUN IT
clean_and_convert_dataset(
    input_path='./multiswebench/data/datasets/dataset.jsonl',
    clean_output_path='./multiswebench/data/datasets/dataset_clean.jsonl',
    final_output_path='codearena_dataset.jsonl'
)
