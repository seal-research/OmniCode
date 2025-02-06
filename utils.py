from CodeArenaInstance import CodeArenaInstance
from datasets import Dataset, load_dataset
from typing import cast
import json
from pathlib import Path
# from constants import KEY_INSTANCE_ID
import pandas as pd
import os
import re
from collections import defaultdict
from swebench.harness.constants import (
    NON_TEST_EXTS,
    KEY_INSTANCE_ID
)

def load_swebench_dataset(name="princeton-nlp/SWE-bench", split="test", instance_ids=None, full: bool = False) -> list[CodeArenaInstance]:
    """
    Load SWE-bench dataset from Hugging Face Datasets or local .json/.jsonl file
    """
    # check that all instance IDs are in the dataset
    if instance_ids:
        instance_ids = set(instance_ids)
    # Load from local .json/.jsonl file
    if name.endswith(".json") or name.endswith(".jsonl"):
        dataset = []
        with open(name, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line) if name.endswith(".jsonl") else json.load(f)
                dataset.append(entry)

        dataset_ids = {instance[KEY_INSTANCE_ID] for instance in dataset}
    else:
        # Load from Hugging Face Datasets
        if name.lower() in {"swe-bench", "swebench", "swe_bench"}:
            name = "princeton-nlp/SWE-bench"
        elif name.lower() in {"swe-bench-lite", "swebench-lite", "swe_bench_lite", "swe-bench_lite", "lite"}:
            name = "princeton-nlp/SWE-bench_Lite"
        elif name.lower() in {"swe-bench_verified"}:
            name = "princeton-nlp/SWE-bench_Verified"

        # Load dataset
        dataset = cast(Dataset, load_dataset(name, split=split))
        if full:
            # Rename columns
            if "test_patch" in dataset.column_names:
                dataset = dataset.rename_column("test_patch", "candidate_test_patch")
            if "patch" in dataset.column_names:
                dataset = dataset.rename_column("patch", "gold_patch")
            
            # Insert a new column 'bad_patch' with the default value 0
            dataset = dataset.add_column("bad_patch", [0] * len(dataset))

        # Collect dataset IDs
        dataset_ids = {instance[KEY_INSTANCE_ID] for instance in dataset}

    if instance_ids:
        if instance_ids - dataset_ids:
            raise ValueError(
                (
                    "Some instance IDs not found in dataset!"
                    f"\nMissing IDs:\n{' '.join(instance_ids - dataset_ids)}"
                )
            )
        dataset = [instance for instance in dataset if instance[KEY_INSTANCE_ID] in instance_ids]
    return [cast(CodeArenaInstance, instance) for instance in dataset]

def merge_and_unpack(expected):
    # Handle case where the input is not a list but a single dictionary
    if isinstance(expected, dict):
        expected = [expected]

    # Initialize a defaultdict to store merged results
    merged = defaultdict(list)

    # Iterate through the list of dictionaries and merge the lists
    for entry in expected:
        for key, value in entry.items():
            if isinstance(value, list):
                merged[key].extend(value)

    # Convert defaultdict back to a regular dictionary for clarity
    merged = {key: list(set(value)) for key, value in merged.items()}  # Remove duplicates if needed
    return merged

def load_CodeArena_prediction_dataset(
    generated_tests_path: str,
    codearena_instances: str, 
    instance_ids: list, 
    custom_dataset_path: str = "data/codearena_instances.jsonl",
    save: bool = False
):
    """
    Process and merge a custom dataset with generated tests and bad patches.
    This function will fix the `model_patch` diffs, merge the datasets, and check for missing predictions.
    """
    import json
    import os
    import pandas as pd

    # Determine file type based on extension
    file_ext = os.path.splitext(custom_dataset_path)[-1].lower()

    custom_data = []
    
    with open(custom_dataset_path, 'r') as f:
        if file_ext == ".jsonl":  # Process line by line for JSONL
            for line in f:
                custom_data.append(json.loads(line.strip()))
        elif file_ext == ".json":  # Load entire JSON file
            custom_data = json.load(f)
        else:
            raise ValueError("Unsupported file format. Please provide a .json or .jsonl file.")

    custom_df = pd.DataFrame(custom_data)

    # Load Generated Tests
    generated_tests = []
    with open(generated_tests_path, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            # Fix the `model_patch` if it starts with `---`
            if entry.get('model_patch', '').startswith('---'):
                entry['model_patch'] = entry['model_patch'].replace('---', 'diff --git', 1)
            generated_tests.append(entry)

    generated_tests_df = pd.DataFrame(generated_tests)

    # Check for missing predictions
    custom_ids = set(custom_df['instance_id'])
    generated_tests_ids = set(generated_tests_df['instance_id'])
    
    missing_preds = custom_ids - generated_tests_ids
    if missing_preds:
        print(f"Warning: Missing predictions for {len(missing_preds)} instance IDs: {missing_preds}")

    # Rename `patch` column in Custom Dataset to `gold_patch`
    if 'patch' in custom_df.columns:
        custom_df.rename(columns={'patch': 'gold_patch'}, inplace=True)

    # Merge Custom Dataset with Generated Tests
    merged_df = pd.merge(
        custom_df,
        generated_tests_df[['instance_id', 'model_patch', 'model_name_or_path']],
        on='instance_id',
        how='left'
    )

    # Rename `model_patch` to `candidate_test_patch`
    merged_df.rename(columns={'model_patch': 'candidate_test_patch'}, inplace=True)

    # Load the additional CodeArena Instances JSONL file
    codearena_instances_data = []
    with open(codearena_instances, 'r') as f:
        for line in f:
            codearena_instances_data.append(json.loads(line.strip()))
    codearena_instances_df = pd.DataFrame(codearena_instances_data)

    # Filter rows where `bad_patch` is not empty
    codearena_instances_filtered = codearena_instances_df[
        codearena_instances_df['bad_patch'].notna() & codearena_instances_df['bad_patch'].str.strip().ne("")
    ]

    # Merge with CodeArena Instances
    merged_df = pd.merge(
        merged_df,
        codearena_instances_filtered[['instance_id', 'bad_patch', 'bad_patch_author', 'Review', 'Review_Author']],
        on='instance_id',
        how='left'
    )

    # Extract `model_name_or_path` for naming the output file
    if 'model_name_or_path' in generated_tests_df.columns:
        model_name_or_path = generated_tests_df['model_name_or_path'].iloc[0]
        sanitized_model_name = model_name_or_path.replace("/", "_").replace("\\", "_").replace(" ", "_")
    else:
        raise ValueError("`model_name_or_path` is missing from the generated tests dataset.")

    # Save as CSV if requested
    if save: 
        output_dir = "TestGeneration_Datasets/"
        output_csv_path = f"{output_dir}custom_dataset_merged_{sanitized_model_name}.csv"
        os.makedirs(output_dir, exist_ok=True)

        merged_df.to_csv(output_csv_path, index=False)
        print(f"Output saved as CSV: {output_csv_path}")

    return merged_df


def get_test_directives(instance: CodeArenaInstance) -> list:
    """
    Get test directives from the test_patch of a task instance

    Args:
        instance (dict): task instance
    Returns:
        directives (list): List of test directives
    """
    # For seq2seq code repos, testing command is fixed
    if instance["repo"] == "swe-bench/humaneval":
        return ["test.py"]

    # Get test directives from test patch and remove non-test files
    diff_pat = r"diff --git a/.* b/(.*)"
    test_patch = instance["candidate_test_patch"]
    directives = re.findall(diff_pat, test_patch)
    directives = [
        d for d in directives if not any(d.endswith(ext) for ext in NON_TEST_EXTS)
    ]

    # For Django tests, remove extension + "tests/" prefix and convert slashes to dots (module referencing)
    if instance["repo"] == "django/django":
        directives_transformed = []
        for d in directives:
            d = d[: -len(".py")] if d.endswith(".py") else d
            d = d[len("tests/") :] if d.startswith("tests/") else d
            d = d.replace("/", ".")
            directives_transformed.append(d)
        directives = directives_transformed

    return directives