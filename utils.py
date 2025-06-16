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
import tempfile
import shutil
import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Node
from unidiff import PatchSet
from git import Repo
from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS

PY_LANGUAGE = Language(tspython.language())

from swebench.harness.constants import (
    NON_TEST_EXTS,
    KEY_INSTANCE_ID
)

import tarfile
from io import BytesIO

def copy_from_container(container, src_path: Path, dest_path: Path):
    """
    Copies a file from the Docker container to the host.

    Args:
        container: Docker container object
        src_path (Path): Path to the file in the container
        dest_path (Path): Path to save the file on the host
    """
    try:
        # Get the file as a tar archive from the container
        bits, _ = container.get_archive(str(src_path))

        # Read the tar stream into a bytes buffer
        tar_stream = BytesIO()
        for chunk in bits:
            tar_stream.write(chunk)
        tar_stream.seek(0)

        # Extract the file from the tar archive
        with tarfile.open(fileobj=tar_stream) as tar:
            # Ensure the destination directory exists
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            # Extract the file
            member = tar.getmembers()[0]  # Assumes single file
            with open(dest_path, "wb") as f:
                f.write(tar.extractfile(member).read())

    except Exception as e:
        raise RuntimeError(f"Failed to copy {src_path} from container: {e}")

def load_swebench_dataset(name="data/codearena_instances.json", split="test", instance_ids=None, full: bool = False) -> list[CodeArenaInstance]:
    """
    Load dataset from local JSON file
    """
    # check that all instance IDs are in the dataset
    if instance_ids:
        instance_ids = set(instance_ids)

    try:
        with open(name, "r", encoding="utf-8") as f:
            dataset = json.load(f)
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

        # Ensure all required fields are present
        for instance in dataset:
            if "model_name_or_path" not in instance:
                instance["model_name_or_path"] = "gold"
            if "bad_patch" not in instance:
                instance["bad_patch"] = 0
            if "candidate_test_patch" not in instance:
                instance["candidate_test_patch"] = instance.get("test_patch", "")
            if "gold_patch" not in instance:
                instance["gold_patch"] = instance.get("patch", "")

        return [cast(CodeArenaInstance, instance) for instance in dataset]
    except Exception as e:
        print(f"Error loading {name}: {e}")
        raise

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
    save: bool = False
):
    """
    Process and merge a custom dataset with generated tests and bad patches.
    This function will fix the `model_patch` diffs, merge the datasets, and check for missing predictions.
    """
    import json
    import os
    import pandas as pd

    # Load Generated Tests
    generated_tests = []
    with open(generated_tests_path, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            if entry['instance_id'] not in instance_ids:
                continue
            # Fix the `model_patch` if it starts with `---`
            if entry.get('model_patch', '').startswith('---'):
                entry['model_patch'] = entry['model_patch'].replace('---', 'diff --git', 1)
            generated_tests.append(entry)

    generated_tests_df = pd.DataFrame(generated_tests)

    # Load the CodeArena Instances file
    with open(codearena_instances, 'r') as f:
        codearena_instances_data = json.load(f)  # Load entire file as JSON

    codearena_instances_df = pd.DataFrame(codearena_instances_data)

    # Filter rows where `bad_patch` is not empty
    patch_label = "bad_patch"
    if "bad_patch" not in codearena_instances_df.columns:
        patch_label = "bad_patches"

    codearena_instances_filtered = codearena_instances_df[
    codearena_instances_df['bad_patches'].notna()
    ].copy()
    # codearena_instances_filtered = codearena_instances_df.copy()

    print(codearena_instances_df.columns)
    # Check for missing predictions
    # codeArena_ids = set(codearena_instances_filtered['instance_id'])
    # generated_tests_ids = set(generated_tests_df['instance_id'])

    # missing_preds = codeArena_ids - generated_tests_ids
    # if missing_preds:
    #     print(f"Warning: Missing predictions for {len(missing_preds)} instance IDs: {missing_preds}")

    # Rename `patch` column to `gold_patch` if needed
    if 'patch' in codearena_instances_filtered.columns:
        codearena_instances_filtered.rename(columns={'patch': 'gold_patch'}, inplace=True)

    # Merge CodeArena Instances with Generated Tests
    merged_df = pd.merge(
        codearena_instances_filtered,
        generated_tests_df[['instance_id', 'model_patch', 'model_name_or_path']],
        on='instance_id',
        how='left'
    )

    # Rename `model_patch` to `candidate_test_patch`
    merged_df.rename(columns={'model_patch': 'candidate_test_patch'}, inplace=True)

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


def get_modified_added_files(patch_string):
    """
    Parse a patch string and return lists of modified and added files.

    Args:
        patch_string (str): String containing the patch/diff content

    Returns:
        tuple: (list of modified files, list of added files)
    """
    # Parse the patch
    patch_set = PatchSet.from_string(patch_string)

    modified_files = []
    added_files = []

    # Iterate through each file in the patch
    for patched_file in patch_set:
        if patched_file.is_added_file:
            added_files.append(patched_file.path)
        elif patched_file.is_modified_file:
            modified_files.append(patched_file.path)

    return modified_files + added_files



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



class DirHandler:
    def __init__(self, dir_path: Path, temp_dir: Path = Path("/tmp")):
        self.dir_path = dir_path
        self.temp_dir = temp_dir

    def __enter__(self):
        """create a temp dir and copy repo into it with tempdir"""
        self.temp_dir = tempfile.mkdtemp()
        shutil.copytree(self.dir_path, self.temp_dir, dirs_exist_ok=True)
        return self.temp_dir

    def __exit__(self, exc_type, exc_value, traceback):
        shutil.rmtree(self.temp_dir)



def git_checkout(repo_path: Path, commit_sha: str) -> Path:
    """
    Checkout a specific commit in a repository
    """
    repo = Repo(repo_path)
    repo.git.checkout(commit_sha)


def get_current_commit(repo_path: Path) -> str:
    """
    Get the current commit of a repository
    """
    repo = Repo(repo_path)
    return repo.head.commit.hexsha


EXTENSION_TO_TS_LANG = {
    ".py": PY_LANGUAGE,
}


def get_identifier_name(node: Node, source_bytes: bytes) -> str | None:
    """
    Given a node (of type function_definition or class_definition)
    find its child whose type is "identifier" and return its text.
    """
    for child in node.children:
        if child.type == "identifier":
            return source_bytes[child.start_byte:child.end_byte].decode("utf8")
    return None



def get_fully_qualified_name(source_code: str, parser: Parser, line_number: int) -> str | None:
    """
    Given the source_code (a string), a tree-sitter parser and a 1-indexed
    line_number, returns the fully qualified name (or None if not found)
    of the function (or method) that the line is part of.
    """

    # Parse the source text. (Note: tree-sitter expects bytes.)
    tree = parser.parse(source_code.encode("utf8"))
    source_bytes = source_code.encode("utf8")
    # Convert external line number (1-indexed) to a point (row, column)
    # figure out leading whitespace to use as column
    line_source_code = source_code.splitlines()[line_number - 1]
    leading_whitespace = len(line_source_code) - len(line_source_code.lstrip())
    target_point = (line_number - 1, leading_whitespace)

    # Find the smallest node that spans our target point.
    node = tree.root_node.descendant_for_point_range(target_point, target_point)
    if node is None:
        return None

    # Walk upward from the node to find any surrounding function or class definitions.
    names = []
    current = node
    while current is not None:
        if current.type in ("function_definition", "class_definition"):
            name = get_identifier_name(current, source_bytes)
            # If found, add it to our list.
            if name:
                names.append(name)
        current = current.parent

    # The names were collected from inner to outer so we reverse them.
    if names:
        # For example, if the node is inside a class and then a method,
        # the fully qualified name becomes "ClassName.function_name".
        return ".".join(reversed(names))
    else:
        return None


def get_function_name(file_path: Path, line_number: int) -> str | None:
    """
    Parse the code file and return the fully qualified name of the function containing the line number
    """
    parser = Parser(EXTENSION_TO_TS_LANG[file_path.suffix])
    source_code = file_path.read_text()
    return get_fully_qualified_name(source_code, parser, line_number)


def get_modified_line_chunks(diff_str: str, repo_path: Path, base_commit: str) -> list[str]:
    """
    Get line number ranges from the original file that were modified in the patch
    """
    with DirHandler(repo_path) as temp_dir:
        git_checkout(temp_dir, base_commit)
        patch_set = PatchSet(diff_str)


def get_modified_functions(diff_str: str, repo_path: Path, base_commit: str) -> list[str]:
    """
    Get the modified functions from a diff string
    """
    with DirHandler(repo_path) as temp_dir:
        git_checkout(temp_dir, base_commit)
        patch_set = PatchSet(diff_str)
        modified_functions = []
        for patched_file in patch_set:
            offset = 1
            for hunk in patched_file:
                for line in hunk:
                    func_name = None
                    if line.is_added:
                        func_name = get_function_name(repo_path / patched_file.path, line.target_line_no - offset)
                        offset += 1
                    if line.is_removed:
                        func_name = get_function_name(repo_path / patched_file.path, line.source_line_no)
                        offset -= 1

                    if func_name is None:
                        continue
                    func_fqn = patched_file.path + "::" + func_name
                    if func_fqn not in modified_functions:
                        modified_functions.append(func_fqn)

        return modified_functions


def update_test_spec_with_specific_test_names(test_spec, repo_path):
    """
    Updates a TestSpec to use specific test function names instead of file paths.

    Args:
        test_spec (TestSpec): The TestSpec to update
        repo_path (Path): Path to the local repository

    Returns:
        TestSpec: The updated TestSpec
    """
    try:
        # Extract necessary information
        base_commit = test_spec.base_commit

        # Extract test patch from the command string in eval_script_list
        test_patch = None
        for cmd in test_spec.eval_script_list:
            if "git apply -v -" in cmd and "EOF_" in cmd:
                # Extract the patch content between the heredoc delimiters
                patch_match = re.search(r"<<'EOF_\d+'\n(.*?)\nEOF_\d+", cmd, re.DOTALL)
                if patch_match:
                    test_patch = patch_match.group(1)
                    break

        if not test_patch:
            print(f"Warning: Could not find test patch in test_spec for {test_spec.instance_id}")
            return test_spec

        # Get modified functions
        modified_functions = get_modified_functions(test_patch, repo_path, base_commit)

        # Filter for test functions
        test_functions = []
        for func_path in modified_functions:
            file_path, func_name = func_path.split(":", 1)

            # Only include test functions
            if "test" in func_name.lower() or "Test" in func_name:
                # Check if it's a test file (not in NON_TEST_EXTS)
                if not any(file_path.endswith(ext) for ext in NON_TEST_EXTS):
                    if test_spec.repo == "django/django":
                        # Apply Django transformations
                        if file_path.endswith(".py"):
                            file_path = file_path[: -len(".py")]
                        if file_path.startswith("tests/"):
                            file_path = file_path[len("tests/"):]
                        file_path = file_path.replace("/", ".")
                        test_functions.append(f"{file_path}.{func_name}")
                    else:
                        # Use pytest format for other repos
                        test_functions.append(f"{file_path}::{func_name}")

        # Only update if we found specific test functions
        if test_functions:
            # Update the test commands in all scripts (eval_script_list, gold_inverted_eval_script_list, bad_inverted_eval_script_list)
            script_lists = [
                test_spec.eval_script_list,
                test_spec.gold_inverted_eval_script_list,
                test_spec.bad_inverted_eval_script_list
            ]

            for script_list in script_lists:
                for i, cmd in enumerate(script_list):
                    # Find the test command
                    test_cmd_base = MAP_REPO_VERSION_TO_SPECS[test_spec.repo][test_spec.version]["test_cmd"]
                    if cmd.startswith(test_cmd_base):
                        # Replace with the new command using specific test functions
                        script_list[i] = f"{test_cmd_base} {' '.join(test_functions)}"

            print(f"Updated test commands for {test_spec.instance_id} with {len(test_functions)} specific test functions")
        else:
            print(f"No test functions found for {test_spec.instance_id}, leaving test commands unchanged")

        return test_spec

    except Exception as e:
        print(f"Error updating test spec for {test_spec.instance_id}: {str(e)}")
        return test_spec  # Return unmodified spec on error
