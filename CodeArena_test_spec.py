from __future__ import annotations

import hashlib
import json
import platform
import re

from dataclasses import dataclass
from typing import Any, Union, cast

from CodeArenaInstance import CodeArenaInstance

from swebench.harness.constants import (
    KEY_INSTANCE_ID,
    FAIL_TO_PASS,
    PASS_TO_PASS,
    MAP_REPO_TO_INSTALL,
    MAP_REPO_VERSION_TO_SPECS,
    USE_X86,
)
from swebench.harness.dockerfiles import (
    get_dockerfile_base,
    get_dockerfile_env,
    get_dockerfile_instance,
)
from swebench.harness.utils import (
    get_requirements,
    get_environment_yml
)

from utils import (
    get_test_directives
)

DIFF_MODIFIED_FILE_REGEX = r"--- a/(.*)"


@dataclass
class TestSpec:
    """
    A dataclass that represents a test specification for a single instance of SWE-bench.
    """
    instance_id: str
    repo: str
    base_commit :str
    version: str
    repo_script_list: list[str]
    eval_script_list: list[str]
    gold_inverted_eval_script_list : list[str]
    bad_inverted_eval_script_list : list[str]
    env_script_list: list[str]
    arch: str

    @property
    def setup_env_script(self):
        return "\n".join(["#!/bin/bash", "set -euxo pipefail"] + self.env_script_list) + "\n"

    @property
    def eval_script(self):
        return "\n".join(["#!/bin/bash", "set -uxo pipefail"] + self.eval_script_list) + "\n"
        # Don't exit early because we need to revert tests at the end

    @property
    def inverted_eval_script_gold(self):
        return "\n".join(["#!/bin/bash", "set -uxo pipefail"] + self.gold_inverted_eval_script_list) + "\n"
        # Don't exit early because we need to revert tests at the end

    @property
    def inverted_eval_script_bad(self):
        return "\n".join(["#!/bin/bash", "set -uxo pipefail"] + self.bad_inverted_eval_script_list) + "\n"
        # Don't exit early because we need to revert tests at the end

    @property
    def install_repo_script(self):
        return "\n".join(["#!/bin/bash", "set -euxo pipefail"] + self.repo_script_list) + "\n"

    @property
    def base_image_key(self):
        return f"sweb.base.{self.arch}:latest"

    @property
    def env_image_key(self):
        """
        The key for the environment image is based on the hash of the environment script list.
        If the environment script list changes, the image will be rebuilt automatically.

        Note that old images are not automatically deleted, so consider cleaning up old images periodically.
        """
        hash_object = hashlib.sha256()
        hash_object.update(str(self.env_script_list).encode("utf-8"))
        hash_value = hash_object.hexdigest()
        val = hash_value[:22]  # 22 characters is still very likely to be unique
        return f"sweb.env.{self.arch}.{val}:latest"

    @property
    def instance_image_key(self):
        return f"sweb.eval.{self.arch}.{self.instance_id}:latest"

    def get_instance_container_name(self, run_id=None):
        if not run_id:
            return f"sweb.eval.{self.instance_id}"
        return f"sweb.eval.{self.instance_id}.{run_id}"

    @property
    def base_dockerfile(self):
        return get_dockerfile_base(self.platform, self.arch)

    @property
    def env_dockerfile(self):
        return get_dockerfile_env(self.platform, self.arch)

    @property
    def instance_dockerfile(self):
        return get_dockerfile_instance(self.platform, self.env_image_key)

    @property
    def platform(self):
        if self.arch == "x86_64":
            return "linux/x86_64"
        elif self.arch == "arm64":
            return "linux/arm64/v8"
        else:
            raise ValueError(f"Invalid architecture: {self.arch}")


def get_test_specs_from_dataset(dataset: Union[list[CodeArenaInstance], list[TestSpec]]) -> list[TestSpec]:
    """
    Idempotent function that converts a list of SWEbenchInstance objects to a list of TestSpec objects.
    """
    if isinstance(dataset[0], TestSpec):
        return cast(list[TestSpec], dataset)
    return list(map(make_test_spec, cast(list[CodeArenaInstance], dataset)))


def make_repo_script_list(specs, repo, repo_directory, base_commit, env_name):
    """
    Create a list of bash commands to set up the repository for testing.
    This is the setup script for the instance image.
    """
    setup_commands = [
        f"git clone -o origin https://github.com/{repo} {repo_directory}",
        f"chmod -R 777 {repo_directory}",  # So nonroot user can run tests
        f"cd {repo_directory}",
        f"git reset --hard {base_commit}",
        # Remove the remote so the agent won't see newer commits.
        "git remote remove origin",
        # Make sure conda is available for later use
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
        'echo "Current environment: $CONDA_DEFAULT_ENV"',
    ]
    if repo in MAP_REPO_TO_INSTALL:
        setup_commands.append(MAP_REPO_TO_INSTALL[repo])

    # Run pre-install set up if provided
    if "pre_install" in specs:
        for pre_install in specs["pre_install"]:
            setup_commands.append(pre_install)

    if "install" in specs:
        setup_commands.append(specs["install"])
    return setup_commands


def replace_uninstallable_packages_requirements_txt(requirement_str: str) -> str:
    """Replaces certain packages in a requirements.txt-like string.
    For example, some packages have been yanked and we need to replace them with compatible alternatives.
    """
    replacements = {
        # See https://github.com/princeton-nlp/SWE-bench/issues/199
        # This package was sinced yanked, so we need to force pip
        # to install it.
        "types-pkg_resources": "types-pkg-resources==0.1.3",
    }
    requirements = [req.strip() for req in requirement_str.split("\n") if req.strip()]
    requirements_replaced = []
    for requirement in requirements:
        if requirement in replacements:
            print(f"Replaced {requirement!r} with {replacements[requirement]!r} (replace_uninstallable_packages)")
            requirements_replaced.append(replacements[requirement])
        else:
            requirements_replaced.append(requirement)
    return "\n".join(requirements_replaced) + "\n"


def make_env_script_list(instance: CodeArenaInstance, specs: dict, env_name: str) -> list[str]:
    """
    Creates the list of commands to set up the conda environment for testing.
    This is the setup script for the environment image.

    Returns:
        list[str]: List of commands to set up the conda environment
    """
    HEREDOC_DELIMITER = "EOF_59812759871"
    reqs_commands = [
        "source /opt/miniconda3/bin/activate",
    ]
    # Create conda environment according to install instructinos
    pkgs = specs.get("packages", "")
    if pkgs == "requirements.txt":
        # Create environment
        cmd = f"conda create -n {env_name} python={specs['python']} -y"
        reqs_commands.append(cmd)

        # Install dependencies
        reqs = replace_uninstallable_packages_requirements_txt(get_requirements(instance))
        path_to_reqs = "$HOME/requirements.txt"
        reqs_commands.append(
            f"cat <<'{HEREDOC_DELIMITER}' > {path_to_reqs}\n{reqs}\n{HEREDOC_DELIMITER}"
        )
        cmd = f"conda activate {env_name} && python -m pip install -r {path_to_reqs}"
        reqs_commands.append(cmd)
        reqs_commands.append(f"rm {path_to_reqs}")
    elif pkgs == "environment.yml":
        # Create environment from yml
        reqs = get_environment_yml(instance, env_name)
        path_to_reqs = "environment.yml"
        reqs_commands.append(
            f"cat <<'{HEREDOC_DELIMITER}' > {path_to_reqs}\n{reqs}\n{HEREDOC_DELIMITER}"
        )
        if "no_use_env" in specs and specs["no_use_env"]:
            # `conda create` based installation
            cmd = f"conda create -c conda-forge -n {env_name} python={specs['python']} -y"
            reqs_commands.append(cmd)

            # Install dependencies
            cmd = f"conda env update -f {path_to_reqs}"
            reqs_commands.append(cmd)
        else:
            # `conda env create` based installation
            cmd = f"conda env create --file {path_to_reqs}"
            reqs_commands.append(cmd)

            cmd = f"conda activate {env_name} && conda install python={specs['python']} -y"
            reqs_commands.append(cmd)

        # Remove environment.yml
        reqs_commands.append(f"rm {path_to_reqs}")
    else:
        # Create environment + install dependencies
        cmd = f"conda create -n {env_name} python={specs['python']} {pkgs} -y"
        reqs_commands.append(cmd)

    reqs_commands.append(f"conda activate {env_name}")

    # Install additional packages if specified
    if "pip_packages" in specs:
        pip_packages = " ".join(specs["pip_packages"])
        cmd = f"python -m pip install {pip_packages}"
        reqs_commands.append(cmd)
    # External pylint dependency to run Style Review inside the docker environment

    reqs_commands.append("python -m pip install pylint")
    return reqs_commands


def make_eval_script_list(instance, specs, env_name, repo_directory, base_commit, test_patch):
    """
    Applies the test patch and runs the tests.
    """
    HEREDOC_DELIMITER = "EOF_114329324912"
    test_files = re.findall(DIFF_MODIFIED_FILE_REGEX, test_patch)
    # Reset test files to the state they should be in before the patch.
    reset_tests_command = f"git checkout {base_commit} {' '.join(test_files)}"
    if(type(test_patch) == str): # check for empty bad patches. In this case do not apply change (defaults to base commit as bad sample)
        apply_test_patch_command = (
            f"git apply -v - <<'{HEREDOC_DELIMITER}'\n{test_patch}\n{HEREDOC_DELIMITER}"
        )
    test_command = " ".join(
        [
            MAP_REPO_VERSION_TO_SPECS[instance["repo"]][instance["version"]]["test_cmd"],
            *get_test_directives(instance),
        ]
    )
    eval_commands = [
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
        f"cd {repo_directory}",
    ]
    if "eval_commands" in specs:
        eval_commands += specs["eval_commands"]
    eval_commands += [
        f"git config --global --add safe.directory {repo_directory}",  # for nonroot user
        f"cd {repo_directory}",
        # This is just informational, so we have a record
        "git status",
        "git show",
        f"git diff {base_commit}",
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
    ]
    if "install" in specs:
        eval_commands.append(specs["install"])
    eval_commands += [
        reset_tests_command,
        apply_test_patch_command,
        test_command,
        reset_tests_command,  # Revert tests after done, leave the repo in the same state as before
    ]
    return eval_commands

def make_inverted_eval_script_list(instance, specs, env_name, repo_directory, base_commit, issue_patch, test_patch):
    """
    Applies the given issue patch (gold or bad) and runs the candidate tests.
    """
    HEREDOC_DELIMITER = "EOF_114329324912"
    test_files = re.findall(DIFF_MODIFIED_FILE_REGEX, test_patch)
    # Reset all files *except* test files to the state they should be in before the patch.
    reset_command = f"git stash push -- {' '.join(test_files)} && git checkout {base_commit} -- . && git stash pop"

    apply_issue_patch_command = (
        f"git apply -v - <<'{HEREDOC_DELIMITER}'\n{issue_patch}\n{HEREDOC_DELIMITER}"
    )
    test_command = " ".join(
        [
            MAP_REPO_VERSION_TO_SPECS[instance["repo"]][instance["version"]]["test_cmd"],
            *get_test_directives(instance),
        ]
    )
    eval_commands = [
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
        f"cd {repo_directory}",
    ]
    if "eval_commands" in specs:
        eval_commands += specs["eval_commands"]
    eval_commands += [
        f"git config --global --add safe.directory {repo_directory}",  # for nonroot user
        f"cd {repo_directory}",
        # This is just informational, so we have a record
        "git status",
        "git show",
        f"git diff {base_commit}",
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
    ]
    if "install" in specs:
        eval_commands.append(specs["install"])
    eval_commands += [
        reset_command,
        apply_issue_patch_command,
        test_command,
        reset_command,  # Revert issue patch after done, leave the repo in the same state as before (maintaining test patch)
    ]
    return eval_commands

def generate_patch_lint_script(repo_directory, base_commit, patch, pylint_output_path, error_output_path, env_name):
    """
    Generate a shell script to run pylint evaluation on modified Python files.
    
    Args:
        repo_directory (str): Path to the git repository
        base_commit (str): Base commit to compare against
        patch (str): Git patch to apply
        pylint_output_path (str): Path to save aggregated pylint results
        error_output_path (str): Path to save detailed error messages
        env_name (str): Conda environment name
    
    Returns:
        list: Shell commands to execute the evaluation
    """
    HEREDOC_DELIMITER = "EOF_114329324912"
    reset_command = f"git stash push --include-untracked && git checkout {base_commit} -- . && git stash pop --index"
    apply_patch_command = f"git apply -v - <<'{HEREDOC_DELIMITER}'\n{patch}\n{HEREDOC_DELIMITER}"

    # Define the entire script as a raw string with proper substitution
    script = rf'''
    # Ensure required tools are available
    for cmd in jq awk pylint; do
        if ! command -v "$cmd" &>/dev/null; then
            echo "$cmd is not installed, exiting." >&2
            exit 1
        fi
    done

    # Create temporary directory for intermediate files
    temp_dir=$(mktemp -d)
    trap 'rm -rf "$temp_dir"' EXIT

    # Get modified Python files
    modified_files=$(git diff --name-only HEAD)
    python_files=$(echo "$modified_files" | grep -E '\.pyx?$' || true)

    if [ -z "$python_files" ]; then
        echo '{{"global_score": 10.0, "total_errors": 0, "total_warnings": 0, "total_conventions": 0}}' > {pylint_output_path}
        echo "[]" > {error_output_path}
        exit 0
    fi

    # Initialize error report
    echo "[]" > {error_output_path}

    # Process each file
    while IFS= read -r file; do
        [ -z "$file" ] && continue
        [ ! -f "$file" ] && continue

        # Run pylint and capture outputs
        pylint "$file" --output-format=json > "$temp_dir/pylint.json" 2>/dev/null || echo "[]" > "$temp_dir/pylint.json"
        pylint "$file" > "$temp_dir/pylint.txt" 2>&1 || true

        # Extract score
        file_score=$(awk '/Your code has been rated at/ {{print $7}}' "$temp_dir/pylint.txt" | sed 's/[^0-9.]//g')
        if [ -z "$file_score" ] || ! [[ "$file_score" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
            file_score="0.0"
        fi

        # Process file results
        jq -n --arg file "$file" \
              --arg score "$file_score" \
              --slurpfile messages "$temp_dir/pylint.json" \
              '{{
                "file": $file,
                "score": ($score | tonumber),
                "messages": $messages[0],
                "error_count": ($messages[0] | map(select(.type=="error")) | length),
                "warning_count": ($messages[0] | map(select(.type=="warning")) | length),
                "convention_count": ($messages[0] | map(select(.type=="convention")) | length)
              }}' > "$temp_dir/file_report.json"

        # Append to main report
        jq -s '.[0] + [.[1]]' {error_output_path} "$temp_dir/file_report.json" > "$temp_dir/new_report.json"
        mv "$temp_dir/new_report.json" {error_output_path}

    done <<< "$python_files"

    # Generate final summary
    jq -r 'reduce .[] as $file (
        {{"global_score": 0, "total_errors": 0, "total_warnings": 0, "total_conventions": 0, "count": 0}};
        {{
            "global_score": (.global_score + ($file.score)),
            "total_errors": (.total_errors + $file.error_count),
            "total_warnings": (.total_warnings + $file.warning_count),
            "total_conventions": (.total_conventions + $file.convention_count),
            "count": (.count + 1)
        }}
    ) | {{
        "global_score": (if .count > 0 then (.global_score / .count) else 10.0 end),
        "total_errors": .total_errors,
        "total_warnings": .total_warnings,
        "total_conventions": .total_conventions
    }}' {error_output_path} > {pylint_output_path}
    '''

    eval_commands = [
        "set -ex",
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
        f"cd {repo_directory}",
        f"git config --global --add safe.directory {repo_directory}",
        reset_command,
        apply_patch_command,
        script.strip(),
        reset_command
    ]
    return eval_commands



def make_test_spec(instance: CodeArenaInstance) -> TestSpec:
    if isinstance(instance, TestSpec):
        return instance
    instance_id = instance[KEY_INSTANCE_ID]
    repo = instance["repo"]
    version = instance["version"]
    base_commit = instance["base_commit"]
    test_patch = instance["candidate_test_patch"]
    gold_issue_patch = instance["gold_patch"]
    bad_issue_patch = instance["bad_patch"]

    def _from_json_or_obj(key: str) -> Any:
        """If key points to string, load with json"""
        if isinstance(instance[key], str):
            return json.loads(instance[key])
        return instance[key]

    env_name = "testbed"
    repo_directory = f"/{env_name}"
    specs = MAP_REPO_VERSION_TO_SPECS[repo][version]

    repo_script_list = make_repo_script_list(specs, repo, repo_directory, base_commit, env_name)
    env_script_list = make_env_script_list(instance, specs, env_name)
    eval_script_list = make_eval_script_list(
        instance, specs, env_name, repo_directory, base_commit, test_patch
    )
    # Inverted Evaluation scripts keep the (candidate) test files consistent and apply (and revert after evaluation) the given issue patch
    inverted_eval_script_list_gold = make_inverted_eval_script_list(
        instance, specs, env_name, repo_directory, base_commit, gold_issue_patch, test_patch)
    inverted_eval_script_list_bad = make_inverted_eval_script_list(
        instance, specs, env_name, repo_directory, base_commit, bad_issue_patch, test_patch)
    if platform.machine() in {"aarch64", "arm64"}:
        # use arm64 unless explicitly specified
        arch = "arm64" if instance_id not in USE_X86 else "x86_64"
    else:
        arch = "x86_64"

    return TestSpec(
        instance_id=instance_id,
        repo=repo,
        base_commit=instance["base_commit"],
        env_script_list=env_script_list,
        repo_script_list=repo_script_list,
        eval_script_list=eval_script_list, # Contains Test Directives
        gold_inverted_eval_script_list=inverted_eval_script_list_gold,
        bad_inverted_eval_script_list=inverted_eval_script_list_bad,
        version=version,
        arch=arch,
    )
