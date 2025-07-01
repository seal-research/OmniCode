from __future__ import annotations

import hashlib
import json
import platform
import re
import os


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
# from swebench.harness.utils import (
#     get_requirements,
#     get_environment_yml
# )
from swebench.harness.test_spec.python import (
    get_requirements,
    get_environment_yml
)

from utils import (
    NON_TEST_EXTS,
    get_modified_added_files,
)

DIFF_MODIFIED_FILE_REGEX = r"--- a/(.*)"


@dataclass
class TestSpec:
    """
    A test specification for a CodeArena instance.
    """
    instance_id: str
    repo: str
    base_commit: str
    version: str
    gold_patch: str
    test_patch: str
    bad_patches: list[str]  # Changed from bad_patch to bad_patches
    repo_script_list: list[str]
    env_script_list: list[str]
    eval_script_list: list[str]
    gold_inverted_eval_script_list: list[str]
    bad_inverted_eval_script_list: list[list[str]]  # Changed to list of lists to handle multiple bad patches
    arch: str

    def __post_init__(self):
        """
        Validate and process the test specification after initialization.
        """
        # Ensure bad_patches is a list
        if not isinstance(self.bad_patches, list):
            self.bad_patches = [self.bad_patches]

        # Filter out empty or None patches
        self.bad_patches = [patch for patch in self.bad_patches if patch and patch != 0]

        # Create inverted eval script lists for each bad patch
        self.bad_inverted_eval_script_list = [
            make_inverted_eval_script_list(
                self,
                MAP_REPO_VERSION_TO_SPECS[self.repo][str(self.version)],
                "testbed",
                f"/testbed",
                self.base_commit,
                bad_patch['patch'],
                self.test_patch
            )
            for bad_patch in self.bad_patches
        ]

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
        """Returns the first bad patch's inverted eval script for backward compatibility"""
        if not self.bad_inverted_eval_script_list:
            return ""
        return "\n".join(["#!/bin/bash", "set -uxo pipefail"] + self.bad_inverted_eval_script_list[0]) + "\n"

    def get_inverted_eval_script_bad(self, index: int = 0) -> str:
        """Returns the inverted eval script for a specific bad patch by index"""
        if not self.bad_inverted_eval_script_list or index >= len(self.bad_inverted_eval_script_list):
            return ""
        return "\n".join(["#!/bin/bash", "set -uxo pipefail"] + self.bad_inverted_eval_script_list[index]) + "\n"

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


def make_env_script_list(instance: CodeArenaInstance,
                         specs: dict,
                         env_name: str) -> list[str]:
    """
    Creates the list of commands to set up the conda environment for testing.
    If conda isn't already present, it downloads & bootstraps Miniconda.
    """
    HEREDOC_DELIMITER = "EOF_59812759871"

    reqs_commands = [
        # 1) if conda isn't on PATH, install Miniconda
        'if ! command -v conda &> /dev/null; then '
        '  echo ">>> Bootstrapping Miniconda"; '
        '  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh && '
        '  bash /tmp/mc.sh -b -p /opt/miniconda3 && '
        '  rm /tmp/mc.sh; '
        'fi',
        # 2) ensure we can run `conda`
        'export PATH="/opt/miniconda3/bin:${PATH}"',
        # 3) initialize shell functions so `conda activate` works
        'source /opt/miniconda3/etc/profile.d/conda.sh',
        'eval "$(conda shell.bash hook)"',
    ]

    pkgs = specs.get("packages", "")
    if pkgs == "requirements.txt":
        reqs_commands.append(f"conda create -n {env_name} python={specs['python']} -y")
        reqs = replace_uninstallable_packages_requirements_txt(get_requirements(instance))
        path_to_reqs = "$HOME/requirements.txt"
        reqs_commands.append(
            f"cat <<'{HEREDOC_DELIMITER}' > {path_to_reqs}\n{reqs}\n{HEREDOC_DELIMITER}"
        )
        reqs_commands.append(
            f"conda activate {env_name} && python -m pip install -r {path_to_reqs}"
        )
        reqs_commands.append(f"rm {path_to_reqs}")

    elif pkgs == "environment.yml":
        reqs = get_environment_yml(instance, env_name)
        path_to_reqs = "environment.yml"
        reqs_commands.append(
            f"cat <<'{HEREDOC_DELIMITER}' > {path_to_reqs}\n{reqs}\n{HEREDOC_DELIMITER}"
        )
        if specs.get("no_use_env", False):
            reqs_commands.append(
                f"conda create -c conda-forge -n {env_name} "
                f"python={specs['python']} -y"
            )
            reqs_commands.append(f"conda env update -f {path_to_reqs}")
        else:
            reqs_commands.append(f"conda env create --file {path_to_reqs}")
            reqs_commands.append(
                f"conda activate {env_name} && "
                f"conda install python={specs['python']} -y"
            )
        reqs_commands.append(f"rm {path_to_reqs}")

    else:
        reqs_commands.append(
            f"conda create -n {env_name} python={specs['python']} {pkgs or ''} -y"
        )

    # Activate the new env
    reqs_commands.append(f"conda activate {env_name}")

    # Extra pip packages
    if "pip_packages" in specs:
        pip_packages = " ".join(specs["pip_packages"])
        reqs_commands.append(f"python -m pip install {pip_packages}")

    # And always install pylint
    # Only install pylint if we've asked for a style review build
    if os.environ.get("STYLE_REVIEW", "0") == "1":
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
            MAP_REPO_VERSION_TO_SPECS[instance["repo"]][str(instance["version"])]["test_cmd"],
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

def get_test_directives(instance: Union[dict, TestSpec]) -> list:
    """
    Get test directives for a specific instance.
    """
    # Handle both dict and TestSpec types
    repo = instance["repo"] if isinstance(instance, dict) else instance.repo
    if isinstance(instance, dict):
    # prefer candidate_test_patch, but fall back to test_patch
        test_patch = instance.get("candidate_test_patch",
                                  instance.get("test_patch"))
    else:
        # for objects, try attribute then fallback to .test_patch
        test_patch = getattr(instance, "candidate_test_patch",
                             getattr(instance, "test_patch", None))

    if repo == "swe-bench/humaneval":
        return []

    # Handle both dict and TestSpec types
    if isinstance(instance, dict):
        if "test_directives" in instance:
            return instance["test_directives"]
        if "test_directive" in instance:
            return [instance["test_directive"]]
    else:
        if hasattr(instance, "test_directives"):
            return instance.test_directives
        if hasattr(instance, "test_directive"):
            return [instance.test_directive]


    # diff_pat = r"diff --git a/.* b/(.*)"
    # directives = re.findall(diff_pat, test_patch)

    directives = get_modified_added_files(test_patch)
    directives = [
        d for d in directives if not any(d.endswith(ext) for ext in NON_TEST_EXTS)
    ]

    # For Django tests, remove extension + "tests/" prefix and convert slashes to dots (module referencing)
    if repo == "django/django":
        directives_transformed = []
        for d in directives:
            d = d[: -len(".py")] if d.endswith(".py") else d
            d = d[len("tests/") :] if d.startswith("tests/") else d
            d = d.replace("/", ".")
            directives_transformed.append(d)
        directives = directives_transformed

    return directives


def make_inverted_eval_script_list(instance, specs, env_name, repo_directory, base_commit, issue_patch, test_patch):
    """
    Applies the given issue patch (gold or bad) and runs the candidate tests.
    """
    HEREDOC_DELIMITER = "EOF_114329324912"
    test_files = re.findall(DIFF_MODIFIED_FILE_REGEX, test_patch)
    # Reset all files *except* test files to the state they should be in before the patch.
    reset_command = f"git stash push -- {' '.join(test_files)} && git checkout {base_commit} -- . && git stash pop"

    if issue_patch != 0:
        apply_issue_patch_command = (
            f"git apply -v - <<'{HEREDOC_DELIMITER}'\n{issue_patch}\n{HEREDOC_DELIMITER}"
        )
    else:
        apply_issue_patch_command = ""

    # Handle both dict and TestSpec types
    repo = instance["repo"] if isinstance(instance, dict) else instance.repo
    version = str(instance["version"] if isinstance(instance, dict) else instance.version)

    test_command = " ".join(
        [
            MAP_REPO_VERSION_TO_SPECS[repo][version]["test_cmd"],
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

def generate_patch_lint_script(repo_directory, base_commit, patch, pylint_output_path, error_output_path, env_name, second_patch=None):
    """
    Generate a shell script to run pylint evaluation on modified Python files.

    Args:
        repo_directory (str): Path to the git repository
        base_commit (str): Base commit to compare against
        patch (str): Git patch to apply
        pylint_output_path (str): Path to save aggregated pylint results
        error_output_path (str): Path to save detailed error messages
        env_name (str): Conda environment name
        second_patch (str, optional): Second patch to apply (for sweagent eval). Defaults to None.

    Returns:
        list: Shell commands to execute the evaluation
    """

    # Define the entire script as a raw string with proper substitution
    script = rf'''
    # Ensure required tools are available
    for cmd in jq awk pylint git; do
        if ! command -v "$cmd" &>/dev/null; then
            echo "$cmd is not installed, exiting." >&2
            exit 1
        fi
    done

    # Create temporary directory for intermediate files
    temp_dir=$(mktemp -d)
    trap 'rm -rf "$temp_dir"' EXIT

    # Ensure output directories exist and are writable
    mkdir -p /tmp
    chmod 777 /tmp
    touch /tmp/pylint_aggregated.json /tmp/pylint_errors.json
    chmod 666 /tmp/pylint_aggregated.json /tmp/pylint_errors.json

    # Initialize git if needed and ensure files are tracked
    if [ ! -d ".git" ]; then
        git init
        git add .
        git config --global --add safe.directory {repo_directory}
        git commit -m "Initial commit"
    fi

    # Save current changes if any
    git stash push --include-untracked || true

    # Checkout base commit
    git fetch || true  # Try to fetch but don't fail if remote not configured
    git checkout {base_commit}

    # Apply the patch
    if ! git apply -v /tmp/changes.patch 2>"$temp_dir/apply.err"; then
        echo "Warning: Failed to apply patch cleanly" >&2
        cat "$temp_dir/apply.err" >&2
    fi

    # Apply second patch if one is provided
    if [ -n "{second_patch}" ]; then
        if ! git apply -v /tmp/changes2.patch 2>"$temp_dir/apply2.err"; then
            echo "Warning: Failed to apply second patch cleanly" >&2
            cat "$temp_dir/apply2.err" >&2
        fi
    fi

    # Get modified Python files - use both git status and patch analysis
    git status --porcelain | grep -E '\.pyx?$' | awk '{{print $2}}' > "$temp_dir/git_files"
    grep -E '^diff --git a/.*\.pyx?$' /tmp/changes.patch | sed -E 's/^diff --git a\/(.*) b\/.*/\\1/' | grep -v '^$' > "$temp_dir/patch_files"
    cat "$temp_dir/git_files" "$temp_dir/patch_files" | sort -u | grep -v '^$' > "$temp_dir/modified_files"
    python_files=$(cat "$temp_dir/modified_files")

    if [ -z "$python_files" ]; then
        echo "No Python files found in the diff"
        echo '{{"global_score": 10.0, "total_errors": 0, "total_warnings": 0, "total_conventions": 0}}' > /tmp/pylint_aggregated.json
        echo "[]" > /tmp/pylint_errors.json
        exit 0
    fi

    # Initialize error report
    echo "[]" > /tmp/pylint_errors.json

    # Process each file
    while IFS= read -r file; do
        [ -z "$file" ] && continue
        [ ! -f "$file" ] && continue

        echo "Processing file: $file"

        # Run pylint and capture outputs
        pylint "$file" --output-format=json > "$temp_dir/pylint.json" 2> "$temp_dir/pylint.err" || true
        pylint "$file" > "$temp_dir/pylint.txt" 2> "$temp_dir/pylint.err" || true

        # Extract score
        file_score=$(awk '/Your code has been rated at/ {{print $7}}' "$temp_dir/pylint.txt" | cut -d'/' -f1)
        if [ -z "$file_score" ] || ! [[ "$file_score" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
            file_score="0.0"
        fi

        # Process file results
        if [ -s "$temp_dir/pylint.json" ]; then
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
            jq -s '.[0] + [.[1]]' /tmp/pylint_errors.json "$temp_dir/file_report.json" > "$temp_dir/new_report.json"
            mv "$temp_dir/new_report.json" /tmp/pylint_errors.json
        else
            echo "Warning: Empty pylint output for $file" >&2
        fi

    done <<< "$python_files"

    # Generate final summary
    if [ -s "/tmp/pylint_errors.json" ]; then
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
        }}' /tmp/pylint_errors.json > /tmp/pylint_aggregated.json

        jq 'map(del(.score))' /tmp/pylint_errors.json > "$temp_dir/cleaned_errors.json" && mv "$temp_dir/cleaned_errors.json" /tmp/pylint_errors.json
    else
        echo '{{"global_score": 10.0, "total_errors": 0, "total_warnings": 0, "total_conventions": 0}}' > /tmp/pylint_aggregated.json
        echo "[]" > /tmp/pylint_errors.json
    fi

    # Verify files exist and are not empty
    if [ ! -s "/tmp/pylint_aggregated.json" ] || [ ! -s "/tmp/pylint_errors.json" ]; then
        echo "Error: Output files are missing or empty" >&2
        exit 1
    fi

    # Ensure files are readable by all
    chmod 666 /tmp/pylint_aggregated.json /tmp/pylint_errors.json

    # Restore original state
    git reset --hard HEAD || true
    git stash pop || true

    echo "Linting completed successfully"
    '''

    eval_commands = [
        "set -ex",
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
        f"cd {repo_directory}",
        # Write patch to file first
        f"cat > /tmp/changes.patch << 'EOF'\n{patch}\nEOF",
        script.strip()
    ]
    return eval_commands



def make_test_spec(instance: CodeArenaInstance) -> TestSpec:
    """
    Create a TestSpec from a CodeArena instance.
    """

    # Extract necessary information
    instance_id = instance[KEY_INSTANCE_ID]
    repo = instance["repo"]
    base_commit = instance["base_commit"]
    version = str(instance["version"])
    # Extract the "gold" patch, falling back to the generic `patch` field if needed
    gold_patch = instance.get("gold_patch", instance.get("patch"))
    if gold_patch is None:
        raise KeyError(f"Instance {instance.get('instance_id')} missing both 'gold_patch' and 'patch'")

    if isinstance(instance, dict):
    # prefer candidate_test_patch, but fall back to test_patch
        test_patch = instance.get("candidate_test_patch",
                                  instance.get("test_patch"))
    else:
        # for objects, try attribute then fallback to .test_patch
        test_patch = getattr(instance, "candidate_test_patch",
                             getattr(instance, "test_patch", None))

    # Get bad patches - now using bad_patches instead of bad_patch
    bad_patches = instance.get("bad_patches", [])  # Default to empty list if not found
    if not isinstance(bad_patches, list):
        bad_patches = [bad_patches]  # Convert single patch to list if needed

    if bad_patches == []:
        bad_patches = [{'idx': 0, 'patch': 0}]

    # Filter out empty or None patches
    # bad_patches = [patch for patch in bad_patches if patch and patch != 0]

    if platform.machine() in {"aarch64", "arm64"}:
        # use arm64 unless explicitly specified
        arch = "arm64" if instance_id not in USE_X86 else "x86_64"
    else:
        arch = "x86_64"

    # Create test spec
    test_spec = TestSpec(
        instance_id=instance_id,
        repo=repo,
        base_commit=base_commit,
        version=version,
        gold_patch=gold_patch,
        test_patch=test_patch,
        bad_patches=bad_patches,  # Pass the list of bad patches
        repo_script_list=make_repo_script_list(MAP_REPO_VERSION_TO_SPECS[repo][str(version)], repo, f"/testbed", base_commit, "testbed"),
        env_script_list=make_env_script_list(instance, MAP_REPO_VERSION_TO_SPECS[repo][str(version)], "testbed"),
        eval_script_list=make_eval_script_list(instance, MAP_REPO_VERSION_TO_SPECS[repo][str(version)], "testbed", f"/testbed", base_commit, test_patch),
        gold_inverted_eval_script_list=make_inverted_eval_script_list(instance, MAP_REPO_VERSION_TO_SPECS[repo][str(version)], "testbed", f"/testbed", base_commit, gold_patch, test_patch),
        bad_inverted_eval_script_list=[
            make_inverted_eval_script_list(
                instance,
                MAP_REPO_VERSION_TO_SPECS[repo][str(version)],
                "testbed",
                f"/testbed",
                base_commit,
                bad_patch['patch'],
                test_patch
            )
            for bad_patch in bad_patches
        ],
        arch=arch
    )

    return test_spec
