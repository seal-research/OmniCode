import hashlib
import json
import platform
import re
import socket

from dataclasses import dataclass
from typing import Any, Union

from swebench.harness.constants import (
    SWEbenchInstance,
    MAP_REPO_VERSION_TO_SPECS,
)
from swebench.harness.dockerfiles import (
    get_dockerfile_base,
    get_dockerfile_env,
    get_dockerfile_instance,
)

DIFF_MODIFIED_FILE_REGEX = r"--- a/(.*)"


@dataclass
class TestSpec:
    """
    A dataclass that represents a test specification for a single instance of SWE-bench.
    """
    instance_id: str
    repo: str
    version: str
    repo_script_list: str
    eval_script_list: str
    env_script_list: str
    arch: str
    FAIL_TO_PASS: list[str]
    PASS_TO_PASS: list[str]

    @property
    def setup_env_script(self):
        return "\n".join(["#!/bin/bash", "set -euxo pipefail"] + self.env_script_list) + "\n"

    @property
    def eval_script(self):
        return "\n".join(["#!/bin/bash", "set -uxo pipefail"] + self.eval_script_list) + "\n"
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
        return get_dockerfile_instance(self.platform, self.env_image_key, self.repo.split("/")[-1])

    @property
    def platform(self):
        if self.arch == "x86_64":
            return "linux/x86_64"
        elif self.arch == "arm64":
            return "linux/arm64/v8"
        else:
            raise ValueError(f"Invalid architecture: {self.arch}")
        

def get_test_specs_from_dataset(dataset: Union[list[SWEbenchInstance], list[TestSpec]]) -> list[TestSpec]:
    """
    Idempotent function that converts a list of SWEbenchInstance objects to a list of TestSpec objects.
    """
    if isinstance(dataset[0], TestSpec):
        return dataset
    return list(map(make_test_spec, dataset))


def make_repo_script_list(repo, repo_directory, root_path, base_commit, env_type):
    """
    Create a list of bash commands to set up the repository for testing.
    This is the setup script for the instance image.
    """
    build_cmds = []
    if env_type == "maven":
        build_cmds.extend([
            "export M2_HOME=/opt/maven",
            "export MAVEN_HOME=/opt/maven",
            "export PATH=${M2_HOME}/bin:${PATH}",
            "mvn -version",
        ])
        build_cmds.append("mvn clean install -Dmaven.test.skip=true")
    elif env_type == "gradle":
        build_cmds.append("./gradlew dependencies")

    setup_commands = [
        # f"git clone -o origin https://github.com/{repo} {repo_directory}",
        f"chmod -R 777 {repo_directory}",  # So nonroot user can run tests
        f"cd {repo_directory}/{root_path}",
        f"git reset --hard {base_commit}",
        f"git remote remove origin",
        *build_cmds,
    ]
    return setup_commands


def make_env_script_list(specs):
    reqs_commands = []
    jdk_version = specs.get("jdk_version", "")
    ENV_TYPE = specs.get("env_type", "")
    if jdk_version:
        reqs_commands = [
            "apt-get update",
            f"apt-get install -y openjdk-{jdk_version}-jdk",
            "java -version",
        ]
    if ENV_TYPE == "maven":
        reqs_commands += [
            f"wget https://dlcdn.apache.org/maven/maven-3/3.9.8/binaries/apache-maven-3.9.8-bin.tar.gz",
            "tar -xvzf apache-maven-*.tar.gz",
            "mv apache-maven-3.9.8 /opt/maven",
            "export M2_HOME=/opt/maven",
            "export MAVEN_HOME=/opt/maven",
            "export PATH=${M2_HOME}/bin:${PATH}",
            "mvn -version",
        ]
    return reqs_commands


def make_eval_script_list(specs, repo_directory, base_commit, test_patch, fail_to_pass):
    """
    Applies the test patch and runs the tests.
    """
    HEREDOC_DELIMITER = "EOF_114329324912"
    ENV_TYPE = specs.get("env_type", "")
    FAIL_TO_PASS = [
        (test_str.split(":")[0], test_str.split(":")[1]) for test_str in fail_to_pass
    ]
    
    test_files = re.findall(DIFF_MODIFIED_FILE_REGEX, test_patch)
    # Reset test files to the state they should be in before the patch.
    reset_tests_command = f"git checkout {base_commit} {' '.join(test_files)}"
    apply_test_patch_command = (
        f"git apply -v - <<'{HEREDOC_DELIMITER}'\n{test_patch}\n{HEREDOC_DELIMITER}"
    )
    eval_commands = []
    if "eval_commands" in specs:
        eval_commands += specs["eval_commands"]
    eval_commands += [
        f"git config --global --add safe.directory {repo_directory}",  # for nonroot user
        # This is just informational, so we have a record
        f"git status",
        f"git show",
        f"git diff {base_commit}",
    ]
    if "install" in specs:
        eval_commands.append(specs["install"])
    
    eval_format = specs.get("eval_cmd", "")
    test_spec = specs.get("test_spec", "test")
    if ENV_TYPE == "maven":
        eval_commands += [
            "export M2_HOME=/opt/maven",
            "export MAVEN_HOME=/opt/maven",
            "export PATH=${M2_HOME}/bin:${PATH}",
            "mvn -version",
        ]
        if "eval_cmd" not in specs:
            eval_format = "mvn clean test -Dsurefire.useFile=false -Dmaven.test.skip=false -D{test_spec}={test_func} -DfailIfNoTests=false -am >{module_escaped}:{test_func}.test.log 2>&1"
        eval_commands += [
            eval_format.format_map({"test_spec": test_spec, "test_func": test_func, "module": module, "module_escaped": module.replace("/", "__")}) for module, test_func in FAIL_TO_PASS
        ]
        # eval_commands += [
        #     f"mvn test -Dmaven.test.skip=false" for module, test_func in FAIL_TO_PASS
        # ]
    elif ENV_TYPE == "gradle":
        if "eval_cmd" not in specs:
            eval_format = "./gradlew {test_spec} --tests {test_func} >{module}:{test_func}.test.log 2>&1"
        eval_commands += [
            eval_format.format_map({"test_spec": test_spec, "module": module, "test_func": test_func}) for module, test_func in FAIL_TO_PASS
        ]
        
    eval_commands = [
        f"cd {repo_directory}",
        reset_tests_command,
        apply_test_patch_command,
        *eval_commands,
        reset_tests_command,  # Revert tests after done, leave the repo in the same state as before
    ]
    return eval_commands


def make_test_spec(instance: SWEbenchInstance) -> TestSpec:
    if isinstance(instance, TestSpec):
        return instance
    instance_id = instance["instance_id"]
    repo = instance["repo"]
    version = instance["version"]
    base_commit = instance["base_commit"]
    problem_statement = instance["problem_statement"]
    hints_text = instance["hints_text"]  # Unused
    test_patch = instance["test_patch"]

    def _from_json_or_obj(key: str) -> Any:
        """If key points to string, load with json"""
        if isinstance(instance[key], str):
            return json.loads(instance[key])
        return instance[key]

    pass_to_pass = _from_json_or_obj("PASS_TO_PASS")
    fail_to_pass = _from_json_or_obj("FAIL_TO_PASS")

    env_name = "testbed"
    repo_directory = f"/{env_name}/{repo.split('/')[-1]}"
    specs = MAP_REPO_VERSION_TO_SPECS[repo][version]
    root_path = specs.get("root_path", "")
    env_type = specs.get("env_type", "")

    repo_script_list = make_repo_script_list(repo, repo_directory, root_path, base_commit, env_type)
    env_script_list = make_env_script_list(specs)
    eval_script_list = make_eval_script_list(
        specs, repo_directory, base_commit, test_patch, fail_to_pass
    )
    if platform.machine() in {"aarch64", "arm64"}:
        # use arm64 unless explicitly specified
        arch = "arm64"
    else:
        arch = "x86_64"

    return TestSpec(
        instance_id=instance_id,
        repo=repo,
        env_script_list=env_script_list,
        repo_script_list=repo_script_list,
        eval_script_list=eval_script_list,
        version=version,
        arch=arch,
        FAIL_TO_PASS=fail_to_pass,
        PASS_TO_PASS=pass_to_pass,
    )
