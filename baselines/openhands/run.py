#!/usr/bin/env python3
"""Generate OmniCode patches with OpenHands SDK.

This script:
1. Reads OmniCode instances from a dataset JSON file.
2. Prepares a per-instance git workspace at the instance base commit.
3. Runs OpenHands SDK against a mode-specific prompt.
4. Writes predictions JSONL compatible with `omnicode.py --BugFixing`.

Setup:
  pip install openhands-sdk pydantic

Example:
    python OmniCode/scripts/run_openhands_sdk_omnicode.py \
    --dataset OmniCode/data/omnicode_instances_python.json \
    --instance-ids astropy__astropy-13236 \
    --model openai/gpt-4.1 \
    --api-key-env OPENAI_API_KEY \
    --output OmniCode/results/openhands_sdk/predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

SDK: dict[str, Any] | None = None


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def load_openhands_sdk() -> dict[str, Any]:
    from pydantic import SecretStr
    from openhands.sdk import Agent, Conversation, LLM, Tool, Workspace
    import openhands.tools  # registers TerminalTool, FileEditorTool, etc. in the discriminated union  # noqa: F401

    return {
        "SecretStr": SecretStr,
        "Agent": Agent,
        "Conversation": Conversation,
        "LLM": LLM,
        "Tool": Tool,
        "Workspace": Workspace,
    }


def run_cmd(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def load_instances(dataset_path: Path) -> list[dict[str, Any]]:
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Dataset must be a JSON array: {dataset_path}")
    return data


def select_instances(
    instances: list[dict[str, Any]], instance_ids: set[str] | None, max_instances: int
) -> list[dict[str, Any]]:
    selected = []
    for inst in instances:
        iid = inst.get("instance_id")
        if not iid:
            continue
        if instance_ids and iid not in instance_ids:
            continue
        selected.append(inst)
        if max_instances > 0 and len(selected) >= max_instances:
            break
    return selected


def ensure_repo_cache(repo: str, repo_cache_root: Path) -> Path:
    cache_dir = repo_cache_root / repo.replace("/", "__")
    cache_dir.parent.mkdir(parents=True, exist_ok=True)

    if not cache_dir.exists():
        url = f"https://github.com/{repo}.git"
        run_cmd(["git", "clone", "--filter=blob:none", url, str(cache_dir)])
    else:
        run_cmd(["git", "-C", str(cache_dir), "fetch", "--all", "--tags", "--prune"])

    return cache_dir


def ensure_commit_available(repo_cache_dir: Path, base_commit: str) -> None:
    probe = run_cmd(
        ["git", "-C", str(repo_cache_dir), "cat-file", "-e", f"{base_commit}^{{commit}}"],
        check=False,
    )
    if probe.returncode == 0:
        return
    run_cmd(["git", "-C", str(repo_cache_dir), "fetch", "--all", "--tags", "--prune"])
    probe2 = run_cmd(
        ["git", "-C", str(repo_cache_dir), "cat-file", "-e", f"{base_commit}^{{commit}}"],
        check=False,
    )
    if probe2.returncode != 0:
        raise RuntimeError(f"Commit {base_commit} not available in repo cache {repo_cache_dir}")


def prepare_instance_workspace(
    repo_cache_dir: Path,
    workspace_root: Path,
    instance_id: str,
    base_commit: str,
    clean_existing: bool,
) -> Path:
    workspace_root.mkdir(parents=True, exist_ok=True)
    workdir = (workspace_root / instance_id).resolve()

    if workdir.exists() and clean_existing:
        run_cmd(["git", "-C", str(repo_cache_dir), "worktree", "remove", "--force", str(workdir)], check=False)
        shutil.rmtree(workdir, ignore_errors=True)

    if not workdir.exists():
        run_cmd(["git", "-C", str(repo_cache_dir), "worktree", "prune"], check=False)
        run_cmd(
            [
                "git",
                "-C",
                str(repo_cache_dir),
                "worktree",
                "add",
                "--detach",
                "--force",
                str(workdir),
                base_commit,
            ]
        )

    run_cmd(["git", "-C", str(workdir), "checkout", "--detach", base_commit])
    run_cmd(["git", "-C", str(workdir), "reset", "--hard", base_commit])
    run_cmd(["git", "-C", str(workdir), "clean", "-fdx"])
    return workdir


def _stylefix_context(instance: dict[str, Any]) -> str | None:
    """Return a formatted string of pylint violations from style_review, or None if none exist."""
    style_review = instance.get("style_review")
    if not isinstance(style_review, list) or not style_review:
        return None

    lines: list[str] = []
    for file_report in style_review:
        messages = file_report.get("messages", [])
        if not messages:
            continue
        lines.append(f"File: {file_report['file']}")
        for msg in messages:
            lines.append(
                f"  Line {msg.get('line', '?')} [{msg.get('symbol', '?')}]: {msg.get('message', '')}"
            )
        lines.append("")

    return "\n".join(lines).strip() or None


def _reviewfix_context(instance: dict[str, Any]) -> tuple[str, str] | None:
    bad_patches = instance.get("bad_patches")
    if not isinstance(bad_patches, list) or len(bad_patches) == 0:
        return None

    # Match sweagent preference when available.
    chosen = None
    for bp in bad_patches:
        if isinstance(bp, dict) and "agentless" in str(bp.get("source", "")):
            chosen = bp
            break
    if chosen is None:
        chosen = bad_patches[0]

    patch = str(chosen.get("patch", "")).strip()
    review = str(chosen.get("review", "")).strip()
    if not patch or not review:
        return None
    return patch, review


def build_prompt(instance: dict[str, Any], mode: str, include_hints: bool, repo_path: str | None = None) -> str:
    iid = instance["instance_id"]
    repo = instance["repo"]
    _problem = instance.get("problem_statement")
    problem = (_problem if isinstance(_problem, str) else "").strip()
    _hints = instance.get("hints_text")
    hints = (_hints if isinstance(_hints, str) else "").strip()
    fail_to_pass = instance.get("FAIL_TO_PASS") or []

    if mode == "stylefix":
        if not problem:
            raise ValueError(
                "stylefix mode requires a non-empty problem_statement"
            )
        env_block = ""
        if repo_path:
            env_block = f"""

Environment:
- Your current working directory is `{repo_path}`.
- The repository is already checked out at `{repo_path}` at the correct base commit. Do NOT clone or copy it.
- All project dependencies are pre-installed and active in the current environment.
- Always use absolute paths starting with `{repo_path}/` when referring to files."""
        return f"""
ROLE: autonomous software-engineer inside **{repo}**

{problem}
{env_block}

Constraints:
- Fix only the listed violations. Do not refactor or change unrelated code.
- Do not commit or create git tags.
- At the end, briefly summarise what you changed.
""".strip()

    if mode == "testgen":
        env_block = ""
        if repo_path:
            env_block = f"""

Environment:
- Your current working directory is `{repo_path}`.
- The repository is already checked out at `{repo_path}` at the correct base commit. Do NOT clone or copy it.
- All project dependencies are pre-installed and active in the current environment.
- Always use absolute paths starting with `{repo_path}/` when referring to files."""
        return f"""
ROLE: autonomous software-engineer inside **{repo}**

<problem_description>
{problem}
</problem_description>

Can you help me implement a test that successfully reproduces the problem specified in the <problem_description>?
The test must be created in the repository's existing test suite and should be runnable with the repository's testing infrastructure/tooling.
Do not make any changes to non-test source code; only add or modify tests.
{env_block}
""".strip()

    if mode == "reviewfix":
        context = _reviewfix_context(instance)
        if context is None:
            raise ValueError(
                "reviewfix mode requires bad_patches with both patch and review"
            )
        bad_patch_text, review_text = context
        env_block = ""
        if repo_path:
            env_block = f"""

Environment:
- Your current working directory is `{repo_path}`.
- The repository is already checked out at `{repo_path}` at the correct base commit. Do NOT clone or copy it.
- All project dependencies are pre-installed and active in the current environment.
- Always use absolute paths starting with `{repo_path}/` when referring to files."""
        return f"""
ROLE: autonomous software-engineer inside **{repo}**

<problem_description>
{problem}
</problem_description>

Here is a previous patch attempt that failed to resolve this issue.

<bad_patch>
{bad_patch_text}
</bad_patch>

Here is a review that attempts to explain why the patch failed:

<review>
{review_text}
</review>

Implement a fix for the given problem description keeping the failed patch and review in mind.
Use insights from them to avoid repeating the same mistakes and to guide your reasoning.
{env_block}
""".strip()

    lines: list[str] = []
    lines.append(f"You are fixing OmniCode BugFixing instance: {iid}")
    lines.append(f"Repository: {repo}")
    lines.append("Goal: implement a minimal source-code fix for the issue below.")
    lines.append("")
    lines.append("Issue description:")
    lines.append(problem)

    if include_hints and hints and hints.lower() != "nan":
        lines.append("")
        lines.append("Hints:")
        lines.append(hints)

    if fail_to_pass:
        lines.append("")
        lines.append("Target failing tests to make pass:")
        for test_name in fail_to_pass[:20]:
            lines.append(f"- {test_name}")

    if repo_path:
        lines.append("")
        lines.append("Environment:")
        lines.append(f"- Your current working directory is `{repo_path}`.")
        lines.append(f"- The repository is already checked out at `{repo_path}` at the correct base commit. Do NOT clone or copy it.")
        lines.append(f"- All project dependencies are pre-installed and active in the current environment.")
        lines.append(f"- Always use absolute paths starting with `{repo_path}/` when referring to files.")

    lines.append("")
    lines.append("Constraints:")
    lines.append("- Keep the fix minimal and scoped to the issue.")
    lines.append("- Do not commit or create git tags.")
    lines.append("- Run relevant tests if feasible.")
    if repo_path:
        lines.append(f"- Make ALL code changes directly in `{repo_path}`. Do NOT copy files or create a separate workspace directory.")
    else:
        lines.append("- Make code changes directly in this workspace.")
    lines.append("- At the end, briefly summarize what you changed.")

    return "\n".join(lines)


def collect_patch(workdir: Path, base_commit: str) -> str:
    unstaged = run_cmd(["git", "-C", str(workdir), "diff", "--binary"], check=False).stdout
    staged = run_cmd(["git", "-C", str(workdir), "diff", "--binary", "--cached"], check=False).stdout
    patch = ""
    if unstaged:
        patch += unstaged
    if staged:
        if patch and not patch.endswith("\n"):
            patch += "\n"
        patch += staged

    if patch.strip():
        return patch

    head = run_cmd(["git", "-C", str(workdir), "rev-parse", "HEAD"], check=False).stdout.strip()
    if head and head != base_commit:
        fp = run_cmd(
            ["git", "-C", str(workdir), "format-patch", "--stdout", f"{base_commit}..HEAD"],
            check=False,
        ).stdout
        if fp.strip():
            return fp

    return ""


# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------

_DOCKER_AGENT_PORT = 3000  # port the agent server listens on inside the container
# The OmniCode images have Python 3.9/3.11; openhands-sdk requires >=3.12.
# We use uv to install Python 3.12 + the three required packages without touching
# the container's existing conda environments.
_DOCKER_UV_INSTALL = "curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --quiet"
# Prepend the testbed conda env so the agent's shell commands use the project's
# Python/pytest without any extra activation step.
_DOCKER_TESTBED_PATH = "export PATH=/opt/miniconda3/envs/testbed/bin:$PATH"
_DOCKER_SERVER_CMD = (
    "$HOME/.local/bin/uv run --python 3.12"
    " --with openhands-sdk --with openhands-workspace --with openhands-tools"
    f" python -m openhands.agent_server --port {_DOCKER_AGENT_PORT} --host 0.0.0.0"
)


def omnicode_image_for_instance(inst: dict[str, Any], dataset_path: Path) -> str:
    """Return the Docker image name for an OmniCode instance.

    Matches the sweagent convention:
      - Java / C++: shared repo-level base image  omnicodeorg/omnicode:{org}_{repo}_base
      - Python:     per-instance image             omnicodeorg/omnicode:{instance_id}
    The language is inferred from the dataset filename (e.g. *_java.json, *_cpp.json).
    """
    lang = dataset_path.stem.split("_")[-1]  # e.g. "python", "java", "cpp"
    if lang in ("java", "cpp"):
        return f"omnicodeorg/omnicode:{inst['repo'].replace('/', '_')}_base"
    # Style instances have a _<number> suffix; use original_instance_id for the image name.
    image_id = inst.get("original_instance_id") or inst["instance_id"]
    return f"omnicodeorg/omnicode:{image_id}"


def docker_container_name(instance_id: str) -> str:
    return f"oh__{instance_id}"


def start_docker_agent_server(
    image: str, container_name: str, host_port: int,
) -> None:
    """Start an OmniCode container with the OpenHands agent server running inside it.

    /testbed starts empty; call populate_testbed_via_archive() after this to load
    the repo content while uv is still installing.
    """
    # Remove any leftover container with the same name.
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)

    startup = f"{_DOCKER_TESTBED_PATH} && {_DOCKER_UV_INSTALL} && {_DOCKER_SERVER_CMD}"
    subprocess.run(
        [
            "docker", "run", "-d",
            "--name", container_name,
            "--platform", "linux/amd64",
            "--workdir", "/testbed",
            "-p", f"{host_port}:{_DOCKER_AGENT_PORT}",
            image,
            "bash", "-c", startup,
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def populate_testbed_via_clone(
    container_name: str, repo: str, base_commit: str
) -> None:
    """Clone the repo at the correct commit into /testbed inside the container.

    Mirrors what SWE-agent does via GithubRepoConfig.  The container already has
    internet access (it downloads uv), so a shallow clone is fast.

    We also add 'workspace/' to .gitignore so the OpenHands agent server's session
    directory (written at /testbed/workspace/) is never included in patches.
    """
    clone_cmd = (
        f"git clone --filter=blob:none https://github.com/{repo}.git /testbed && "
        f"git -C /testbed checkout {base_commit} && "
        # Keep workspace/ out of any git diff so patch collection stays clean.
        # Use .git/info/exclude (git-local, never tracked) rather than .gitignore.
        "echo 'workspace/' >> /testbed/.git/info/exclude"
    )
    result = subprocess.run(
        ["docker", "exec", container_name, "bash", "-c", clone_cmd],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to populate /testbed in container {container_name}:\n"
            f"{result.stderr}"
        )


def wait_for_agent_server_ready(host_port: int, timeout: int = 180) -> None:
    """Poll the agent server health endpoint until it responds or timeout is reached."""
    url = f"http://localhost:{host_port}/health"
    deadline = time.time() + timeout
    last_exc: Exception = RuntimeError("timeout before first attempt")
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    return
        except Exception as exc:
            last_exc = exc
        time.sleep(2)
    raise RuntimeError(
        f"Agent server on port {host_port} not ready after {timeout}s: {last_exc}"
    )


def stop_remove_container(container_name: str) -> None:
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)


def collect_patch_from_docker(container_name: str) -> str:
    """Collect uncommitted changes from /testbed inside the running container."""
    unstaged = subprocess.run(
        ["docker", "exec", container_name, "git", "-C", "/testbed", "diff", "--binary"],
        capture_output=True, text=True, check=False,
    ).stdout
    staged = subprocess.run(
        ["docker", "exec", container_name, "git", "-C", "/testbed", "diff", "--binary", "--cached"],
        capture_output=True, text=True, check=False,
    ).stdout
    patch = ""
    if unstaged:
        patch += unstaged
    if staged:
        if patch and not patch.endswith("\n"):
            patch += "\n"
        patch += staged

    if patch.strip():
        return patch

    # Also try new files (untracked)
    untracked_check = subprocess.run(
        ["docker", "exec", container_name, "git", "-C", "/testbed", "status", "--porcelain"],
        capture_output=True, text=True, check=False,
    ).stdout
    if untracked_check.strip():
        # There are untracked changes; add everything except ignored paths.
        subprocess.run(
            ["docker", "exec", container_name, "git", "-C", "/testbed", "add", "-A",
             "--", ":!workspace"],
            capture_output=True, text=True, check=False,
        )
        staged2 = subprocess.run(
            ["docker", "exec", container_name, "git", "-C", "/testbed", "diff", "--binary", "--cached"],
            capture_output=True, text=True, check=False,
        ).stdout
        if staged2.strip():
            return staged2

    return ""


def clean_run_log_failures(run_log_path: Path) -> None:
    """Rewrite run_log keeping only status=ok entries (called on --resume)."""
    if not run_log_path.exists():
        return
    ok_lines = []
    with run_log_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("status") == "ok":
                    ok_lines.append(line)
            except json.JSONDecodeError:
                continue
    with run_log_path.open("w") as f:
        for line in ok_lines:
            f.write(line + "\n")


def existing_prediction_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            iid = obj.get("instance_id")
            if isinstance(iid, str):
                ids.add(iid)
    return ids


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=True) + "\n")


def run_openhands_on_instance(
    llm: Any,
    prompt: str,
    workspace: Any,  # str path for LocalWorkspace, or Workspace object for RemoteWorkspace
    model_name: str,
) -> tuple[str, dict[str, Any]]:
    assert SDK is not None, "SDK must be loaded before running instances"
    agent = SDK["Agent"](
        llm=llm,
        tools=[
            SDK["Tool"](name="terminal"),
            SDK["Tool"](name="file_editor"),
            SDK["Tool"](name="task_tracker"),
        ],
    )
    conversation = SDK["Conversation"](agent=agent, workspace=workspace)

    try:
        conversation.send_message(prompt)
        result = conversation.run()
    finally:
        conversation.close()

    full_output = str(result) if result is not None else ""
    return full_output, {
        "raw_result_type": type(result).__name__,
        "model": model_name,
    }


def write_instance_artifacts(
    artifacts_root: Path,
    instance_id: str,
    model_name: str,
    full_output: str,
    patch: str,
) -> None:
    inst_dir = artifacts_root / instance_id
    inst_dir.mkdir(parents=True, exist_ok=True)

    pred_obj = {
        "instance_id": instance_id,
        "model_name_or_path": model_name,
        "full_output": full_output,
        "model_patch": patch,
    }
    (inst_dir / f"{instance_id}.pred").write_text(
        json.dumps(pred_obj, ensure_ascii=True), encoding="utf-8"
    )
    (inst_dir / f"{instance_id}.patch").write_text(patch, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate OmniCode patches with OpenHands SDK")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("OmniCode/data/omnicode_instances_python.json"),
        help="Path to OmniCode dataset JSON",
    )
    parser.add_argument(
        "--instance-ids",
        nargs="*",
        default=None,
        help="Optional instance IDs to run",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=0,
        help="Maximum number of selected instances (0 = all selected)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("OmniCode/results/openhands_sdk/all_preds.jsonl"),
        help="Aggregated predictions JSONL output path (baseline-style all_preds)",
    )
    parser.add_argument(
        "--run-log",
        type=Path,
        default=Path("OmniCode/results/openhands_sdk/run_log.jsonl"),
        help="Run log JSONL output path",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("OmniCode/results/openhands_sdk"),
        help="Directory for per-instance artifacts (<id>/<id>.pred and <id>.patch)",
    )
    parser.add_argument(
        "--repo-cache-root",
        type=Path,
        default=Path("OmniCode/.openhands_repo_cache"),
        help="Directory for cached git repositories",
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path("OmniCode/.openhands_workspaces"),
        help="Directory for per-instance workspaces",
    )
    parser.add_argument(
        "--clean-existing-workspace",
        action="store_true",
        help="Delete existing instance workspace and recreate it",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip instances already present in --output",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Sleep between instances",
    )
    parser.add_argument(
        "--num-retries",
        type=int,
        default=1,
        help="Retries per instance when run fails or produces empty patch",
    )
    parser.add_argument(
        "--include-hints",
        action="store_true",
        help="Include hints_text in prompt when available",
    )
    parser.add_argument(
        "--mode",
        choices=["bugfixing", "testgen", "reviewfix", "stylefix"],
        default="bugfixing",
        help="Prompt mode",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print selected instances without running OpenHands",
    )

    parser.add_argument(
        "--docker",
        action="store_true",
        help=(
            "Run the agent inside the OmniCode Docker container for each instance "
            "(omnicodeorg/omnicode:{instance_id}). The container must be pullable and "
            "the host must have Docker installed. The agent server is installed inside "
            "the container at startup."
        ),
    )
    parser.add_argument(
        "--docker-port",
        type=int,
        default=3000,
        help="Host port forwarded to the agent server inside the container (default: 3000)",
    )

    parser.add_argument("--model", required=True, help="OpenHands SDK model id")
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing model API key",
    )
    parser.add_argument("--base-url", default=None, help="Optional model API base URL")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Path to .env file (loaded before reading --api-key-env)",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel instances to run (default: 1)",
    )
    parser.add_argument(
        "--target-successes",
        type=int,
        default=0,
        help="Stop writing patches after this many successes (0 = no target, write all)",
    )

    return parser.parse_args()


def process_one_instance(
    inst: dict,
    args: argparse.Namespace,
    llm: Any,
    port: int,
) -> dict:
    """Process one instance; return result dict with success, pred_obj, log_obj, artifact_args."""
    iid = inst["instance_id"]
    repo = inst.get("repo")
    base_commit = inst.get("base_commit")
    started = time.time()

    if not repo or not base_commit:
        return {
            "success": False,
            "pred_obj": None,
            "log_obj": {
                "instance_id": iid,
                "status": "error",
                "error": "Missing repo or base_commit",
            },
            "artifact_args": None,
        }

    attempts = max(1, args.num_retries)
    last_error = None

    for attempt in range(1, attempts + 1):
        container_name = None
        try:
            prompt = build_prompt(
                inst,
                mode=args.mode,
                include_hints=args.include_hints,
                repo_path="/testbed" if args.docker else None,
            )

            if args.docker:
                image = omnicode_image_for_instance(inst, args.dataset)
                container_name = docker_container_name(iid)
                start_docker_agent_server(image, container_name, port)
                populate_testbed_via_clone(container_name, repo, base_commit)
                wait_for_agent_server_ready(port)
                workspace = SDK["Workspace"](host=f"http://localhost:{port}")
                workspace_label = f"docker:{image}"
            else:
                repo_cache_dir = ensure_repo_cache(repo, args.repo_cache_root)
                ensure_commit_available(repo_cache_dir, base_commit)
                workdir = prepare_instance_workspace(
                    repo_cache_dir=repo_cache_dir,
                    workspace_root=args.workspace_root,
                    instance_id=iid,
                    base_commit=base_commit,
                    clean_existing=(args.clean_existing_workspace or attempt > 1),
                )
                workspace = str(workdir)
                workspace_label = str(workdir)

            full_output, sdk_meta = run_openhands_on_instance(
                llm=llm,
                prompt=prompt,
                workspace=workspace,
                model_name=args.model,
            )

            patch = (
                collect_patch_from_docker(container_name)
                if args.docker
                else collect_patch(workdir, base_commit=base_commit)
            )

            if not patch.strip():
                last_error = "empty patch"
                continue

            elapsed = round(time.time() - started, 2)
            pred_obj = {
                "instance_id": iid,
                "model_name_or_path": args.model,
                "full_output": full_output,
                "model_patch": patch,
            }
            log_obj = {
                "instance_id": iid,
                "repo": repo,
                "base_commit": base_commit,
                "workspace": workspace_label,
                "status": "ok",
                "attempt": attempt,
                "patch_bytes": len(patch.encode("utf-8")),
                "elapsed_seconds": elapsed,
                **sdk_meta,
            }
            return {
                "success": True,
                "pred_obj": pred_obj,
                "log_obj": log_obj,
                "artifact_args": (args.artifacts_dir, iid, args.model, full_output, patch),
            }

        except Exception as exc:
            last_error = str(exc)
        finally:
            if args.docker and container_name:
                stop_remove_container(container_name)

    elapsed = round(time.time() - started, 2)
    return {
        "success": False,
        "pred_obj": None,
        "log_obj": {
            "instance_id": iid,
            "repo": repo,
            "base_commit": base_commit,
            "status": "error",
            "error": last_error or "unknown",
            "elapsed_seconds": elapsed,
        },
        "artifact_args": None,
    }


def main() -> int:
    global SDK
    args = parse_args()

    instances = load_instances(args.dataset)
    selected = select_instances(
        instances,
        set(args.instance_ids) if args.instance_ids else None,
        args.max_instances,
    )

    if args.resume:
        done = existing_prediction_ids(args.output)
        selected = [inst for inst in selected if inst.get("instance_id") not in done]
        # Clean failed entries from run_log so it only shows successes
        clean_run_log_failures(args.run_log)

    print(f"Selected instances: {len(selected)}")
    if not selected:
        return 0

    if args.dry_run:
        for inst in selected:
            print(inst.get("instance_id"))
        return 0

    load_env_file(args.env_file)
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        print(f"Missing API key env var: {args.api_key_env}", file=sys.stderr)
        return 2

    SDK = load_openhands_sdk()
    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "api_key": SDK["SecretStr"](api_key),
    }
    if args.base_url:
        llm_kwargs["base_url"] = args.base_url
    llm = SDK["LLM"](**llm_kwargs)

    # Thread-safety primitives
    write_lock = threading.Lock()
    success_count = 0
    target = args.target_successes  # 0 means no target

    # Progress display (tqdm if available, else plain print)
    total_tried = 0
    try:
        from tqdm import tqdm
        pbar = tqdm(total=len(selected), desc="instances", unit="inst", leave=True)

        def update_progress(success: bool) -> None:
            nonlocal total_tried
            total_tried += 1
            pbar.update(1)
            desc = f"instances ({success_count} ok)"
            if target:
                desc += f"/{target}"
            pbar.set_description(desc)
    except ImportError:
        pbar = None

        def update_progress(success: bool) -> None:
            nonlocal total_tried
            total_tried += 1
            print(
                f"  [{total_tried}/{len(selected)}] {'ok' if success else 'error'} "
                f"| successes: {success_count}" + (f"/{target}" if target else "")
            )

    def handle_result(result: dict) -> None:
        """Write result to output files. Must be called under write_lock."""
        nonlocal success_count
        log_obj = result["log_obj"]
        # Always log to run_log
        append_jsonl(args.run_log, log_obj)
        if result["success"]:
            # Only write to output if target not yet reached
            if target == 0 or success_count < target:
                append_jsonl(args.output, result["pred_obj"])
                art = result["artifact_args"]
                write_instance_artifacts(*art)
                success_count += 1

    # Port pool for parallel Docker workers
    port_q: queue.Queue[int] = queue.Queue()
    for i in range(max(1, args.workers)):
        port_q.put(args.docker_port + i)

    def worker_fn(inst: dict) -> dict:
        port = port_q.get()
        try:
            return process_one_instance(inst, args, llm, port)
        finally:
            port_q.put(port)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(worker_fn, inst): inst for inst in selected}
        target_reached = False
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                inst = futures[future]
                result = {
                    "success": False,
                    "pred_obj": None,
                    "log_obj": {
                        "instance_id": inst.get("instance_id"),
                        "status": "error",
                        "error": str(exc),
                    },
                    "artifact_args": None,
                }
            with write_lock:
                handle_result(result)
            update_progress(result["success"])
            if target and not target_reached and success_count >= target:
                target_reached = True
                for f in futures:
                    f.cancel()

    if pbar:
        pbar.close()

    print(f"\nSuccesses: {success_count}" + (f"/{target}" if target else f"/{len(selected)}"))
    print(f"Predictions written to: {args.output}")
    print(f"Run log written to: {args.run_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
