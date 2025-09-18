#!/usr/bin/env python
from __future__ import annotations
# --------------------------------------------------------------------------- #
#  aider_runner.py – run Aider on Swe-Bench / CodeArena instances             #
# --------------------------------------------------------------------------- #

import json, logging, os, shutil, subprocess, tempfile
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


# ----------------------------- prompt builder ------------------------------ #
def build_prompt(
    instance: dict,
    mode: str,
) -> str:
    """
    Return the appropriate prompt text for the given mode.
    """
    base = instance["problem_statement"].strip()
    repo = instance["repo"]

    if mode.startswith("bugfixing"):
        return base

    if mode.startswith("testgen"):
        return f"""
ROLE: autonomous software-engineer inside **{repo}**

<problem_description>
{base}
</problem_description>

Can you help me implement a test that successfully reproduces the problem specified in the <problem_description>?
The test must be created in the repository's existing test suite and should be runable with the repository's testing infrastructure / tooling.
Do not make any changes to the non-test code in the repository since we only need to create a reproduction test.
""".strip()
    
    if mode.startswith("reviewfix"):
        if 'bad_patches' not in instance or len(instance['bad_patches']) == 0:
            logger.warning(f"Instance {instance['instance_id']} does not have any bad patches, cannot generate faux problem statement.")
            return None
        bad_patch = instance['bad_patches'][0]    
        problem_statement = instance['problem_statement']
        bad_patch_text = bad_patch['patch']
        review = bad_patch['review']
        return f"""
ROLE: autonomous software-engineer inside **{repo}**
        
<problem_description>
{problem_statement}
</problem_description>

Here is a previous patch attempt that failed to resolve this issue.

<bad_patch>
{bad_patch_text}
</bad_patch>

Here is a review that attempts to explain why the patch failed:

<review>
{review}
</review>

Implement a fix for the given problem description keeping the failed patch and review in mind. Use insight from them to **avoid repeating the same mistakes** and to **guide your reasoning**."""


    raise ValueError(f"Unsupported mode '{mode}'")


# -------------------------- logging / global vars -------------------------- #
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ----------------------------- core function -------------------------------- #
def run_aider_single(
    instance: dict,
    model_name: str,
    api_key: str,
    output_dir: Path,
    model_provider: str,
    mode: str,
) -> Tuple[Optional[str], dict]:

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # clone & checkout ---------------------------------------------------- #
        repo_url = f"https://github.com/{instance['repo']}"
        logger.info("Cloning %s", repo_url)
        subprocess.run(["git", "clone", repo_url, temp_path], check=True)

        subprocess.run(
            ["git", "checkout", instance["base_commit"]],
            cwd=temp_path, check=True)

        # write prompt -------------------------------------------------------- #
        prompt_path = temp_path / "problem.txt"
        prompt_text = build_prompt(
            instance,
            mode,
        )
        prompt_path.write_text(prompt_text, encoding="utf-8")
        logger.info("Prompt written to %s", prompt_path)

        # build aider command ------------------------------------------------- #
        aider_cmd = [
            "aider",
            "--message-file", str(prompt_path),
            "--model", model_name,
            "--no-auto-commits", "--no-gitignore", "--no-pretty",
            "--no-stream",
            "--yes-always",
            "--encoding", "utf-8",
        ]
        if mode == "testgen":
            aider_cmd += [
                "--no-gui", "--no-browser", "--no-auto-test", "--verbose"
            ]

        timeout_sec = 1200

        env = {**os.environ}
        env[f"{model_provider}_API_KEY"] = api_key
        env[f"{model_provider}_MODEL"] = model_name
        env["PYTHONIOENCODING"] = "utf-8"
        env["AIDER_NO_PROMPT"] = "1"

        logger.info("Running Aider …")
        try:
            result = subprocess.run(
                aider_cmd,
                cwd=temp_path,
                capture_output=True, text=True,
                env=env, timeout=timeout_sec, check=True,
                encoding="utf-8", errors="replace",
            )
        except subprocess.TimeoutExpired as e:
            tail_out = (e.stdout or "")[-1500:]
            tail_err = (e.stderr or "")[-500:]
            logger.error("Timeout after %s\n…stdout…\n%s\n…stderr…\n%s",
                         e.timeout, tail_out, tail_err)
            return "timeout", {}
        except subprocess.CalledProcessError as e:
            logger.error("Aider failed: %s", e.stderr or e.stdout)
            return "aider error", {}
        except Exception as e:
            logger.error("Unexpected: %s", e)
            return "unexpected error", {}

        # capture diff -------------------------------------------------------- #
        diff = subprocess.run(
            ["git", "diff"], cwd=temp_path,
            capture_output=True, text=True, check=True,
            encoding="utf-8", errors="replace",
        ).stdout

        inst_dir = output_dir / instance["instance_id"]
        inst_dir.mkdir(parents=True, exist_ok=True)
        (inst_dir / "fix.patch").write_text(diff)

        meta = {
            "instance_id": instance["instance_id"],
            "mode": mode,
            "model_name": model_name,
            "full_output": result.stdout,
            "model_patch": diff,
        }
        (inst_dir / f"{instance['instance_id']}.pred").write_text(json.dumps(meta))
        return None, meta


NUM_RETRIES = 3

# ----------------------------- batch driver --------------------------------- #
def main(
    input_tasks_path: Path,
    output_dir_path: Path,
    model_name: str,
    api_key: str,
    model_provider: str,
    instance_ids: list[str] | None,
    mode: str,
):

    # load dataset ----------------------------------------------------------- #
    if input_tasks_path.exists():
        if input_tasks_path.suffix.endswith("json"):
            data = json.loads(input_tasks_path.read_text())
        elif input_tasks_path.suffix.endswith("jsonl"):
            data = [json.loads(l) for l in input_tasks_path.read_text().splitlines()]
        elif input_tasks_path.suffix.endswith("csv"):
            data = pd.read_csv(input_tasks_path).to_dict("records")
        else:
            raise RuntimeError(f"Unsupported {input_tasks_path.suffix}")
    else:
        data = load_dataset(str(input_tasks_path))
        if isinstance(data, dict):
            data = data["test"]

    if instance_ids:
        data = [d for d in data if d["instance_id"] in instance_ids]

    output_dir_path.mkdir(parents=True, exist_ok=True)
    preds_path = output_dir_path / "all_preds.jsonl"
    done = set()
    if preds_path.exists():
        done = {json.loads(l)["instance_id"] for l in preds_path.read_text().splitlines()}

    # iterate ---------------------------------------------------------------- #
    for inst in tqdm(data, desc=f"Inference with {model_name}"):
        if inst["instance_id"] in done:
            continue

        for i in range(NUM_RETRIES):
            err, res = run_aider_single(
                inst,
                model_name, api_key, output_dir_path,
                model_provider, mode,
            )
            if res['model_patch'] is not None and res['model_patch'] != '':
                break

        if err:
            logger.error("%s: %s", inst["instance_id"], err)
            continue

        with preds_path.open("a+") as sink:
            sink.write(json.dumps(res) + "\n")
            sink.flush()


# ------------------------------ CLI entry ----------------------------------- #
if __name__ == "__main__":
    from argparse import ArgumentParser
    

    p = ArgumentParser()
    p.add_argument("-i", "--input_tasks", required=True)
    p.add_argument("-o", "--output_dir", required=True)
    p.add_argument("-m", "--model_name", required=True)
    p.add_argument("-k", "--api_key", default=None)
    p.add_argument("--model_provider", required=True)
    p.add_argument("--mode", default="bugfixing",
                   choices=["bugfixing", "testgen"])
    p.add_argument("--instance_ids", default=None)
    args = p.parse_args()

    main(
        input_tasks_path=Path(args.input_tasks),
        output_dir_path=Path(args.output_dir),
        model_name=args.model_name,
        api_key=args.api_key,
        mode=args.mode,
        model_provider=args.model_provider.upper(),
        instance_ids=args.instance_ids.split(",") if args.instance_ids else None,
    )
