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
    working_dir: str | None = None,
    pylint_feedback: str | None = None,
) -> str:
    """
    Return the appropriate prompt text for the given mode.
    """
    base = instance["problem_statement"].strip()
    repo = instance["repo"]

    if mode == "bugfixing":
        return base

    if mode == "testgen":
        return f"""
ROLE: autonomous software-engineer inside **{repo}**

GOAL: write thorough pytest unit tests only.
- Cover the behaviour / bug described below
- Include at least one test that fails before a fix
- Put tests in the existing suite if present.

CONTEXT:
{base}

OUTPUT: add tests; finish with **ALL TESTS ADDED**.
""".strip()

    if mode == "stylereview":
        feedback = pylint_feedback or base
        return f"""
You have recently generated a patch to resolve an issue within this repository.
Pylint has been run on the modified files and has produced the following
feedback:
<lint_report>
{feedback.strip()}
</lint_report>

Please resolve the Pylint feedback to the best of your ability, while
preserving the functionality of the code.
""".strip()

    if mode == "codereview":
        bp_raw = instance.get("bad_patches", [])
        if isinstance(bp_raw, str):
            try:
                bp_raw = json.loads(bp_raw)
            except Exception:
                bp_raw = []

        blocks = []
        for item in bp_raw:
            idx   = item.get("idx", "?")
            patch = item.get("patch", "").strip()
            blocks.append(
                f"[Candidate patch #{idx} – did **not** fix the bug]\n"
                "```diff\n" + patch + "\n```"
            )
        bad_patches = "\n\n".join(blocks) or "_none supplied_"
        working_dir = working_dir or "<repo>"

        return f"""
<uploaded_files>
{working_dir}
</uploaded_files>
I've uploaded a Python code repository in **{working_dir}**.

Pull-request description
------------------------
{base}

Failed candidate patches
------------------------
{bad_patches}

Your job
--------
Analyse why the above attempts failed.
Make the minimal changes to **non-test** files so the PR requirements are met.
You may create and run reproduction scripts under `bash`.
When done, apply your fix.
""".strip()

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
    pylint_feedback: str | None = None,
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
            working_dir=str(temp_path) if mode == "codereview" else None,
            pylint_feedback=pylint_feedback,
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

        if mode == "codereview":
            review_src = temp_path / "REVIEW.md"
            if review_src.exists():
                shutil.copy(review_src, inst_dir / "REVIEW.md")

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
    style_feedback_path: Path | None,
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

    # read pylint feedback file once ---------------------------------------- #
    pylint_feedback = None
    if mode == "stylereview" and style_feedback_path:
        pylint_feedback = Path(style_feedback_path).read_text(encoding="utf-8")

    output_dir_path.mkdir(parents=True, exist_ok=True)
    preds_path = output_dir_path / "all_preds.jsonl"
    done = set()
    if preds_path.exists():
        done = {json.loads(l)["instance_id"] for l in preds_path.read_text().splitlines()}

    # iterate ---------------------------------------------------------------- #
    with preds_path.open("a+") as sink:
        for inst in tqdm(data, desc=f"Inference with {model_name}"):
            if inst["instance_id"] in done:
                continue

            for i in range(NUM_RETRIES):
                err, res = run_aider_single(
                    inst,
                    model_name, api_key, output_dir_path,
                    model_provider, mode,
                    pylint_feedback=pylint_feedback,
                )
                if res['model_patch'] is not None and res['model_patch'] != '':
                    break


            if err:
                logger.error("%s: %s", inst["instance_id"], err)
                continue

            sink.write(json.dumps(res) + "\n")
            sink.flush()


# ------------------------------ CLI entry ----------------------------------- #
if __name__ == "__main__":
    from argparse import ArgumentParser
    

    p = ArgumentParser()
    p.add_argument("-i", "--input_tasks", required=True)
    p.add_argument("-o", "--output_dir", required=True)
    p.add_argument("-m", "--model_name",
                   default="gemini/gemini-2.0-flash")
    p.add_argument("-k", "--api_key", default=None)
    p.add_argument("--model_provider", default="gemini")
    p.add_argument("--mode", default="bugfixing",
                   choices=["bugfixing", "testgen", "stylereview", "codereview"])
    p.add_argument("--instance_ids", default=None)
    p.add_argument("--style_feedback", default=None,
                   help="Path to a pylint/ruff feedback file (used with --mode stylereview)")
    args = p.parse_args()

    main(
        input_tasks_path=Path(args.input_tasks),
        output_dir_path=Path(args.output_dir),
        model_name=args.model_name,
        api_key=args.api_key,
        mode=args.mode,
        model_provider=args.model_provider.upper(),
        instance_ids=args.instance_ids.split(",") if args.instance_ids else None,
        style_feedback_path=Path(args.style_feedback) if args.style_feedback else None,
    )
