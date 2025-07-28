#!/usr/bin/env python
"""
acr_runner.py  —  AutoCodeRover batch driver (patched 2025-05-14)

Key improvements
----------------
1. --acr-root   • expands ~, resolves relative/.., defaults to script dir
2. model names  • "provider/model-id"  →  "model-id"  (ACR-friendly)
3. debug logs   • Captures ACR stdout/stderr to <output-dir>/<task>.log
"""
from __future__ import annotations
import argparse, json, logging, os, shutil, subprocess, tempfile, textwrap
from pathlib import Path
import pandas as pd
import sys
from typing import Optional

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
import os
log.info(f"Python executable: {sys.executable}")

def clone_repo(clone_url: str, commit: str, dst: Path) -> None:
    import os
    import shutil
    
    # Clean up existing directory if it exists
    if dst.exists():
        log.info(f"Removing existing directory: {dst}")
        shutil.rmtree(dst)
    
    log.info(f"Environment variables for git: {os.environ}")
    log.info(f"Cloning repo: {clone_url} into {dst}")
    try:
        result = subprocess.run(
            ["git", "clone", clone_url, str(dst)],
            check=True, capture_output=True, text=True, timeout=600
        )
        log.info(f"git clone stdout: {result.stdout}")
        log.info(f"git clone stderr: {result.stderr}")
        print(f"git clone stdout: {result.stdout}")
        print(f"git clone stderr: {result.stderr}")
    except subprocess.TimeoutExpired as e:
        log.error("git clone timed out!")
        print(f"git clone timed out! stdout: {e.stdout}, stderr: {e.stderr}")
        raise
    except subprocess.CalledProcessError as e:
        log.error(f"git clone failed: {e.stderr}")
        print(f"git clone failed! stdout: {e.stdout}, stderr: {e.stderr}")
        raise
    log.info(f"Repo cloned. Checking out commit: {commit}")
    if subprocess.run(["git", "checkout", commit], cwd=dst).returncode == 0:
        log.info(f"Checked out commit {commit} successfully.")
        return
    log.info(f"Commit not found in shallow clone. Fetching commit: {commit}")
    subprocess.run(["git", "fetch", "origin"], cwd=dst, check=True)
    log.info(f"Fetched all from origin. Checking out again: {commit}")
    subprocess.run(["git", "checkout", commit], cwd=dst, check=True)
    log.info(f"Checked out commit {commit} after fetch.")


def apply_patch(repo: Path, patch: str) -> None:
    """Apply unified diff with fuzzy fallback."""
    tmp = repo / ".acr_tmp.diff"
    tmp.write_text(patch)
    try:
        subprocess.run(["git", "apply", tmp.name], cwd=repo,
                       check=True, capture_output=True)
    except subprocess.CalledProcessError:
        subprocess.run(["patch", "-p1", "--batch", "--fuzz=5", "-i", tmp.name],
                       cwd=repo, check=True)
    tmp.unlink(missing_ok=True)


def normalise_model(model: str) -> str:
    """
    Convert model names to ACR-compatible format.
    Examples
        gemini/gemini-2.0-flash   → gemini/gemini-2.0-flash (keep as is)
        gemini/gemini-1.5-pro     → gemini/gemini-1.5-pro (keep as is)
        vertex/gemini-2.0-flash   → gemini/gemini-2.0-flash (convert provider)
        openai/gpt-4o-2024-05-13  → gpt-4o-2024-05-13 (strip provider)
    """
    # For Gemini models, convert vertex to gemini provider
    if model.startswith("vertex/gemini"):
        return model.replace("vertex/", "gemini/")
    
    # For OpenAI models, strip the provider prefix
    if model.startswith("openai/"):
        return model.split("/", 1)[1]
    
    # For other models with providers, strip the prefix
    if "/" in model and not model.startswith("gemini/"):
        return model.split("/", 1)[1]
    
    return model


def run_custom_testgen(task: dict, model: str, out_dir: Path, acr_root: Path) -> dict | None:
    """
    Custom testgen workflow that uses ACR's search capabilities but bypasses the TestAgent.
    This directly prompts the model for pytest tests after gathering context.
    """
    log = logging.getLogger(__name__)
    
    # Setup paths
    task_id = task["instance_id"]
    repo_dir = acr_root / "acr_tmp" / f"acr_{task_id}_u_{task_id}"
    work_dir = acr_root / "acr_tmp" / f"acr_{task_id}_u_{task_id}_work"
    
    try:
        # Clone repository
        clone_repo(f"https://github.com/{task['repo']}", task["base_commit"], repo_dir)
        
        # Create work directory
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # Create meta.json for ACR
        meta = {
            "task_id": task_id,
            "repo": task["repo"],
            "base_commit": task["base_commit"],
            "problem_statement": task["problem_statement"]
        }
        (work_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        
        # Add ACR's auto-code-rover directory to Python path
        import sys
        acr_app_path = acr_root / "auto-code-rover"
        if str(acr_app_path) not in sys.path:
            sys.path.insert(0, str(acr_app_path))
        
        # Initialize the model first (required for search)
        from app.model.gpt import common
        from app.model.register import register_all_models
        
        # Register models and set the selected model
        register_all_models()
        common.set_model(model)
        
        # Use ACR's SearchManager to gather context
        from app.search.search_manage import SearchManager
        from app.task import PlainTask
        from app.data_structures import MessageThread
        
        # Create a task object
        acr_task = PlainTask(
            commit_hash=task["base_commit"],
            local_path=str(repo_dir),
            problem_statement=task["problem_statement"]
        )
        
        # Initialize SearchManager
        search_manager = SearchManager(str(repo_dir), str(work_dir))
        
        # Run search to gather context (without reproducer)
        bug_locations, search_thread = search_manager.search_iterative(
            acr_task, 
            "",  # No SBFL result
            "",  # No reproducer result  
            None  # No reproduced test content
        )
        
        # Get the search conversation for context
        search_context = ""
        for msg in search_thread.messages:
            if msg["role"] == "assistant":
                search_context += f"Assistant: {msg['content']}\n\n"
            elif msg["role"] == "user":
                search_context += f"User: {msg['content']}\n\n"
        
        # Create a direct prompt for test generation
        format_example = '''
```
<file>tests/test_modeling.py</file>
<original>
# Empty if creating new file
</original>
<patched>
import pytest
from astropy.modeling.core import separability_matrix

def test_separability_matrix_nested_compound():
    """Test that separability_matrix handles nested CompoundModel correctly."""
    # Your test implementation here
    pass
</patched>
```
'''
        
        testgen_prompt = f"""
You are a software engineer tasked with writing comprehensive pytest unit tests for a bug or feature described below.

TASK: Write pytest unit tests that:
1. Reproduce the bug or test the feature described in the issue
2. Include at least one test that fails before the fix is applied (to demonstrate the bug)
3. Include tests that pass after the fix is applied (to verify the fix works)
4. Follow pytest best practices and conventions
5. Be placed in the appropriate test file within the existing test suite
6. Use descriptive test names that explain what is being tested
7. Include proper setup and teardown if needed
8. Test both the failing case and edge cases

ISSUE DESCRIPTION:
{task["problem_statement"].strip()}

CODEBASE CONTEXT (from ACR search):
{search_context}

REQUIREMENTS:
- Write the tests as a git patch that can be applied to the repository
- The patch should create or modify test files as needed
- Tests should be comprehensive and cover the described functionality
- Use pytest fixtures and assertions appropriately
- Include comments explaining the test logic where helpful
- Make sure the tests are self-contained and don't depend on external state

CRITICAL: You MUST format your response exactly as follows for each test file modification:

```
<file>path/to/test/file.py</file>
<original>
# Original code here (if modifying existing file)
# Leave empty if creating new file
</original>
<patched>
# New or modified test code here
import pytest
# Your test functions here
</patched>
```

For example:{format_example}

IMPORTANT: 
- You MUST use the exact tags: <file>, <original>, and <patched>
- You MUST wrap each modification in triple backticks (```)
- You MUST provide the actual file path in the <file> tag
- You MUST include the actual test code in the <patched> section
- You MUST close tags properly: </file>, </original>, and </patched>
- Do NOT include any explanations outside the code blocks
- Do NOT use markdown formatting inside the code blocks
- Do NOT use incomplete tags like "patched>" - use "</patched>"

Please generate a complete test suite that thoroughly covers the described issue using this exact format.
"""
        
        # Use ACR's model to generate the test (model already initialized above)
        
        # Create a message thread for the test generation
        test_thread = MessageThread()
        test_thread.add_system("You are an experienced software engineer responsible for writing comprehensive pytest unit tests.")
        test_thread.add_user(testgen_prompt)
        
        # Generate the test
        response, *_ = common.SELECTED_MODEL.call(test_thread.to_msg())
        
        # Debug: Log the model response
        log.info(f"Model response length: {len(response)}")
        log.info(f"Model response preview: {response[:500]}...")
        log.info(f"Full model response: {response}")
        
        # Extract the patch from the response
        from app.post_process import convert_response_to_diff, ExtractStatus
        
        # Create a meta.json file that convert_response_to_diff expects
        meta = {
            "task_info": {"base_commit": task["base_commit"]},
            "setup_info": {"repo_path": str(repo_dir)}
        }
        meta_file = work_dir / "meta.json"
        meta_file.write_text(json.dumps(meta))
        
        # Convert the response to a diff
        # For test generation, we want to allow test file modifications
        # So we'll manually parse the edits and skip the test file filtering
        from app.agents.patch_utils import parse_edits
        import app.utils as apputils
        
        # For test generation, we'll create a simple patch format directly from the response
        # This avoids the complexity of trying to apply edits to the repository
        try:
            raw_edits = parse_edits(response)
            if not raw_edits:
                log.warning("No edits could be parsed from the response")
                extract_status = ExtractStatus.RAW_PATCH_BUT_UNPARSED
                summary = "No edits could be parsed from the response"
                patch_content = ""
            else:
                # Create a simple unified diff format directly from the edits
                patch_lines = []
                for edit in raw_edits:
                    # Add file header
                    patch_lines.append(f"--- a/{edit.filename}")
                    patch_lines.append(f"+++ b/{edit.filename}")
                    
                    # Add the patch content
                    if edit.before.strip():
                        # If there's original content, show it as removed
                        for line in edit.before.split('\n'):
                            patch_lines.append(f"-{line}")
                    
                    # Add the new content
                    for line in edit.after.split('\n'):
                        patch_lines.append(f"+{line}")
                    
                    patch_lines.append("")  # Empty line between files
                
                patch_content = "\n".join(patch_lines)
                if patch_content.strip():
                    extract_status = ExtractStatus.APPLICABLE_PATCH
                    summary = "Successfully created patch from parsed edits"
                else:
                    extract_status = ExtractStatus.MATCHED_BUT_EMPTY_DIFF
                    summary = "No content in parsed edits"
                    patch_content = ""
                        
        except Exception as e:
            log.warning(f"Failed to parse edits: {e}")
            extract_status = ExtractStatus.RAW_PATCH_BUT_UNPARSED
            summary = f"Failed to parse edits: {e}"
            patch_content = ""
        
        if extract_status != ExtractStatus.APPLICABLE_PATCH:
            log.warning(f"Failed to extract applicable patch: {extract_status} - {summary}")
            patch_content = ""
        
        if not patch_content:
            log.error("Failed to convert response to patch")
            return None
        
        # Save the patch
        patch_file = out_dir / f"{task_id}" / "test.patch"
        patch_file.parent.mkdir(parents=True, exist_ok=True)
        patch_file.write_text(patch_content)
        
        # Create the prediction file
        pred_file = out_dir / f"{task_id}" / f"{task_id}.pred"
        pred_content = {
            "instance_id": task_id,
            "model": model,
            "response": response,
            "patch": patch_content
        }
        pred_file.write_text(json.dumps(pred_content, indent=2))
        
        log.info(f"Custom testgen completed successfully for {task_id}")
        return {
            "instance_id": task_id,
            "status": "success",
            "patch_file": str(patch_file),
            "pred_file": str(pred_file)
        }
        
    except Exception as e:
        log.error(f"Custom testgen failed for {task_id}: {e}")
        return None
    finally:
        # Cleanup - remove both work directory and repository directory
        import shutil
        if work_dir.exists():
            shutil.rmtree(work_dir)
        if repo_dir.exists():
            shutil.rmtree(repo_dir)

def run_single(task: dict, model: str, out_dir: Path, acr_root: Path, mode: str = "bugfixing", style_feedback: str | None = None, agentic: bool = False) -> dict | None:
    """
    Run a single task with the specified mode and settings.
    """
    import os
    import tempfile
    import textwrap
    import shutil
    
    log = logging.getLogger(__name__)
    
    # Setup temporary directories
    tmp_root = Path('./acr_tmp')
    tmp_root.mkdir(parents=True, exist_ok=True)
    task_id = task["instance_id"]
    work = Path(tempfile.mkdtemp(prefix=f"acr_{task_id}_", dir=tmp_root))
    repo_dir = work / "repo"
    issue_txt = work / "issue.txt"

    log.info(f"=== Starting task: {task_id} (mode: {mode}) ===")
    log.info(f"Temporary work dir: {work}")
    
    try:
        log.info(f"Task repo: {task['repo']}, base_commit: {task['base_commit']}")
        clone_repo(f"https://github.com/{task['repo']}", task["base_commit"], repo_dir)
        log.info(f"Repo ready at {repo_dir}")
        if task.get("patch"):
            log.info(f"Applying patch to repo {repo_dir}")
            apply_patch(repo_dir, task["patch"])
            log.info(f"Patch applied.")
        log.info(f"Writing issue statement to {issue_txt}")
        issue_txt.write_text(textwrap.dedent(task["problem_statement"]))
        log.info(f"Issue statement written.")

        # Normalize model name
        model_id = normalise_model(model)
        log.info(f"Normalized model: {model_id}")
        
        if mode == "testgen" and agentic:
            # Use custom testgen workflow that bypasses TestAgent
            log.info(f"Running custom testgen workflow with ACR search capabilities")
            return run_custom_testgen(task, model_id, out_dir, acr_root)
        
        if agentic:
            # Use full ACR workflow with agentic capabilities
            log.info(f"Running in FULL AGENTIC mode with search and exploration capabilities")
            
            # Modify issue content based on mode (exactly like Aider's build_prompt)
            if mode == "testgen":
                # For testgen, we need to make the TestAgent fail gracefully so ACR can continue to patch generation
                # The TestAgent looks for reproduction steps, so we'll modify the issue to not contain clear reproduction steps
                original_issue = task["problem_statement"].strip()
                
                # Completely rewrite the issue to be a feature request, not a bug report
                # This should make the TestAgent respond with "has-reproducible-example": false
                modified_issue = f"""
Feature Request: Add comprehensive test coverage for the following functionality.

{original_issue}

This is a feature request for adding test coverage. There are no reproduction steps or bug reports involved. The goal is to create comprehensive pytest unit tests that cover the described functionality.

Requirements:
- Write pytest unit tests that cover the described functionality
- Include at least one test that fails before a fix (to demonstrate the bug)
- Include tests that pass after the fix is applied
- Place tests in the appropriate test file within the existing test suite
- Follow pytest conventions and best practices
- Use descriptive test names and proper assertions
- The output should be a git patch that creates or modifies test files

Please generate comprehensive tests that thoroughly cover the described functionality.
""".strip()
                issue_txt.write_text(modified_issue)
            elif mode == "stylereview":
                feedback_text = style_feedback if style_feedback else task["problem_statement"]
                
                # Handle different types for style feedback
                if isinstance(feedback_text, list):
                    # Handle list of dictionaries (common format for style feedback)
                    if feedback_text and isinstance(feedback_text[0], dict):
                        # Extract relevant fields from each dictionary
                        feedback_lines = []
                        for item in feedback_text:
                            if isinstance(item, dict):
                                # Common fields in style feedback dictionaries
                                if 'message' in item:
                                    feedback_lines.append(item['message'])
                                elif 'text' in item:
                                    feedback_lines.append(item['text'])
                                elif 'description' in item:
                                    feedback_lines.append(item['description'])
                                else:
                                    # Fallback: convert dict to string
                                    feedback_lines.append(str(item))
                            else:
                                feedback_lines.append(str(item))
                        feedback_text = "\n".join(feedback_lines)
                    else:
                        # Handle list of strings
                        feedback_text = "\n".join(str(item) for item in feedback_text)
                elif isinstance(feedback_text, str):
                    feedback_text = feedback_text.strip()
                else:
                    feedback_text = str(feedback_text)
                
                modified_issue = f"""
You have recently generated a patch to resolve an issue within this repository.
Pylint has been run on the modified files and has produced the following
feedback:
<lint_report>
{feedback_text}
</lint_report>

Please resolve the Pylint feedback to the best of your ability, while
preserving the functionality of the code.
""".strip()
                issue_txt.write_text(modified_issue)
            elif mode == "codereview":
                # For codereview, we need to handle bad patches like Aider does
                bp_raw = task.get("bad_patches", [])
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

                modified_issue = f"""
<uploaded_files>
{str(repo_dir)}
</uploaded_files>
I've uploaded a Python code repository in **{str(repo_dir)}**.

Pull-request description
------------------------
{task["problem_statement"].strip()}

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
                issue_txt.write_text(modified_issue)
            # For bugfixing mode, use the original issue statement
            
            if agentic:
                # Create a unique run directory for this execution to avoid conflicts
                import time
                timestamp = int(time.time())
                run_name = f"acr-run-{timestamp}"
                acr_output_dir = acr_root / "auto-code-rover" / "results" / run_name
                
                cmd = ["python", "-m", "app.main", "local-issue",
                       "--output-dir", str(acr_output_dir),
                       "--model", model_id,
                       "--task-id", task_id,
                       "--local-repo", str(repo_dir),
                       "--issue-file", str(issue_txt),
                       "--model-temperature", "0.2"]
        else:
            # Pure prompting mode is not supported - only agentic mode
            log.error(f"Pure prompting mode is not supported. Use --agentic flag for full ACR capabilities.")
            return None
        
        # Set environment variable to enable LiteLLM debug mode
        env = os.environ.copy()
        env["LITELLM_DEBUG"] = "1"
        
        # Check if GEMINI_API_KEY is set
        if "GEMINI_API_KEY" in env:
            log.info(f"GEMINI_API_KEY is set (length: {len(env['GEMINI_API_KEY'])})")
            log.info(f"GEMINI_API_KEY starts with: {env['GEMINI_API_KEY'][:10]}...")
        else:
            log.error("GEMINI_API_KEY is not set in environment!")

        log.info(f"Prepared ACR command: {' '.join(cmd)} (cwd={acr_root / 'auto-code-rover'})")
        proc = subprocess.run(cmd, cwd=acr_root / "auto-code-rover",
                              capture_output=True, text=True, env=env)

        log_file = out_dir / f"{task_id}.log"
        log.info(f"Writing subprocess output to {log_file}")
        log_file.write_text(proc.stdout + "\n" + proc.stderr)

        log.info(f"ACR return code: {proc.returncode}")
        if proc.stdout:
            log.info(f"ACR stdout:\n{proc.stdout}")
        if proc.stderr:
            log.info(f"ACR stderr:\n{proc.stderr}")

        if proc.returncode != 0:
            log.error(f"ACR exited {proc.returncode} on {task_id} (see {log_file})")
            return None

        if agentic:
            # Agentic mode: Look for ACR's generated patches in the specific run directory we created
            acr_run_dir = acr_output_dir
            log.info(f"Looking for ACR results in {acr_run_dir}")
            
            if not acr_run_dir.exists():
                log.error(f"ACR run directory {acr_run_dir} does not exist")
                return None
            
            # Look for the task in different possible locations
            task_found = False
            patch_content = None
            
            # Check applicable_patch directory first
            applicable_dir = acr_run_dir / "applicable_patch"
            if applicable_dir.exists():
                for task_dir in applicable_dir.iterdir():
                    if task_dir.name.startswith(f"{task_id}_"):
                        log.info(f"Found task in applicable_patch: {task_dir}")
                        # Look for selected_patch.json
                        selected_patch_file = task_dir / "selected_patch.json"
                        if selected_patch_file.exists():
                            acr_patch_data = json.loads(selected_patch_file.read_text())
                            if "patch" in acr_patch_data:
                                patch_content = acr_patch_data["patch"]
                                task_found = True
                                break
            
            # If not found in applicable_patch, check other directories
            if not task_found:
                for subdir in ["raw_patch_but_unparsed", "raw_patch_but_unmatched"]:
                    check_dir = acr_run_dir / subdir
                    if check_dir.exists():
                        for task_dir in check_dir.iterdir():
                            if task_dir.name.startswith(f"{task_id}_"):
                                log.info(f"Found task in {subdir}: {task_dir}")
                                # Look for selected_patch.json
                                selected_patch_file = task_dir / "selected_patch.json"
                                if selected_patch_file.exists():
                                    acr_patch_data = json.loads(selected_patch_file.read_text())
                                    if "patch" in acr_patch_data:
                                        patch_content = acr_patch_data["patch"]
                                        task_found = True
                                        break
                        if task_found:
                            break
            
            if not task_found:
                log.error(f"Task {task_id} not found in any ACR output directory")
                log.info(f"Available directories in {acr_run_dir}:")
                if acr_run_dir.exists():
                    for item in acr_run_dir.iterdir():
                        log.info(f"  - {item.name}")
            if log_file.exists():
                log.info(f"Log file contents for {task_id}:\n{log_file.read_text()}")
                return None
            
            # Create Aider-compatible output directory
            aider_output_dir = out_dir / task_id
            aider_output_dir.mkdir(exist_ok=True)
            
            # Generate fix.patch file (Aider format)
            if patch_content:
                patch_file = aider_output_dir / "fix.patch"
                patch_file.write_text(patch_content)
                log.info(f"Generated Aider-compatible patch: {patch_file}")
            
            # Generate .pred file (Aider format)
            pred_data = {
                "instance_id": task_id,
                "mode": mode,
                "model_name": model_id,
                "full_output": proc.stdout + "\n" + proc.stderr,
                "model_patch": patch_content or ""
            }
            pred_file = aider_output_dir / f"{task_id}.pred"
            pred_file.write_text(json.dumps(pred_data))
            log.info(f"Generated Aider-compatible prediction: {pred_file}")
            
            return {"patch": patch_content}
    except Exception as e:
        log.error(f"Exception occurred in run_single for {task_id}: {e}", exc_info=True)
        return None
    finally:
        # Always clean up the temporary directory
        try:
            if work.exists():
                log.info(f"Cleaning up temporary directory: {work}")
                shutil.rmtree(work, ignore_errors=True)
        except Exception as cleanup_error:
            log.error(f"Failed to clean up {work}: {cleanup_error}")


# --------------------------------------------------------------------------- #
# Batch driver                                                                #
# --------------------------------------------------------------------------- #
def load_tasks(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in path.read_text().splitlines()]
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        return data["test"] if isinstance(data, dict) and "test" in data else data
    if path.suffix == ".csv":
        return pd.read_csv(path).to_dict("records")
    raise ValueError(f"Unsupported task file: {path}")


def main(args):
    print("Running ACR main")
    tasks = load_tasks(args.input_tasks)
    if args.instance_ids:
        subset = set(args.instance_ids.split(","))
        tasks = [t for t in tasks if t["instance_id"] in subset]

    # Create mode-specific output directory
    mode_output_dir = args.output_dir / f"acr_{args.mode}_outputs"
    mode_output_dir.mkdir(parents=True, exist_ok=True)
    
    done = set()
    all_preds = mode_output_dir / "all_preds.jsonl"
    if all_preds.exists():
        done = {json.loads(l)["instance_id"] for l in all_preds.read_text().splitlines()}

    for t in tasks:
        if t["instance_id"] in done:
            continue
        
        # Get style feedback from dataset for stylereview mode
        style_feedback = None
        if args.mode == "stylereview":
            if args.style_feedback:
                # Use provided style feedback file
                style_feedback = args.style_feedback.read_text(encoding="utf-8")
            elif "style_review" in t:
                # Use style feedback from dataset
                style_feedback = t["style_review"]
            else:
                log.warning(f"No style feedback found for {t['instance_id']} in stylereview mode")
        
        patch = run_single(t, args.model_name, mode_output_dir, args.acr_root, args.mode, style_feedback, args.agentic)
        with all_preds.open("a") as fp:
            fp.write(json.dumps({
                "instance_id": t["instance_id"],
                "model_name_or_path": args.model_name,
                "selected_patch": patch
            }) + "\n")


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-tasks", type=Path, required=True)
    parser.add_argument("-o", "--output-dir",  type=Path, required=True)
    parser.add_argument("-m", "--model-name",  required=True,
                        help="gemini-2.0-flash, gpt-4o-2024-05-13, etc.")
    parser.add_argument("--instance-ids", help="comma-separated subset of task ids")
    parser.add_argument("--acr-root",
                        type=lambda p: Path(p).expanduser().resolve(),
                        default=Path(__file__).parent.resolve(),
                        help="Directory that contains AutoCodeRover's app/ folder "
                             "(defaults to the script's folder)")
    parser.add_argument("--mode", choices=["bugfixing", "testgen", "stylereview", "codereview"], default="bugfixing",
                        help="Mode for the ACR runner (default: bugfixing)")
    parser.add_argument("--style-feedback", type=Path, help="File containing style feedback for stylereview mode")
    parser.add_argument("--agentic", action="store_true", help="Run ACR in agentic mode (required - only agentic mode is supported)")
    main(parser.parse_args())
