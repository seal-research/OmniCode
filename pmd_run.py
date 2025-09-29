#!/usr/bin/env python3
"""
pmd_run.py

Usage:
  python pmd_run.py --max_workers 1 --instance_ids <org>/<repo>:<pull> --review_type pmd

What it does:
- Parses instance ids like "apache/dubbo:10638" (comma-separated allowed)
- Ensures directories: data/java_style_review/<org>/<repo>/style_review/style-review-<pull>/
- Clones the repo into .../repo if not already present (assumes GitHub URL https://github.com/<org>/<repo>.git)
- Writes the provided bash PMD style review script into a file
- Generates an Apptainer (Singularity) definition that installs Java, xmllint, jq, git and downloads PMD
- Attempts to build an Apptainer image and execute the script inside it with /workspace bound to host outputdir parent
- Falls back to running the script on the host if Apptainer isn't available or the build fails
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

# -------------------------
# The PMD style review bash script (exactly as provided)
# -------------------------
PMD_BASH_SCRIPT = r"""#!/bin/bash
set -e

run_style_review() {
    local patch_file="$1"
    local output_dir="$2"

    echo "=== Starting style review ==="
    echo "Patch file: $patch_file"
    echo "Output directory: $output_dir"
    echo "Current working directory: $(pwd)"
    echo "Workspace contents (ls -la /workspace):"
    ls -la /workspace/ 2>/dev/null || echo "No /workspace directory"
    echo "Full directory tree (find /workspace):"
    find /workspace -type d 2>/dev/null || echo "No directories found"
    echo "All files in /workspace (ls -lR /workspace):"
    ls -lR /workspace 2>/dev/null || echo "No files found"

    # Make a safe directory derived from output_dir — never use raw input path directly
    safe_output_dir="/workspace/output_dir_$(date +%s%N)"
    echo "Safe output directory: $safe_output_dir"

    echo "Using mkdir"
    mkdir -p "$safe_output_dir"

    # Initialize default results immediately
    echo '{
        "global_score": 10.0,
        "total_errors": 0,
        "total_warnings": 0
    }' > "$safe_output_dir/style_report.json"
    echo "[]" > "$safe_output_dir/style_errors.json"

    # Handle patch application with comprehensive error handling
    PATCH_STATUS="not_attempted"
    if [ -f "$patch_file" ] && [ "$patch_file" != "/dev/null" ]; then
        echo "Applying patch: $patch_file"
        echo "Patch file contents (first 10 lines):"
        head -10 "$patch_file" 2>/dev/null || echo "Could not read patch file"
        patch_errors_file="$safe_output_dir/patch_errors.log"
        if (cd /workspace/repo && git apply --check "$patch_file" 2>"$patch_errors_file"); then
            if (cd /workspace/repo && git apply --reject --whitespace=fix "$patch_file" 2>>"$patch_errors_file"); then
                echo "Patch applied successfully" | tee -a "$safe_output_dir/patch_status.log"
                PATCH_STATUS="applied"
            else
                echo "Patch partially applied or with warnings. See $patch_errors_file for details." | tee -a "$safe_output_dir/patch_status.log"
                PATCH_STATUS="partial"
            fi
        else
            echo "Patch could NOT be applied at all. See $patch_errors_file for details." | tee -a "$safe_output_dir/patch_status.log"
            PATCH_STATUS="failed"
            fi
        echo "PATCH_STATUS=$PATCH_STATUS" | tee -a "$safe_output_dir/patch_status.log"
    elif [ "$patch_file" = "/dev/null" ]; then
        echo "No patch to apply (original state)" | tee -a "$safe_output_dir/patch_status.log"
        PATCH_STATUS="none"
        echo "PATCH_STATUS=$PATCH_STATUS" | tee -a "$safe_output_dir/patch_status.log"
    else
        echo "No patch file found at $patch_file" > "$safe_output_dir/error.log"
        echo "Continuing with analysis without patch..." | tee -a "$safe_output_dir/patch_status.log"
        PATCH_STATUS="missing"
        echo "PATCH_STATUS=$PATCH_STATUS" | tee -a "$safe_output_dir/patch_status.log"
    fi

    # Find Java files to analyze - try multiple approaches
    echo "Finding Java files to analyze..."
    # For original state (no patch), analyze all Java files in the repository
    if [ "$patch_file" = "/dev/null" ]; then
        echo "Original state analysis - looking for all Java files in repository..."
        # Search for all Java files in the repository
        java_dirs=$(find /workspace/repo -type d -name java 2>/dev/null | head -10 || true)
        if [ -z "$java_dirs" ]; then
            java_dirs="/workspace/repo"
        fi
    else
        # For patched state, analyze all Java files as well
        java_dirs=$(find /workspace/repo -type d -name java 2>/dev/null | head -10 || true)
        if [ -z "$java_dirs" ]; then
            java_dirs="/workspace/repo"
        fi
    fi

    # Find all Java files to analyze (for accurate total_files count)
    all_java_files=$(find $java_dirs -name "*.java" -type f 2>/dev/null)
    total_files=$(echo "$all_java_files" | wc -w)
    total_errors=0
    echo "[]" > "$safe_output_dir/style_errors.json"

    temp_dir=$(mktemp -d)
    trap 'rm -rf "$temp_dir"' EXIT
    pmd_error_log="$safe_output_dir/pmd_error.log"
    pmd_output_xml="$temp_dir/pmd_output.xml"
    echo "Running PMD on: $java_dirs"
    # Use -r option to suppress progress bar warning and output to file
    if ! pmd check -d $java_dirs -R /workspace/pmd-ruleset.xml -f xml -r "$pmd_output_xml" 2> "$pmd_error_log"; then
        echo "PMD failed to analyze some files. See $pmd_error_log for details."
    fi

    # Count total violations directly from the XML for robust scoring
    if [ -s "$pmd_output_xml" ]; then
        total_errors=$(grep -c '<violation ' "$pmd_output_xml")
    fi

    # Parse PMD XML output and build style_errors.json (per-file breakdown)
    if [ -s "$pmd_output_xml" ]; then
        # Use xmllint to extract all <file> nodes and their violations
        xmllint --xpath '//file' "$pmd_output_xml" 2>/dev/null | \
        awk -v q="\"" 'BEGIN{RS="<file ";FS="</file>"} NR>1{print "<file " $1}' | while read -r file_block; do
            file_path=$(echo "$file_block" | grep -o 'name="[^"]*"' | head -1 | cut -d'"' -f2)
            error_count=$(echo "$file_block" | grep -c '<violation ')
            file_score=$(echo "scale=1; 10 - $error_count * 0.5" | bc 2>/dev/null || echo "10.0")
            if (( $(echo "$file_score < 0" | bc -l 2>/dev/null || echo "0") )); then
                file_score="0.0"
            fi
            # Extract all violation details for this file
            error_json="["
            while read -r vline; do
                # Extract attributes and message
                beginline=$(echo "$vline" | grep -o 'beginline="[^"]*"' | cut -d'"' -f2)
                begincolumn=$(echo "$vline" | grep -o 'begincolumn="[^"]*"' | cut -d'"' -f2)
                rule=$(echo "$vline" | grep -o 'rule="[^"]*"' | cut -d'"' -f2)
                message=$(echo "$vline" | sed -n 's/.*<violation[^>]*>\(.*\)<\/violation>.*/\1/p' | sed 's/"/\\"/g')
                if [ -n "$error_json" ] && [ "$error_json" != "[" ]; then
                    error_json+=",";
                fi
                error_json+="{\"line\": ${beginline:-0}, \"column\": ${begincolumn:-0}, \"type\": \"error\", \"message\": \"${message}\", \"source\": \"${rule}\"}"
            done < <(echo "$file_block" | grep '<violation ')
            error_json+="]"
            # Write file report JSON
            file_report="{\n  \"file\": \"$file_path\", \"score\": $file_score, \"error_count\": $error_count, \"messages\": $error_json\n}"
            jq -s '.[0] + [.[1]]' "$safe_output_dir/style_errors.json" <(echo "$file_report") > "$temp_dir/tmp.json" 2>/dev/null || true
            if [ -f "$temp_dir/tmp.json" ]; then
                mv "$temp_dir/tmp.json" "$safe_output_dir/style_errors.json" 2>/dev/null || true
            fi
        done
    fi

    # Calculate global score with error handling
    global_score=10.0
    if [ "$total_files" -gt 0 ]; then
        global_score=$(echo "scale=1; 10 - ($total_errors / $total_files) * 0.5" | bc 2>/dev/null || echo "10.0")
        if (( $(echo "$global_score < 0" | bc -l 2>/dev/null || echo "0") )); then
            global_score="0.0"
        fi
    fi

    echo "Final statistics: total_files=$total_files, total_errors=$total_errors, global_score=$global_score"
    echo "{\n    \"global_score\": $global_score,\n    \"total_errors\": $total_errors,\n    \"total_warnings\": 0\n}" > "$safe_output_dir/style_report.json"

    # Copy results to the specified output directory with comprehensive error handling
    if [ -n "$output_dir" ]; then
        echo "Copying results to: $output_dir"
        mkdir -p "$output_dir" 2>/dev/null || true
        cp "$safe_output_dir/style_report.json" "$output_dir/original_style_report.json" 2>/dev/null || true
        cp "$safe_output_dir/style_errors.json" "$output_dir/original_style_errors.json" 2>/dev/null || true
        [ -f "$safe_output_dir/patch_warning.log" ] && cp "$safe_output_dir/patch_warning.log" "$output_dir/patch_warning.log" 2>/dev/null || true
        [ -f "$safe_output_dir/patch_errors.log" ] && cp "$safe_output_dir/patch_errors.log" "$output_dir/patch_errors.log" 2>/dev/null || true
        [ -f "$safe_output_dir/error.log" ] && cp "$safe_output_dir/error.log" "$output_dir/error.log" 2>/dev/null || true
        [ -f "$pmd_error_log" ] && cp "$pmd_error_log" "$output_dir/pmd_error.log" 2>/dev/null || true
        # Copy the full PMD XML output
        [ -f "$pmd_output_xml" ] && cp "$pmd_output_xml" "$output_dir/pmd_output.xml" 2>/dev/null || true
        [ -f "$safe_output_dir/patch_status.log" ] && cp "$safe_output_dir/patch_status.log" "$output_dir/patch_status.log" 2>/dev/null || true
    fi

    echo "\n==== FULL PMD VIOLATION XML OUTPUT ===="
    if [ -f "$pmd_output_xml" ]; then
        cat "$pmd_output_xml"
    else
        echo "No PMD XML output found."
    fi
    echo "==== END OF PMD VIOLATION XML OUTPUT ===="

    echo "Style review completed successfully"
    echo "=== Style review finished ==="
    return 0
}

# Call the function with the provided arguments
run_style_review "$@"
"""

# -------------------------
# Apptainer/Singularity definition template
# -------------------------
APPTAINER_DEF = """Bootstrap: docker
From: ubuntu:22.04

%labels
    Author pmd_run_script
    Version 1.0
%end

%post
    apt-get update -y
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        openjdk-17-jre-headless wget unzip git jq libxml2-utils ca-certificates \
        python3 python3-pip curl

    # create a place for pmd
    mkdir -p /opt/pmd

    # Try to download PMD binary (latest). If it fails, container will still have tools installed.
    echo "Attempting to download PMD..."
    set -e
    cd /opt
    PMD_ZIP_URL="https://github.com/pmd/pmd/releases/latest/download/pmd-bin.zip"
    if wget -q -O pmd-bin.zip "$PMD_ZIP_URL"; then
        unzip -q pmd-bin.zip -d /opt
        # the zip usually expands to pmd-bin-<version>
        for d in /opt/pmd-bin-*; do
            if [ -d "$d" ]; then
                mv "$d" /opt/pmd || true
                break
            fi
        done
        # fallback if move didn't happen
        [ -d /opt/pmd ] || mkdir -p /opt/pmd
        ln -sf /opt/pmd/bin/run.sh /usr/local/bin/pmd || true
    else
        echo "Warning: Could not download PMD during build. Ensure PMD is available at runtime."
    fi

    # make sure /opt/run_style_review.sh exists (files section will copy it)
    chmod +x /opt/run_style_review.sh || true

%files
    run_style_review.sh /opt/run_style_review.sh

%environment
    export PATH=/opt/pmd/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

%runscript
    echo "This container contains the PMD run script at /opt/run_style_review.sh"
    echo "To run: apptainer exec --bind <host_dir>:/workspace <this_image> bash /opt/run_style_review.sh /dev/null /workspace/<outputdir>"
%end
"""

# -------------------------
# Helpers
# -------------------------
def run_cmd(cmd, **kwargs):
    """Run a subprocess command and return CompletedProcess. Raises subprocess.CalledProcessError on error."""
    print(f"+ {' '.join(cmd)}")
    return subprocess.run(cmd, check=True, **kwargs)


def ensure_git_clone(org: str, repo: str, repo_dir: Path):
    if repo_dir.exists() and (repo_dir / ".git").exists():
        print(f"Repo already present at {repo_dir} — fetching latest (shallow).")
        try:
            run_cmd(["git", "-C", str(repo_dir), "fetch", "--all", "--depth", "1"])
            run_cmd(["git", "-C", str(repo_dir), "reset", "--hard", "origin/HEAD"])
        except subprocess.CalledProcessError:
            print("Failed to update existing repo — leaving as-is.")
    else:
        print(f"Cloning https://github.com/{org}/{repo}.git into {repo_dir}")
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            run_cmd(["git", "clone", "--depth", "1", f"https://github.com/{org}/{repo}.git", str(repo_dir)])
        except subprocess.CalledProcessError:
            print(f"Warning: clone failed for https://github.com/{org}/{repo}.git. Create the repo directory manually if it's private.")
            repo_dir.mkdir(parents=True, exist_ok=True)


def build_and_run_apptainer(script_path: Path, output_dir: Path, workspace_parent: Path):
    """
    Build an Apptainer image (sandbox directory) and run the script inside it.
    Returns True if run succeeded, False otherwise.
    """

    # check if apptainer exists
    apptainer_exec = shutil.which("apptainer") or shutil.which("singularity")
    if not apptainer_exec:
        print("Apptainer (or singularity) not found on PATH. Skipping container build/run.")
        return False

    print(f"Using Apptainer binary at: {apptainer_exec}")

    with tempfile.TemporaryDirectory() as build_dir:
        build_dir = Path(build_dir)
        def_file = build_dir / "Apptainer.def"
        # write the script into build_dir so %files copies it
        build_script = build_dir / "run_style_review.sh"
        shutil.copy(script_path, build_script)
        def_file.write_text(APPTAINER_DEF)
        image_sandbox = build_dir / "image.sandbox"

        # Build the sandbox (may require privileges)
        try:
            print("Building Apptainer sandbox image (may require root)...")
            run_cmd([apptainer_exec, "build", "--sandbox", str(image_sandbox), str(def_file)])
        except subprocess.CalledProcessError as e:
            print(f"Apptainer build failed: {e}. Trying to continue by using a remote image or skipping container path.")
            return False

        # Run the script inside the built sandbox, binding the output directory's parent to /workspace
        try:
            bind_spec = f"{workspace_parent}:{'/workspace'}"
            out_dir_container = "/workspace/" + output_dir.relative_to(workspace_parent).as_posix()
            print(f"Executing script inside container; binding {workspace_parent} -> /workspace")
            run_cmd([apptainer_exec, "exec", "--bind", bind_spec, str(image_sandbox),
                     "bash", "/opt/run_style_review.sh", "/dev/null", out_dir_container],
                    cwd=str(workspace_parent))
            return True
        except subprocess.CalledProcessError as e:
            print(f"Execution inside Apptainer failed: {e}")
            return False


def run_script_directly(script_path: Path, output_dir: Path, repo_dir: Path):
    """
    Run the script directly on the host. We bind-mount by ensuring the script sees /workspace as the workspace_parent.
    We'll execute it with working dir = repo_dir.parent (so relative paths behave).
    """
    print("Attempting to run the style review script directly on the host.")
    # create a temporary dir to act as a workspace root and symlink the repo into it at /workspace/repo
    with tempfile.TemporaryDirectory() as tmp_workspace:
        tmp_workspace = Path(tmp_workspace)
        # create /workspace/repo inside temp workspace (just copy or symlink)
        workspace_repo = tmp_workspace / "repo"
        if not workspace_repo.exists():
            try:
                # attempt a lightweight copy if repo_dir is on the same filesystem
                if repo_dir.exists():
                    print(f"Linking repo {repo_dir} into temporary workspace {workspace_repo}")
                    try:
                        os.symlink(str(repo_dir.resolve()), str(workspace_repo))
                    except Exception:
                        # fallback to copytree (may be heavy)
                        shutil.copytree(str(repo_dir), str(workspace_repo))
                else:
                    workspace_repo.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"Warning: failed to prepare workspace repo symlink/copy: {e}")
        # create a pmd-ruleset.xml placeholder if not present (script expects /workspace/pmd-ruleset.xml)
        pmd_rules = tmp_workspace / "pmd-ruleset.xml"
        if not pmd_rules.exists():
            pmd_rules.write_text("<ruleset name='default'></ruleset>")

        out_dir_in_workspace = Path("/workspace") / output_dir.relative_to(output_dir.anchor)  # will be translated via bind

        # Run the script with /workspace mapped to tmp_workspace using env var and working dir
        env = os.environ.copy()
        # Execute script with cwd = tmp_workspace so it can find repo under /workspace/repo
        try:
            # run bash with script path, args: /dev/null and output dir path inside /workspace
            subprocess.run(["bash", str(script_path), "/dev/null", str(output_dir)], check=True, cwd=str(tmp_workspace), env=env)
            print("Script executed on host (results should be in the given output directory if the script wrote to it).")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Direct execution failed: {e}")
            return False


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Run PMD style review via Apptainer (or host fallback).")
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--instance_ids", type=str, required=True,
                        help="Comma-separated list of <org>/<repo>:<pull_number> entries")
    parser.add_argument("--review_type", type=str, required=True, choices=["pmd"],
                        help="Type of review to perform (must be 'pmd')")
    args = parser.parse_args()

    if args.review_type.lower() != "pmd":
        print("Only 'pmd' review_type is supported by this script.")
        sys.exit(2)

    instance_list = [x.strip() for x in args.instance_ids.split(",") if x.strip()]
    if not instance_list:
        print("No instance_ids provided.")
        sys.exit(2)

    # Write the bash script to a safe temp file
    tmp = tempfile.mkdtemp(prefix="pmd_run_")
    script_path = Path(tmp) / "run_style_review.sh"
    script_path.write_text(PMD_BASH_SCRIPT)
    script_path.chmod(0o755)

    # We'll attempt apptainer build & run; fallback to host execution if necessary
    any_failure = False
    for instance in instance_list:
        # parse org/repo:pull
        if ":" not in instance:
            print(f"Skipping malformed instance id (missing colon): {instance}")
            any_failure = True
            continue
        repo_part, pull_number = instance.rsplit(":", 1)
        if "/" not in repo_part:
            print(f"Skipping malformed repo part (expected org/repo): {repo_part}")
            any_failure = True
            continue
        org, repo = repo_part.split("/", 1)
        # construct directories
        out_base = Path("data") / "java_style_review" / org / repo / "style_review"
        out_dir = out_base / f"style-review-{pull_number}"
        repo_dir = out_dir / "repo"

        print(f"\n== Processing instance: {instance} ==")
        print(f"Output dir will be: {out_dir}")
        # ensure parent directories exist
        out_dir.mkdir(parents=True, exist_ok=True)

        # clone repo if needed
        ensure_git_clone(org, repo, repo_dir)

        # PMD ruleset: allow user to provide a pmd-ruleset.xml at top-level of repo clone; if not present, create a minimal one
        ruleset_host = out_dir / "pmd-ruleset.xml"
        if not ruleset_host.exists():
            # if the cloned repo has one at root, copy it into out_dir
            candidate = repo_dir / "pmd-ruleset.xml"
            if candidate.exists():
                shutil.copy(candidate, ruleset_host)
            else:
                # create minimal placeholder that matches the script expectations
                ruleset_host.write_text("<ruleset name='default'></ruleset>")

        # The script expects to write into the provided output directory under /workspace
        # We'll design workspace_parent to be the 'data/java_style_review/<org>/<repo>/style_review' so that container /workspace maps to that
        workspace_parent = out_base.resolve()

        # Attempt Apptainer flow
        appt_success = build_and_run_apptainer(script_path, out_dir.resolve(), workspace_parent)
        if not appt_success:
            print("Apptainer path failed or unavailable — attempting to run the script directly on host.")
            host_success = run_script_directly(script_path, out_dir.resolve(), repo_dir.resolve())
            if not host_success:
                print(f"ERROR: Both Apptainer and host execution failed for instance {instance}. Check logs.")
                any_failure = True
            else:
                print(f"Host execution succeeded for instance {instance}.")
        else:
            print(f"Apptainer execution succeeded for instance {instance}.")

    # cleanup
    try:
        shutil.rmtree(tmp)
    except Exception:
        pass

    if any_failure:
        print("\nOne or more instances failed. Exit code 1.")
        sys.exit(1)
    else:
        print("\nAll instances processed. Exit code 0.")
        sys.exit(0)


if __name__ == "__main__":
    main()
