#!/usr/bin/env python3
"""
mswebench_check_pass_to_pass_java_friendly.py

Same usage as previous script, but adds Maven/Gradle detection and runs tests
with the appropriate Java build tool (mvn/gradle).
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from typing import List, Tuple, Dict, Optional

# ---------- Helpers ----------
def run_cmd(cmd: List[str], cwd: Optional[str] = None, stream_output: bool = True, env=None) -> Tuple[int, str, str]:
    print(f"\n>>> Running: {' '.join(shlex.quote(x) for x in cmd)} (cwd={cwd or os.getcwd()})\n", flush=True)
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    stdout_lines = []
    stderr_lines = []

    while True:
        out = proc.stdout.readline()
        err = proc.stderr.readline()
        if out:
            stdout_lines.append(out)
            if stream_output:
                print(out, end="", flush=True)
        if err:
            stderr_lines.append(err)
            if stream_output:
                print(err, end="", flush=True)
        if out == "" and err == "" and proc.poll() is not None:
            break

    out_rest, err_rest = proc.communicate()
    if out_rest:
        stdout_lines.append(out_rest)
        if stream_output:
            print(out_rest, end="", flush=True)
    if err_rest:
        stderr_lines.append(err_rest)
        if stream_output:
            print(err_rest, end="", flush=True)

    stdout = "".join(stdout_lines)
    stderr = "".join(stderr_lines)
    return proc.returncode, stdout, stderr

# ---------- mswebench JSON helpers ----------
def load_instances(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def find_instance(instances, org: str, repo: str, pr: int):
    want = f"{org}/{repo}"
    for inst in instances:
        if inst.get("repo") == want and int(inst.get("pull_number", -1)) == int(pr):
            return inst
    return None

# ---------- Build-system detection ----------
def detect_build_system(repo_path: str) -> str:
    """Return 'maven', 'gradle', or 'unknown'."""
    if os.path.exists(os.path.join(repo_path, "pom.xml")):
        return "maven"
    if os.path.exists(os.path.join(repo_path, "build.gradle")) or os.path.exists(os.path.join(repo_path, "build.gradle.kts")):
        return "gradle"
    return "unknown"

def which_wrapper_or_tool(repo_path: str, wrapper_name: str, tool_name: str) -> Optional[str]:
    """Prefer wrapper in repo (./mvnw or ./gradlew), else fallback to tool in PATH (mvn/gradle)."""
    wrapper_path = os.path.join(repo_path, wrapper_name)
    if os.path.isfile(wrapper_path) and os.access(wrapper_path, os.X_OK):
        return wrapper_path
    # fall back to system tool if available
    from shutil import which
    tool_path = which(tool_name)
    return tool_path

# ---------- Java test runners ----------
def run_maven_test(repo_path: str, test_name: str) -> Tuple[str, str]:
    """
    Run a single test with Maven. Returns (status, output).
    status in {"passed","failed","missing","error"}.
    """
    mvn = which_wrapper_or_tool(repo_path, "mvnw", "mvn")
    if not mvn:
        return "error", "mvn/mvnw not found on PATH and no mvnw in repo."

    # Maven's -Dtest accepts FQCN in many cases, but to be safer we pass the FQCN first.
    # Use -DfailIfNoTests=false so the invocation returns success if no test matched (we'll mark 'missing').
    cmd = [mvn, "-Dtest=" + test_name, "-DfailIfNoTests=false", "test", "-DtrimStackTrace=false"]
    rc, out, err = run_cmd(cmd, cwd=repo_path)
    combined = out + err

    if rc == 0:
        # Maven exit 0: tests passed (or none matched but failIfNoTests=false)
        # We need to detect whether the test actually ran. Search surefire output for matching test class.
        # Check surefire output for "There are no tests to run." or surefire reports.
        if re.search(r"(?i)there are no tests to run", combined) or re.search(r"(?i)no tests were executed", combined):
            return "missing", combined
        # Check surefire for failures summary
        if re.search(r"\[INFO\] BUILD SUCCESS", combined) and not re.search(r"Failures: 0, Errors: 0", combined) and not re.search(r"Tests run: \d+, Failures: 0, Errors: 0", combined):
            # Hmm - ambiguous; fall back to treat as passed if build success and no failure indicators
            return "passed", combined
        # Look for surefire summary line
        # Example: "Tests run: 10, Failures: 0, Errors: 0, Skipped: 0"
        m = re.search(r"Tests run:\s*\d+,\s*Failures:\s*(\d+),\s*Errors:\s*(\d+)", combined)
        if m:
            failures = int(m.group(1))
            errors = int(m.group(2))
            if failures == 0 and errors == 0:
                return "passed", combined
            else:
                return "failed", combined
        # Conservative: treat as passed if rc==0 and no explicit failure found
        return "passed", combined
    else:
        # Non-zero exit code -> failure or build/test error.
        # If output contains "There are no tests to run" treat as missing
        if re.search(r"(?i)there are no tests to run", combined) or re.search(r"(?i)no tests were executed", combined):
            return "missing", combined
        return "failed", combined

def run_gradle_test(repo_path: str, test_name: str) -> Tuple[str, str]:
    """
    Run a single test with Gradle. Returns (status, output).
    Gradle --tests accepts FQCN or patterns.
    """
    gradle = which_wrapper_or_tool(repo_path, "gradlew", "gradle")
    if not gradle:
        return "error", "gradle/gradlew not found on PATH and no gradlew in repo."

    # Gradle tends to require the test class pattern; we pass the FQCN intact.
    cmd = [gradle, "test", "--no-daemon", "--console", "plain", "--tests", test_name]
    rc, out, err = run_cmd(cmd, cwd=repo_path)
    combined = out + err

    if rc == 0:
        # Check for "No tests found for given includes"
        if re.search(r"No tests found for given includes", combined) or re.search(r"(?i)0 tests completed", combined):
            return "missing", combined
        # Otherwise assume passed
        return "passed", combined
    else:
        # Non-zero -> could be failure or missing
        if re.search(r"No tests found for given includes", combined) or re.search(r"(?i)0 tests completed", combined):
            return "missing", combined
        return "failed", combined

# ---------- Main evaluation ----------
def evaluate_pass_to_pass(repo_path: str, pass_to_pass: List[str]) -> Dict[str, str]:
    results = {}
    if not pass_to_pass:
        print("No PASS_TO_PASS tests listed.")
        return results

    build_system = detect_build_system(repo_path)
    print(f"Detected build system: {build_system}")

    if build_system == "unknown":
        print("No pom.xml or Gradle build file detected. Trying CTest/pytest fallback (old behavior).")
        # Optionally, you could reuse previous CTest/pytest logic here, but for Java repos this is unlikely.
        # Mark everything as error to be explicit
        for t in pass_to_pass:
            results[t] = "error"
        return results

    # Optionally build the project first for Maven (mvn test will build automatically)
    # We'll attempt per-test runs (this can be slow but is explicit).
    for test in pass_to_pass:
        if build_system == "maven":
            status, output = run_maven_test(repo_path, test)
        else:
            status, output = run_gradle_test(repo_path, test)
        results[test] = status

    return results

def summarize_and_exit(results: Dict[str, str]):
    if not results:
        print("No tests were evaluated.")
        sys.exit(2)

    passed = [t for t, s in results.items() if s == "passed"]
    failed = [t for t, s in results.items() if s == "failed"]
    missing = [t for t, s in results.items() if s == "missing"]
    error = [t for t, s in results.items() if s == "error"]

    print("\n\n=== SUMMARY ===")
    print(f"Total listed tests: {len(results)}")
    print(f"Passed: {len(passed)}")
    print(f"Failed: {len(failed)}")
    print(f"Missing: {len(missing)}")
    print(f"Errors (tooling): {len(error)}")

    if passed:
        print("\nPassed tests:")
        for t in passed:
            print("  ", t)
    if failed:
        print("\nFailed tests:")
        for t in failed:
            print("  ", t)
    if missing:
        print("\nMissing tests:")
        for t in missing:
            print("  ", t)
    if error:
        print("\nErrors:")
        for t in error:
            print("  ", t)

    if len(failed) == 0 and len(missing) == 0 and len(error) == 0:
        print("\nALL PASS_TO_PASS tests passed ✅")
        sys.exit(0)
    else:
        print("\nNOT ALL PASS_TO_PASS tests passed ❌")
        sys.exit(1)

def main():
    p = argparse.ArgumentParser(description="Check PASS_TO_PASS tests from mswebench_instances.json for a repo (Java-friendly)")
    p.add_argument("--instances", required=True, help="mswebench_instances.json path")
    p.add_argument("--repo-path", required=True, help="Path to local checkout of the repo (where tests will run)")
    p.add_argument("--org", required=True, help="GitHub org (used to find the instance)")
    p.add_argument("--repo", required=True, help="Repo name (used to find the instance)")
    p.add_argument("--pr", required=True, type=int, help="Pull request number (pull_number in JSON)")
    args = p.parse_args()

    if not os.path.isfile(args.instances):
        print(f"Instances file not found: {args.instances}", file=sys.stderr)
        sys.exit(2)
    instances = load_instances(args.instances)
    inst = find_instance(instances, args.org, args.repo, args.pr)
    if inst is None:
        print(f"No instance found for {args.org}/{args.repo} pr={args.pr}", file=sys.stderr)
        sys.exit(2)

    pass_to_pass = inst.get("PASS_TO_PASS", [])
    print(f"Found instance for {args.org}/{args.repo} pr={args.pr}. PASS_TO_PASS count: {len(pass_to_pass)}")

    if not os.path.isdir(args.repo_path):
        print(f"Repo path not found: {args.repo_path}", file=sys.stderr)
        sys.exit(2)

    results = evaluate_pass_to_pass(args.repo_path, pass_to_pass)
    summarize_and_exit(results)

if __name__ == "__main__":
    main()
