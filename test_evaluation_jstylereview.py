#!/usr/bin/env python3
"""
sweagent_run_and_test.py

Runs PASS_TO_PASS test class names by invoking Maven/Gradle from the repo root
using reactor-aware flags so BOM/parent POM resolution works (fixes the
'Non-resolvable import POM' errors you saw).
"""
import argparse
import json
import os
import subprocess
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger('sweagent')


def run(cmd: List[str], cwd: Path = None, timeout: int = 1800) -> Dict[str, Any]:
    try:
        proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, timeout=timeout)
        return {"ret": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
    except subprocess.TimeoutExpired as e:
        return {"ret": -124, "stdout": e.stdout or "", "stderr": (e.stderr or "") + "\n[timeout]"}
    except Exception as e:
        return {"ret": -1, "stdout": "", "stderr": str(e)}


# ----------------- build-tool helpers -----------------


def detect_build_tool(repo_path: Path) -> Tuple[Optional[str], Optional[str]]:
    mvnw = repo_path / 'mvnw'
    pom = repo_path / 'pom.xml'
    gradlew = repo_path / 'gradlew'
    build_gradle = repo_path / 'build.gradle'
    build_gradle_kts = repo_path / 'build.gradle.kts'

    if mvnw.exists():
        return ('maven', str(mvnw.resolve()))
    if pom.exists():
        return ('maven', 'mvn')
    if gradlew.exists():
        return ('gradle', str(gradlew.resolve()))
    if build_gradle.exists() or build_gradle_kts.exists():
        return ('gradle', 'gradle')
    return (None, None)


def find_module_for_class(repo_path: Path, test_class: str) -> Optional[Path]:
    simple_name = test_class.split('.')[-1]
    candidates = list(repo_path.rglob(simple_name + '.java'))
    # prefer exact package match
    fqpath = '/'.join(test_class.split('.')) + '.java'
    for cand in candidates:
        try:
            rel = str(cand.relative_to(repo_path)).replace(os.sep, '/')
            if fqpath in rel:
                cur = cand.parent
                while cur != repo_path and cur.exists():
                    if (cur / 'pom.xml').exists() or (cur / 'build.gradle').exists() or (cur / 'build.gradle.kts').exists():
                        return cur
                    cur = cur.parent
        except Exception:
            continue
    # fallback: any candidate -> climb to module root
    for cand in candidates:
        cur = cand.parent
        while cur != repo_path and cur.exists():
            if (cur / 'pom.xml').exists() or (cur / 'build.gradle').exists() or (cur / 'build.gradle.kts').exists():
                return cur
            cur = cur.parent
    return None


def _relpath_for_module(repo_root: Path, module_dir: Path) -> str:
    """Return module relative path (posix) from repo_root."""
    try:
        rel = module_dir.relative_to(repo_root)
        return rel.as_posix()
    except Exception:
        return ''


def _gradle_project_path_from_rel(rel: str) -> str:
    """Convert relative path 'a/b/c' -> ':a:b:c' for Gradle project path."""
    if not rel:
        return ''
    return ':' + rel.replace('/', ':')


def make_test_commands(repo_path: Path, test_classes: List[str]) -> List[Dict[str, Any]]:
    """Return list of {'cmd': str, 'cwd': Path} to execute from repo root so reactor works."""
    tool, runner = detect_build_tool(repo_path)
    if tool is None:
        logger.warning("No Maven/Gradle detected in repo - skipping tests.")
        return []

    commands = []
    for tc in [t.strip() for t in test_classes if isinstance(t, str) and t.strip()]:
        module_dir = find_module_for_class(repo_path, tc) or repo_path
        rel = _relpath_for_module(repo_path, module_dir)
        if tool == 'maven':
            # run from repo root so reactor can provide BOM/parent
            # -pl accepts module path (directory or artifact id). Using relative path works.
            # -am ensures required modules are built.
            # Use failIfNoTests=false so missing class doesn't break everything.
            if rel:
                cmd = f"{runner} -pl {rel} -am -Dtest={tc} -DfailIfNoTests=false test"
            else:
                cmd = f"{runner} -Dtest={tc} -DfailIfNoTests=false test"
            commands.append({"cmd": cmd, "cwd": repo_path})
        else:  # gradle
            # Gradle: use project path like :module:submodule:test
            gp = _gradle_project_path_from_rel(rel)
            if gp:
                # e.g. ./gradlew :module:test --tests "FQCN"
                cmd = f"{runner} {gp}:test --tests \"{tc}\""
            else:
                cmd = f"{runner} test --tests \"{tc}\""
            commands.append({"cmd": cmd, "cwd": repo_path})
    return commands


# ----------------- test execution -----------------


def run_tests(commands: List[Dict[str, Any]], instance_id: str, timeout_each: int = 1800) -> List[Dict[str, Any]]:
    results = []
    for i, item in enumerate(commands):
        cmd = item.get('cmd')
        cwd = Path(item.get('cwd', '.'))
        logger.info(f"[{instance_id}] Running {i+1}/{len(commands)} in {cwd}: {cmd}")
        try:
            proc = subprocess.run(cmd, cwd=str(cwd), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout_each)
            results.append({"command": cmd, "ret": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr, "cwd": str(cwd)})
        except subprocess.TimeoutExpired as e:
            results.append({"command": cmd, "ret": -124, "stdout": e.stdout or "", "stderr": (e.stderr or "") + "\n[timeout]", "cwd": str(cwd)})
        except Exception as e:
            results.append({"command": cmd, "ret": -1, "stdout": "", "stderr": str(e), "cwd": str(cwd)})
    return results


# ----------------- git / patch / main flow -----------------


def ensure_clone(repo_full: str, workspace: Path) -> Path:
    org, repo = repo_full.split('/')
    dest = workspace / f"{org}_{repo}"
    if dest.exists():
        logger.info(f"Repo exists: {dest}")
        return dest
    url = f"https://github.com/{org}/{repo}.git"
    logger.info(f"Cloning {url} -> {dest}")
    r = run(["git", "clone", "--depth", "1", url, str(dest)])
    if r["ret"] != 0:
        logger.warning("Shallow clone failed, trying full clone")
        r2 = run(["git", "clone", url, str(dest)])
        if r2["ret"] != 0:
            raise RuntimeError(f"git clone failed: {r2['stderr']}")
    return dest


def checkout_commit(repo_path: Path, base_commit: str, pr_number: int) -> bool:
    if base_commit:
        logger.info(f"Checking out {base_commit}")
        run(["git", "fetch", "origin", base_commit], cwd=repo_path)
        r2 = run(["git", "checkout", "-f", base_commit], cwd=repo_path)
        if r2["ret"] == 0:
            return True
    logger.info(f"Falling back to PR head refs/pull/{pr_number}/head")
    r = run(["git", "fetch", "origin", f"pull/{pr_number}/head:pr/{pr_number}"], cwd=repo_path)
    if r["ret"] == 0:
        r2 = run(["git", "checkout", "-f", f"pr/{pr_number}"], cwd=repo_path)
        return r2["ret"] == 0
    return False


def apply_patch(repo_path: Path, patch_text: str, instance_id: str) -> Dict[str, Any]:
    if not patch_text:
        return {"applied": False, "reason": "empty patch"}
    import tempfile, os
    tf = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".diff")
    tf.write(patch_text)
    tf.close()
    tfpath = tf.name
    r = run(["git", "apply", "--index", tfpath], cwd=repo_path)
    if r["ret"] == 0:
        run(["git", "add", "-A"], cwd=repo_path)
        run(["git", "commit", "-m", f"apply patch {instance_id}"], cwd=repo_path)
        os.unlink(tfpath)
        return {"applied": True, "method": "git apply --index", "stdout": r["stdout"], "stderr": r["stderr"]}
    r2 = run(["git", "apply", tfpath], cwd=repo_path)
    if r2["ret"] == 0:
        run(["git", "add", "-A"], cwd=repo_path)
        run(["git", "commit", "-m", f"apply patch {instance_id}"], cwd=repo_path)
        os.unlink(tfpath)
        return {"applied": True, "method": "git apply", "stdout": r2["stdout"], "stderr": r2["stderr"]}
    r3 = run(["git", "am", "--keep-cr", tfpath], cwd=repo_path)
    err = r.get("stderr", "") + "\n" + r2.get("stderr", "") + "\n" + r3.get("stderr", "")
    os.unlink(tfpath)
    return {"applied": False, "reason": "all methods failed", "stderr": err}


def main(swefile: str, mswefile: str, workspace: str, outdir: str):
    swe = json.load(open(swefile, "r"))
    mswe = json.load(open(mswefile, "r"))
    workspace = Path(workspace).expanduser().resolve()
    outdir = Path(outdir).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    # index mswe
    mswe_index: Dict[tuple, Dict[str, Any]] = {}
    for e in mswe:
        key = (e.get("repo"), int(e.get("pull_number")) if e.get("pull_number") is not None else None)
        mswe_index[key] = e

    from collections import defaultdict
    groups = defaultdict(list)
    for e in swe:
        key = (e.get("repo"), int(e.get("pull_number")) if e.get("pull_number") is not None else None)
        groups[key].append(e)

    overall_results = []

    for key, instances in groups.items():
        repo_full, pr_number = key
        if repo_full is None:
            logger.warning(f"Skipping group with missing repo: {key}")
            continue
        logger.info(f"Processing {repo_full}#{pr_number} ({len(instances)} instances)")
        try:
            repo_path = ensure_clone(repo_full, workspace)
        except Exception as exc:
            logger.error(f"Clone failed: {exc}")
            for inst in instances:
                overall_results.append({"instance_id": inst.get("instance_id"), "repo": repo_full, "pull_number": pr_number, "error": str(exc)})
            continue

        # clean and checkout
        run(["git", "reset", "--hard"], cwd=repo_path)
        run(["git", "clean", "-fdx"], cwd=repo_path)
        base_commit = instances[0].get("base_commit", "")
        if not checkout_commit(repo_path, base_commit, pr_number):
            logger.error("Checkout failed, skipping group.")
            for inst in instances:
                overall_results.append({"instance_id": inst.get("instance_id"), "repo": repo_full, "pull_number": pr_number, "error": "checkout failed"})
            continue

        # get PASS_TO_PASS as class names
        mswe_entry = mswe_index.get((repo_full, pr_number), None)
        test_classes = []
        if mswe_entry:
            p = mswe_entry.get("PASS_TO_PASS", None)
            if isinstance(p, list):
                test_classes = [str(x).strip() for x in p if str(x).strip()]
            elif isinstance(p, str):
                test_classes = [line.strip() for line in p.splitlines() if line.strip()]

        # prepare commands (repo-root-aware)
        test_commands = make_test_commands(repo_path, test_classes) if test_classes else []

        for inst in instances:
            instance_id = inst.get("instance_id")
            logger.info(f"--- Instance {instance_id} ---")
            # reset to base
            run(["git", "checkout", "-f", "HEAD"], cwd=repo_path)
            run(["git", "reset", "--hard"], cwd=repo_path)
            run(["git", "clean", "-fdx"], cwd=repo_path)

            apply_res = apply_patch(repo_path, inst.get("patch", ""), instance_id)

            test_results = []
            if test_commands:
                test_results = run_tests(test_commands, instance_id)
                total = len(test_results)
                passed = sum(1 for r in test_results if r.get("ret") == 0)
                pass_pct = (passed / total * 100.0) if total > 0 else None
                logger.info(f"[{instance_id}] Passed {passed}/{total} ({pass_pct:.1f}%)" if total > 0 else f"[{instance_id}] No tests run")
            else:
                logger.warning("No tests or build tool not detected; skipping tests for this group.")
                total = 0
                passed = 0
                pass_pct = None

            out = {
                "instance_id": instance_id,
                "repo": repo_full,
                "pull_number": pr_number,
                "base_commit": inst.get("base_commit"),
                "apply_result": apply_res,
                "test_results": test_results,
                "tests_total": total,
                "tests_passed": passed,
                "pass_percentage": round(pass_pct, 2) if pass_pct is not None else None,
                "problem_statement": inst.get("problem_statement"),
            }
            with open(outdir / f"{instance_id}.json", "w") as f:
                json.dump(out, f, indent=2)
            overall_results.append(out)

            # reset repo to base before next instance
            checkout_commit(repo_path, base_commit, pr_number)
            run(["git", "reset", "--hard"], cwd=repo_path)
            run(["git", "clean", "-fdx"], cwd=repo_path)

    with open(outdir / "sweagent_run_results.json", "w") as f:
        json.dump(overall_results, f, indent=2)
    logger.info("Done. Results written.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--swefile", required=True)
    p.add_argument("--mswefile", required=True)
    p.add_argument("--workspace", default="workspace")
    p.add_argument("--outdir", default="sweagent_results")
    args = p.parse_args()
    main(args.swefile, args.mswefile, args.workspace, args.outdir)
