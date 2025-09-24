#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import subprocess
import time
from typing import Dict, List, Set, Optional


def load_predictions_jsonl(pred_path: Path, model_substr: Optional[str] = None) -> Dict[str, dict]:
    preds: Dict[str, dict] = {}
    for line in pred_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        iid = obj.get("instance_id")
        if isinstance(iid, str):
            if model_substr and isinstance(obj.get("model_name_or_path"), str):
                if model_substr not in obj["model_name_or_path"]:
                    continue
            preds[iid] = obj
    return preds


def parse_instance_id_from_eval_filename(name: str) -> str | None:
    # Examples:
    #   openrouter__anthropic__claude-sonnet-4.claude_sonnet_acr_eval_codereview_pytest-dev__pytest-10081.json
    if not name.endswith(".json"):
        return None
    base = name[:-5]
    if "__" not in base:
        return None
    return base.split("__")[-1]


def collect_evaluated_ids(swebench_eval_dir: Path, model_substr: str = "claude", mode_substr: str = "codereview") -> Set[str]:
    done: Set[str] = set()
    for p in swebench_eval_dir.glob("*.json"):
        name = p.name
        if model_substr in name and mode_substr in name:
            iid = parse_instance_id_from_eval_filename(name)
            if iid:
                done.add(iid)
    return done


def find_predictions_path(base_dir: Path) -> Optional[Path]:
    # Common candidates
    candidates = [
        base_dir / "acr_codereview_outputs" / "all_preds.jsonl",
        base_dir / "codereview" / "all_preds.jsonl",
        base_dir / "codereview" / "codereview_predictions_converted.jsonl",
        base_dir / "all_preds.jsonl",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fallback: search shallowly
    try:
        for p in base_dir.rglob("all_preds.jsonl"):
            if any(tok in p.as_posix() for tok in ["codereview", "acr_codereview_outputs"]):
                return p
        first = next(base_dir.rglob("all_preds.jsonl"), None)
        return first
    except Exception:
        return None


def make_slurm_script(
    ids_file: Path,
    run_id: str,
    predictions_path: Path,
    log_dir: Path,
    cpus: int = 8,
    mem: str = "16G",
    time_limit: str = "02:00:00",
    sbatch_bin: str = "sbatch",
) -> str:
    # Build script without f-strings to avoid brace parsing issues
    script = "#!/usr/bin/env -S bash --noprofile --norc\n\n"
    script += "set -euo pipefail\n\n"
    script += f"SBATCH_BIN=\"{sbatch_bin}\"\n"
    script += f"INSTANCE_FILE=\"{ids_file.as_posix()}\"\n"
    script += f"RUN_ID=\"{run_id}\"\n"
    script += f"LOG_DIR=\"{log_dir.as_posix()}\"\n\n"
    script += f"CPUS={cpus}\nMEM={mem}\nTIME_LIMIT=\"{time_limit}\"\n\n"
    script += "mkdir -p \"$LOG_DIR\"\n\n"
    script += "echo \"Starting CLAUDE CodeReview evaluation...\"\n\n"
    script += "while IFS= read -r ID || [[ -n \"$ID\" ]]; do\n"
    script += "    SAN_ID=\"${ID//\\/\\/__}\"\n"
    script += "    SAN_ID=\"${SAN_ID//:/_}\"\n"
    script += "    JOB_NAME=\"${RUN_ID}_codereview_${SAN_ID}\"\n\n"
    script += "    echo \"Submitting job for instance_id=${ID}  (job-name=${JOB_NAME})\"\n\n"
    script += "    \"$SBATCH_BIN\" --job-name=\"${JOB_NAME}\" \\\n"
    script += "           --cpus-per-task=\"${CPUS}\" \\\n"
    script += "           --gres=gpu:1 \\\n"
    script += "           --mem=\"${MEM}\" \\\n"
    script += "           --time=\"${TIME_LIMIT}\" \\\n"
    script += "           --constraint=gpu \\\n"
    script += "           --export=NONE \\\n"
    script += "           --output=\"${LOG_DIR}/%x_%j.out\" \\\n"
    script += "           --error=\"${LOG_DIR}/%x_%j.err\" \\\n"
    wrap = (
        "(cd $(pwd) && export PATH=/share/apps/singularity/3.7.0/bin:$PATH; unset LD_PRELOAD; unset LD_LIBRARY_PATH; "
        "python codearena.py --CodeReview "
        "--predictions_path " + predictions_path.as_posix() + " "
        "--run_id ${JOB_NAME} "
        "--max_workers 1 --mswe_phase all --force_rebuild False --clean True --use_apptainer True "
        "--instance_ids ${ID} --g2 True;)"
    )
    script += "           --wrap=\"" + wrap.replace("\"", "\\\"") + "\"\n"
    script += "done < \"" + ids_file.as_posix() + "\"\n\n"
    script += "echo \"Completed CLAUDE CodeReview evaluation submission\"\n"
    return script


def main() -> None:
    ap = argparse.ArgumentParser(description="Resume evaluation-only for Claude CodeReview (no regeneration)")
    ap.add_argument("--acr-results-dir", type=Path, required=True, help="Path to Claude ACR results (contains acr_codereview_outputs/all_preds.jsonl)")
    ap.add_argument("--swebench-eval-dir", type=Path, default=Path("swebench_eval"), help="Flat per-instance eval JSONs directory")
    ap.add_argument("--output-dir", type=Path, default=Path("evaluation_setup_claude"), help="Where to write IDs file and optional SLURM script")
    ap.add_argument("--run-id", type=str, default="claude_sonnet_acr_eval_codereview", help="Base run_id for evaluation jobs")
    ap.add_argument("--cpus", type=int, default=8)
    ap.add_argument("--mem", type=str, default="16G")
    ap.add_argument("--time-limit", type=str, default="02:00:00")
    ap.add_argument("--sbatch-bin", type=str, default="sbatch")
    ap.add_argument("--direct-submit", action="store_true", help="Submit one sbatch job per instance (parallel on cluster)")
    ap.add_argument("--sequential", action="store_true", help="Submit a single sbatch job that evaluates all IDs sequentially")
    ap.add_argument("--wait", action="store_true", help="Wait for all submitted jobs to finish, then summarize")
    ap.add_argument("--summary", action="store_true", help="Write a global summary for Claude CodeReview after completion")
    ap.add_argument("--summary-out", type=Path, default=None, help="Optional summary output JSON path")
    ap.add_argument("--predictions-path", type=str, default=None, help="Explicit predictions path; use 'gold' for gold patches or path to all_preds.jsonl")
    ap.add_argument("--model-substr", type=str, default="claude", help="Model substring for filtering preds and eval files")
    args = ap.parse_args()

    # Determine predictions argument: prefer explicit, else autodetect, else fallback to 'gold'
    predictions_arg: str
    if args.predictions_path:
        predictions_arg = args.predictions_path
    else:
        found = find_predictions_path(args.acr_results_dir)
        predictions_arg = found.as_posix() if found else "gold"

    all_ids: Set[str]
    if predictions_arg == "gold":
        # With gold predictions we don't have per-instance file; evaluate all Claude IDs not done.
        # We'll derive IDs solely from evaluated/missing sets relative to swebench_eval names.
        # To be conservative, require user to rely on evaluated files; without preds we cannot know all IDs.
        # In this fallback, we do not filter from preds.
        all_ids = set()
    else:
        preds = load_predictions_jsonl(Path(predictions_arg), model_substr=args.model_substr)
        all_ids = set(preds.keys())
    done_ids = collect_evaluated_ids(args.swebench_eval_dir, model_substr=args.model_substr, mode_substr="codereview")
    if all_ids:
        todo_ids = sorted(all_ids - done_ids)
    else:
        # No preds file; try to infer todo by scanning instance ids present in predictions filenames under acr-results-dir
        # and excluding done_ids. If none found, just exit gracefully.
        inferred: Set[str] = set()
        try:
            for p in args.acr_results_dir.rglob("*.json"):
                name = p.name
                if args.model_substr in name and "codereview" in name and "__" in name:
                    iid = parse_instance_id_from_eval_filename(name)
                    if iid:
                        inferred.add(iid)
        except Exception:
            pass
        if not inferred:
            print("No predictions file and could not infer instance IDs; nothing to submit.")
            return
        todo_ids = sorted(inferred - done_ids)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ids_file = args.output_dir / "codereview_instance_ids.txt"
    ids_file.write_text("\n".join(todo_ids) + ("\n" if todo_ids else ""), encoding="utf-8")

    print(f"Wrote instance IDs: {ids_file}  (count={len(todo_ids)})")

    # Sequential submission: single job iterating through all IDs
    if args.sequential and todo_ids:
        log_dir = args.output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        job_name = f"{args.run_id}_codereview_seq"
        # Build a small loop that runs codearena for each ID sequentially
        loop_lines = [
            "set -euo pipefail",
            "while IFS= read -r ID || [[ -n \"$ID\" ]]; do",
            "  echo Running $ID",
            "  python codearena.py --CodeReview "
            f"    --predictions_path {predictions_arg} "
            f"    --run_id {job_name} "
            "    --max_workers 1 --mswe_phase all --force_rebuild False --clean True --use_apptainer True "
            "    --instance_ids $ID --g2 True",
            "done < \"" + ids_file.as_posix() + "\"",
        ]
        loop_script = " && ".join(loop_lines)
        cmd = [
            args.sbatch_bin,
            "--job-name", job_name,
            "--cpus-per-task", str(args.cpus),
            "--gres", "gpu:1",
            "--mem", args.mem,
            "--time", args.time_limit,
            "--constraint", "gpu",
            "--export", "NONE",
            "--output", str((log_dir / "%x_%j.out").as_posix()),
            "--error", str((log_dir / "%x_%j.err").as_posix()),
            "--wrap", f"(cd {Path.cwd()} && export PATH=/share/apps/singularity/3.7.0/bin:$PATH; unset LD_PRELOAD; unset LD_LIBRARY_PATH; {loop_script})",
        ]
        print("Submitting sequential job:", " ".join(cmd))
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(res.stdout)
        if res.stderr:
            print(res.stderr)

    # If direct-submit, submit individual jobs and optionally wait
    job_ids: list[str] = []
    if args.direct_submit and todo_ids:
        log_dir = args.output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        for iid in todo_ids:
            san_id = iid.replace("/", "__").replace(":", "_")
            job_name = f"{args.run_id}_codereview_{san_id}"
            wrap = (
                "(cd " + str(Path.cwd()) +
                " && export PATH=/share/apps/singularity/3.7.0/bin:$PATH; unset LD_PRELOAD; unset LD_LIBRARY_PATH; "
                f"python codearena.py --CodeReview --predictions_path {predictions_arg} --run_id {job_name} "
                "--max_workers 1 --mswe_phase all --force_rebuild False --clean True --use_apptainer True "
                f"--instance_ids {iid} --g2 True;)"
            )
            cmd = [
                args.sbatch_bin,
                "--job-name", job_name,
                "--cpus-per-task", str(args.cpus),
                "--gres", "gpu:1",
                "--mem", args.mem,
                "--time", args.time_limit,
                "--constraint", "gpu",
                "--export", "NONE",
                "--output", str((log_dir / "%x_%j.out").as_posix()),
                "--error", str((log_dir / "%x_%j.err").as_posix()),
                "--wrap", wrap,
            ]
            print("Submitting:", " ".join(cmd))
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            # Expect: Submitted batch job <JOBID>
            jobid = res.stdout.strip().split()[-1] if res.stdout.strip() else ""
            if jobid.isdigit():
                job_ids.append(jobid)
        print(f"Submitted {len(job_ids)} jobs.")

    # Wait for completion if requested
    if args.wait and job_ids:
        print("Waiting for jobs to finish...")
        remaining = set(job_ids)
        while remaining:
            try:
                check = subprocess.run([
                    "squeue", "-h", "-j", ",".join(sorted(remaining))
                ], capture_output=True, text=True)
                # squeue -h prints nothing if no jobs found
                if check.stdout.strip():
                    # Still running
                    time.sleep(30)
                else:
                    break
            except Exception:
                time.sleep(30)
                continue
        print("All submitted jobs appear to have finished.")

    # Produce global summary if requested
    if args.summary:
        summ_cmd = [
            sys.executable,
            "analysis_scripts/summarize_swebench_eval.py",
            str(args.swebench_eval_dir),
            "--mode", "codereview",
            "--model-substr", "claude",
        ]
        if args.summary_out:
            summ_cmd += ["--out", str(args.summary_out)]
        print("\nGenerating global summary for Claude CodeReview...")
        res = subprocess.run(summ_cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.stderr:
            print(res.stderr)


if __name__ == "__main__":
    main()


