import os
import subprocess
import time
import json
from pathlib import Path


MAX_ATTEMPTS = 3

def process(
    jobname,
    mode,
    instance_list_path,
    output_file,
    mem,
):
    all_instances = open(instance_list_path, 'r', encoding='utf-8').read().splitlines()

    num_attempts = 0

    while num_attempts < MAX_ATTEMPTS:
        start_time = time.time()

        completed_instances = []
        if Path(output_file).exists():
            completed_instances = [
                json.loads(line)['instance_id']
                for line in open(output_file, "r").read().splitlines()
            ]
        instances = [i for i in all_instances if i not in completed_instances]
        print(f"{len(all_instances)=}, {len(instances)=}")

        if len(instances) == 0:
            print(f"Done!")
            break

        def eval_instance(instance): 
            print("Processing predict instance:", instance)
            run_id = f"{jobname}_{instance}"

            wrap_cmd = [
                "python codearena.py",
                f"--{mode}",
                f"--predictions_path {output_file}",
                f"--run_id {run_id}",
                f"--max_workers 1",
                f"--mswe_phase all",
                f"--force_rebuild True",
                "--clean True",
                f"--use_apptainer True",
                f"--timeout 10000",
                f"-g2 True",
                f"--instance_ids {instance}",
            ]
            wrap_cmd_str = " ".join(wrap_cmd)
            cmd = [
                "sbatch", f"--job-name={run_id}_pred",
                "--cpus-per-task=2",
                f"--mem={mem}",
                "--gres=gpu:1", 
                "--time=2:00:00",
                f"--output=slurm_logs/%x_%j.out",
                f"--error=slurm_logs/%x_%j.err",
                f'--wrap="{wrap_cmd_str}"'
            ]    
            result = subprocess.run(" ".join(cmd), shell=True, text=True,
                                capture_output=True)
            if result.returncode != 0:
                print("sbatch failed:", result.stderr.strip())
            else:
                print(f"{instance}: {result.stdout.strip()}")

        for instance in instances:
            # 10 jobs + header + empty line
            while len(subprocess.run(["squeue"], capture_output=True, text=True).stdout.split("\n")) == 22:
                time.sleep(180)
            eval_instance(instance)
            print(f"🔄 Starting predict instance {instance}")


        while len(subprocess.run(["squeue"], capture_output=True, text=True).stdout.split("\n")) != 2: # header + empty line
            time.sleep(30)
        print("Predict done.")
        cost = time.time() - start_time
        print(f"Total time: {cost/60:.2f} minutes")
        num_attempts += 1

if __name__=='__main__':

    jobname = "aider_gemini_tg_java"
    process(
        jobname = f"eval_{jobname}",
        mode = "MSWEBugFixing",
        instance_list_path = "data/g2_sane_java_instances.txt",
        output_file = f"logs/baselines/{jobname}/all_preds.jsonl",
        mem = "32G",
    )

    jobname = "aider_gemini_tg_cpp"
    process(
        jobname = f"eval_{jobname}",
        mode = "MSWEBugFixing",
        instance_list_path = "data/g2_sane_cpp_instances.txt",
        output_file = f"logs/baselines/{jobname}/all_preds.jsonl",
        mem = "32G",
    )