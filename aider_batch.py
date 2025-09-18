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
    input_data_path,
    model,
    model_provider,
    output_dir,
    output_file,
    mem,
):
    api_key = os.getenv("OMNICODE_API_KEY", None)
    if api_key is None:
        raise RuntimeError(f"Could not find OMNICODE_API_KEY environment variable")

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

        def pred_instance(instance): 
            print("Processing predict instance:", instance)
            run_id = f"{jobname}_{instance}"
            wrap_cmd = [
                "python baselines/aider/aider_regular.py",
                f"-i {input_data_path}",
                f"-o {output_dir}",
                f"--model_name {model}",
                f"-k {api_key}",
                f"--model_provider {model_provider}",
                f"--mode {mode}",
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
            pred_instance(instance)
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
        jobname = jobname,
        mode = "testgen",
        instance_list_path = "data/g2_sane_java_instances.txt",
        input_data_path = "data/codearena_instances_java.json",
        model = "openrouter/google/gemini-2.5-flash",
        model_provider = "openrouter",
        output_dir = f"logs/baselines/{jobname}",
        output_file = f"logs/baselines/{jobname}/all_preds.jsonl",
        mem = "32G",
    )

    jobname = "aider_gemini_tg_cpp"
    process(
        jobname = jobname,
        mode = "testgen",
        instance_list_path = "data/g2_sane_cpp_instances.txt",
        input_data_path = "data/codearena_instances_cpp.json",
        model = "openrouter/google/gemini-2.5-flash",
        model_provider = "openrouter",
        output_dir = f"logs/baselines/{jobname}",
        output_file = f"logs/baselines/{jobname}/all_preds.jsonl",
        mem = "32G",
    )