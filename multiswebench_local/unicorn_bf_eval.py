import json
import subprocess
import time, os, shutil

start_time = time.time()

api_key = "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"
programming_language = "cpp"  # "cpp" or "java"
mode = "bugfixing"
model = "openrouter/deepseek/deepseek-chat-v3.1"  # "openrouter/deepseek/deepseek-chat-v3.1" or "gpt-4o" or "gpt-4o-mini"
mem = "32G"
output_file = "baselines/sweagent/logs/sweagent_outputs/all_preds_cpp.jsonl"
task = "evaluate"  # "predict" or "evaluate" or "both"

instances = []
with open(output_file) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            instances.append(data["instance_id"])
        except json.JSONDecodeError:
            print("Skipping corrupt line: %r", line)

def eval_instance(instance):
    print("Processing evaluate instance:", instance)
    run_id = instance.split('__')[1]
    wrap_cmd = ["export TMPDIR=/scratch/dd732/tmp && mkdir -p /scratch/dd732/tmp &&",
                "python codearena.py",
                "--MSWEBugFixing",
                f"--predictions_path {output_file}",
                f"--run_id {run_id}",
                "--max_workers 1",
                "--mswe_phase all",
                "--force_rebuild True",
                "--clean True",
                "--use_apptainer True",
                f"--instance_ids {instance}",
                "--timeout 10000",
                "--g2 True"]
    wrap_cmd_str = " ".join(wrap_cmd)
    cmd = [
        "sbatch", f"--job-name={run_id}_eval",
        "--cpus-per-task=2",
        f"--mem={mem}",
        #"--gres=gpu=1",
        "--nodelist=dutta-compute-01",
        "--time=1:00:00",
        f"--output=slurm_logs/end2end_{mode}_{programming_language}_{model.split("/")[-1]}/%x_%j.out",
        f"--error=slurm_logs/end2end_{mode}_{programming_language}_{model.split("/")[-1]}/%x_%j.err",
        f'--wrap="{wrap_cmd_str}"'
    ]
    cmd_str = " ".join(cmd)
    ret = os.system(cmd_str)

    if ret >> 8 != 0:
        print(f"Error processing instance {instance}: {ret >> 8}")
    else:
        print(f"Successfully processed instance {instance}")

if task == "evaluate" or task == "both" and programming_language != "python":
    # old:
# eval_instances = instances
# while eval_instances:
#     run_id_dict = {}
#     for instance in eval_instances:
#         ...
#     while len(subprocess.run(["squeue"], ...).stdout.split("\n")) != 3:
#         time.sleep(30)
#     ...
#     next_instances = []
#     for instance in eval_instances:
#         ...
#         if not success: next_instances.append(instance)
#     eval_instances = next_instances

# new: single pass (no retry)
    eval_instances = instances
    run_id_dict = {}
    for instance in eval_instances:
        while len(subprocess.run(["squeue"], capture_output=True, text=True).stdout.split("\n")) == 127:
            time.sleep(180)
        eval_instance(instance)
        print(f":arrows_counterclockwise: Starting evaluate instance {instance}")

        run_id = instance.split('__')[1] + "_eval"
        while True:
            run_info = subprocess.run(["squeue", "-n", run_id], capture_output=True, text=True).stdout.split("\n")[1].split()
            if run_info and run_info[4] == "R":
                run_id_dict[run_id] = run_info
                break

    # wait for all jobs to finish
    while len(subprocess.run(["squeue"], capture_output=True, text=True).stdout.split("\n")) != 3:
        time.sleep(30)
    print("Evaluate done.")

    # check results once, but DO NOT re-add to eval_instances
    for instance in eval_instances:
        run_id = instance.split('__')[1] + "_eval"
        if run_id not in run_id_dict:
            print(instance, "record error")
            continue
        each_info = run_id_dict[run_id]
        with open(f"slurm_logs/end2end_{mode}_{programming_language}_{model.split('/')[-1]}/{run_id}_{each_info[0]}.out", "r") as f:
            text = f.read()
        if "Resolved instances: 1" not in text and "Unresolved instances: 1" not in text and "Invalid patch: None" not in text:
            print(instance, "NOT resolved (no retry will be attempted)")

        

    cost = time.time() - start_time
    print(f"Total time: {cost/60:.2f} minutes")