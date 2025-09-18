import codearena
import os
import argparse
import json
import sys

parser = argparse.ArgumentParser(description="Verify Bad Patch")
parser.add_argument( "--instance_id", help="Instance ID")
parser.add_argument("--results_folder", help="run evaluation folder name (after logs/run_evaluation/)")
# parser.add_argument("--dataset_name", default="data/codearena_instances.json",
#                     help="Name of the dataset")
parser.add_argument("--dataset_name", default="data/agentless_results/result_1.jsonl",
                    help="Name of the dataset")
parser.add_argument("--language", default="python")
parser.add_argument("--model", default="none", help="Model used to generate the patch")
args = parser.parse_args()

if args.language == 'python':
    results_dir = os.path.join('logs/run_evaluation', args.results_folder)
elif args.language == 'java' or args.language == 'cpp':
    results_dir = os.path.join('multiswebench_runs/BugFixing')

if not os.path.exists(results_dir):
    print('Results folder does not exist (likely due to patch being empty)', results_dir)
    sys.exit(1)

if args.language == 'python':
    full_dir = os.path.join(results_dir, 'agentless', args.instance_id)
    work_dir = full_dir
    report_path = os.path.join(full_dir, 'report.json')
elif args.language == 'java' or args.language == 'cpp':
    full_dir = os.path.join(results_dir, 'output', f"{args.results_folder}")
    work_dir_temp = os.path.join(results_dir, 'workdir', f"{args.results_folder}")
    for root, dirs, files in os.walk(work_dir_temp):
        if 'fix.patch' in files:
            # check if directory two up is evals
            if os.path.basename(os.path.dirname(root)) == 'evals':
                work_dir = root
                break
    report_path = os.path.join(full_dir, 'final_report.json')

if not os.path.exists(report_path):
    print('Report file does not exist (likely due to error in building image)', report_path)
    sys.exit(1)

with open(report_path, 'r') as f:
    report = json.load(f)

if args.language == 'python':
    unresolved = not report[args.instance_id]['resolved']
elif args.language == 'java' or args.language == 'cpp':
    unresolved = report['unresolved_instances'] > 0
    gen_report_log = os.path.join(results_dir, 'logs', f"{args.results_folder}", "gen_report.log")
    if os.path.exists(gen_report_log):
        with open(gen_report_log, 'r') as f:
            gen_report = f.read()
        if "There is no valid fix patch result" in gen_report:
            print('No valid fix patch:', results_dir)
            sys.exit(1) # The patch failed to apply, but it was not due to a bug in the patch itself
        elif "Invalid f2p_tests" in gen_report:
            print('Invalid f2p_tests:', results_dir)
            sys.exit(1) # The patch failed to apply, but it was not due to a bug in the patch itself
        elif "Invalid p2p_tests" in gen_report:
            print('Invalid p2p_tests:', results_dir)
            sys.exit(1) # The patch failed to apply, but it was not due to a bug in the patch itself
        elif "No fix for failed test" in gen_report:
            print("No fix for failed test:", results_dir)
            reason = "No fix for failed test"
        elif "Test passed in test patch but failed in fix patch" in gen_report:
            print("Test passed in test patch but failed in fix patch:", results_dir)
            reason = "Test passed in test patch but failed in fix patch"
        else:
            print("unknown reason", results_dir)
            reason = "Unknown reason"
if not unresolved:
    print('Solved task:', results_dir)
    sys.exit(1)
else:
    with open('data/multiswebench_data/mswebench_instances.json', 'r') as f:
        dataset = json.load(f)
    # load the patch from predictions path
    if args.language == 'python':
        patch_path = os.path.join(work_dir, 'patch.diff')
    elif args.language == 'java' or args.language == 'cpp':
        patch_path = os.path.join(work_dir, 'fix.patch')
    with open(patch_path, 'r') as f:
        patch = f.read()
    # add patch to dataset
    try:
        task_ix = [i for i in range(len(dataset)) if dataset[i]['instance_id'] == args.instance_id][0]
    except:
        print("!!! instance not in output file already")
    if 'bad_patches' in dataset[task_ix]:
        index = len(dataset[task_ix]['bad_patches']) + 1
        res = {
            "instance_id": args.instance_id,
            "idx": index,
            "source": f"agentless_{args.model}",
            "patch": patch,
            "reason": reason
        }
    else:
        res = {
            "instance_id": args.instance_id,
            "idx": 1,
            "source": f"agentless_{args.model}",
            "patch": patch,
            "reason": reason
        }

    # save dataset back to jsonl file
    if not os.path.exists(args.dataset_name):
        # create the directory if it doesn't exist
        os.makedirs(os.path.dirname(args.dataset_name), exist_ok=True)
        with open(args.dataset_name, 'w') as f:
            json.dump(res, f)
            f.write("\n")
    else:
        with open(args.dataset_name, 'a') as f:
            json.dump(res, f)
            f.write("\n")
    # save gold patch to gold.diff for easy comparison
    gold_path = os.path.join(work_dir, 'gold.diff')
    with open(gold_path, 'w+') as f:
        f.write(dataset[task_ix]['patch'])

    print('Bad patch successfully added to dataset from:', results_dir)

    found_bad_patch = True
    sys.exit(0)

