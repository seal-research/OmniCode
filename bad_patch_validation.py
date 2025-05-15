import codearena
import os
import argparse
import json
import sys

parser = argparse.ArgumentParser(description="Verify Bad Patch")
parser.add_argument( "--instance_id", help="Instance ID")
parser.add_argument("--results_folder", help="run evaluation folder name (after logs/run_evaluation/)")
parser.add_argument("--dataset_name", default="data/codearena_instances.json",
                    help="Name of the dataset")
args = parser.parse_args()

results_dir = os.path.join('logs/run_evaluation', args.results_folder)

if not os.path.exists(results_dir):
    print('Results folder does not exist (likely due to patch being empty)', results_dir)
    sys.exit(1)

full_dir = os.path.join(results_dir, 'agentless', args.instance_id)
report_path = os.path.join(full_dir, 'report.json')

if not os.path.exists(report_path):
    print('Report file does not exist (likely due to error in building image)', report_path)
    sys.exit(1)

with open(report_path, 'r') as f:
    report = json.load(f)

resolved = report[args.instance_id]['resolved']
if resolved:
    print('Solved task:', results_dir)
    sys.exit(1)
else:

    with open(args.dataset_name, 'r') as f:
        dataset = json.load(f)

    # load the patch from predictions path
    patch_path = os.path.join(full_dir, 'patch.diff')
    with open(patch_path, 'r') as f:
        patch = f.read()

    # add patch to dataset
    task_ix = [i for i in range(len(dataset)) if dataset[i]['instance_id'] == args.instance_id][0]
    # if 'bad_patches' in dataset[task_ix]:
    #     dataset[task_ix]['bad_patches'].append(patch)
    # else:
    #     dataset[task_ix]['bad_patches'] = [patch]
    dataset[task_ix]['bad_patch'] = patch

    # save dataset back to json file
    with open(args.dataset_name, 'w') as f:
        json.dump(dataset, f, indent=4)

    # save gold patch to gold.diff for easy comparison
    gold_path = os.path.join(full_dir, 'gold.diff')
    with open(gold_path, 'w+') as f:
        f.write(dataset[task_ix]['patch'])

    print('Bad patch successfully added to dataset from:', results_dir)

    found_bad_patch = True
    sys.exit(0)

