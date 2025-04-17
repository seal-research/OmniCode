import codearena
import os
import argparse
import json
import sys

parser = argparse.ArgumentParser(description="Verify Bad Patch")
parser.add_argument( "--instance_id", help="Instance ID")
parser.add_argument("--results_folder_prefix",
                    help="run evaluation folder prefix pattern")
parser.add_argument("--dataset_name", default="data/codearena_instances.json",
                    help="Name of the dataset")
args = parser.parse_args()

# check each of the logs to see if the patch solved.
results_dirs = [d for d in os.listdir('logs/run_evaluation/') if d.startswith(args.results_folder_prefix)]
if len(results_dirs) == 0:
    print('No results found with prefix', args.results_folder_prefix)
    sys.exit(1)

print('Looking for bad patches...')
found_bad_patch = False
for results_dir in results_dirs:
    full_dir = os.path.join('logs/run_evaluation', results_dir, 'agentless', args.instance_id)
    report_path = os.path.join(full_dir, 'report.json')
    with open(report_path, 'r') as f:
        report = json.load(f)

    resolved = report[args.instance_id]['resolved']
    if resolved:
        print('Solved task:', results_dir)
    if not resolved:
        print('Bad patch found in', results_dir)

        with open(args.dataset_name, 'r') as f:
            dataset = json.load(f)

        # load the patch from predictions path
        patch_path = os.path.join(full_dir, 'patch.diff')
        with open(patch_path, 'r') as f:
            patch = f.read()

        # add patch to dataset
        task_ix = [i for i in range(len(dataset)) if dataset[i]['instance_id'] == args.instance_id][0]
        dataset[task_ix]['bad_patch'] = patch

        # save dataset back to json file
        with open(args.dataset_name, 'w') as f:
            json.dump(dataset, f, indent=4)

        print('Added bad patch to dataset.')

        # save gold patch to gold.diff for easy comparison
        gold_path = os.path.join(full_dir, 'gold.diff')
        with open(gold_path, 'w+') as f:
            f.write(dataset[task_ix]['patch'])
        print('Saved gold patch to', gold_path, 'for easy comparison.')

        found_bad_patch = True
        break
        sys.exit(0)

if found_bad_patch:
    sys.exit(0)
else:
    print('All patches solved the task (not what we want).')
    sys.exit(1)
