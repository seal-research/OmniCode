import codearena
import argparse
import json
import sys

parser = argparse.ArgumentParser(description="Verify Bad Patch")
parser.add_argument(
    "--predictions_path",
    required=True,
    help="Paths to predictions file",
)
parser.add_argument("--run_id", required=True, help="Run ID for the evaluation")
parser.add_argument(
    "--instance_id",
    help="Instance ID",
)

parser.add_argument("--dataset_name", default="data/codearena_instances.json", help="Name of the dataset")

args = parser.parse_args()

report_file = f'logs/run_evaluation/{args.run_id}/agentless/{args.instance_id}/report.json'

with open(report_file, 'r') as f:
    report = json.load(f)

resolved = report[args.instance_id]['resolved']

if not resolved:
    print('Patch is confirmed bad. Adding to dataset.')

    with open(args.dataset_name, 'r') as f:
        dataset = json.load(f)

    # load the patch from predictions path
    with open(args.predictions_path, 'r') as f:
        patch = json.load(f)['model_patch']

    # add patch to dataset
    task_ix = [i for i in range(len(dataset)) if dataset[i]['instance_id'] == args.instance_id]
    dataset[task_ix]['bad_patch'] = patch

    # save dataset back to json file
    with open(args.dataset_name, 'w') as f:
        json.dump(dataset, f, indent=4)

    print('Added bad patch to dataset.')
    sys.exit(0)
else:
    print('Patch resolves all tests (which is not what we want).')
    sys.exit(1)
