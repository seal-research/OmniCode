import os
outputs_dir = 'logs/run_evaluation/check_bad_patch_1/agentless_bad_patch'
# get dirs in outputs_dir
dirs = [d for d in os.listdir(outputs_dir) if os.path.isdir(os.path.join(outputs_dir, d))]
result_dict = {}
for dir in dirs:
    instance_id = str(dir)
    # see if dir contains report.json
    report_path = os.path.join(outputs_dir, dir, 'report.json')
    if os.path.exists(report_path):
        with open(report_path) as f:
            report = json.load(f)
            resolved = report[instance_id]['resolved']
            if resolved:
                result_dict[instance_id] = 'resolved'
            else:
                result_dict[instance_id] = 'unresolved'
    else:
        result_dict[instance_id] = 'error'


from pathlib import Path

def analyse(
    results_dir: str | Path,
):
    result_dir = Path(result_dir)
    


if __name__ == '__main__':

    import fire
    fire.Fire(analyse)