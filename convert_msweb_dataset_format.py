import json
import os
from pathlib import Path


if __name__ == "__main__":
    
    dataset_path = Path("multiswebench/data/datasets/dataset.jsonl")

    with open(dataset_path, "r") as msweb_data:
        json_list = list(msweb_data)

        new_format = []
        for json_str in json_list:
            instance = json.loads(json_str)

            # print(instance["f2p_tests"].keys())

            # look at get_modified_file on test_patch from add_data.py file for f2p
            # need repo, problem description, gold_patch (not p2p or f2p)

            new_json = {"repo":f"{instance["org"]}/{instance["repo"]}", "pull_number": instance["number"], "instance_id":instance["instance_id"], "issue_numbers":[], "base_commit":instance["base"]["sha"], "patch":instance["fix_patch"], "test_patch":instance["test_patch"], "problem_statement":f"{instance["title"]}\n{instance["body"]}", "hints_text":"", "created_at":"", "version":"", "PASS_TO_PASS":[], "FAIL_TO_PASS":[]}


            new_format.append(new_json)

    output_path = Path("data/components/msweb_codearena_formatted.json")
    with open(output_path, "w") as f:
        json.dump(new_format, f)
            