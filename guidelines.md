# CodeArena updates 3/11

Writing up a little more detailed documentation for the processs of adding tasks.

### Downloading PR's
- add github auth token: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens
- `cd SWE-bench/swebench/collect` and run command:
```
python get_tasks_pipeline.py \
  --repos "keras-team/keras" \
  --path_prs "/path/to/codearena/data/new_repos/PRs" \
  --path_tasks "/path/to/codearena/data/new_repos/Tasks" \
  --cutoff_date "20180101"
 ```
Note `--cutoff_date` command to ignore PR's before 2018.

Github rate-limits API queries, resetting every hour. Best solution is to run overnight, but if you want it to finish earlier, you can use the `--max_pulls` flag.

### Adding dependencies
Use the `add_data.py` streamlit app to generate and add task instances to Codearena. Use the provided area in the app to setup the correct dependencies
(further instructions?)

If it's hard (takes more than a couple hours) to get the dependencies installed, try another repo.

### Verifying tasks
Validate that your have added instances correctly by making sure that `codearena.py --BugFixing --predictions_path gold --instance_ids <ids for instances you are adding>` results in all successes

Also, manually check that the task "looks good":
- The main thing to check is that the test cases for the PR don't test additional functionality that the issue does not ask for. This might be hard to figure out, but just try to filter out obvious cases. Aim to spend no more than a few minutes per task on this.

(Add other things to check for here?)

### Adding tasks
If any tasks weren't validated successfully (from the gold predictions or manual filtering) remove them from `codearena_instances.json`.
