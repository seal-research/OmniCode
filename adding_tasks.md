# Tracker

## TODOs

- [ ] Complete adding base instances in `data/components`
- [ ] Add bad patches for each base instance
- [ ] Add reviews for bad patches
- [ ] Unify data files in `data/components` into a single file
- [ ] Add code to run the _lint fixing_ task
- [ ] Run baselines for all tasks
- [ ] Adding Java tasks

## Data Onboarding Notes

### Phase 1: Onboarding Base Tasks

#### Downloading PR's
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

#### Adding dependencies
Use the `add_data.py` streamlit app to generate and add task instances to Codearena. Use the provided area in the app to setup the correct dependencies.
For examples/inspiration you can look at the configs in `SWE-bench/swebench/harness/constants.py`.

If it's hard (takes more than a couple hours) to get the dependencies installed, try another repo.

#### Verifying tasks
Validate that your have added instances correctly by making sure that `python codearena.py --BugFixing --predictions_path gold --instance_ids <ids for instances you are adding>` results in all successes.

Also, manually check that the task "looks good":
- Look at the issue that the PR solves, and make sure it gives a well specified problem
- Make sure the test cases don't test additional functionality that the issue does not ask for.
- Just try to filter out obvious cases. Aim to spend no more than a few minutes per task on this.

#### Adding tasks
If any tasks weren't validated successfully (from the gold predictions or manual filtering) remove them from `codearena_instances.json`.

### Phase 2: Adding bad patches

#### Setting up SWE-Agent

- Follow [these instructions](https://swe-agent.com/latest/installation/source/) to install SWE-Agent
- Get an API to use a model with SWE-Agent.
  - Gemini provides some free credits. Setup an account at [https://aistudio.google.com/](https://aistudio.google.com/) using you non-edu email and get an API key at [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)
  - Add this key as an environment variable `export GEMINI_API_KEY=<your key>`

#### Running SWE-Agent to generate patches

Use the following command (substituing paths and instance ids) to run SWE-Agent on a specific issue in CodeArena:

```
python baselines/sweagent/sweagent_regular.py -i data/codearena_instances.json -o baselines/sweagent/sweagent_outputs --instance_ids fastapi__fastapi-1534 -m gemini/gemini-2.0-flash
```

This should generate an instance specific log inside `baselines/sweagent/logs/sweagent_outputs/<instance_id>`.
You can inspect the various logs (the most useful being the `.traj` file) to understand SWE-Agent execution.
The above command would also have updated `baselines/sweagent/logs/sweagent_outputs/all_preds.jsonl`, adding the prediction data as a new line at the end of the file.
Since this data is in a slightly different format from the prediction data expected by codearena, we need to run the following cleaning script to clean it up -

```
python clean_sweagent_outputs.py /baselines/sweagent/logs/sweagent_outputs/all_preds.jsonl
```

This will modify `all_preds.jsonl` in place.


#### Using Agentless to generate bad patches

##### Installation
1. Clone Simon's fork of the Agentless repo (into `codearena/baselines/`) and create a conda environment for it:

```bash
git clone https://github.com/simonalford42/Agentless.git
cd Agentless
conda create -n agentless python=3.11
conda activate agentless
pip install -r requirements.txt
```

2. Make sure you have an OpenAI API key set to the `OPENAI_API_KEY` environment variable and that Docker is working.

##### To run Agentless:
Make sure you've pulled from our Agentless repo copy to have the most up to date version.
Run `run_agentless.sh $target_id` passing in the target id that you wish to generate a patch for.
Notes:
- `run_agentless.sh` will run the method via a sequence of python commands for the different steps. The script was derived from the instructions and commands at https://github.com/simonalford42/Agentless/blob/main/README_swebench.md, see that page for full explanation and instructions.
- The script will ask you to trust custom code. This is to load the codearena instances into a local huggingface dataset (`codearena_local.py`) to interface with Agentless.
- The `run_agentless.sh` script will also check whether the generated patch is a good bad patch, and add the bad patch to the codearena task dataset if it is.
- Make sure to also manually check that the patch looks okay by inspecting it. A .diff file is saved by codearena bug fixing under `codearena/logs/run_evaluation/check_pad_patch/agentless/$target_id/patch.diff`.

Dev notes:
- I removed the reproduction tests so that it is always able to come up with a patch. Without doing this, the agent would usually not be able to come up with anything that passes the reproduction stage.
- This uses OpenAI credits, but if you use 4o mini and 1 sample it isn't very expensive. For me it takes ~5-10 minutes and 10 cents to get one sample for a task. You can track usage at https://platform.openai.com/settings/organization/usage.
- Currently not sure whether running multiple times to get more samples is possible or not. In my experience, if it generates a successful patch, rerunning will still generate a successful patch.

#### Evaluate the generated patches with CodeArena

Run CodeArena with BugFixing mode to ensure that the generated patch is an incorrect fix -

```
python codearena.py --BugFixing --predictions_path baselines/sweagent/logs/sweagent_outputs/all_preds.jsonl --instance_ids fastapi__fastapi-1534 --run_id test
```

Additionally, manually review the patch to ensure that it is a reasonable attempt at fixing the issue.


#### Add the bad patches to CodeArena

Use the `add_sweagent_bad_patches.py` script to add bad patches to your data file in `data/components/` in the following way -

```
python add_sweagent_bad_patches.py baselines/sweagent/logs/sweagent_outputs/all_preds.jsonl data/components/fastapi_instances.json
```
