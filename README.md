# OmniCode

Welcome to **OmniCode[CodeArena]**! This repository allows you to evaluate performance on various Software Development Activities for different models and datasets. Below, you'll find the commands to test your setup and evaluate the results. CodeArena requires you to have docker set up and running prior to executing Evaluation.

## Setup

CodeArena requires `Python 3.13` and its dependecies can be installed via `pip install -r requirements.txt`

CodeArena is currently set up to work with a specific swebench and multiswebench version which can be installed using:

```bash
git clone git@github.com:seal-research/SWE-bench.git
cd SWE-bench
pip install .
cd ..
```

```bash
git clone https://github.com/seal-research/multi-swe-bench.git
cd multi-swe-bench
pip install .
```

or if you are comfortable using git submodules you can use:

```bash
git submodule update --init --recursive
cd <submodule_path>
pip install .
```

> NOTE: Running `pip install .` in multi-swe-bench installs multi-swe-bench as package. If you make changes to multi-swe-bench and wish to run/test the changes locally, you can re-run `pip install .` in the multi-sweb-bench folder to update the package for your local CodeArena.

## CodeArena Evaluation

To run the full CodeArena benchmark you can pass the corresponding flags to the evaluation command line tool.

The `codearena` command allows you to run multiple code evaluation benchmarks, such as `TestGeneration`, `StyleReview` and `CodeReview`. We further support CodeReview, which exposes the model to an inital bad patch and requires to incorporate external review feedback to correct this. You can specify flags to choose which benchmarks to execute. The command also supports running multiple benchmarks in one go.

### Example 1: Running `BugFixing` for a single instance

CodeArena with the `--BugFixing` flag can be used to evaluate whether a patch resolves the test for a particular issue.
In the following command, we pass in the `--predictions_patch gold` to indicate that we want to evaluate on the correct patch as a sanity check.
Passing in the path to actual predictions here will enable evaluating on generated patches.
This command with build the docker image and run the evaluation on the instance `astropy__astropy-13033` (which is a bug in the astropy library).

```bash
python codearena.py --BugFixing --predictions_path gold --run_id BugFixing --instance_ids astropy__astropy-13033 --use_apptainer False --g2 False
```

### Example 2: Running `TestGeneration` for single instance

The following command with the `--TestGeneration` flag can be used to evaluate generated tests. The path to generated tests can be specified with `--predictions_path`

```bash
   python codearena.py --TestGeneration --predictions_path gold --language python --max_workers 1 --run_id BadPatchTest --use_apptainer False --instance_ids astropy__astropy-14995
```

## Supported Tasks

In this section you will find instructions on the different specifications of our tasks **Bug Fixing**, **Test Generation**, **Style Review**, and **Review Fixing**!

---

### Bug Fixing (`--BugFixing`)

- **Description**: The agent receives a repository and PR description, identifies and applies minimal source code changes (excluding tests) to meet the specified requirements. It verifies the fix by reproducing the issue, applying the fix, re-running the relevant test, and ensuring completeness.
- **Evaluation**: Success is measured by the fix passing all relevant tests without introducing unintended changes.
- **Use Case**: Ideal for evaluating a model’s ability to make minimal, correct, and test-verified code changes.

---

### Test Generation (`--TestGeneration`)

- **Description**: The agent receives a repository and a problem description, then writes a new test in the repository’s test suite that reproduces the reported issue using the existing testing framework (e.g., pytest).
- **Evaluation**: Success is measured by the test failing on incorrect implementations and passing on correct ones.
- **Use Case**: Useful for assessing a model's ability to generate meaningful, differentiating test cases.

---

### Style Review (`--StyleReview`)

- **Description**: The agent runs a style check on a given instance, applies fixes for detected issues, and verifies functionality remains unaffected by re-running relevant tests.
- **Evaluation**: Success is measured by the reduction of style violations without breaking functionality.
- **Use Case**: Designed for scenarios where code quality and adherence to style guidelines are important.

---

### Review Fixing (`--BugFixing`)

- **Description**: The agent receives a problem description, a failed patch, and a review explaining the failure. It uses this context to avoid repeating mistakes and implements an improved fix. The evaluation is the same as BugFixing since we check whether the predicted patch passes the final tests.
- **Evaluation**: Success is measured by whether the improved patch resolves the issue while avoiding pitfalls highlighted in the review.
- **Use Case**: Especially relevant for testing a model’s ability to apply reviewer feedback to refine implementations.

---

## Java Support

- **Note**: Bug Fixing and Test Generation agents also support Java repositories, including Java-specific build and test tooling. Please note that this is an experimental feature and may not always function correctly. In order to set up Java support, a few additional steps are needed:

<!-- Datasets are currently included in repo -->
<!-- 0. Download data from huggingface (it is expected to be placed under multiswebench_local/mswebench_dataset) -->

1. Add desired repo into `target_repos` and `repo_file_map` in `multiswebench_local/prepare_eval`
2. From the multiswebench_local directory, `run python prepare_eval.py`
3. From the codearena directory, run `python codearena.py --MSWEBugFixing --predictions_path gold --run_id mswebench_test --max_workers 1 --instance_ids "INSERT YOUR INSTANCE HERE EX: elastic/logstash:17021" --mswe_phase all --force_rebuild True --clean True`

For now, you should stick with the original three java repos (elastic/logstash, alibaba/fastjson, mockito/mockito), since there may be some issues with the new ones that were just added very recently.

The process often takes a while. The logging is a bit different than the normal swebench btw, it instead writes to a dedicated location under multiswebench_runs.

Custom preds file can look like this for example:

```json
[
  {
    "id": "mockito/mockito:3424",
    "org": "mockito",
    "repo": "mockito",
    "number": 3424,
    "patch": "diff --git a..."
  }
]
```

Should be saved in a json format and can replace gold in the example call above.

### MSWEBugFixing for newly onboarded Java Tasks

Prerequisites:

0. Multiswebench `[org]__[repo]_dataset.jsonl` for new instance should be present
1. Add desired repo into `target_repos` and `repo_file_map` in `multiswebench_local/prepare_eval`
2. From the multiswebench_local directory, `run python prepare_eval.py`

Example Command:

```bash
# run without /scratch and apptainer
python codearena.py --MSWEBugFixing --predictions_path gold --run_id mswebench_bugfixing_test --max_workers 1 --instance_ids mockito__mockito_3173 --mswe_phase all --force_rebuild True --clean True --use_apptainer False --g2 False
# run with /scratch and apptainer
python codearena.py --MSWEBugFixing --predictions_path gold --run_id mswebench_bugfixing_test --max_workers 1 --instance_ids mockito__mockito_3173 --mswe_phase all --force_rebuild True --clean True --use_apptainer True --g2 True
```

### Java Test Generation

Test Generation for Java follows mostly the same format as Test Generation for python. However, the output files are in a different format and all instances must also exist in Multi-SWE-Bench's dataset.

Use the `--MSWETestGeneration` flag to run test generation for Java repos supported by multi-swe-bench.

#### Example Command

You can run test generation testing as follows. The tags work how they work for python test generation.

```bash
python codearena.py --MSWETestGeneration --dataset_name data/multiswebench_data/mswebench_instances.json --predictions_path gold --run_id MSWE_TestGen --instance_ids alibaba__fastjson2_2775
```

#### Example Command to run MSWETestGeneration on newly onboarded instances:

```bash
python codearena.py --MSWETestGeneration --dataset_name data/codearena_instances_java.json --predictions_path gold --run_id MSWE_TestGenGuava --instance_ids google__guava_6586
```

### Java Style Review

Java style review has been configured to work using two different types of tools: Checkstyle and PMD

#### Example Command to run Java Style Review:

```bash
python codearena.py --StyleReview --predictions_path gold --run_id mswe_java_style_review --max_workers 1 --instance_ids "apache/dubbo:10638" --mswe_phase all --force_rebuild True --review_type [pmd,checkstyle]
```

#### File Formats

The format for any prediction path other than `gold` should be as follows.

```Java
{"instance_id": "alibaba__fastjson2_2775",
"model_name_or_path": "gpt",
"full_output": "",
"model_patch": ...}
```

An example for a java instance in the same codearena format as the python example would be as follows. While the format is shared with the python instances, most of the fields are unused for the Multi-SWE-Bench instances. Additionally, all instances must also exist inside the Multi-SWE-Bench dataset.

```Java
[
  {
    "repo": "alibaba/fastjson2",
    "pull_number": 2775,
    "instance_id": "alibaba__fastjson2_2775",
    "issue_numbers": [],
    "base_commit": "12b40c7ba3e7c30e35977195770c80beb34715c5",
    "patch": ...,
    "test_patch": ...,
    "hints_text": "",
    "created_at": "",
    "version": "",
    "PASS_TO_PASS": [],
    "FAIL_TO_PASS": [],
    "bad_patches": [...]
  }
]
```

#### Result Breakdown

Results will be in `mswebench_runs/TestGeneration/`. There is a folder for each patch in the codearena instance (gold, and each bad patch in the bad patches list). Each of these folders has the files from a multi-swe-bench run, as generated by multi-swe-bench. Additionally, outside of the folders is a `report.json` file. This gives an overall report on which test cases passed and failed for each instance, all in one place.

#### Overall Status and Notes for Future Work

1. Multiple Instance at once

   - The current pipeline hasn't been tested with multiple instances being passed at the same time. It is not clear that this will work as intended with the current format.
   - Additionally, the `report.json` is not setup to work well in that case either.

2. Create `codearena.json` file for Multi_SWE instances.

   - In the current pipeline, you need a `codearena.json`-like file with all the instances you want to run. However, these instance are limited to existing ones inside multi-SWE-Bench. Thus, it would make sense to transfer all existing instance ids to a `.json` file to be used.

3. Ensuring generalizability inside Multi-SWE-Bench

   - The system hasn't been tested on a variety of instances. It is not clear that it will work on repositories other than `alibaba/fastjson2` or even with instance ids other than `alibaba__fastjson2_2775`.
   - This is largely due to the difficulty in creating generated test cases in Java.

4. Language Expansion

   - Theoretically, the pipeline should not need to be changed to work for other languages supported by Multi-SWE-Bench. However, this remains untested.

## LLM API Key

You can generate a free API key for the Gemini LLM by following the instructions at https://ai.google.dev/gemini-api/docs/api-key. This key is required to run the evaluation tasks that involve LLMs. Note that the free tier has rate limits, so don't run too many tasks in parallel.

## Running SWE-AGENT

We have configured a basic swe-agent implementation to test on our repository.

Install SWE-agent with the following command -

```bash
git clone git@github.com:seal-research/SWE-agent.git
cd SWE-agent
pip install -e .
cd ..
git clone git@github.com:seal-research/SWE-ReX.git
cd SWE-ReX
pip install -e .
cd ..
```

```bash
# run without /scratch and apptainer
python baselines/sweagent/sweagent_regular.py --input_tasks data/codearena_instances.json --api_key [KEY] --output_dir baselines/sweagent/logs/sweagent_outputs --use_apptainer False --instance_ids astropy__astropy-13033 --mode [bugfixing, testgen, bugfixing-java, testgen-java, stylereview, reviewfix] --g2 False --output_file baselines/sweagent/logs/sweagent_outputs/all_preds.jsonl --model_name openrouter/google/gemini-2.5-flash 
# run with /scratch and apptainer
python baselines/sweagent/sweagent_regular.py --input_tasks data/codearena_instances.json --api_key [KEY] --output_dir /scratch/$USER/baselines/sweagent/logs/sweagent_outputs --use_apptainer True --instance_ids astropy__astropy-13033 --mode [bugfixing, testgen, bugfixing-java, testgen-java, stylereview, reviewfix] --g2 True --output_file baselines/sweagent/logs/sweagent_outputs/all_preds.jsonl --model_name openrouter/google/gemini-2.5-flash 
```

### Running SWE-Agent for Java Instances

Prerequisites:

- Instance should be present in `data/codearena_instances_java.json`
- Base image should already built in your local docker (e.g. MSWEBugFixing)

Example command:

```bash
python baselines/sweagent/sweagent_regular.py --input_tasks data/codearena_instances_java.json --api_key [key] --output_dir baselines/sweagent/logs/sweagent_outputs --use_apptainer False --instance_ids google__guava_6586 --mode [bugfixing-java, testgen-java] --g2 False --output_file baselines/sweagent/logs/sweagent_outputs/all_preds.jsonl --model_name openrouter/google/gemini-2.5-flash 
```

### Running SWE-Agent for Java Style Review

Prerequisites:

- Dataset should be downloaded from
  https://drive.google.com/file/d/1ZVg-rVXU9hPN0iO1qsxm-Ru7a5AUmwJU/view?usp=drive_link (PMD)
  https://drive.google.com/file/d/15yDXDq9S-mOOYoNT0na7MqiodvJ8Rf4g/view?usp=drive_link (Checkstyle)
- Base image should already built in your local docker (e.g. MSWEBugFixing)

```bash
python baselines/sweagent/convert_style_errors_to_sweagent_from_dataset.py --org apache --repo dubbo --pr_number 10638 --style_tool pmd --output sweagent_input.json
```

OR (to use only errors of files modified in gold patch)

```bash
python baselines/sweagent/convert_filtered_style_errors_to_sweagent_from_dataset.py --org apache --repo dubbo --pr_number 10638 --style_tool pmd --output sweagent_input.json
```

```bash
python baselines/sweagent/sweagent_regular.py -i sweagent_input.json -o sweagent_pmd_apache_dubbo_10638_results --mode stylereview --style_tool pmd --model_name "gemini/gemini-2.5-flash" --api_key $GEMINI_API_KEY
```

## Adding Bad Patches

### Option 1: Agentless Generation

#### General Setup:

There is a codearena_local dataset used by agentless. This dataset does not automatically update when there are changes to `mswebench_instances.json` or `codearena_instances.json`. If you change these files, you must make a trivial change to `baselines/Agentless/codearena_local/codearena_local.py` in order for them to be reflected in Agentless.

#### Usage:

In the submodule folder under `baselines/Agentless` you can modify the values inside `run.sh`. There are existing examples in this file for openrouter. The general structure is as follows:

```bash
run_id=$1
instance=$2
dataset=$3
use_apptainer=true
runs=25

bash full_bad_patch_gen.sh "$instance" "$runs" "$run_id" {model name here (e.g. gemma-2-9b-it, llama-3-8b-instruct)} {provider name here (e.g. google, openrouter)} 'codearena_local' {coding language here (e.g. python, java, cpp)} "$dataset" "$use_apptainer"
```

`run_id`, `instance`, and `dataset` are taken as arguments when running `bash run.sh`. The `run_id` is the name of the run. `instance` is one of the instance_ids from the codearena dataset. `dataset` is the location where you want successfully generated bad patches to be placed. This should include the file name, not just the directory.

#### Adding bad patches back to dataset:

Wherever you have chosen to put your bad patches, they should be in one .jsonl file. You may chose to add these back to the dataset however you chose, but note that the entries in the .jsonl have several key values. The extra keys include a reason for the bad patch having failed a test and the instance id for the bad patch. Both of this values can be omitted when adding back to the dataset.

#### Adding reviews:

At this point, the dataset will include bad patch entries but they are not yet complete. You should add reviews for each of the bad patches as well. You can accomplish this by running:

```bash
python baselines/badpatchllm/generate_review.py \
--output_dir {your chosen output directory} \
--api_key {your api key (at this time must be a gemini api key)} \
--input_tasks {data/multiswebench_data/mswebench_instances.json or data/codearena_instances.json}\
--num_reviews_per_patch 1 \
--instance_ids {instance ids that you added bad patches for}
```

### Option 2: LLM Sourced Generation

```bash
python baselines/badpatchllm/generate_bad.py \
    -o baselines/badpatchllm/logs/gemini_outputs \
    --instance_ids astropy__astropy-13033 \
    -m [gemini-2.5-flash-preview-4-17]  (recommended] \
    -k [KEY] \
    --run_id test \
    -n 3 \
    -d data/codearena_instances.json \
```

Note: Raw diff files will also be outputted and found under the user specified output directory for ease of use.

### Generating Reviews

```bash
python baselines/badpatchllm/generate_review.py \
    --input_tasks data/codearena_instances.json \
    --api_key [KEY] \
    --output_dir baselines/badpatchllm/logs/gemini_outputs \
    --instance_ids astropy__astropy-13033
```

You will need to move the data back to the original dataset. The reviews will not be added in place.

## Deploying experiments on Google Cloud

To enable more reproducable and parallel evaluation, we also provide experimental support for launching jobs in Google Cloud via `gc/multivm.py`. For example the following command will run SWE-agent on instances specified in the `instances_to_run.txt` file. Details about available modes can be found at the end of `gc/utils.py`

Note that this requires a base vm to be set up with the CodeArena and SWE-agent dependencies in your Google Cloud project.

```bash
python gc/multivm.py \
   instances_to_run.txt \
   sweagent-bf \
   --base_vm sedsbase \
   --base_zone us-east1-b \
   --zone us-east1-b \
   --num_vms 1 \
   --key <api key>
```

## Status

### Benchmark Construction Infrastructure

<div align="center">

|                 | Python (Tasks) | Java (Tasks) |
| --------------- | -------------- | ------------ |
| Base Instances  | Complete       | Complete     |
| Test Generation | Complete       | In progress  |
| Code Review     | Complete       | In progress  |
| Style Review    | Complete       | In progress  |

</div>

### Instances Onboarded

<div align="center">

|                 | Python (Tasks) | Java (Tasks) |
| --------------- | -------------- | ------------ |
| Base Instances  | 716            | 128/128      |
| Test Generation | 300/716        | 0/128        |
| Review Response | 300/716        | 0/128        |
| Style Review    | 716/716        | 0/128        |

</div>

#### Python Instances Breakdown

<div align="center">

| Repo                      | Count |
| ------------------------- | ----- |
| astropy/astropy           | 22    |
| django/django             | 231   |
| matplotlib/matplotlib     | 34    |
| mwaskom/seaborn           | 2     |
| pallets/flask             | 1     |
| psf/requests              | 8     |
| pydata/xarray             | 22    |
| pylint-dev/pylint         | 10    |
| pytest-dev/pytest         | 19    |
| scikit-learn/scikit-learn | 32    |
| sphinx-doc/sphinx         | 44    |
| sympy/sympy               | 75    |
| ytdl-org/youtube-dl       | 10    |
| scrapy/scrapy             | 41    |
| keras-team/keras          | 83    |
| camel-ai/camel            | 21    |
| celery/celery             | 12    |
| fastapi/fastapi           | 26    |
| statsmodels/statsmodels   | 23    |

</div>
