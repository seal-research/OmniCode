# OmniCode

Welcome to **OmniCode**! This is benchmark for evaluating various LLM powered agents on various Software Developemnt activities . Below, you'll find the commands to test your setup and evaluate the results.

OmniCode synthetically builds multiple tasks out of a base dataset to holistically evaluate software
engineering agents. Four different types of tasks that we consider: Bug fixing, test generation, responding to
code review, and enforcing style guidelines.

<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/46a4e55c-d8fd-4940-a7ad-26ea746f6c54" />

## Supported Tasks

In this section, you will find details of the different specifications of our tasks: **Bug Fixing**, **Test Generation**, **Style Review**, and **Review Response**!

---

### Bug Fixing Evaluation (`--BugFixing`)

- **Description**: The agent receives a repository and PR description, identifies and applies minimal source code changes (excluding tests) to meet the specified requirements. It verifies the fix by reproducing the issue, applying the fix, re-running the relevant test, and ensuring completeness.
- **Evaluation**: Success is measured by the fix passing all relevant tests without introducing unintended changes.
- **Use Case**: Ideal for evaluating a model’s ability to make minimal, correct, and test-verified code changes.

---

### Test Generation Evaluation (`--TestGeneration`)

- **Description**: The agent receives a repository and a problem description, then writes a new test in the repository’s test suite that reproduces the reported issue using the existing testing framework (e.g., pytest).
- **Evaluation**: Success is measured by the test failing on incorrect implementations and passing on correct ones.
- **Use Case**: Useful for assessing a model's ability to generate meaningful, differentiating test cases.

<p align="center">
<img width="500" alt="image" src="https://github.com/user-attachments/assets/6a59e91d-a824-4fb7-a18c-74b2cf0ecd7e" />
</p>

---

### Style Review Evaluation (`--StyleReview`)

- **Description**: The agent runs a style check on a given instance, applies fixes for detected issues, and verifies functionality remains unaffected by re-running relevant tests.
- **Evaluation**: Success is measured by the reduction of style violations without breaking functionality.
- **Use Case**: Designed for scenarios where code quality and adherence to style guidelines are important.

<p align="center">
<img width="500" height="210" alt="image" src="https://github.com/user-attachments/assets/5eb69bcd-8266-4f06-bedd-b3b0aaf3886e" />
</p>

---

### Review Response Evaluation (`--BugFixing`)

- **Description**: The agent receives a problem description, a failed patch, and a review explaining the failure. It uses this context to avoid repeating mistakes and implements an improved fix. The evaluation is the same as BugFixing since we check whether the predicted patch passes the final tests.
- **Evaluation**: Success is measured by whether the improved patch resolves the issue while avoiding pitfalls highlighted in the review.
- **Use Case**: Especially relevant for testing a model’s ability to apply reviewer feedback to refine implementations.

<p align="center">
<img width="300" height="550" alt="image" src="https://github.com/user-attachments/assets/485f47ac-7d40-4421-bd70-2b684689322e" />
</p>

## Setup

### Environment
OmniCode requires `Python 3.13` and its dependencies can be installed via `pip install -r requirements.txt`

### Clone
We have some submodules have to be clone as well. You can clone our repo by: 
```bash
git clone --recursive git@github.com:seal-research/OmniCode.git
cd OmniCode
```

### Dataset
Our dataset is currently located on HuggingFace at ([seal-research/OmniCode](https://huggingface.co/datasets/seal-research/OmniCode/tree/main)). 
To use OmniCode, you will have to pull/download the data from our hugging face repo to the `./data` directory. 

```bash
pip install -U hf
hf download seal-research/OmniCode \
  --repo-type dataset \
  --local-dir data
```

```
OmniCode/   
└── data/
   ├── omnicode_instances_python.json
   ├── omnicode_instances_java.json
   ├── omnicode_instances_cpp.json
   ├── omnicode_style_instances_python.json
   ├── omnicode_style_instances_java.json
   └── omnicode_style_instances_cpp.json
```

### Submodules
OmniCode is currently set up to work with a specific swebench and multiswebench version, which can be installed using:

```bash
cd SWE-bench
pip install .
cd ..
```

```bash
cd multi-swe-bench
pip install .
cd ..
```

Or if you are comfortable using git submodules, you can use:

```bash
git submodule update --init --recursive
cd <submodule_path>
pip install .
```

> NOTE: Running `pip install .` in multi-swe-bench installs multi-swe-bench as a package. If you make changes to multi-swe-bench and wish to run/test the changes locally, you can re-run `pip install .` in the multi-sweb-bench folder to update the package for your local OmniCode.

### Apptainer

Compared to Docker, Apptainer is designed to run containers without requiring root privileges, making it more suitable for shared or restricted environments (e.g., HPC clusters).
Docker is commonly used in service and cloud deployments but usually relies on root or privileged daemons.
By supporting Apptainer, OmniCode enables containerized workflows for users who cannot use Docker due to permission or security constraints.

Follow the [official instruction](https://apptainer.org/docs/admin/main/installation.html#install-from-pre-built-packages) to install Apptainer first. When you want to use apptainer mode, turn --use_apptainer parameter into True in your command. If --use_apptainer is False, OmniCode would use Docker automatically. 

## OmniCode Evaluation

To run the full OmniCode benchmark, you can pass the corresponding flags to the evaluation command line tool.

The `omnicode` command allows you to run multiple code evaluation benchmarks, such as `TestGeneration`, `StyleReview` and `ReviewResponse`. You can specify flags to choose which benchmarks to execute. The command also supports running multiple benchmarks in one go.

### Example 1: Running `BugFixing` for a single instance

OmniCode with the `--BugFixing` flag can be used to evaluate whether a patch resolves the test for a particular issue.
In the following command, we pass in the `--predictions_patch gold` to indicate that we want to evaluate on the correct patch as a sanity check.
Passing in the path to actual predictions here will enable evaluating on generated patches.
This command with build the docker image and run the evaluation on the instance `astropy__astropy-13033` (which is a bug in the astropy library).

```bash
python omnicode.py --BugFixing --dataset_name data/omnicode_instances_python.json --predictions_path gold --run_id BugFixing --instance_ids astropy__astropy-13236 --use_apptainer False
```

### Example 2: Running `TestGeneration` for a single instance

The following command with the `--TestGeneration` flag can be used to evaluate generated tests. The path to generated tests can be specified with `--predictions_path`

```bash
   python omnicode.py --TestGeneration --dataset_name data/omnicode_instances_python.json --predictions_path gold --language python --max_workers 1 --run_id BadPatchTest --use_apptainer False --instance_ids astropy__astropy-14995
```

---

## Java Support

- **Note**: Bug Fixing and Test Generation agents also support Java repositories, including Java-specific build and test tooling. Please note that this is an experimental feature and may not always function correctly. In order to set up Java support, a few additional steps are needed:

<!-- Datasets are currently included in repo -->
<!-- 0. Download data from HuggingFace (it is expected to be placed under multiswebench_local/mswebench_dataset) -->

1. Add desired repo into `target_repos` and `repo_file_map` in `multiswebench_local/prepare_eval`
2. From the multiswebench_local directory, `run python prepare_eval.py`
3. From the omnicode directory, run `python omnicode.py --MSWEBugFixing --predictions_path gold --run_id mswebench_test --max_workers 1 --instance_ids elastic__logstash_17021 --mswe_phase all --force_rebuild True --clean True --use_apptainer False`

For now, you should stick with the original three java repos (elastic/logstash, alibaba/fastjson, mockito/mockito), since there may be some issues with the new ones that were just added very recently.

The process often takes a while. The logging is a bit different than the normal swebench btw, it instead writes to a dedicated location under multiswebench_runs.

Custom preds file can look like this, for example:

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

Should be saved in a JSON format and can replace gold in the example call above.

### MSWEBugFixing for newly onboarded Java Tasks

Prerequisites:

0. Multiswebench `[org]__[repo]_dataset.jsonl` for new instance should be present
1. Add desired repo into `target_repos` and `repo_file_map` in `multiswebench_local/prepare_eval`
2. From the multiswebench_local directory, `run python prepare_eval.py`

Example Command:

```bash
python omnicode.py --MSWEBugFixing --predictions_path gold --run_id mswebench_bugfixing_test --max_workers 1 --instance_ids google__gson_1093 --mswe_phase all --force_rebuild True --clean True --use_apptainer False
```

### Java Test Generation

Test Generation for Java follows mostly the same format as Test Generation for Python. However, the output files are in a different format, and all instances must also exist in Multi-SWE-Bench's dataset.

Use the `--MSWETestGeneration` flag to run test generation for Java repos supported by multi-swe-bench.

#### Example Command

You can run test generation testing as follows. The tags work how they work for python test generation.

```bash
python omnicode.py --MSWETestGeneration --dataset_name data/multiswebench_data/mswebench_instances.json --predictions_path gold --run_id MSWE_TestGen --instance_ids alibaba__fastjson2_2775 --use_apptainer False
```

#### Example Command to run MSWETestGeneration on newly onboarded instances:

```bash
python omnicode.py --MSWETestGeneration --dataset_name data/omnicode_instances_java.json --predictions_path gold --run_id MSWE_TestGenGuava --instance_ids google__guava_6586 --use_apptainer False
```

### Java Style Review

Java style review has been configured to work using two different types of tools: Checkstyle and PMD

#### Example Command to run Java Style Review:

```bash
python omnicode.py --stylereview-java-pmd --predictions_path gold --run_id mswe_java_style_review --max_workers 1 --instance_ids "apache__dubbo_10638" --mswe_phase all --force_rebuild True --review_type [pmd,checkstyle] --use_apptainer False
```
The above call will not work for environments which do not support Docker, hence the following is preferred:

```bash
python multiswebench_local/multi_swe_bench/harness/style_review/pmd_runner.py --org [org] --repo [repo] --pull [pull_number] --base-commit [base_commit]
```

For Evaluating Patches (Style Violation Reduction Evaluation):

```bash
python legacy/code/evaluate_splitted_java_style_review.py --org [org] --repo [repo] --pull_number [pr] --swe-file [json file containing patches]
```

### CPP Style Review

CPP style review has been configured to work using clang tidy

#### Example Command to run CPP Style Review:

```bash
python multiswebench_local/multi_swe_bench/harness/CPP_style_review/style_reviewcpp.py --repo-url [url] --pr [pr] --clang-tidy-config multiswebench_local/multi_swe_bench/harness/CPP_style_review/.clang-tidy --work-dir [dir] --out results1.json --instance-id [id] --swe-results [file]
```

For Evaluating Patches (Style Violation Reduction Evaluation):

```bash
python legacy/code/evaluate_splitted_cpp_style_review.py --org [org] --repo repo --pr --swe-file [json file containing patches]
```


#### File Formats

The format for any prediction path other than `gold` should be as follows.

```Java
{"instance_id": "alibaba__fastjson2_2775",
"model_name_or_path": "gpt",
"full_output": "",
"model_patch": ...}
```

An example for a Java instance in the same Omnicode format as the Python example would be as follows. While the format is shared with the Python instances, most of the fields are unused for the Multi-SWE-Bench instances. Additionally, all instances must also exist inside the Multi-SWE-Bench dataset.

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

Results will be in `mswebench_runs/TestGeneration/`. There is a folder for each patch in the omnicode instance (gold) and each bad patch in the bad patches list. Each of these folders has the files from a multi-swe-bench run, as generated by multi-swe-bench. Additionally, outside of the folders is a `report.json` file. This gives an overall report on which test cases passed and failed for each instance, all in one place.

## LLM API Key

You can generate a free API key for the Gemini LLM by following the instructions at https://ai.google.dev/gemini-api/docs/api-key. This key is required to run the evaluation tasks that involve LLMs. Note that the free tier has rate limits, so don't run too many tasks in parallel.

## Running SWE-AGENT

We have configured a basic swe-agent implementation to test on our repository.

Install SWE-agent with the following command -

```bash
cd SWE-ReX
pip install -e .
cd ..
cd SWE-agent
pip install -e .
cd ..
```

```bash
# run without apptainer
python baselines/sweagent/sweagent_regular.py --input_tasks data/omnicode_instances_python.json --api_key [KEY] --output_dir baselines/sweagent/logs/sweagent_outputs --use_apptainer False --instance_ids astropy__astropy-13236 --mode [bugfixing, testgen, bugfixing-java, bugfixing-cpp, testgen-java, stylereview-python,stylereview-java-checkstyle,stylereview-java-pmd,stylereview-cpp-clangtidy, reviewfix] --output_file baselines/sweagent/logs/sweagent_outputs/all_preds.jsonl --model_name openrouter/google/gemini-2.5-flash 
# run with apptainer
python baselines/sweagent/sweagent_regular.py --input_tasks data/omnicode_instances_python.json --api_key [KEY] --output_dir /scratch/$USER/baselines/sweagent/logs/sweagent_outputs --use_apptainer True --instance_ids astropy__astropy-13236 --mode [bugfixing, testgen, bugfixing-java, , bugfixing-cpp, testgen-java, stylereview-python,stylereview-java-checkstyle,stylereview-java-pmd,stylereview-cpp-clangtidy, reviewfix] --output_file baselines/sweagent/logs/sweagent_outputs/all_preds.jsonl --model_name openrouter/google/gemini-2.5-flash 
```

### Running SWE-Agent for Java Instances

Prerequisites:

- Instance should be present in `data/omnicode_instances_java.json`
- Base image should already built in your local docker (e.g. MSWEBugFixing)

Example command:

```bash
python baselines/sweagent/sweagent_regular.py --input_tasks data/omnicode_instances_java.json --api_key [key] --output_dir baselines/sweagent/logs/sweagent_outputs --use_apptainer False --instance_ids google__guava_6586 --mode [bugfixing-java, testgen-java] --output_file baselines/sweagent/logs/sweagent_outputs/all_preds.jsonl --model_name openrouter/google/gemini-2.5-flash 
```

## Adding Bad Patches

### Option 1: Agentless Generation

#### General Setup:

There is a codearena_local dataset used by agentless. This dataset does not automatically update when there are changes to `mswebench_instances.json` or `omnicode_instances_python.json`. If you change these files, you must make a trivial change to `synthetic_datagen/Agentless/codearena_local/codearena_local.py` in order for them to be reflected in Agentless.

#### Usage:

In the submodule folder under `synthetic_datagen/Agentless`, you can modify the values inside `run.sh`. There are existing examples in this file for OpenRouter. The general structure is as follows:

```bash
run_id=$1
instance=$2
dataset=$3
use_apptainer=false
runs=25

bash full_bad_patch_gen.sh "$instance" "$runs" "$run_id" {model name here (e.g. gemma-2-9b-it, llama-3-8b-instruct)} {provider name here (e.g. google, openrouter)} 'codearena_local' {coding language here (e.g. python, java, cpp)} "$dataset" "$use_apptainer"
```

`run_id`, `instance`, and `dataset` are taken as arguments when running `bash run.sh`. The `run_id` is the name of the run. `instance` is one of the instance_ids from the omnicode dataset. `dataset` is the location where you want successfully generated bad patches to be placed. This should include the file name, not just the directory.

#### Adding bad patches back to the dataset:

Wherever you have chosen to put your bad patches, they should be in one .jsonl file. You may chose to add these back to the dataset however you chose, but note that the entries in the .jsonl have several key values. The extra keys include a reason for the bad patch having failed a test and the instance id for the bad patch. Both of these values can be omitted when adding back to the dataset.

#### Adding reviews:

At this point, the dataset will include bad patch entries, but they are not yet complete. You should add reviews for each of the bad patches as well. You can accomplish this by running:

```bash
python synthetic_datagen/badpatchllm/generate_review.py \
--output_dir {your chosen output directory} \
--api_key {your api key} \
--input_tasks {data/multiswebench_data/mswebench_instances.json or data/omnicode_instances_python.json}\
--num_reviews_per_patch 1 \
--instance_ids {instance ids that you added bad patches for}
```

### Option 2: LLM Sourced Generation

```bash
python synthetic_datagen/badpatchllm/generate_bad.py \
    -o synthetic_datagen/badpatchllm/logs/gemini_outputs \
    --instance_ids astropy__astropy-13033 \
    -m [gemini-2.5-flash-preview-4-17]  (recommended] \
    -k [KEY] \
    --run_id test \
    -n 3 \
    -d data/omnicode_instances_python.json \
```

Note: Raw diff files will also be outputted and found under the user specified output directory for ease of use.

### Generating Reviews

```bash
python synthetic_datagen/badpatchllm/generate_review.py \
    --input_tasks data/omnicode_instances_python.json \
    --api_key [KEY] \
    --output_dir synthetic_datagen/badpatchllm/logs/gemini_outputs \
    --instance_ids astropy__astropy-13033
```

You will need to move the data back to the original dataset. The reviews will not be added in place.


## Status

### Available Instances

<div align="center">

|                 | Python (Tasks) | Java (Tasks) | Cpp (Tasks)  |
| --------------- | -------------- | ------------ | ------------ |
| Base Instances  | 273            | 109          | 112          |
| Test Generation | 164            | 44           | 79           |
| Review Response | 164            | 44           | 79           |
| Style Review    | 144            | 124          | 147          |

</div>

<!-- #### Python Instances Breakdown

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

</div> -->
