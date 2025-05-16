# OmniCode

Welcome to **OmniCode[CodeArena]**! This repository allows you to evaluate performance on various Software Development Activities for different models and datasets. Below, you'll find the commands to test your setup and evaluate the results. CodeArena requires you to have docker set up and running prior to executing Evaluation.

### Setup
CodeArena is currently set up to work with a specific swebench version which can be installed using

`pip install git+https://github.com/swe-bench/SWE-bench@e0b9bf9#egg=swebench`

### Command to Test Setup for Test Generation

> **Note**: If you omit the `--instance_ids` parameter, the evaluation will run on the full dataset. However, keep in mind that this may take a significant amount of time, especially for larger datasets.

### CodeArena Evaluation

To run the full CodeArena benchmark you can pass the corresponding flags to the evaluation command line tool.

The `codearena` command allows you to run multiple code evaluation benchmarks, such as `TestGeneration` and `CodeReview`. You can specify flags to choose which benchmarks to execute. The command also supports running both benchmarks simultaneously and has an option for `CodeMigration` (currently not supported).

### Example 1: Run `TestGeneration` with a single given value for `instance_ids`

```bash
   python codearena.py \
    --TestGeneration \
    --dataset_name princeton-nlp/SWE-bench_Verified \
    --predictions_path predictions.json \
    --max_workers 1 \
    --run_id BadPatchTest \
    --instance_ids astropy__astropy-13033´
```

### Example 2: Run both `TestGeneration` and `CodeReview` on the whole dataset (by passing no instance_ids)

```bash
   python codearena.py \
    --TestGeneration --CodeReview \
    --dataset_name princeton-nlp/SWE-bench_Verified \
    --predictions_path predictions_tg.json predictions_cr.json \
    --max_workers 2 \
    --run_id MyRun
```

### Visualisation and Data annotation

The GUI based tools require the `streamlit` python package which can be installed via `pip install streamlit`
The `viz.py` script can be run with `streamlit run viz.py` and allows for inspection of the contents of each instance as well as adding data annotations for each instance.
The `add_data.py` script can be run with `streamlit run add_data.py` and allows for onboarding of instances from additional repositories, where the user may need to interactively provide the dependency and setup metadata for the new repositories.

### Supported Tasks

#### Test Generation
* **Description**: Write a test patch that passes for a correct (gold) patch and fails for an incorrect (bad) patch. Test Generation aims to create test cases that can effectively differentiate between working and broken implementations.
* **Gold Predictions Support**: Test Generation can also be run using gold predictions, where the original test patch, gold patch, and base commit (as a bad patch stand-in) are used. This allows for evaluating test generation capabilities using known-good examples.
* **Evaluation**: Success is measured by the test's ability to pass for correct implementations while failing for incorrect ones.

#### Code Review
* **Description**: A bad patch for an issue is written by a developer, and a review comment is written by another developer to guide the model toward the correct solution. This simulates real-world code review scenarios.
* **Evaluation**: This task is evaluated using the standard SWE-bench approach, measuring the effectiveness of review comments in guiding towards correct implementations.
* **Use Case**: Particularly useful for assessing a model's ability to understand code context and provide constructive feedback.

### Running OmniCode SWE-Agent

In this section you will find instructions on how to run bug fixing, test generation, style review, and review fixing!

Bug Fixing: 
The agent receives a repository and PR description, identifies and applies minimal source code changes (excluding tests) to meet the specified requirements. It verifies the fix by reproducing the issue, applying the fix, re-running the relevant test, and ensuring completeness.

Test Generation: 
This agent receives a repository and a problem description, then writes a new test in the repository’s test suite that reproduces the reported issue using the existing testing framework (e.g., pytest). 
Style Review: 

Review Fixing: 
This agent receives a problem description, a failed patch, and a review explaining at a high level the fix for the failed patch. It uses this context to implement a better fix while avoiding the mistakes identified in the review.

Style Review: 
This agent runs a style check on a given instance. It then uses the results of the style check to fix as many of the stylistic issues as possible. It is then ran again on the relevant tests to ensure functionality is unaffected. 

Note: There is support for bug fixing and test generation in java as well.

```bash
python baselines/sweagent/sweagent_regular.py \
    --input_tasks data/codearena_instances.json \
    --api_key [KEY] \
    --output_dir baselines/sweagent/logs/sweagent_outputs \
    --instance_ids astropy__astropy-13033 \
    --mode [bugfixing, testgen, bugfixing-java, testgen-java, stylereview, reviewfix]
```

### Adding Bad Patches

#### Option 1: Agentless Generation
Follow instructions found here: https://github.com/seal-research/codearena/blob/main/adding_tasks.md

#### Option 2: LLM Sourced Generation
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

## Status

### Benchmark Construction Infrastructure

<div align="center">

|                 | Python (Tasks) | Java (Tasks) |
|-----------------|----------------|--------------|
| Base Instances  | Complete       | Complete     |
| Test Generation | Complete      | In progress    |
| Code Review     | Complete       | In progress   |
| Style Review    | Complete       | In progress   |


</div>

### Instances Onboarded

<div align="center">

|                 | Python (Tasks) | Java (Tasks) |
|-----------------|----------------|--------------|
| Base Instances  | 716            | 3/128            |
| Test Generation | 0/716          | 0/128            |
| Review Response | 0/716          | 0/128            |
| Style Review    | 716/716       | 0/128   |


</div>


#### Python Instances Breakdown

<div align="center">

| Repo | Count |
|------|-------|
| astropy/astropy | 22 |
| django/django  | 231 |
| matplotlib/matplotlib |  34 |
| mwaskom/seaborn | 2 |
| pallets/flask |  1 |
| psf/requests  |  8 | 
| pydata/xarray |  22 |
| pylint-dev/pylint     |  10 |
| pytest-dev/pytest    |   19 |
| scikit-learn/scikit-learn |      32| 
| sphinx-doc/sphinx   |    44 |
| sympy/sympy   |  75 | 
| ytdl-org/youtube-dl  |    10 |
| scrapy/scrapy  | 41 | 
| keras-team/keras    |    83 |
| camel-ai/camel |  21 |
| celery/celery   | 12 |
| fastapi/fastapi | 26 |
| statsmodels/statsmodels | 23 |

</div>

### Baseline Results


#### Python

<div align="center">

|                 | Bug Fixing | Test Generation | Review Response | Style Review |
|-----------------|----------------|--------------|----------------|--------------|
| Agentless       |  2/5          |   1/5       |                  |            |
| SWE-Agent       |               |               |                  |            |
| Aider           |               |               |                  |            |

</div>


#### Java

<div align="center">

|                 | Bug Fixing | Test Generation | Review Response | Style Review |
|-----------------|----------------|--------------|----------------|--------------|
| Agentless       |               |               |                  |            |
| SWE-Agent       |               |               |                  |            |
| Aider           |               |               |                  |            |

</div>


