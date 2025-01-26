# CodeArena

Welcome to **CodeArena**! This repository allows you to evaluate performance on various Software Development Activities for different models and datasets. Below, you'll find the commands to test your setup and evaluate the results. CodeArena requires you to have docker set up and running prior to executing Evaluation.

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

#### Code Migration (Coming Soon)
* **Description**: Translate code between programming languages while maintaining functionality and idiomaticity.
* **Scope**: Initially focusing on translations between major programming languages.
* **Future Plans**: Will support automated verification of translated code functionality.

### Contained Tasks

We use SWE-Bench_Verified as an initial data source and have manually created bad patches and Reviews for them. 

<div align="center">

|                | Python (Tasks) | Java (Tasks) | C++ (Tasks) |
|----------------|----------------|--------------|-------------|
| Test Generation | ✅ (15)         | ✖️ (0)        | ✖️ (0)       |
| Code Review     | ✅ (15)         | ✖️ (0)        | ✖️ (0)       |
| Code Migration  | ✖️ (0)         | ✖️ (0)        | ✖️ (0)       |


</div>

   
