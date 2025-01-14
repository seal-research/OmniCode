# CodeArena

Welcome to **CodeArena**! This repository allows you to evaluate performance on various Software Development Activities for different models and datasets. Below, you'll find the commands to test your setup and evaluate the results. CodeArena requires you to have docker set up and running prior to executing Evaluation.

### Setup
CodeArena is currently set up to work with a specific swebench version which can be installed using 

`pip install git+https://github.com/swe-bench/SWE-bench@e0b9bf9#egg=swebench`

### Command to Test Setup for Test Generation

> **Note**: If you omit the `--instance_ids` parameter, the evaluation will run on the full dataset. However, keep in mind that this may take a significant amount of time, especially for larger datasets.

1. **Validate on a Specific Instance ID using standard Baseline:**

   Run the evaluation on a specific instance using gold Tests and patches with the following command:

   ```bash
   python -m run_evaluation_GenTests \
       --predictions_path gold \
       --max_workers 1 \
       --instance_ids sympy__sympy-20590 \
       --run_id validate-gold

2. **Evaluating Model Predictions**
   Given model predictions can be evaluated in the following manner (assumes you have bad patches available)

   ```bash
   python -m run_evaluation_GenTests \
    --dataset_name princeton-nlp/SWE-bench_Verified \
    --predictions_path <predictions_path> \ # Expected to be .json or .jsonl
    --max_workers 1 \
    --run_id BadPatchTest \
    --instance_ids astropy__astropy-13033

### CodeArena Evaluation

To run the full CodeArena benchmark you can pass the corresponding flags to the evaluation commands.

### CodeArena Benchmark Command

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
    --predictions_path predictions.json \
    --max_workers 2 \
    --run_id MyRun
```


### Contained Tasks

<div align="center">

|                | Python (Tasks) | Java (Tasks) | C++ (Tasks) |
|----------------|----------------|--------------|-------------|
| Test Generation | ✅ (15)         | ✖️ (0)        | ✖️ (0)       |
| Code Review     | ✅ (3)         | ✖️ (0)        | ✖️ (0)       |
| Code Migration  | ✖️ (0)         | ✖️ (0)        | ✖️ (0)       |


</div>

   
