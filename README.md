# CodeArena

Welcome to **CodeArena**! This repository allows you to evaluate performance on various Software Development Activities for different models and datasets. Below, you'll find the commands to test your setup and evaluate the results. CodeArena requires you to have docker set up and running prior to executing Evaluation.

> **Note**: If you omit the `--instance_ids` parameter, the evaluation will run on the full dataset. However, keep in mind that this may take a significant amount of time, especially for larger datasets.

### Command to Test Setup for Test Generation

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

### Contained Tasks

<div align="center">

|                | Python (Tasks) | Java (Tasks) | C++ (Tasks) |
|----------------|----------------|--------------|-------------|
| Test Generation | ✅ (15)         | ✖️ (0)        | ✖️ (0)       |
| Code Review     | ✅ (3)         | ✖️ (0)        | ✖️ (0)       |

</div>

   
