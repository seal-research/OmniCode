# AutoCodeRover for CodeArena

This directory contains the AutoCodeRover (ACR) integration for processing CodeArena instances. ACR is an automated code repair system that can perform multiple tasks: bug fixing, test generation, style review, and code review.

## Overview

AutoCodeRover processes software engineering tasks by:
1. **Bug Fixing**: Automatically generating patches to fix bugs
2. **Test Generation**: Creating test cases for given functionality
3. **Style Review**: Providing code style and formatting suggestions
4. **Code Review**: Analyzing code quality and suggesting improvements

## Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key or OpenRouter API key
- CodeArena instances dataset

### Setup
1. Set your API key:
   ```bash
   export OPENROUTER_API_KEY="your-api-key-here"
   ```

2. Configure the batch processing:
   ```bash
   # Edit batch_config.json to point to your dataset
   {
     "input_tasks": "../../data/filtered_dataset/instances.json",
     "output_dir": "../../results/acr_batch_results",
     "model_name": "openrouter/meta-llama/llama-4-scout",
     "modes": ["bugfixing", "testgen", "stylereview", "codereview"]
   }
   ```

### Running on Filtered CodeArena Instances

To process the filtered CodeArena dataset:

```bash
# Single instance processing
python acr_runner.py --task_file ../../data/filtered_dataset/instances.json --instance_id <instance_id>

# Batch processing (recommended for large datasets)
python acr_batch_runner.py --config batch_config.json
```

### Output Structure

Results are organized by mode:
```
results/acr_batch_results/
├── bugfixing/
│   ├── instance1.patch
│   └── bugfixing_results.jsonl
├── testgen/
│   ├── instance1.patch
│   └── testgen_results.jsonl
├── stylereview/
│   ├── instance1.patch
│   └── stylereview_results.jsonl
└── codereview/
    ├── instance1.patch
    └── codereview_results.jsonl
```

## Configuration Options

- **input_tasks**: Path to CodeArena instances JSON file
- **output_dir**: Directory for saving results
- **model_name**: LLM model to use (e.g., "openrouter/meta-llama/llama-4-scout")
- **modes**: List of modes to run (bugfixing, testgen, stylereview, codereview)
- **max_retries**: Number of retry attempts for failed tasks
- **timeout_seconds**: Timeout per task (default: 3600 seconds)

## Cluster Deployment

For large-scale processing on compute clusters, use:
```bash
./run_batch_cluster.sh
```

See `BATCH_DEPLOYMENT.md` for detailed cluster deployment instructions.

## Files

- `acr_batch_runner.py`: Main batch processing script
- `acr_runner.py`: Single instance runner
- `batch_config.json`: Configuration file
- `run_batch_cluster.sh`: Cluster deployment script
- `auto-code-rover/`: Core ACR implementation 