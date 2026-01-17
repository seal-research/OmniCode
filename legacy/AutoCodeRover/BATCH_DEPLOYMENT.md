# AutoCodeRover Batch Processing - Cluster Deployment Guide

This guide explains how to deploy and run the AutoCodeRover batch processing system on remote compute clusters.

## Overview

The batch processing system runs all four ACR modes (bugfixing, testgen, stylereview, codereview) on all instances in your dataset, with robust error handling, progress tracking, and intermediate result saving suitable for long-running cluster jobs.

## Files Overview

- `acr_batch_runner.py` - Main batch processing script
- `batch_config.json` - Configuration file
- `run_batch_cluster.sh` - Cluster-friendly shell script
- `acr_runner.py` - Original ACR runner (unchanged)

## Directory Structure

After running, you'll have this structure:
```
results/acr_batch_results/
├── bugfixing/
│   ├── instance1.patch
│   ├── instance2.patch
│   └── bugfixing_results.jsonl
├── testgen/
│   ├── instance1.patch
│   ├── instance2.patch
│   └── testgen_results.jsonl
├── stylereview/
│   ├── instance1.patch
│   ├── instance2.patch
│   └── stylereview_results.jsonl
├── codereview/
│   ├── instance1.patch
│   ├── instance2.patch
│   └── codereview_results.jsonl
└── batch_progress.json
```

## Prerequisites

1. **Python Environment**: Ensure you have the same Python environment as your local ACR setup
2. **Dependencies**: All ACR dependencies must be installed
3. **API Key**: Set `OPENROUTER_API_KEY` environment variable
4. **Dataset**: Ensure your `instances.json` file is accessible

## Configuration

### 1. Edit `batch_config.json`

Modify the configuration file for your cluster environment:

```json
{
  "input_tasks": "../../data/filtered_dataset/instances.json",
  "output_dir": "../../results/acr_batch_results",
  "model_name": "openrouter/meta-llama/llama-4-scout",
  "acr_root": ".",
  "modes": ["bugfixing", "testgen", "stylereview", "codereview"],
  "intermediate_save_interval": 5,
  "save_intermediate": true,
  "log_level": "INFO",
  "max_retries": 3,
  "timeout_seconds": 3600
}
```

### 2. Key Configuration Options

- `input_tasks`: Path to your instances.json file
- `output_dir`: Where to save results
- `model_name`: The model to use (e.g., "openrouter/meta-llama/llama-4-scout")
- `modes`: Which modes to run (can be subset)
- `intermediate_save_interval`: How often to save intermediate results
- `max_retries`: Number of retries for failed tasks
- `timeout_seconds`: Timeout per task (default: 1 hour)

## Deployment Steps

### 1. Upload Files to Cluster

Upload these files to your cluster:
```
baselines/AutoCodeRover/
├── acr_batch_runner.py
├── batch_config.json
├── run_batch_cluster.sh
├── acr_runner.py (existing)
└── auto-code-rover/ (existing ACR directory)
```

### 2. Set Environment Variables

```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

### 3. Make Script Executable

```bash
chmod +x run_batch_cluster.sh
```

### 4. Run the Batch Processor

#### Option A: Using the Shell Script (Recommended)
```bash
./run_batch_cluster.sh
```

#### Option B: Direct Python Execution
```bash
python acr_batch_runner.py --config batch_config.json
```

#### Option C: With Custom Parameters
```bash
python acr_batch_runner.py \
    --config batch_config.json \
    --instance-ids "instance1,instance2" \
    --modes bugfixing stylereview \
    --timeout-seconds 7200
```

## Cluster-Specific Considerations

### 1. Job Submission (SLURM Example)

Create a SLURM job script `run_batch.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=acr_batch
#SBATCH --output=acr_batch_%j.out
#SBATCH --error=acr_batch_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Load modules (adjust for your cluster)
module load python/3.9
module load git

# Set environment variables
export OPENROUTER_API_KEY="your-api-key-here"

# Run the batch processor
cd /path/to/baselines/AutoCodeRover
./run_batch_cluster.sh
```

Submit the job:
```bash
sbatch run_batch.slurm
```

### 2. PBS/Torque Example

```bash
#!/bin/bash
#PBS -N acr_batch
#PBS -o acr_batch.out
#PBS -e acr_batch.err
#PBS -l walltime=24:00:00
#PBS -l mem=32gb
#PBS -l nodes=1:ppn=4

cd $PBS_O_WORKDIR
export OPENROUTER_API_KEY="your-api-key-here"
./run_batch_cluster.sh
```

### 3. Docker/Singularity (if needed)

If your cluster requires containerization:

```bash
# Build Docker image
docker build -t acr-batch .

# Run with Singularity
singularity exec acr-batch.sif ./run_batch_cluster.sh
```

## Monitoring and Debugging

### 1. Progress Tracking

The system saves progress to `batch_progress.json`:
```json
{
  "start_time": 1640995200.0,
  "total_tasks": 100,
  "completed_tasks": 45,
  "failed_tasks": 2,
  "mode_progress": {
    "bugfixing": {
      "completed": 45,
      "failed": 2,
      "total": 100
    }
  }
}
```

### 2. Log Files

- Main log: `logs/batch_run_YYYYMMDD_HHMMSS.log`
- Individual task logs: In ACR's results directories

### 3. Intermediate Results

Each mode saves results every N tasks (configurable):
- `bugfixing_results.jsonl`
- `testgen_results.jsonl`
- `stylereview_results.jsonl`
- `codereview_results.jsonl`

### 4. Resuming from Failure

The system is designed to be resumable. If interrupted:
1. Check `batch_progress.json` for current state
2. Restart the script - it will continue from where it left off
3. Intermediate results are preserved

## Performance Optimization

### 1. Timeout Settings

- **Small datasets**: `timeout_seconds: 1800` (30 minutes)
- **Large datasets**: `timeout_seconds: 7200` (2 hours)
- **Complex tasks**: `timeout_seconds: 10800` (3 hours)

### 2. Retry Settings

- **Stable clusters**: `max_retries: 2`
- **Unstable clusters**: `max_retries: 5`

### 3. Save Intervals

- **Frequent saves**: `intermediate_save_interval: 3`
- **Less frequent**: `intermediate_save_interval: 10`

## Troubleshooting

### Common Issues

1. **API Key Not Set**
   ```
   Error: OPENROUTER_API_KEY environment variable not set
   ```
   Solution: Set the environment variable before running

2. **Input File Not Found**
   ```
   Error: Input tasks file not found
   ```
   Solution: Check the path in `batch_config.json`

3. **Timeout Errors**
   ```
   TimeoutError: Task instance1 timed out after 3600 seconds
   ```
   Solution: Increase `timeout_seconds` in config

4. **Memory Issues**
   ```
   MemoryError: ...
   ```
   Solution: Reduce batch size or increase cluster memory allocation

### Debug Mode

Run with debug logging:
```bash
python acr_batch_runner.py --config batch_config.json --log-level DEBUG
```

## Results Analysis

### 1. Success Rate

Check the final summary in logs:
```
BATCH PROCESSING COMPLETED!
Total time: 86400.00 seconds (24.00 hours)
Total tasks processed: 1000
Successful: 950
Failed: 50
Success rate: 95.0%
```

### 2. Individual Patch Files

Each successful task creates a `.patch` file:
```json
{
  "patch": "diff --git a/file.py b/file.py\n...",
  "instance_id": "repo__issue-123",
  "mode": "bugfixing"
}
```

### 3. Results Summary

Each mode has a results file with detailed information:
```json
{
  "instance_id": "repo__issue-123",
  "mode": "bugfixing",
  "model_name": "openrouter/meta-llama/llama-4-scout",
  "patch_saved": true,
  "success": true,
  "processing_time": 45.2,
  "attempts": 1
}
```

## Security Considerations

1. **API Key**: Never commit API keys to version control
2. **Logs**: Review logs for sensitive information before sharing
3. **Permissions**: Ensure proper file permissions on cluster

## Support

For issues:
1. Check the log files first
2. Verify configuration settings
3. Test with a small subset of instances
4. Check cluster resource allocation 