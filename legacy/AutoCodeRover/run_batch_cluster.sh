#!/bin/bash

# Cluster-friendly batch runner for AutoCodeRover
# This script is designed to run on remote compute clusters

set -e  # Exit on any error

# Configuration - modify these for your cluster environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/batch_config.json"
LOG_DIR="${SCRIPT_DIR}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create log directory
mkdir -p "$LOG_DIR"

# Log file for this run
LOG_FILE="${LOG_DIR}/batch_run_${TIMESTAMP}.log"

echo "================================================" | tee -a "$LOG_FILE"
echo "AutoCodeRover Batch Processing - Cluster Mode" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "================================================" | tee -a "$LOG_FILE"

# Check if we're in the right directory
if [ ! -f "${SCRIPT_DIR}/acr_batch_runner.py" ]; then
    echo "Error: acr_batch_runner.py not found in ${SCRIPT_DIR}" | tee -a "$LOG_FILE"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found: $CONFIG_FILE" | tee -a "$LOG_FILE"
    exit 1
fi

# Check environment variables
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "Warning: OPENROUTER_API_KEY environment variable not set" | tee -a "$LOG_FILE"
    echo "Make sure to set it before running the batch processor" | tee -a "$LOG_FILE"
fi

# Load configuration from JSON (requires jq)
if command -v jq &> /dev/null; then
    echo "Loading configuration from $CONFIG_FILE..." | tee -a "$LOG_FILE"
    INPUT_TASKS=$(jq -r '.input_tasks' "$CONFIG_FILE")
    OUTPUT_DIR=$(jq -r '.output_dir' "$CONFIG_FILE")
    MODEL_NAME=$(jq -r '.model_name' "$CONFIG_FILE")
    ACR_ROOT=$(jq -r '.acr_root' "$CONFIG_FILE")
    INTERMEDIATE_SAVE_INTERVAL=$(jq -r '.intermediate_save_interval' "$CONFIG_FILE")
    MAX_RETRIES=$(jq -r '.max_retries' "$CONFIG_FILE")
    TIMEOUT_SECONDS=$(jq -r '.timeout_seconds' "$CONFIG_FILE")
    
    echo "Configuration loaded:" | tee -a "$LOG_FILE"
    echo "  Input tasks: $INPUT_TASKS" | tee -a "$LOG_FILE"
    echo "  Output directory: $OUTPUT_DIR" | tee -a "$LOG_FILE"
    echo "  Model: $MODEL_NAME" | tee -a "$LOG_FILE"
    echo "  ACR root: $ACR_ROOT" | tee -a "$LOG_FILE"
    echo "  Intermediate save interval: $INTERMEDIATE_SAVE_INTERVAL" | tee -a "$LOG_FILE"
    echo "  Max retries: $MAX_RETRIES" | tee -a "$LOG_FILE"
    echo "  Timeout: $TIMEOUT_SECONDS seconds" | tee -a "$LOG_FILE"
else
    echo "Warning: jq not found, using default configuration" | tee -a "$LOG_FILE"
    INPUT_TASKS="../../data/filtered_dataset/instances.json"
    OUTPUT_DIR="../../results/acr_batch_results"
    MODEL_NAME="openrouter/meta-llama/llama-4-scout"
    ACR_ROOT="."
    INTERMEDIATE_SAVE_INTERVAL=5
    MAX_RETRIES=3
    TIMEOUT_SECONDS=3600
fi

# Check if input file exists
if [ ! -f "$INPUT_TASKS" ]; then
    echo "Error: Input tasks file not found: $INPUT_TASKS" | tee -a "$LOG_FILE"
    echo "Please check the path in your configuration file" | tee -a "$LOG_FILE"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting batch processing..." | tee -a "$LOG_FILE"
echo "This may take several hours depending on the dataset size." | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Run the batch processor with all output going to the log file
cd "$SCRIPT_DIR"
python acr_batch_runner.py \
    --config "$CONFIG_FILE" \
    --intermediate-save-interval "$INTERMEDIATE_SAVE_INTERVAL" \
    --max-retries "$MAX_RETRIES" \
    --timeout-seconds "$TIMEOUT_SECONDS" \
    2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo "" | tee -a "$LOG_FILE"
echo "================================================" | tee -a "$LOG_FILE"
echo "Batch processing finished at: $(date)" | tee -a "$LOG_FILE"
echo "Exit code: $EXIT_CODE" | tee -a "$LOG_FILE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Batch processing completed successfully!" | tee -a "$LOG_FILE"
else
    echo "ERROR: Batch processing failed with exit code $EXIT_CODE" | tee -a "$LOG_FILE"
fi

echo "Results saved to: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "================================================" | tee -a "$LOG_FILE"

exit $EXIT_CODE 