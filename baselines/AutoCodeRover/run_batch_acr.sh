#!/bin/bash

# Batch runner script for AutoCodeRover
# This script runs all modes on all instances in the dataset

set -e  # Exit on any error

# Configuration
INPUT_TASKS="../../data/filtered_dataset/instances.json"
OUTPUT_DIR="../../results/acr_batch_results"
MODEL_NAME="openrouter/meta-llama/llama-4-scout"
ACR_ROOT="$(pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  AutoCodeRover Batch Processing      ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo -e "  Input tasks: ${GREEN}${INPUT_TASKS}${NC}"
echo -e "  Output directory: ${GREEN}${OUTPUT_DIR}${NC}"
echo -e "  Model: ${GREEN}${MODEL_NAME}${NC}"
echo -e "  ACR root: ${GREEN}${ACR_ROOT}${NC}"
echo ""

# Check if input file exists
if [ ! -f "$INPUT_TASKS" ]; then
    echo -e "${RED}Error: Input tasks file not found: ${INPUT_TASKS}${NC}"
    echo "Please make sure the path is correct relative to this script."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo -e "${YELLOW}Starting batch processing...${NC}"
echo ""

# Run the batch processor
python acr_batch_runner.py \
    -i "$INPUT_TASKS" \
    -o "$OUTPUT_DIR" \
    -m "$MODEL_NAME" \
    --acr-root "$ACR_ROOT" \
    --intermediate-save-interval 5

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Batch processing completed!         ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}Results saved to:${NC}"
echo -e "  ${GREEN}${OUTPUT_DIR}${NC}"
echo ""
echo -e "${YELLOW}Directory structure:${NC}"
echo -e "  ${GREEN}${OUTPUT_DIR}/bugfixing/${NC}     - Bug fixing patches"
echo -e "  ${GREEN}${OUTPUT_DIR}/testgen/${NC}       - Test generation patches"
echo -e "  ${GREEN}${OUTPUT_DIR}/stylereview/${NC}   - Style review patches"
echo -e "  ${GREEN}${OUTPUT_DIR}/codereview/${NC}    - Code review patches"
echo ""
echo -e "${YELLOW}Each mode directory contains:${NC}"
echo -e "  - Individual patch files: ${GREEN}<instance_id>.patch${NC}"
echo -e "  - Results summary: ${GREEN}<mode>_results.jsonl${NC}" 