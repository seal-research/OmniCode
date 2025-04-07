#!/bin/bash

# List of Celery instance IDs
INSTANCES=(
    "celery__celery-8462"
    "celery__celery-8432"
    "celery__celery-8374"
    "celery__celery-8098"
    "celery__celery-7945"
    "celery__celery-7734"
    "celery__celery-7609"
    "celery__celery-7608"
)

# Check if GEMINI_API_KEY is set
if [ -z "$GEMINI_API_KEY" ]; then
    echo "Error: GEMINI_API_KEY environment variable is not set"
    exit 1
fi

# Run generation for each instance
for instance in "${INSTANCES[@]}"; do
    echo "Generating bad patches for instance: $instance"
    python baselines/badpatchllm/generate.py \
        -i data/codearena_instances.json \
        -o baselines/badpatchllm/logs/gemini_outputs \
        --instance_ids "$instance" \
        -m gemini-2.0-flash \
        -k "$GEMINI_API_KEY"
    
    # Add a small delay between requests to avoid rate limiting
    sleep 2
    echo "Completed generation for: $instance"
    echo "----------------------------------------"
done

echo "All instances completed!" 