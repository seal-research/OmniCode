#!/bin/bash
# Run with the codearena environment already activated
# If you're in a different environment, manually run:
# conda activate codearena

# Use the INSTANCE_ID environment variable if set, otherwise use default
INSTANCE_ID=${INSTANCE_ID:-celery__celery-8806}

# Run validation
rm -rf logs
python codearena.py --BugFixing --predictions_path gold --instance_ids $INSTANCE_ID --run_id test