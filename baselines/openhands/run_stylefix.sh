#!/usr/bin/env bash
# Run OpenHands SDK baseline for python_stylefix, targeting 10 successes.

set -euo pipefail

conda run -n seds python OmniCode/baselines/openhands/run.py \
  --env-file .env \
  --model openrouter/google/gemini-2.5-flash \
  --api-key-env OPENROUTER_API_KEY \
  --base-url https://openrouter.ai/api/v1 \
  --num-retries 2 --docker --max-instances 20 --target-successes 10 --workers 5 --resume \
  --dataset OmniCode/data/omnicode_style_instances_python.json --mode stylefix \
  --output      OmniCode/results/openhands_sdk/python_stylefix/all_preds.jsonl \
  --artifacts-dir OmniCode/results/openhands_sdk/python_stylefix \
  --run-log     OmniCode/results/openhands_sdk/python_stylefix/run_log.jsonl
