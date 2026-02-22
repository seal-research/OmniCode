#!/usr/bin/env bash
# Run OpenHands SDK baseline for remaining combos (java_stylefix + cpp).
# All python_* and java bugfixing/testgen/reviewfix combos are complete.

set -euo pipefail

BASE="conda run -n seds python OmniCode/baselines/openhands/run.py \
  --env-file .env \
  --model openrouter/google/gemini-2.5-flash \
  --api-key-env OPENROUTER_API_KEY \
  --base-url https://openrouter.ai/api/v1 \
  --num-retries 2 --docker --max-instances 20 --target-successes 10 --workers 5 --resume"

# ---------------------------------------------------------------------------
# C++
# ---------------------------------------------------------------------------
$BASE --dataset OmniCode/data/omnicode_instances_cpp.json --mode bugfixing \
  --output      OmniCode/results/openhands_sdk/cpp_bugfixing/all_preds.jsonl \
  --artifacts-dir OmniCode/results/openhands_sdk/cpp_bugfixing \
  --run-log     OmniCode/results/openhands_sdk/cpp_bugfixing/run_log.jsonl

$BASE --dataset OmniCode/data/omnicode_instances_cpp.json --mode testgen \
  --output      OmniCode/results/openhands_sdk/cpp_testgen/all_preds.jsonl \
  --artifacts-dir OmniCode/results/openhands_sdk/cpp_testgen \
  --run-log     OmniCode/results/openhands_sdk/cpp_testgen/run_log.jsonl

$BASE --dataset OmniCode/data/omnicode_instances_cpp.json --mode reviewfix \
  --output      OmniCode/results/openhands_sdk/cpp_reviewfix/all_preds.jsonl \
  --artifacts-dir OmniCode/results/openhands_sdk/cpp_reviewfix \
  --run-log     OmniCode/results/openhands_sdk/cpp_reviewfix/run_log.jsonl
