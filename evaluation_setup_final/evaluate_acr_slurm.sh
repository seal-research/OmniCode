#!/usr/bin/env bash

set -euo pipefail

INSTANCE_FILE="data/instance_IDs.txt"
RUN_ID="claude_sonnet_acr_eval"
LOG_DIR="evaluation_setup_final/logs"

CPUS=8 
MEM=16G         
TIME_LIMIT="02:00:00"

mkdir -p "${LOG_DIR}"

# Change to the codearena directory
cd /home/cbb89/codearena/codearena/OmniCode

# BUGFIXING evaluation
echo "Starting bugfixing evaluation..."

while IFS= read -r ID || [[ -n "${ID}" ]]; do
    SAN_ID="${ID//\//__}"      # 1)  /  →  __
    SAN_ID="${SAN_ID//:/_}"    # 2)  :  →  _
    JOB_NAME="${RUN_ID}_bugfixing_${SAN_ID}"

    echo "Submitting job for instance_id=${ID}  (job-name=${JOB_NAME})"

    sbatch --job-name="${JOB_NAME}" \
           --exclusive \
           --cpus-per-task="${CPUS}" \
           --gres=gpu:1 \
           --mem="${MEM}" \
           --time="${TIME_LIMIT}" \
           --output="${LOG_DIR}/%x_%j.out" \
           --error="${LOG_DIR}/%x_%j.err" \
           --wrap="(cd /home/cbb89/codearena/codearena/OmniCode && export SWEBENCH_BUILD_DIR=/scratch/cbb89/logs/build_images && export SWEBENCH_CACHE_DIR=/scratch/cbb89/logs/cache && python codearena.py --BugFixing \
                    --predictions_path gold \
                    --run_id ${JOB_NAME} \
                    --max_workers 1 \
                    --mswe_phase all \
                    --force_rebuild False \
                    --clean True \
                    --use_apptainer True \
                    --instance_ids ${ID} \
                    --g2 True;)"
done < "evaluation_setup_final/bugfixing_instance_ids.txt"

echo "Completed bugfixing evaluation submission"

# CODEREVIEW evaluation
echo "Starting codereview evaluation..."

while IFS= read -r ID || [[ -n "${ID}" ]]; do
    SAN_ID="${ID//\//__}"      # 1)  /  →  __
    SAN_ID="${SAN_ID//:/_}"    # 2)  :  →  _
    JOB_NAME="${RUN_ID}_codereview_${SAN_ID}"

    echo "Submitting job for instance_id=${ID}  (job-name=${JOB_NAME})"

    sbatch --job-name="${JOB_NAME}" \
           --exclusive \
           --cpus-per-task="${CPUS}" \
           --gres=gpu:1 \
           --mem="${MEM}" \
           --time="${TIME_LIMIT}" \
           --output="${LOG_DIR}/%x_%j.out" \
           --error="${LOG_DIR}/%x_%j.err" \
           --wrap="(cd /home/cbb89/codearena/codearena/OmniCode && export SWEBENCH_BUILD_DIR=/scratch/cbb89/logs/build_images && export SWEBENCH_CACHE_DIR=/scratch/cbb89/logs/cache && python codearena.py --CodeReview \
                    --predictions_path gold \
                    --run_id ${JOB_NAME} \
                    --max_workers 1 \
                    --mswe_phase all \
                    --force_rebuild False \
                    --clean True \
                    --use_apptainer True \
                    --instance_ids ${ID} \
                    --g2 True;)"
done < "evaluation_setup_final/codereview_instance_ids.txt"

echo "Completed codereview evaluation submission"

# STYLEREVIEW evaluation
echo "Starting stylereview evaluation..."

while IFS= read -r ID || [[ -n "${ID}" ]]; do
    SAN_ID="${ID//\//__}"      # 1)  /  →  __
    SAN_ID="${SAN_ID//:/_}"    # 2)  :  →  _
    JOB_NAME="${RUN_ID}_stylereview_${SAN_ID}"

    echo "Submitting job for instance_id=${ID}  (job-name=${JOB_NAME})"

    sbatch --job-name="${JOB_NAME}" \
           --exclusive \
           --cpus-per-task="${CPUS}" \
           --gres=gpu:1 \
           --mem="${MEM}" \
           --time="${TIME_LIMIT}" \
           --output="${LOG_DIR}/%x_%j.out" \
           --error="${LOG_DIR}/%x_%j.err" \
           --wrap="(cd /home/cbb89/codearena/codearena/OmniCode && export SWEBENCH_BUILD_DIR=/scratch/cbb89/logs/build_images && export SWEBENCH_CACHE_DIR=/scratch/cbb89/logs/cache && python codearena.py --StyleReview \
                    --predictions_path gold \
                    --run_id ${JOB_NAME} \
                    --max_workers 1 \
                    --mswe_phase all \
                    --force_rebuild False \
                    --clean True \
                    --use_apptainer True \
                    --instance_ids ${ID} \
                    --g2 True;)"
done < "evaluation_setup_final/stylereview_instance_ids.txt"

echo "Completed stylereview evaluation submission"

# TESTGEN evaluation
echo "Starting testgen evaluation..."

while IFS= read -r ID || [[ -n "${ID}" ]]; do
    SAN_ID="${ID//\//__}"      # 1)  /  →  __
    SAN_ID="${SAN_ID//:/_}"    # 2)  :  →  _
    JOB_NAME="${RUN_ID}_testgen_${SAN_ID}"

    echo "Submitting job for instance_id=${ID}  (job-name=${JOB_NAME})"

    sbatch --job-name="${JOB_NAME}" \
           --exclusive \
           --cpus-per-task="${CPUS}" \
           --gres=gpu:1 \
           --mem="${MEM}" \
           --time="${TIME_LIMIT}" \
           --output="${LOG_DIR}/%x_%j.out" \
           --error="${LOG_DIR}/%x_%j.err" \
           --wrap="(cd /home/cbb89/codearena/codearena/OmniCode && export SWEBENCH_BUILD_DIR=/scratch/cbb89/logs/build_images && export SWEBENCH_CACHE_DIR=/scratch/cbb89/logs/cache && python codearena.py --TestGeneration \
                    --predictions_path gold \
                    --run_id ${JOB_NAME} \
                    --max_workers 1 \
                    --mswe_phase all \
                    --force_rebuild False \
                    --clean True \
                    --use_apptainer True \
                    --instance_ids ${ID} \
                    --g2 True;)"
done < "evaluation_setup_final/testgen_instance_ids.txt"

echo "Completed testgen evaluation submission"
