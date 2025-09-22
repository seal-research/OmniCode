#!/usr/bin/env -S bash --noprofile --norc

set -euo pipefail

export PATH=/share/apps/singularity/3.7.0/bin:$PATH
unset LD_PRELOAD
unset LD_LIBRARY_PATH

# Locate sbatch even in a minimal environment
if command -v sbatch >/dev/null 2>&1; then
    SBATCH_BIN="$(command -v sbatch)"
else
    for d in /usr/bin /usr/local/bin /opt/slurm/bin /usr/lib/slurm /cm/shared/apps/slurm/current/bin /cm/local/apps/slurm/*/bin; do
        if [ -x "$d/sbatch" ]; then
            SBATCH_BIN="$d/sbatch"
            break
        fi
    done
fi

if [ -z "${SBATCH_BIN:-}" ]; then
    echo "ERROR: sbatch not found. Add its directory to PATH or set SBATCH_BIN."
    exit 1
fi

INSTANCE_FILE="data/instance_IDs.txt"
RUN_ID="llama_scout_acr_eval"
LOG_DIR="evaluation_setup_scout/logs"

CPUS=8 
MEM=16G         
TIME_LIMIT="02:00:00"

mkdir -p "${LOG_DIR}"

# Set environment variables for SWE-bench
# Try different scratch locations, fallback to home directory
if [ -d "/scratch" ] && [ -w "/scratch" ]; then
    SCRATCH_BASE="/scratch"
elif [ -d "/tmp" ] && [ -w "/tmp" ]; then
    SCRATCH_BASE="/tmp"
else
    SCRATCH_BASE="$HOME"
fi

export SWEBENCH_BUILD_DIR="${SCRATCH_BASE}/logs/build_images"
export SWEBENCH_CACHE_DIR="${SCRATCH_BASE}/logs/cache"

# Create directories and symlink
mkdir -p "${SCRATCH_BASE}/logs/build_images/def"
mkdir -p "${SCRATCH_BASE}/logs/build_images/base"
mkdir -p "${SCRATCH_BASE}/logs/build_images/env"
mkdir -p "${SCRATCH_BASE}/logs/build_images/instances"
mkdir -p "${SCRATCH_BASE}/logs/run_evaluation"
mkdir -p "${SCRATCH_BASE}/logs/run_validation"

# Create symlink to fix hardcoded /scratch/logs path
ln -sf "${SCRATCH_BASE}/logs" /scratch/logs 2>/dev/null || true

# Change to the codearena directory
cd /home/cbb89/codearena/codearena/OmniCode

# BUGFIXING evaluation
echo "Starting bugfixing evaluation..."

while IFS= read -r ID || [[ -n "${ID}" ]]; do
    SAN_ID="${ID//\//__}"      # 1)  /  →  __
    SAN_ID="${SAN_ID//:/_}"    # 2)  :  →  _
    JOB_NAME="${RUN_ID}_bugfixing_${SAN_ID}"

    echo "Submitting job for instance_id=${ID}  (job-name=${JOB_NAME})"

    "${SBATCH_BIN}" --job-name="${JOB_NAME}" \
           --cpus-per-task="${CPUS}" \
           --gres=gpu:1 \
           --mem="${MEM}" \
           --time="${TIME_LIMIT}" \
           --constraint=gpu \
           --export=NONE \
           --output="${LOG_DIR}/%x_%j.out" \
           --error="${LOG_DIR}/%x_%j.err" \
           --wrap="(cd /home/cbb89/codearena/codearena/OmniCode && export PATH=/share/apps/singularity/3.7.0/bin:$PATH; unset LD_PRELOAD; unset LD_LIBRARY_PATH; python codearena.py --BugFixing \
                    --predictions_path gold \
                    --run_id ${JOB_NAME} \
                    --max_workers 1 \
                    --mswe_phase all \
                    --force_rebuild False \
                    --clean True \
                    --use_apptainer True \
                    --instance_ids ${ID} \
                    --g2 True;)"
done < "evaluation_setup_scout/bugfixing_instance_ids.txt"

echo "Completed bugfixing evaluation submission"

# CODEREVIEW evaluation
echo "Starting codereview evaluation..."

while IFS= read -r ID || [[ -n "${ID}" ]]; do
    SAN_ID="${ID//\//__}"      # 1)  /  →  __
    SAN_ID="${SAN_ID//:/_}"    # 2)  :  →  _
    JOB_NAME="${RUN_ID}_codereview_${SAN_ID}"

    echo "Submitting job for instance_id=${ID}  (job-name=${JOB_NAME})"

    "${SBATCH_BIN}" --job-name="${JOB_NAME}" \
           --cpus-per-task="${CPUS}" \
           --gres=gpu:1 \
           --mem="${MEM}" \
           --time="${TIME_LIMIT}" \
           --constraint=gpu \
           --export=NONE \
           --output="${LOG_DIR}/%x_%j.out" \
           --error="${LOG_DIR}/%x_%j.err" \
           --wrap="(cd /home/cbb89/codearena/codearena/OmniCode && export PATH=/share/apps/singularity/3.7.0/bin:$PATH; unset LD_PRELOAD; unset LD_LIBRARY_PATH; python codearena.py --CodeReview \
                    --predictions_path gold \
                    --run_id ${JOB_NAME} \
                    --max_workers 1 \
                    --mswe_phase all \
                    --force_rebuild False \
                    --clean True \
                    --use_apptainer True \
                    --instance_ids ${ID} \
                    --g2 True;)"
done < "evaluation_setup_scout/codereview_instance_ids.txt"

echo "Completed codereview evaluation submission"

# STYLEREVIEW evaluation
echo "Starting stylereview evaluation..."

while IFS= read -r ID || [[ -n "${ID}" ]]; do
    SAN_ID="${ID//\//__}"      # 1)  /  →  __
    SAN_ID="${SAN_ID//:/_}"    # 2)  :  →  _
    JOB_NAME="${RUN_ID}_stylereview_${SAN_ID}"

    echo "Submitting job for instance_id=${ID}  (job-name=${JOB_NAME})"

    "${SBATCH_BIN}" --job-name="${JOB_NAME}" \
           --cpus-per-task="${CPUS}" \
           --gres=gpu:1 \
           --mem="${MEM}" \
           --time="${TIME_LIMIT}" \
           --constraint=gpu \
           --export=NONE \
           --output="${LOG_DIR}/%x_%j.out" \
           --error="${LOG_DIR}/%x_%j.err" \
           --wrap="(cd /home/cbb89/codearena/codearena/OmniCode && export PATH=/share/apps/singularity/3.7.0/bin:$PATH; unset LD_PRELOAD; unset LD_LIBRARY_PATH; python codearena.py --StyleReview \
                    --predictions_path gold \
                    --run_id ${JOB_NAME} \
                    --max_workers 1 \
                    --mswe_phase all \
                    --force_rebuild False \
                    --clean True \
                    --use_apptainer True \
                    --instance_ids ${ID} \
                    --g2 True;)"
done < "evaluation_setup_scout/stylereview_instance_ids.txt"

echo "Completed stylereview evaluation submission"

# TESTGEN evaluation
echo "Starting testgen evaluation..."

while IFS= read -r ID || [[ -n "${ID}" ]]; do
    SAN_ID="${ID//\//__}"      # 1)  /  →  __
    SAN_ID="${SAN_ID//:/_}"    # 2)  :  →  _
    JOB_NAME="${RUN_ID}_testgen_${SAN_ID}"

    echo "Submitting job for instance_id=${ID}  (job-name=${JOB_NAME})"

    "${SBATCH_BIN}" --job-name="${JOB_NAME}" \
           --cpus-per-task="${CPUS}" \
           --gres=gpu:1 \
           --mem="${MEM}" \
           --time="${TIME_LIMIT}" \
           --constraint=gpu \
           --export=NONE \
           --output="${LOG_DIR}/%x_%j.out" \
           --error="${LOG_DIR}/%x_%j.err" \
           --wrap="(cd /home/cbb89/codearena/codearena/OmniCode && export PATH=/share/apps/singularity/3.7.0/bin:$PATH; unset LD_PRELOAD; unset LD_LIBRARY_PATH; python codearena.py --TestGeneration \
                    --predictions_path gold \
                    --run_id ${JOB_NAME} \
                    --max_workers 1 \
                    --mswe_phase all \
                    --force_rebuild False \
                    --clean True \
                    --use_apptainer True \
                    --instance_ids ${ID} \
                    --g2 True;)"
done < "evaluation_setup_scout/testgen_instance_ids.txt"

echo "Completed testgen evaluation submission"
