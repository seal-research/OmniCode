#!/bin/bash
#SBATCH --job-name=unicorn_job
#SBATCH --output=unicorn_%j.out
#SBATCH --error=unicorn_%j.err
#SBATCH --time=13:00:00             # walltime (adjust)
#SBATCH --nodes=4                   # number of nodes
#SBATCH --ntasks=1                  # number of tasks
#SBATCH --cpus-per-task=4           # CPU cores per task
#SBATCH --mem=16G                   # memory per node
#SBATCH --partition=dutta         # adjust to your cluster's partition/queue

# --- Setup environment ---
export PATH=/share/apps/software/anaconda3/bin:$PATH
export PATH=/share/apps/software/apptainer/apptainer-1.4.0/bin:$PATH
export PATH="$PATH:/share/dutta/software/LLVM-21.1.1-Linux-X64/bin"
# Load conda
source /share/apps/software/anaconda3/etc/profile.d/conda.sh

# Activate environment
conda activate /home/dd732/myenv
export OPENROUTER_API_KEY="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"
# Navigate to working directory
cd /home/dd732/OmniCode || exit 1

# --- Run your Python script --
