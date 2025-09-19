#!/bin/bash
#SBATCH --job-name=scrapy-map
#SBATCH --output=/share/dutta/ejt82/test_mapper/logs/%x_%j.out    # logs/scrapy-map_JOBID.out
#SBATCH --error=/share/dutta/ejt82/test_mapper/logs/%x_%j.err     # logs/scrapy-map_JOBID.err
#SBATCH --time=06:00:00            # adjust runtime
#SBATCH --partition=dutta            # change if you need gpu or special partition
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# ---- Ensure logs directory exists ----
mkdir -p logs

# ---- Start fresh coverage ----
coverage erase

# ---- Count tests ----
i=1
total=$(pytest --collect-only -q | cut -d ':' -f1 | sort -u | grep '\.py$' | wc -l)

# ---- Loop over test files ----
for testfile in $(pytest --collect-only -q | cut -d ':' -f1 | sort -u | grep '\.py$'); do
    echo "[$i/$total] Running $testfile"

    # Run tests inside tox (py environment must exist in tox.ini)
    tox -e py -- --cov=scrapy --cov-branch "$testfile"

    # Export coverage JSON
    coverage json -o coverage.json

    # Update mapping
    python build_mapping.py "$testfile"

    i=$((i+1))
done
