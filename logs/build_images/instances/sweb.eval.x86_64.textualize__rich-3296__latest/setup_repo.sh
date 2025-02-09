#!/bin/bash
set -euxo pipefail

git clone -o origin https://github.com/textualize/rich /testbed
chmod -R 777 /testbed
cd /testbed
git reset --hard fde5d6eee3b0437eaecdcbf6f8b11aeab3a5d503
git remote remove origin
source /opt/miniconda3/bin/activate
conda activate testbed

# Debug step: check environment
echo "After conda activate, the environment is: $CONDA_DEFAULT_ENV"

# Adding ls here
ls -la /testbed

echo "Current environment: $CONDA_DEFAULT_ENV"
poetry lock
poetry install --with dev
