#!/bin/bash
set -euxo pipefail
git clone -o origin https://github.com/sherlock-project/sherlock /testbed
chmod -R 777 /testbed
cd /testbed
git reset --hard fba27cd709d684c0f5a4f644c8db71a3de6b10cb
git remote remove origin
source /opt/miniconda3/bin/activate
conda activate testbed
echo "Current environment: $CONDA_DEFAULT_ENV"
pip install -e .
