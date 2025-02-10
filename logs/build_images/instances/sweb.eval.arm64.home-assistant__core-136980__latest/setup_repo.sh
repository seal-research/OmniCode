#!/bin/bash
set -euxo pipefail
git clone -o origin https://github.com/home-assistant/core /testbed
chmod -R 777 /testbed
cd /testbed
git reset --hard fc979cd564ee2d5fd27e05b32fa6f11b343ee4d5
git remote remove origin
source /opt/miniconda3/bin/activate
conda activate testbed
echo "Current environment: $CONDA_DEFAULT_ENV"
pip install -e '.[all, dev, test]'
