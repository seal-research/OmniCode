#!/bin/bash
set -euxo pipefail
git clone -o origin https://github.com/statsmodels/statsmodels /testbed
chmod -R 777 /testbed
cd /testbed
git reset --hard 8600926f2f22e58779a667d82047a90318b20431
git remote remove origin
source /opt/miniconda3/bin/activate
conda activate testbed
echo "Current environment: $CONDA_DEFAULT_ENV"
pip install -e '.[all, dev, test]'
