#!/bin/bash
set -euxo pipefail
git clone -o origin https://github.com/apache/airflow /testbed
chmod -R 777 /testbed
cd /testbed
git reset --hard 4833b53705acfc4bd0a26bf3e4dd4fc7a22b0bfa
git remote remove origin
source /opt/miniconda3/bin/activate
conda activate testbed
echo "Current environment: $CONDA_DEFAULT_ENV"
pip install -e '.[all,dev,test]' --no-deps && pip install -e '.[devel]' --upgrade --upgrade-strategy eager && pip install --no-deps -r <(pip freeze)
