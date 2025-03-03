#!/bin/bash
set -euxo pipefail
git clone -o origin https://github.com/apache/airflow /testbed
chmod -R 777 /testbed
cd /testbed
git reset --hard a10ae15440b812e146d57de1a5d5a02b3ec9c4c7
git remote remove origin
source /opt/miniconda3/bin/activate
conda activate testbed
echo "Current environment: $CONDA_DEFAULT_ENV"
pip install -e '.[all,dev,test]' --no-deps && pip install -e '.[devel]' --upgrade --upgrade-strategy eager && pip install --no-deps -r <(pip freeze)
