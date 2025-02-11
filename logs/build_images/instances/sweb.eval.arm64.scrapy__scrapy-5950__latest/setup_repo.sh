#!/bin/bash
set -euxo pipefail
git clone -o origin https://github.com/scrapy/scrapy /testbed
chmod -R 777 /testbed
cd /testbed
git reset --hard 510574216d70ec84d75639ebcda360834a992e47
git remote remove origin
source /opt/miniconda3/bin/activate
conda activate testbed
echo "Current environment: $CONDA_DEFAULT_ENV"
pip install -e '.'
