#!/bin/bash
set -euxo pipefail
git clone -o origin https://github.com/scrapy/scrapy /testbed
chmod -R 777 /testbed
cd /testbed
git reset --hard a5da77d01dccbc91206d053396fb5b80e1a6b15b
git remote remove origin
source /opt/miniconda3/bin/activate
conda activate testbed
echo "Current environment: $CONDA_DEFAULT_ENV"
pip install -e '.'
