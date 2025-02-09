#!/bin/bash
set -euxo pipefail
git clone -o origin https://github.com/scrapy/scrapy /testbed
chmod -R 777 /testbed
cd /testbed
git reset --hard c330a399dcc69f6d51fcfbe397fbc42b5a9ee323
git remote remove origin
source /opt/miniconda3/bin/activate
conda activate testbed
echo "Current environment: $CONDA_DEFAULT_ENV"
pip install -e '.'
