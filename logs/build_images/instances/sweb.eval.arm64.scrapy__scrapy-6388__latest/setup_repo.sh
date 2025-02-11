#!/bin/bash
set -euxo pipefail
git clone -o origin https://github.com/scrapy/scrapy /testbed
chmod -R 777 /testbed
cd /testbed
git reset --hard 2b9e32f1ca491340148e6a1918d1df70443823e6
git remote remove origin
source /opt/miniconda3/bin/activate
conda activate testbed
echo "Current environment: $CONDA_DEFAULT_ENV"
pip install -e '.'
