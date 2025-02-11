#!/bin/bash
set -euxo pipefail
git clone -o origin https://github.com/scrapy/scrapy /testbed
chmod -R 777 /testbed
cd /testbed
git reset --hard 8055a948dc2544c4d8ebe7aa1c6227e19b1583ac
git remote remove origin
source /opt/miniconda3/bin/activate
conda activate testbed
echo "Current environment: $CONDA_DEFAULT_ENV"
pip install -e '.'
