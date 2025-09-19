git clone https://github.com/scrapy/scrapy.git
cp build_mapping.py scrapy/
cp map.json scrapy/
cd scrapy

# 1. Create env
conda create -n scrapy python=3.12 pytest pytest-cov coverage
conda activate scrapy

# 2. Install test deps
pip install -e .

# 3. erase past coverage runs
coverage erase

# # 4. Run tests with per-test coverage
i=1
total=$(pytest --collect-only -q | cut -d ':' -f1 | sort -u | grep '\.py$' | wc -l)
for testfile in $(pytest --collect-only -q | cut -d ':' -f1 | sort -u | grep '\.py$'); do
    echo "[$i/$total] Running $testfile"
    tox -e py -- --cov=scrapy --cov-branch "$testfile"
    coverage json -o coverage.json
    python build_mapping.py "$testfile"
    i=$((i+1))
done
