git clone https://github.com/camel-ai/camel.git
cp build_mapping.py camel/
cp map.json camel/
cd camel

# 1. Create env
conda create -n camel python=3.13 pytest pytest-cov coverage
conda activate camel

# 2. Install test deps
pip install -e '.[all, dev, test]'
pip install gradio

# 3. erase past coverage runs
coverage erase

# # 4. Run tests with per-test coverage
i=1
total=$(pytest --collect-only -q | cut -d ':' -f1 | sort -u | grep '\.py$' | wc -l)
for testfile in $(pytest --collect-only -q | cut -d ':' -f1 | sort -u | grep '\.py$'); do
    echo "[$i/$total] Running $testfile"
    coverage run --source=camel --context=test -m pytest -q "$testfile"
    coverage json -o coverage.json
    python build_mapping.py "$testfile"
    i=$((i+1))
done
