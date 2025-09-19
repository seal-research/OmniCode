git clone https://github.com/astropy/astropy.git
cp build_mapping.py astropy/
cp map.json astropy/
cd astropy

# 1. Create env
conda create -n astropy python=3.13 pytest pytest-cov coverage
conda activate astropy

# 2. Install test deps
pip install -e .[test]

# 3. erase past coverage runs
coverage erase

# # 4. Run tests with per-test coverage
i=1
total=$(pytest --collect-only -q | cut -d ':' -f1 | sort -u | grep '\.py$' | wc -l)
for testfile in $(pytest --collect-only -q | cut -d ':' -f1 | sort -u | grep '\.py$'); do
    echo "[$i/$total] Running $testfile"
    coverage run --source=astropy --context=test -m pytest -q "$testfile"
    coverage json -o coverage.json
    python build_mapping.py "$testfile"
    i=$((i+1))
done
