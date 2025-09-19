git clone https://github.com/ytdl-org/youtube-dl.git
cp build_mapping.py youtube-dl/
cp map.json youtube-dl/
cd youtube-dl

# 1. Create env
conda create -n youtube-dl python=3.7 pytest pytest-cov coverage
conda activate youtube-dl

# 2. Install test deps
pip install -e '.[all, dev, test]'

# 3. erase past coverage runs
coverage erase

# # 4. Run tests with per-test coverage
i=1
total=$(pytest --collect-only -q | cut -d ':' -f1 | sort -u | grep '\.py$' | wc -l)
for testfile in $(pytest --collect-only -q | cut -d ':' -f1 | sort -u | grep '\.py$'); do
    echo "[$i/$total] Running $testfile"
    coverage run --source=youtube_dl --context=test -m pytest -q "$testfile"
    coverage json -o coverage.json
    python build_mapping.py "$testfile"
    i=$((i+1))
done