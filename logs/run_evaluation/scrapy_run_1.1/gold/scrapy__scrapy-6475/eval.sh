#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git diff 67ab8d4650c1e9212c9508803c7b5265e166cbaa
source /opt/miniconda3/bin/activate
conda activate testbed
pip install -e '.'
git checkout 67ab8d4650c1e9212c9508803c7b5265e166cbaa tests/test_engine.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/test_engine.py b/tests/test_engine.py
index 86526420f83..2ebc0b5e449 100644
--- a/tests/test_engine.py
+++ b/tests/test_engine.py
@@ -499,7 +499,6 @@ def signal_handler(request: Request, spider: Spider) -> None:
     assert scheduler.enqueued == [
         keep_request
     ], f"{scheduler.enqueued!r} != [{keep_request!r}]"
-    assert "dropped request <GET https://drop.example>" in caplog.text
     crawler.signals.disconnect(signal_handler, request_scheduled)
 
 

EOF_114329324912
pytest -rA --tb=long tests/test_engine.py
git checkout 67ab8d4650c1e9212c9508803c7b5265e166cbaa tests/test_engine.py
