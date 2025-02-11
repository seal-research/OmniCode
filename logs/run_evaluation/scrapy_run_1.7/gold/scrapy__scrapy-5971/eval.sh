#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git diff 8055a948dc2544c4d8ebe7aa1c6227e19b1583ac
source /opt/miniconda3/bin/activate
conda activate testbed
pip install -e '.'
git checkout 8055a948dc2544c4d8ebe7aa1c6227e19b1583ac tests/test_commands.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/test_commands.py b/tests/test_commands.py
index 014f50e92e5..3df9b35af00 100644
--- a/tests/test_commands.py
+++ b/tests/test_commands.py
@@ -919,6 +919,64 @@ def start_requests(self):
         log = self.get_log(spider_code, args=args)
         self.assertIn("[myspider] DEBUG: FEEDS: {'stdout:': {'format': 'json'}}", log)
 
+    @skipIf(platform.system() == "Windows", reason="Linux only")
+    def test_absolute_path_linux(self):
+        spider_code = """
+import scrapy
+
+class MySpider(scrapy.Spider):
+    name = 'myspider'
+
+    start_urls = ["data:,"]
+
+    def parse(self, response):
+        yield {"hello": "world"}
+        """
+        temp_dir = mkdtemp()
+
+        args = ["-o", f"{temp_dir}/output1.json:json"]
+        log = self.get_log(spider_code, args=args)
+        self.assertIn(
+            f"[scrapy.extensions.feedexport] INFO: Stored json feed (1 items) in: {temp_dir}/output1.json",
+            log,
+        )
+
+        args = ["-o", f"{temp_dir}/output2.json"]
+        log = self.get_log(spider_code, args=args)
+        self.assertIn(
+            f"[scrapy.extensions.feedexport] INFO: Stored json feed (1 items) in: {temp_dir}/output2.json",
+            log,
+        )
+
+    @skipIf(platform.system() != "Windows", reason="Windows only")
+    def test_absolute_path_windows(self):
+        spider_code = """
+import scrapy
+
+class MySpider(scrapy.Spider):
+    name = 'myspider'
+
+    start_urls = ["data:,"]
+
+    def parse(self, response):
+        yield {"hello": "world"}
+        """
+        temp_dir = mkdtemp()
+
+        args = ["-o", f"{temp_dir}\\output1.json:json"]
+        log = self.get_log(spider_code, args=args)
+        self.assertIn(
+            f"[scrapy.extensions.feedexport] INFO: Stored json feed (1 items) in: {temp_dir}\\output1.json",
+            log,
+        )
+
+        args = ["-o", f"{temp_dir}\\output2.json"]
+        log = self.get_log(spider_code, args=args)
+        self.assertIn(
+            f"[scrapy.extensions.feedexport] INFO: Stored json feed (1 items) in: {temp_dir}\\output2.json",
+            log,
+        )
+
 
 @skipIf(platform.system() != "Windows", "Windows required for .pyw files")
 class WindowsRunSpiderCommandTest(RunSpiderCommandTest):

EOF_114329324912
pytest -rA --tb=long tests/test_commands.py
git checkout 8055a948dc2544c4d8ebe7aa1c6227e19b1583ac tests/test_commands.py
