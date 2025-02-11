#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git diff 2b9e32f1ca491340148e6a1918d1df70443823e6
source /opt/miniconda3/bin/activate
conda activate testbed
pip install -e '.'
git checkout 2b9e32f1ca491340148e6a1918d1df70443823e6 tests/test_contracts.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/test_contracts.py b/tests/test_contracts.py
index 1459e0b5fd5..c9c12f0d804 100644
--- a/tests/test_contracts.py
+++ b/tests/test_contracts.py
@@ -182,6 +182,19 @@ def custom_form(self, response):
         """
         pass
 
+    def invalid_regex(self, response):
+        """method with invalid regex
+        @ Scrapy is awsome
+        """
+        pass
+
+    def invalid_regex_with_valid_contract(self, response):
+        """method with invalid regex
+        @ scrapy is awsome
+        @url http://scrapy.org
+        """
+        pass
+
 
 class CustomContractSuccessSpider(Spider):
     name = "custom_contract_success_spider"
@@ -385,6 +398,21 @@ def test_scrapes(self):
         message = "ContractFail: Missing fields: name, url"
         assert message in self.results.failures[-1][-1]
 
+    def test_regex(self):
+        spider = TestSpider()
+        response = ResponseMock()
+
+        # invalid regex
+        request = self.conman.from_method(spider.invalid_regex, self.results)
+        self.should_succeed()
+
+        # invalid regex with valid contract
+        request = self.conman.from_method(
+            spider.invalid_regex_with_valid_contract, self.results
+        )
+        self.should_succeed()
+        request.callback(response)
+
     def test_custom_contracts(self):
         self.conman.from_spider(CustomContractSuccessSpider(), self.results)
         self.should_succeed()

EOF_114329324912
pytest -rA --tb=long tests/test_contracts.py
git checkout 2b9e32f1ca491340148e6a1918d1df70443823e6 tests/test_contracts.py
