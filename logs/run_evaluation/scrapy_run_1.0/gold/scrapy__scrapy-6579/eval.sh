#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git diff c330a399dcc69f6d51fcfbe397fbc42b5a9ee323
source /opt/miniconda3/bin/activate
conda activate testbed
pip install -e '.'
git checkout c330a399dcc69f6d51fcfbe397fbc42b5a9ee323 tests/test_item.py tests/test_utils_url.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/test_item.py b/tests/test_item.py
index 5a8ee095e61..4804128417a 100644
--- a/tests/test_item.py
+++ b/tests/test_item.py
@@ -1,7 +1,8 @@
 import unittest
+from abc import ABCMeta
 from unittest import mock
 
-from scrapy.item import ABCMeta, Field, Item, ItemMeta
+from scrapy.item import Field, Item, ItemMeta
 
 
 class ItemTest(unittest.TestCase):
diff --git a/tests/test_utils_url.py b/tests/test_utils_url.py
index 94a59f8835e..314082742cf 100644
--- a/tests/test_utils_url.py
+++ b/tests/test_utils_url.py
@@ -6,7 +6,7 @@
 from scrapy.linkextractors import IGNORED_EXTENSIONS
 from scrapy.spiders import Spider
 from scrapy.utils.misc import arg_to_iter
-from scrapy.utils.url import (
+from scrapy.utils.url import (  # type: ignore[attr-defined]
     _is_filesystem_path,
     _public_w3lib_objects,
     add_http_if_no_scheme,

EOF_114329324912
pytest -rA --tb=long tests/test_item.py tests/test_utils_url.py
git checkout c330a399dcc69f6d51fcfbe397fbc42b5a9ee323 tests/test_item.py tests/test_utils_url.py
