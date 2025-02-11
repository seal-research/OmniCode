#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git diff a5da77d01dccbc91206d053396fb5b80e1a6b15b
source /opt/miniconda3/bin/activate
conda activate testbed
pip install -e '.'
git checkout a5da77d01dccbc91206d053396fb5b80e1a6b15b tests/spiders.py tests/test_utils_log.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/spiders.py b/tests/spiders.py
index 94969db993d..ea419afbdac 100644
--- a/tests/spiders.py
+++ b/tests/spiders.py
@@ -4,6 +4,7 @@
 
 import asyncio
 import time
+from typing import Optional
 from urllib.parse import urlencode
 
 from twisted.internet import defer
@@ -78,6 +79,28 @@ def errback(self, failure):
         self.t2_err = time.time()
 
 
+class LogSpider(MetaSpider):
+    name = "log_spider"
+
+    def log_debug(self, message: str, extra: Optional[dict] = None):
+        self.logger.debug(message, extra=extra)
+
+    def log_info(self, message: str, extra: Optional[dict] = None):
+        self.logger.info(message, extra=extra)
+
+    def log_warning(self, message: str, extra: Optional[dict] = None):
+        self.logger.warning(message, extra=extra)
+
+    def log_error(self, message: str, extra: Optional[dict] = None):
+        self.logger.error(message, extra=extra)
+
+    def log_critical(self, message: str, extra: Optional[dict] = None):
+        self.logger.critical(message, extra=extra)
+
+    def parse(self, response):
+        pass
+
+
 class SlowSpider(DelaySpider):
     name = "slow"
 
diff --git a/tests/test_utils_log.py b/tests/test_utils_log.py
index eae744df5e4..a8d0808222e 100644
--- a/tests/test_utils_log.py
+++ b/tests/test_utils_log.py
@@ -1,18 +1,26 @@
+import json
 import logging
+import re
 import sys
 import unittest
+from io import StringIO
+from typing import Any, Dict, Mapping, MutableMapping
+from unittest import TestCase
 
+import pytest
 from testfixtures import LogCapture
 from twisted.python.failure import Failure
 
 from scrapy.extensions import telnet
 from scrapy.utils.log import (
     LogCounterHandler,
+    SpiderLoggerAdapter,
     StreamLogger,
     TopLevelFormatter,
     failure_to_exc_info,
 )
 from scrapy.utils.test import get_crawler
+from tests.spiders import LogSpider
 
 
 class FailureToExcInfoTest(unittest.TestCase):
@@ -106,3 +114,180 @@ def test_redirect(self):
         with LogCapture() as log:
             print("test log msg")
         log.check(("test", "ERROR", "test log msg"))
+
+
+@pytest.mark.parametrize(
+    ("base_extra", "log_extra", "expected_extra"),
+    (
+        (
+            {"spider": "test"},
+            {"extra": {"log_extra": "info"}},
+            {"extra": {"log_extra": "info", "spider": "test"}},
+        ),
+        (
+            {"spider": "test"},
+            {"extra": None},
+            {"extra": {"spider": "test"}},
+        ),
+        (
+            {"spider": "test"},
+            {"extra": {"spider": "test2"}},
+            {"extra": {"spider": "test"}},
+        ),
+    ),
+)
+def test_spider_logger_adapter_process(
+    base_extra: Mapping[str, Any], log_extra: MutableMapping, expected_extra: Dict
+):
+    logger = logging.getLogger("test")
+    spider_logger_adapter = SpiderLoggerAdapter(logger, base_extra)
+
+    log_message = "test_log_message"
+    result_message, result_kwargs = spider_logger_adapter.process(
+        log_message, log_extra
+    )
+
+    assert result_message == log_message
+    assert result_kwargs == expected_extra
+
+
+class LoggingTestCase(TestCase):
+    def setUp(self):
+        self.log_stream = StringIO()
+        handler = logging.StreamHandler(self.log_stream)
+        logger = logging.getLogger("log_spider")
+        logger.addHandler(handler)
+        logger.setLevel(logging.DEBUG)
+        self.handler = handler
+        self.logger = logger
+        self.spider = LogSpider()
+
+    def tearDown(self):
+        self.logger.removeHandler(self.handler)
+
+    def test_debug_logging(self):
+        log_message = "Foo message"
+        self.spider.log_debug(log_message)
+        log_contents = self.log_stream.getvalue()
+
+        assert log_contents == f"{log_message}\n"
+
+    def test_info_logging(self):
+        log_message = "Bar message"
+        self.spider.log_info(log_message)
+        log_contents = self.log_stream.getvalue()
+
+        assert log_contents == f"{log_message}\n"
+
+    def test_warning_logging(self):
+        log_message = "Baz message"
+        self.spider.log_warning(log_message)
+        log_contents = self.log_stream.getvalue()
+
+        assert log_contents == f"{log_message}\n"
+
+    def test_error_logging(self):
+        log_message = "Foo bar message"
+        self.spider.log_error(log_message)
+        log_contents = self.log_stream.getvalue()
+
+        assert log_contents == f"{log_message}\n"
+
+    def test_critical_logging(self):
+        log_message = "Foo bar baz message"
+        self.spider.log_critical(log_message)
+        log_contents = self.log_stream.getvalue()
+
+        assert log_contents == f"{log_message}\n"
+
+
+class LoggingWithExtraTestCase(TestCase):
+    def setUp(self):
+        self.log_stream = StringIO()
+        handler = logging.StreamHandler(self.log_stream)
+        formatter = logging.Formatter(
+            '{"levelname": "%(levelname)s", "message": "%(message)s", "spider": "%(spider)s", "important_info": "%(important_info)s"}'
+        )
+        handler.setFormatter(formatter)
+        logger = logging.getLogger("log_spider")
+        logger.addHandler(handler)
+        logger.setLevel(logging.DEBUG)
+        self.handler = handler
+        self.logger = logger
+        self.spider = LogSpider()
+        self.regex_pattern = re.compile(r"^<LogSpider\s'log_spider'\sat\s[^>]+>$")
+
+    def tearDown(self):
+        self.logger.removeHandler(self.handler)
+
+    def test_debug_logging(self):
+        log_message = "Foo message"
+        extra = {"important_info": "foo"}
+        self.spider.log_debug(log_message, extra)
+        log_contents = self.log_stream.getvalue()
+        log_contents = json.loads(log_contents)
+
+        assert log_contents["levelname"] == "DEBUG"
+        assert log_contents["message"] == log_message
+        assert self.regex_pattern.match(log_contents["spider"])
+        assert log_contents["important_info"] == extra["important_info"]
+
+    def test_info_logging(self):
+        log_message = "Bar message"
+        extra = {"important_info": "bar"}
+        self.spider.log_info(log_message, extra)
+        log_contents = self.log_stream.getvalue()
+        log_contents = json.loads(log_contents)
+
+        assert log_contents["levelname"] == "INFO"
+        assert log_contents["message"] == log_message
+        assert self.regex_pattern.match(log_contents["spider"])
+        assert log_contents["important_info"] == extra["important_info"]
+
+    def test_warning_logging(self):
+        log_message = "Baz message"
+        extra = {"important_info": "baz"}
+        self.spider.log_warning(log_message, extra)
+        log_contents = self.log_stream.getvalue()
+        log_contents = json.loads(log_contents)
+
+        assert log_contents["levelname"] == "WARNING"
+        assert log_contents["message"] == log_message
+        assert self.regex_pattern.match(log_contents["spider"])
+        assert log_contents["important_info"] == extra["important_info"]
+
+    def test_error_logging(self):
+        log_message = "Foo bar message"
+        extra = {"important_info": "foo bar"}
+        self.spider.log_error(log_message, extra)
+        log_contents = self.log_stream.getvalue()
+        log_contents = json.loads(log_contents)
+
+        assert log_contents["levelname"] == "ERROR"
+        assert log_contents["message"] == log_message
+        assert self.regex_pattern.match(log_contents["spider"])
+        assert log_contents["important_info"] == extra["important_info"]
+
+    def test_critical_logging(self):
+        log_message = "Foo bar baz message"
+        extra = {"important_info": "foo bar baz"}
+        self.spider.log_critical(log_message, extra)
+        log_contents = self.log_stream.getvalue()
+        log_contents = json.loads(log_contents)
+
+        assert log_contents["levelname"] == "CRITICAL"
+        assert log_contents["message"] == log_message
+        assert self.regex_pattern.match(log_contents["spider"])
+        assert log_contents["important_info"] == extra["important_info"]
+
+    def test_overwrite_spider_extra(self):
+        log_message = "Foo message"
+        extra = {"important_info": "foo", "spider": "shouldn't change"}
+        self.spider.log_error(log_message, extra)
+        log_contents = self.log_stream.getvalue()
+        log_contents = json.loads(log_contents)
+
+        assert log_contents["levelname"] == "ERROR"
+        assert log_contents["message"] == log_message
+        assert self.regex_pattern.match(log_contents["spider"])
+        assert log_contents["important_info"] == extra["important_info"]

EOF_114329324912
pytest -rA --tb=long tests/spiders.py tests/test_utils_log.py
git checkout a5da77d01dccbc91206d053396fb5b80e1a6b15b tests/spiders.py tests/test_utils_log.py
