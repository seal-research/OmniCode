#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git diff 510574216d70ec84d75639ebcda360834a992e47
source /opt/miniconda3/bin/activate
conda activate testbed
pip install -e '.'
git checkout 510574216d70ec84d75639ebcda360834a992e47 tests/test_middleware.py tests/test_utils_deprecate.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/test_addons.py b/tests/test_addons.py
new file mode 100644
index 00000000000..5d053ed52d9
--- /dev/null
+++ b/tests/test_addons.py
@@ -0,0 +1,158 @@
+import itertools
+import unittest
+from typing import Any, Dict
+
+from scrapy import Spider
+from scrapy.crawler import Crawler, CrawlerRunner
+from scrapy.exceptions import NotConfigured
+from scrapy.settings import BaseSettings, Settings
+from scrapy.utils.test import get_crawler
+
+
+class SimpleAddon:
+    def update_settings(self, settings):
+        pass
+
+
+def get_addon_cls(config: Dict[str, Any]) -> type:
+    class AddonWithConfig:
+        def update_settings(self, settings: BaseSettings):
+            settings.update(config, priority="addon")
+
+    return AddonWithConfig
+
+
+class CreateInstanceAddon:
+    def __init__(self, crawler: Crawler) -> None:
+        super().__init__()
+        self.crawler = crawler
+        self.config = crawler.settings.getdict("MYADDON")
+
+    @classmethod
+    def from_crawler(cls, crawler: Crawler):
+        return cls(crawler)
+
+    def update_settings(self, settings):
+        settings.update(self.config, "addon")
+
+
+class AddonTest(unittest.TestCase):
+    def test_update_settings(self):
+        settings = BaseSettings()
+        settings.set("KEY1", "default", priority="default")
+        settings.set("KEY2", "project", priority="project")
+        addon_config = {"KEY1": "addon", "KEY2": "addon", "KEY3": "addon"}
+        testaddon = get_addon_cls(addon_config)()
+        testaddon.update_settings(settings)
+        self.assertEqual(settings["KEY1"], "addon")
+        self.assertEqual(settings["KEY2"], "project")
+        self.assertEqual(settings["KEY3"], "addon")
+
+
+class AddonManagerTest(unittest.TestCase):
+    def test_load_settings(self):
+        settings_dict = {
+            "ADDONS": {"tests.test_addons.SimpleAddon": 0},
+        }
+        crawler = get_crawler(settings_dict=settings_dict)
+        manager = crawler.addons
+        self.assertIsInstance(manager.addons[0], SimpleAddon)
+
+    def test_notconfigured(self):
+        class NotConfiguredAddon:
+            def update_settings(self, settings):
+                raise NotConfigured()
+
+        settings_dict = {
+            "ADDONS": {NotConfiguredAddon: 0},
+        }
+        crawler = get_crawler(settings_dict=settings_dict)
+        manager = crawler.addons
+        self.assertFalse(manager.addons)
+
+    def test_load_settings_order(self):
+        # Get three addons with different settings
+        addonlist = []
+        for i in range(3):
+            addon = get_addon_cls({"KEY1": i})
+            addon.number = i
+            addonlist.append(addon)
+        # Test for every possible ordering
+        for ordered_addons in itertools.permutations(addonlist):
+            expected_order = [a.number for a in ordered_addons]
+            settings = {"ADDONS": {a: i for i, a in enumerate(ordered_addons)}}
+            crawler = get_crawler(settings_dict=settings)
+            manager = crawler.addons
+            self.assertEqual([a.number for a in manager.addons], expected_order)
+            self.assertEqual(crawler.settings.getint("KEY1"), expected_order[-1])
+
+    def test_create_instance(self):
+        settings_dict = {
+            "ADDONS": {"tests.test_addons.CreateInstanceAddon": 0},
+            "MYADDON": {"MYADDON_KEY": "val"},
+        }
+        crawler = get_crawler(settings_dict=settings_dict)
+        manager = crawler.addons
+        self.assertIsInstance(manager.addons[0], CreateInstanceAddon)
+        self.assertEqual(crawler.settings.get("MYADDON_KEY"), "val")
+
+    def test_settings_priority(self):
+        config = {
+            "KEY": 15,  # priority=addon
+        }
+        settings_dict = {
+            "ADDONS": {get_addon_cls(config): 1},
+        }
+        crawler = get_crawler(settings_dict=settings_dict)
+        self.assertEqual(crawler.settings.getint("KEY"), 15)
+
+        settings = Settings(settings_dict)
+        settings.set("KEY", 0, priority="default")
+        runner = CrawlerRunner(settings)
+        crawler = runner.create_crawler(Spider)
+        self.assertEqual(crawler.settings.getint("KEY"), 15)
+
+        settings_dict = {
+            "KEY": 20,  # priority=project
+            "ADDONS": {get_addon_cls(config): 1},
+        }
+        settings = Settings(settings_dict)
+        settings.set("KEY", 0, priority="default")
+        runner = CrawlerRunner(settings)
+        crawler = runner.create_crawler(Spider)
+        self.assertEqual(crawler.settings.getint("KEY"), 20)
+
+    def test_fallback_workflow(self):
+        FALLBACK_SETTING = "MY_FALLBACK_DOWNLOAD_HANDLER"
+
+        class AddonWithFallback:
+            def update_settings(self, settings):
+                if not settings.get(FALLBACK_SETTING):
+                    settings.set(
+                        FALLBACK_SETTING,
+                        settings.getwithbase("DOWNLOAD_HANDLERS")["https"],
+                        "addon",
+                    )
+                settings["DOWNLOAD_HANDLERS"]["https"] = "AddonHandler"
+
+        settings_dict = {
+            "ADDONS": {AddonWithFallback: 1},
+        }
+        crawler = get_crawler(settings_dict=settings_dict)
+        self.assertEqual(
+            crawler.settings.getwithbase("DOWNLOAD_HANDLERS")["https"], "AddonHandler"
+        )
+        self.assertEqual(
+            crawler.settings.get(FALLBACK_SETTING),
+            "scrapy.core.downloader.handlers.http.HTTPDownloadHandler",
+        )
+
+        settings_dict = {
+            "ADDONS": {AddonWithFallback: 1},
+            "DOWNLOAD_HANDLERS": {"https": "UserHandler"},
+        }
+        crawler = get_crawler(settings_dict=settings_dict)
+        self.assertEqual(
+            crawler.settings.getwithbase("DOWNLOAD_HANDLERS")["https"], "AddonHandler"
+        )
+        self.assertEqual(crawler.settings.get(FALLBACK_SETTING), "UserHandler")
diff --git a/tests/test_middleware.py b/tests/test_middleware.py
index 00ff746ee5a..a42c7b3d1e2 100644
--- a/tests/test_middleware.py
+++ b/tests/test_middleware.py
@@ -39,7 +39,7 @@ def close_spider(self, spider):
         pass
 
     def __init__(self):
-        raise NotConfigured
+        raise NotConfigured("foo")
 
 
 class TestMiddlewareManager(MiddlewareManager):
diff --git a/tests/test_utils_deprecate.py b/tests/test_utils_deprecate.py
index 2d9210410d4..eedb6f6af9c 100644
--- a/tests/test_utils_deprecate.py
+++ b/tests/test_utils_deprecate.py
@@ -296,3 +296,7 @@ def test_unmatched_path_stays_the_same(self):
             output = update_classpath("scrapy.unmatched.Path")
         self.assertEqual(output, "scrapy.unmatched.Path")
         self.assertEqual(len(w), 0)
+
+    def test_returns_nonstring(self):
+        for notastring in [None, True, [1, 2, 3], object()]:
+            self.assertEqual(update_classpath(notastring), notastring)

EOF_114329324912
pytest -rA --tb=long tests/test_addons.py tests/test_middleware.py tests/test_utils_deprecate.py
git checkout 510574216d70ec84d75639ebcda360834a992e47 tests/test_middleware.py tests/test_utils_deprecate.py
