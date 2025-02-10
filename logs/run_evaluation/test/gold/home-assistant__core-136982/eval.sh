#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git diff fc979cd564ee2d5fd27e05b32fa6f11b343ee4d5
source /opt/miniconda3/bin/activate
conda activate testbed
pip install -e '.[all, dev, test]'
git checkout fc979cd564ee2d5fd27e05b32fa6f11b343ee4d5 tests/components/swiss_public_transport/snapshots/test_sensor.ambr tests/components/swiss_public_transport/test_sensor.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/components/swiss_public_transport/snapshots/test_sensor.ambr b/tests/components/swiss_public_transport/snapshots/test_sensor.ambr
index dbd689fc8f6ce3..b8ad82c7b79951 100644
--- a/tests/components/swiss_public_transport/snapshots/test_sensor.ambr
+++ b/tests/components/swiss_public_transport/snapshots/test_sensor.ambr
@@ -192,7 +192,7 @@
     'state': '2024-01-06T17:05:00+00:00',
   })
 # ---
-# name: test_all_entities[sensor.zurich_bern_duration-entry]
+# name: test_all_entities[sensor.zurich_bern_line-entry]
   EntityRegistryEntrySnapshot({
     'aliases': set({
     }),
@@ -204,7 +204,7 @@
     'disabled_by': None,
     'domain': 'sensor',
     'entity_category': None,
-    'entity_id': 'sensor.zurich_bern_duration',
+    'entity_id': 'sensor.zurich_bern_line',
     'has_entity_name': True,
     'hidden_by': None,
     'icon': None,
@@ -214,34 +214,32 @@
     'name': None,
     'options': dict({
     }),
-    'original_device_class': <SensorDeviceClass.DURATION: 'duration'>,
+    'original_device_class': None,
     'original_icon': None,
-    'original_name': 'Duration',
+    'original_name': 'Line',
     'platform': 'swiss_public_transport',
     'previous_unique_id': None,
     'supported_features': 0,
-    'translation_key': None,
-    'unique_id': 'Zürich Bern_duration',
-    'unit_of_measurement': <UnitOfTime.SECONDS: 's'>,
+    'translation_key': 'line',
+    'unique_id': 'Zürich Bern_line',
+    'unit_of_measurement': None,
   })
 # ---
-# name: test_all_entities[sensor.zurich_bern_duration-state]
+# name: test_all_entities[sensor.zurich_bern_line-state]
   StateSnapshot({
     'attributes': ReadOnlyDict({
       'attribution': 'Data provided by transport.opendata.ch',
-      'device_class': 'duration',
-      'friendly_name': 'Zürich Bern Duration',
-      'unit_of_measurement': <UnitOfTime.SECONDS: 's'>,
+      'friendly_name': 'Zürich Bern Line',
     }),
     'context': <ANY>,
-    'entity_id': 'sensor.zurich_bern_duration',
+    'entity_id': 'sensor.zurich_bern_line',
     'last_changed': <ANY>,
     'last_reported': <ANY>,
     'last_updated': <ANY>,
-    'state': '10',
+    'state': 'T10',
   })
 # ---
-# name: test_all_entities[sensor.zurich_bern_line-entry]
+# name: test_all_entities[sensor.zurich_bern_platform-entry]
   EntityRegistryEntrySnapshot({
     'aliases': set({
     }),
@@ -253,7 +251,7 @@
     'disabled_by': None,
     'domain': 'sensor',
     'entity_category': None,
-    'entity_id': 'sensor.zurich_bern_line',
+    'entity_id': 'sensor.zurich_bern_platform',
     'has_entity_name': True,
     'hidden_by': None,
     'icon': None,
@@ -265,30 +263,30 @@
     }),
     'original_device_class': None,
     'original_icon': None,
-    'original_name': 'Line',
+    'original_name': 'Platform',
     'platform': 'swiss_public_transport',
     'previous_unique_id': None,
     'supported_features': 0,
-    'translation_key': 'line',
-    'unique_id': 'Zürich Bern_line',
+    'translation_key': 'platform',
+    'unique_id': 'Zürich Bern_platform',
     'unit_of_measurement': None,
   })
 # ---
-# name: test_all_entities[sensor.zurich_bern_line-state]
+# name: test_all_entities[sensor.zurich_bern_platform-state]
   StateSnapshot({
     'attributes': ReadOnlyDict({
       'attribution': 'Data provided by transport.opendata.ch',
-      'friendly_name': 'Zürich Bern Line',
+      'friendly_name': 'Zürich Bern Platform',
     }),
     'context': <ANY>,
-    'entity_id': 'sensor.zurich_bern_line',
+    'entity_id': 'sensor.zurich_bern_platform',
     'last_changed': <ANY>,
     'last_reported': <ANY>,
     'last_updated': <ANY>,
-    'state': 'T10',
+    'state': '0',
   })
 # ---
-# name: test_all_entities[sensor.zurich_bern_platform-entry]
+# name: test_all_entities[sensor.zurich_bern_transfers-entry]
   EntityRegistryEntrySnapshot({
     'aliases': set({
     }),
@@ -300,7 +298,7 @@
     'disabled_by': None,
     'domain': 'sensor',
     'entity_category': None,
-    'entity_id': 'sensor.zurich_bern_platform',
+    'entity_id': 'sensor.zurich_bern_transfers',
     'has_entity_name': True,
     'hidden_by': None,
     'icon': None,
@@ -312,30 +310,30 @@
     }),
     'original_device_class': None,
     'original_icon': None,
-    'original_name': 'Platform',
+    'original_name': 'Transfers',
     'platform': 'swiss_public_transport',
     'previous_unique_id': None,
     'supported_features': 0,
-    'translation_key': 'platform',
-    'unique_id': 'Zürich Bern_platform',
+    'translation_key': 'transfers',
+    'unique_id': 'Zürich Bern_transfers',
     'unit_of_measurement': None,
   })
 # ---
-# name: test_all_entities[sensor.zurich_bern_platform-state]
+# name: test_all_entities[sensor.zurich_bern_transfers-state]
   StateSnapshot({
     'attributes': ReadOnlyDict({
       'attribution': 'Data provided by transport.opendata.ch',
-      'friendly_name': 'Zürich Bern Platform',
+      'friendly_name': 'Zürich Bern Transfers',
     }),
     'context': <ANY>,
-    'entity_id': 'sensor.zurich_bern_platform',
+    'entity_id': 'sensor.zurich_bern_transfers',
     'last_changed': <ANY>,
     'last_reported': <ANY>,
     'last_updated': <ANY>,
     'state': '0',
   })
 # ---
-# name: test_all_entities[sensor.zurich_bern_transfers-entry]
+# name: test_all_entities[sensor.zurich_bern_trip_duration-entry]
   EntityRegistryEntrySnapshot({
     'aliases': set({
     }),
@@ -347,7 +345,7 @@
     'disabled_by': None,
     'domain': 'sensor',
     'entity_category': None,
-    'entity_id': 'sensor.zurich_bern_transfers',
+    'entity_id': 'sensor.zurich_bern_trip_duration',
     'has_entity_name': True,
     'hidden_by': None,
     'icon': None,
@@ -356,29 +354,34 @@
     }),
     'name': None,
     'options': dict({
+      'sensor.private': dict({
+        'suggested_unit_of_measurement': <UnitOfTime.HOURS: 'h'>,
+      }),
     }),
-    'original_device_class': None,
+    'original_device_class': <SensorDeviceClass.DURATION: 'duration'>,
     'original_icon': None,
-    'original_name': 'Transfers',
+    'original_name': 'Trip duration',
     'platform': 'swiss_public_transport',
     'previous_unique_id': None,
     'supported_features': 0,
-    'translation_key': 'transfers',
-    'unique_id': 'Zürich Bern_transfers',
-    'unit_of_measurement': None,
+    'translation_key': 'trip_duration',
+    'unique_id': 'Zürich Bern_duration',
+    'unit_of_measurement': <UnitOfTime.HOURS: 'h'>,
   })
 # ---
-# name: test_all_entities[sensor.zurich_bern_transfers-state]
+# name: test_all_entities[sensor.zurich_bern_trip_duration-state]
   StateSnapshot({
     'attributes': ReadOnlyDict({
       'attribution': 'Data provided by transport.opendata.ch',
-      'friendly_name': 'Zürich Bern Transfers',
+      'device_class': 'duration',
+      'friendly_name': 'Zürich Bern Trip duration',
+      'unit_of_measurement': <UnitOfTime.HOURS: 'h'>,
     }),
     'context': <ANY>,
-    'entity_id': 'sensor.zurich_bern_transfers',
+    'entity_id': 'sensor.zurich_bern_trip_duration',
     'last_changed': <ANY>,
     'last_reported': <ANY>,
     'last_updated': <ANY>,
-    'state': '0',
+    'state': '0.003',
   })
 # ---
diff --git a/tests/components/swiss_public_transport/test_sensor.py b/tests/components/swiss_public_transport/test_sensor.py
index 4afdd88c9de2f8..6e8327282773e7 100644
--- a/tests/components/swiss_public_transport/test_sensor.py
+++ b/tests/components/swiss_public_transport/test_sensor.py
@@ -83,7 +83,7 @@ async def test_fetching_data(
         hass.states.get("sensor.zurich_bern_departure_2").state
         == "2024-01-06T17:05:00+00:00"
     )
-    assert hass.states.get("sensor.zurich_bern_duration").state == "10"
+    assert hass.states.get("sensor.zurich_bern_trip_duration").state == "0.003"
     assert hass.states.get("sensor.zurich_bern_platform").state == "0"
     assert hass.states.get("sensor.zurich_bern_transfers").state == "0"
     assert hass.states.get("sensor.zurich_bern_delay").state == "0"

EOF_114329324912
pytest -rA --tb=long tests/components/swiss_public_transport/snapshots/test_sensor.ambr tests/components/swiss_public_transport/test_sensor.py
git checkout fc979cd564ee2d5fd27e05b32fa6f11b343ee4d5 tests/components/swiss_public_transport/snapshots/test_sensor.ambr tests/components/swiss_public_transport/test_sensor.py
