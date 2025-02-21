#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git diff 4833b53705acfc4bd0a26bf3e4dd4fc7a22b0bfa
source /opt/miniconda3/bin/activate
conda activate testbed
pip install -e '.[all,dev,test]' --no-deps && pip install -e '.[devel]' --upgrade --upgrade-strategy eager && pip install --no-deps -r <(pip freeze)
git checkout 4833b53705acfc4bd0a26bf3e4dd4fc7a22b0bfa tests/api_fastapi/core_api/routes/public/test_task_instances.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/api_fastapi/core_api/routes/public/test_task_instances.py b/tests/api_fastapi/core_api/routes/public/test_task_instances.py
index a28565d451d6d..761f3074aadbc 100644
--- a/tests/api_fastapi/core_api/routes/public/test_task_instances.py
+++ b/tests/api_fastapi/core_api/routes/public/test_task_instances.py
@@ -2238,7 +2238,20 @@ def test_should_respond_200_with_reset_dag_run(self, test_client, session):
         assert response.json()["total_entries"] == 6
         assert failed_dag_runs == 0
 
-    def test_should_respond_200_with_dag_run_id(self, test_client, session):
+    @pytest.mark.parametrize(
+        "target_logical_date, response_logical_date",
+        [
+            pytest.param(DEFAULT_DATETIME_1, "2020-01-01T00:00:00Z", id="date"),
+            pytest.param(None, None, id="null"),
+        ],
+    )
+    def test_should_respond_200_with_dag_run_id(
+        self,
+        test_client,
+        session,
+        target_logical_date,
+        response_logical_date,
+    ):
         dag_id = "example_python_operator"
         payload = {
             "dry_run": False,
@@ -2247,29 +2260,14 @@ def test_should_respond_200_with_dag_run_id(self, test_client, session):
             "only_running": True,
             "dag_run_id": "TEST_DAG_RUN_ID_0",
         }
-        task_instances = [
-            {"logical_date": DEFAULT_DATETIME_1, "state": State.RUNNING},
-            {
-                "logical_date": DEFAULT_DATETIME_1 + dt.timedelta(days=1),
-                "state": State.RUNNING,
-            },
-            {
-                "logical_date": DEFAULT_DATETIME_1 + dt.timedelta(days=2),
-                "state": State.RUNNING,
-            },
-            {
-                "logical_date": DEFAULT_DATETIME_1 + dt.timedelta(days=3),
-                "state": State.RUNNING,
-            },
-            {
-                "logical_date": DEFAULT_DATETIME_1 + dt.timedelta(days=4),
-                "state": State.RUNNING,
-            },
-            {
-                "logical_date": DEFAULT_DATETIME_1 + dt.timedelta(days=5),
-                "state": State.RUNNING,
-            },
-        ]
+        if target_logical_date:
+            task_instances = [
+                {"logical_date": target_logical_date + dt.timedelta(days=i), "state": State.RUNNING}
+                for i in range(6)
+            ]
+        else:
+            self.ti_extras["run_after"] = DEFAULT_DATETIME_1
+            task_instances = [{"logical_date": target_logical_date, "state": State.RUNNING} for _ in range(6)]
 
         self.create_task_instances(
             session,
@@ -2296,7 +2294,7 @@ def test_should_respond_200_with_dag_run_id(self, test_client, session):
                 "executor_config": "{}",
                 "hostname": "",
                 "id": mock.ANY,
-                "logical_date": "2020-01-01T00:00:00Z",
+                "logical_date": response_logical_date,
                 "map_index": -1,
                 "max_tries": 0,
                 "note": "placeholder-note",

EOF_114329324912
pytest -rA --tb=long tests/api_fastapi/core_api/routes/public/test_task_instances.py
git checkout 4833b53705acfc4bd0a26bf3e4dd4fc7a22b0bfa tests/api_fastapi/core_api/routes/public/test_task_instances.py
