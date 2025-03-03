#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git diff a10ae15440b812e146d57de1a5d5a02b3ec9c4c7
source /opt/miniconda3/bin/activate
conda activate testbed
pip install -e '.[all,dev,test]' --no-deps && pip install -e '.[devel]' --upgrade --upgrade-strategy eager && pip install --no-deps -r <(pip freeze)
git checkout a10ae15440b812e146d57de1a5d5a02b3ec9c4c7 tests/api_fastapi/core_api/routes/public/test_dag_run.py tests/api_fastapi/core_api/routes/ui/test_dags.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/api_fastapi/core_api/routes/public/test_dag_run.py b/tests/api_fastapi/core_api/routes/public/test_dag_run.py
index 99a2ae83acce0..df4dc708d94d1 100644
--- a/tests/api_fastapi/core_api/routes/public/test_dag_run.py
+++ b/tests/api_fastapi/core_api/routes/public/test_dag_run.py
@@ -66,6 +66,8 @@
 START_DATE1 = datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc)
 LOGICAL_DATE1 = datetime(2024, 2, 16, 0, 0, tzinfo=timezone.utc)
 LOGICAL_DATE2 = datetime(2024, 2, 20, 0, 0, tzinfo=timezone.utc)
+RUN_AFTER1 = datetime(2024, 2, 16, 0, 0, tzinfo=timezone.utc)
+RUN_AFTER2 = datetime(2024, 2, 20, 0, 0, tzinfo=timezone.utc)
 START_DATE2 = datetime(2024, 4, 15, 0, 0, tzinfo=timezone.utc)
 LOGICAL_DATE3 = datetime(2024, 5, 16, 0, 0, tzinfo=timezone.utc)
 LOGICAL_DATE4 = datetime(2024, 5, 25, 0, 0, tzinfo=timezone.utc)
@@ -397,6 +399,14 @@ def test_bad_limit_and_offset(self, test_client, query_params, expected_detail):
                 },
                 [DAG1_RUN1_ID, DAG1_RUN2_ID],
             ),
+            (
+                DAG1_ID,
+                {
+                    "run_after_gte": RUN_AFTER1.isoformat(),
+                    "run_after_lte": RUN_AFTER2.isoformat(),
+                },
+                [DAG1_RUN1_ID, DAG1_RUN2_ID],
+            ),
             (
                 DAG2_ID,
                 {
@@ -436,11 +446,27 @@ def test_bad_filters(self, test_client):
             "logical_date_gte": "invalid",
             "start_date_gte": "invalid",
             "end_date_gte": "invalid",
+            "run_after_gte": "invalid",
             "logical_date_lte": "invalid",
             "start_date_lte": "invalid",
             "end_date_lte": "invalid",
+            "run_after_lte": "invalid",
         }
         expected_detail = [
+            {
+                "type": "datetime_from_date_parsing",
+                "loc": ["query", "run_after_gte"],
+                "msg": "Input should be a valid datetime or date, input is too short",
+                "input": "invalid",
+                "ctx": {"error": "input is too short"},
+            },
+            {
+                "type": "datetime_from_date_parsing",
+                "loc": ["query", "run_after_lte"],
+                "msg": "Input should be a valid datetime or date, input is too short",
+                "input": "invalid",
+                "ctx": {"error": "input is too short"},
+            },
             {
                 "type": "datetime_from_date_parsing",
                 "loc": ["query", "logical_date_gte"],
@@ -577,6 +603,7 @@ def test_invalid_order_by_raises_400(self, test_client):
                 "state", [DAG1_RUN2_ID, DAG1_RUN1_ID, DAG2_RUN1_ID, DAG2_RUN2_ID], id="order_by_state"
             ),
             pytest.param("dag_id", DAG_RUNS_LIST, id="order_by_dag_id"),
+            pytest.param("run_after", DAG_RUNS_LIST, id="order_by_run_after"),
             pytest.param("logical_date", DAG_RUNS_LIST, id="order_by_logical_date"),
             pytest.param("dag_run_id", DAG_RUNS_LIST, id="order_by_dag_run_id"),
             pytest.param("start_date", DAG_RUNS_LIST, id="order_by_start_date"),
diff --git a/tests/api_fastapi/core_api/routes/ui/test_dags.py b/tests/api_fastapi/core_api/routes/ui/test_dags.py
index 5e7cee8096cf0..fc6d509356972 100644
--- a/tests/api_fastapi/core_api/routes/ui/test_dags.py
+++ b/tests/api_fastapi/core_api/routes/ui/test_dags.py
@@ -53,6 +53,7 @@ def setup_dag_runs(self, session=None) -> None:
                     run_type=DagRunType.MANUAL,
                     start_date=start_date,
                     logical_date=start_date,
+                    run_after=start_date,
                     state=(DagRunState.FAILED if i % 2 == 0 else DagRunState.SUCCESS),
                     triggered_by=DagRunTriggeredByType.TEST,
                 )
@@ -90,16 +91,16 @@ def test_recent_dag_runs(self, test_client, query_params, expected_ids, expected
             "dag_run_id",
             "dag_id",
             "state",
-            "logical_date",
+            "run_after",
         ]
         for recent_dag_runs in body["dags"]:
             dag_runs = recent_dag_runs["latest_dag_runs"]
             # check date ordering
-            previous_logical_date = None
+            previous_run_after = None
             for dag_run in dag_runs:
                 # validate the response
                 for key in required_dag_run_key:
                     assert key in dag_run
-                if previous_logical_date:
-                    assert previous_logical_date > dag_run["logical_date"]
-                previous_logical_date = dag_run["logical_date"]
+                if previous_run_after:
+                    assert previous_run_after > dag_run["run_after"]
+                previous_run_after = dag_run["run_after"]

EOF_114329324912
pytest -rA --tb=long tests/api_fastapi/core_api/routes/public/test_dag_run.py tests/api_fastapi/core_api/routes/ui/test_dags.py
git checkout a10ae15440b812e146d57de1a5d5a02b3ec9c4c7 tests/api_fastapi/core_api/routes/public/test_dag_run.py tests/api_fastapi/core_api/routes/ui/test_dags.py
