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
git checkout fc979cd564ee2d5fd27e05b32fa6f11b343ee4d5 tests/components/onedrive/conftest.py tests/components/onedrive/test_backup.py
git apply -v - <<'EOF_114329324912'
diff --git a/tests/components/onedrive/conftest.py b/tests/components/onedrive/conftest.py
index 65142217017431..649966a7828978 100644
--- a/tests/components/onedrive/conftest.py
+++ b/tests/components/onedrive/conftest.py
@@ -176,3 +176,10 @@ def mock_instance_id() -> Generator[AsyncMock]:
         return_value="9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0",
     ):
         yield
+
+
+@pytest.fixture(autouse=True)
+def mock_asyncio_sleep() -> Generator[AsyncMock]:
+    """Mock asyncio.sleep."""
+    with patch("homeassistant.components.onedrive.backup.asyncio.sleep", AsyncMock()):
+        yield
diff --git a/tests/components/onedrive/test_backup.py b/tests/components/onedrive/test_backup.py
index 3492202d3feb08..162ecb7d92ae74 100644
--- a/tests/components/onedrive/test_backup.py
+++ b/tests/components/onedrive/test_backup.py
@@ -8,8 +8,10 @@
 from json import dumps
 from unittest.mock import Mock, patch
 
+from httpx import TimeoutException
 from kiota_abstractions.api_error import APIError
 from msgraph.generated.models.drive_item import DriveItem
+from msgraph_core.models import LargeFileUploadSession
 import pytest
 
 from homeassistant.components.backup import DOMAIN as BACKUP_DOMAIN, AgentBackup
@@ -255,6 +257,140 @@ async def test_broken_upload_session(
     assert "Failed to start backup upload" in caplog.text
 
 
+@pytest.mark.parametrize(
+    "side_effect",
+    [
+        APIError(response_status_code=500),
+        TimeoutException("Timeout"),
+    ],
+)
+async def test_agents_upload_errors_retried(
+    hass_client: ClientSessionGenerator,
+    caplog: pytest.LogCaptureFixture,
+    mock_drive_items: MagicMock,
+    mock_config_entry: MockConfigEntry,
+    mock_adapter: MagicMock,
+    side_effect: Exception,
+) -> None:
+    """Test agent upload backup."""
+    client = await hass_client()
+    test_backup = AgentBackup.from_dict(BACKUP_METADATA)
+
+    mock_adapter.send_async.side_effect = [
+        side_effect,
+        LargeFileUploadSession(next_expected_ranges=["2-"]),
+        LargeFileUploadSession(next_expected_ranges=["2-"]),
+    ]
+
+    with (
+        patch(
+            "homeassistant.components.backup.manager.BackupManager.async_get_backup",
+        ) as fetch_backup,
+        patch(
+            "homeassistant.components.backup.manager.read_backup",
+            return_value=test_backup,
+        ),
+        patch("pathlib.Path.open") as mocked_open,
+        patch("homeassistant.components.onedrive.backup.UPLOAD_CHUNK_SIZE", 3),
+    ):
+        mocked_open.return_value.read = Mock(side_effect=[b"test", b""])
+        fetch_backup.return_value = test_backup
+        resp = await client.post(
+            f"/api/backup/upload?agent_id={DOMAIN}.{mock_config_entry.unique_id}",
+            data={"file": StringIO("test")},
+        )
+
+    assert resp.status == 201
+    assert mock_adapter.send_async.call_count == 3
+    assert f"Uploading backup {test_backup.backup_id}" in caplog.text
+    mock_drive_items.patch.assert_called_once()
+
+
+async def test_agents_upload_4xx_errors_not_retried(
+    hass_client: ClientSessionGenerator,
+    caplog: pytest.LogCaptureFixture,
+    mock_drive_items: MagicMock,
+    mock_config_entry: MockConfigEntry,
+    mock_adapter: MagicMock,
+) -> None:
+    """Test agent upload backup."""
+    client = await hass_client()
+    test_backup = AgentBackup.from_dict(BACKUP_METADATA)
+
+    mock_adapter.send_async.side_effect = APIError(response_status_code=404)
+
+    with (
+        patch(
+            "homeassistant.components.backup.manager.BackupManager.async_get_backup",
+        ) as fetch_backup,
+        patch(
+            "homeassistant.components.backup.manager.read_backup",
+            return_value=test_backup,
+        ),
+        patch("pathlib.Path.open") as mocked_open,
+        patch("homeassistant.components.onedrive.backup.UPLOAD_CHUNK_SIZE", 3),
+    ):
+        mocked_open.return_value.read = Mock(side_effect=[b"test", b""])
+        fetch_backup.return_value = test_backup
+        resp = await client.post(
+            f"/api/backup/upload?agent_id={DOMAIN}.{mock_config_entry.unique_id}",
+            data={"file": StringIO("test")},
+        )
+
+    assert resp.status == 201
+    assert mock_adapter.send_async.call_count == 1
+    assert f"Uploading backup {test_backup.backup_id}" in caplog.text
+    assert mock_drive_items.patch.call_count == 0
+    assert "Backup operation failed" in caplog.text
+
+
+@pytest.mark.parametrize(
+    ("side_effect", "error"),
+    [
+        (APIError(response_status_code=500), "Backup operation failed"),
+        (TimeoutException("Timeout"), "Backup operation timed out"),
+    ],
+)
+async def test_agents_upload_fails_after_max_retries(
+    hass_client: ClientSessionGenerator,
+    caplog: pytest.LogCaptureFixture,
+    mock_drive_items: MagicMock,
+    mock_config_entry: MockConfigEntry,
+    mock_adapter: MagicMock,
+    side_effect: Exception,
+    error: str,
+) -> None:
+    """Test agent upload backup."""
+    client = await hass_client()
+    test_backup = AgentBackup.from_dict(BACKUP_METADATA)
+
+    mock_adapter.send_async.side_effect = side_effect
+
+    with (
+        patch(
+            "homeassistant.components.backup.manager.BackupManager.async_get_backup",
+        ) as fetch_backup,
+        patch(
+            "homeassistant.components.backup.manager.read_backup",
+            return_value=test_backup,
+        ),
+        patch("pathlib.Path.open") as mocked_open,
+        patch("homeassistant.components.onedrive.backup.UPLOAD_CHUNK_SIZE", 3),
+    ):
+        mocked_open.return_value.read = Mock(side_effect=[b"test", b""])
+        fetch_backup.return_value = test_backup
+        resp = await client.post(
+            f"/api/backup/upload?agent_id={DOMAIN}.{mock_config_entry.unique_id}",
+            data={"file": StringIO("test")},
+        )
+
+    assert resp.status == 201
+    assert mock_adapter.send_async.call_count == 6
+    assert f"Uploading backup {test_backup.backup_id}" in caplog.text
+    assert mock_drive_items.patch.call_count == 0
+    assert error in caplog.text
+
+
 async def test_agents_download(
     hass_client: ClientSessionGenerator,
     mock_drive_items: MagicMock,
@@ -282,7 +418,7 @@ async def test_agents_download(
             APIError(response_status_code=500),
             "Backup operation failed",
         ),
-        (TimeoutError(), "Backup operation timed out"),
+        (TimeoutException("Timeout"), "Backup operation timed out"),
     ],
 )
 async def test_delete_error(

EOF_114329324912
pytest -rA --tb=long tests/components/onedrive/conftest.py tests/components/onedrive/test_backup.py
git checkout fc979cd564ee2d5fd27e05b32fa6f11b343ee4d5 tests/components/onedrive/conftest.py tests/components/onedrive/test_backup.py
