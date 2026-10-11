import os
import io
import time
from pathlib import Path
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from src.api.app import app
from src.core.config import config
from src.core.database import SessionLocal, init_db
from src.core.rate_limit import rate_limiter
from src.core.upload import get_temp_upload_dir
from src.models.db import ScanRecord, JobState, AlertHistory

client = TestClient(app)

def setup_module():
    init_db()

def test_sqlite_persistence_via_api():
    rate_limiter.reset()
    payload = {
        "message_text": "Guaranteed 100% loss recovery call! DM to mirror trades immediately.",
        "platform": "whatsapp"
    }
    res = client.post("/api/scan/message", json=payload)
    assert res.status_code == 200

    # Verify record in SQLite database
    db = SessionLocal()
    try:
        rec = db.query(ScanRecord).filter(ScanRecord.scan_type == "message").order_by(ScanRecord.id.desc()).first()
        assert rec is not None
        assert "Guaranteed" in rec.input_target
        assert rec.risk_score > 0.5

        job = db.query(JobState).filter(JobState.job_id == rec.scan_id).first()
        assert job is not None
        assert job.status == "COMPLETED"

        # Verify query endpoints
        job_res = client.get(f"/api/jobs/{rec.scan_id}")
        assert job_res.status_code == 200
        assert job_res.json()["job_id"] == rec.scan_id
        assert job_res.json()["status"] == "COMPLETED"

        scans_res = client.get("/api/scans/recent?limit=5")
        assert scans_res.status_code == 200
        assert len(scans_res.json()) > 0
    finally:
        db.close()

def test_alert_history_persistence():
    rate_limiter.reset()
    payload = {
        "message_text": "100% guaranteed profit rocket call! DM to mirror.",
        "target_url": "https://groww-bonus-reward.xyz"
    }
    res = client.post("/api/scan/comprehensive", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert len(data["synthesized_alerts"]) > 0

    # Verify alert saved in database
    db = SessionLocal()
    try:
        alert = db.query(AlertHistory).order_by(AlertHistory.id.desc()).first()
        assert alert is not None
        assert alert.severity in ["WARNING", "CRITICAL"]

        # Verify recent alerts endpoint queries from DB
        recent_res = client.get("/api/alerts/recent?limit=5")
        assert recent_res.status_code == 200
        assert len(recent_res.json()) > 0
    finally:
        db.close()

def test_media_upload_valid_disk_stream():
    rate_limiter.reset()
    temp_dir = get_temp_upload_dir()
    initial_tmp_count = len(list(temp_dir.glob("*.tmp")))

    # Valid MP4 header: 4 bytes length + 'ftyp' + brand
    valid_mp4_bytes = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2mp41" + b"\x00" * 1024
    files = {"file": ("test_clip.mp4", valid_mp4_bytes, "video/mp4")}

    res = client.post("/api/scan/media-file", files=files, data={"sampling_fps": 2.0})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "COMPLETED"
    assert data["filename"] == "test_clip.mp4"

    # Verify temp file was cleanly removed
    remaining_tmp_count = len(list(temp_dir.glob("*.tmp")))
    assert remaining_tmp_count == initial_tmp_count

def test_media_upload_invalid_extension():
    rate_limiter.reset()
    temp_dir = get_temp_upload_dir()
    initial_tmp_count = len(list(temp_dir.glob("*.tmp")))

    files = {"file": ("malicious_script.exe", b"MZ\x90\x00binary", "application/octet-stream")}
    res = client.post("/api/scan/media-file", files=files)
    assert res.status_code == 400
    assert "Unsupported file extension" in res.json()["detail"]

    # Verify no leaked temp file
    assert len(list(temp_dir.glob("*.tmp"))) == initial_tmp_count

def test_media_upload_invalid_magic_bytes():
    rate_limiter.reset()
    temp_dir = get_temp_upload_dir()
    initial_tmp_count = len(list(temp_dir.glob("*.tmp")))

    # Extension is .mp4 but content is plain text
    fake_mp4_bytes = b"This is a text file renamed to mp4, not a real video stream."
    files = {"file": ("forged_video.mp4", fake_mp4_bytes, "video/mp4")}

    res = client.post("/api/scan/media-file", files=files)
    assert res.status_code == 400
    assert "magic bytes signature" in res.json()["detail"]

    # Verify no leaked temp file
    assert len(list(temp_dir.glob("*.tmp"))) == initial_tmp_count

def test_media_upload_oversized_limit():
    rate_limiter.reset()
    temp_dir = get_temp_upload_dir()
    initial_tmp_count = len(list(temp_dir.glob("*.tmp")))

    # Temporarily set max_upload_mb to 1MB in config
    orig_max = config._raw_config.setdefault("uploads", {})["max_upload_mb"]
    try:
        config._raw_config["uploads"]["max_upload_mb"] = 1
        # Create 1.5MB buffer
        oversized_bytes = b"\x00\x00\x00\x18ftypisom" + (b"A" * (1024 * 1024 + 500 * 1024))
        files = {"file": ("big_video.mp4", oversized_bytes, "video/mp4")}

        res = client.post("/api/scan/media-file", files=files)
        assert res.status_code == 413
        assert "exceeds maximum limit" in res.json()["detail"]
    finally:
        config._raw_config["uploads"]["max_upload_mb"] = orig_max

    # Verify temp file was cleaned up even when 413 was raised
    assert len(list(temp_dir.glob("*.tmp"))) == initial_tmp_count

def test_rate_limiting_enforcement():
    rate_limiter.reset()
    # Test with custom limit threshold
    for _ in range(5):
        rate_limiter.check(request=None, custom_limit=5)

    # 6th request must raise HTTP 429
    import pytest
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc_info:
        rate_limiter.check(request=None, custom_limit=5)
    assert exc_info.value.status_code == 429
    assert "Rate limit exceeded" in exc_info.value.detail
    assert "Retry-After" in exc_info.value.headers

    rate_limiter.reset()

def test_cors_explicit_configuration():
    origins = config.cors_origins
    assert isinstance(origins, list)
    # Ensure wildcard * is not the default
    assert "*" not in origins
