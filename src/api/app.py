import os
import sys
import json
import uuid
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timezone

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Depends, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from src.core.config import config
from src.core.logging import logger
from src.core.database import init_db, get_db
from src.core.upload import disk_stream_upload, cleanup_stale_uploads
from src.core.rate_limit import rate_limiter, rate_limit_dependency
from src.models.db import ScanRecord, JobState, AlertHistory
from src.models.media import MediaScanRequest, MediaScanResponse
from src.models.advisory import MessageScanRequest, MessageScanResponse
from src.models.infrastructure import AppScanRequest, AppScanResponse
from src.models.verdict import (
    ComprehensiveScanRequest,
    ComprehensiveScanResponse,
    AlertNotification,
    SebiScoresDossier
)
from src.services.deepfake_engine import DeepfakeEngine
from src.services.impersonation_engine import ImpersonationEngine
from src.services.infrastructure_engine import TradingAppEngine
from src.services.alert_engine import AlertEngine

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes SQLite database tables and cleans stale uploads."""
    init_db()
    removed = cleanup_stale_uploads()
    if removed > 0:
        logger.info(f"Cleaned up {removed} stale temporary upload files.")
    yield

# Initialize FastAPI App
app = FastAPI(
    title="FinGuard 2.0 — Real-Time Investor Fraud & Impersonation Detection System",
    description="SEBI Investor-Protection Aligned System detecting Financial Deepfakes, Impersonation Messaging, and Fake Trading Infrastructure.",
    version="2.0-foundation-hardened",
    lifespan=lifespan
)

# Replace unrestricted CORS with explicit environment-based allowed origins
origins = config.cors_origins
logger.info(f"Configured CORS allowed origins: {origins}")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static and temp directories exist
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Instantiate detection engines
deepfake_engine = DeepfakeEngine()
impersonation_engine = ImpersonationEngine()
trading_app_engine = TradingAppEngine()
alert_engine = AlertEngine()

# Ensure database schema is initialized
init_db()

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serves the FinGuard 2.0 Command Center Dashboard."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>FinGuard 2.0 API Online</h1><p>UI is initializing.</p>")

@app.get("/api/health")
async def health_check():
    """System health check and module status."""
    return {
        "status": "HEALTHY",
        "version": "2.0-foundation-hardened",
        "phase": "Milestone: Foundation Hardening",
        "security": {
            "cors_origins": config.cors_origins,
            "rate_limit_per_minute": config.rate_limit_per_minute,
            "max_upload_mb": config.max_upload_mb
        },
        "persistence": {
            "database_url": config.database_url,
            "driver": "SQLAlchemy / SQLite"
        },
        "engines": {
            "deepfake_engine": {
                "status": "READY",
                "mode": deepfake_engine.mode,
                "benchmark_targets": ["FaceForensics++", "DFDC"]
            },
            "impersonation_engine": {
                "status": "READY",
                "sebi_registry_records": len(impersonation_engine.registry_data)
            },
            "trading_app_engine": {
                "status": "READY",
                "whitelisted_brokers": len(trading_app_engine.whitelist)
            },
            "alert_engine": {
                "status": "READY",
                "active_alerts_count": len(alert_engine.recent_alerts)
            }
        }
    }

@app.get("/api/config")
async def get_system_config():
    """Returns active configuration and threshold calibration."""
    return {
        "thresholds": config.thresholds,
        "benchmarks": config.benchmarks,
        "engines": config.engines,
        "security": {
            "cors_origins": config.cors_origins,
            "rate_limit_per_minute": config.rate_limit_per_minute,
            "rate_limit_enabled": config.rate_limit_enabled
        },
        "uploads": {
            "max_upload_mb": config.max_upload_mb,
            "allowed_video_extensions": config.get("uploads.allowed_video_extensions"),
            "allowed_audio_extensions": config.get("uploads.allowed_audio_extensions")
        }
    }

@app.get("/api/benchmarks/info")
async def get_benchmarks_info():
    """Returns information about deepfake evaluation benchmarks (FaceForensics++ & DFDC)."""
    return {
        "evaluation_guidelines": "Official academic datasets used for deepfake model validation prior to deployment.",
        "benchmarks": config.benchmarks
    }

@app.post("/api/scan/media", response_model=MediaScanResponse, dependencies=[Depends(rate_limit_dependency)])
async def scan_media_json(request: MediaScanRequest, db: Session = Depends(get_db)):
    """Scans media file by URL or parameter request with persistence."""
    res = deepfake_engine.scan_media(request)

    # Persist JobState and ScanRecord
    job_rec = JobState(job_id=res.job_id, status="COMPLETED", result_payload=res.model_dump_json())
    scan_rec = ScanRecord(
        scan_id=res.job_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        scan_type="media",
        input_target=request.media_url or "URL Request",
        risk_score=max(res.visual_manipulation_score, res.acoustic_manipulation_score),
        verdict="CRITICAL_FRAUD" if res.is_deepfake else "SAFE",
        processing_time_ms=res.processing_time_ms,
        raw_payload=res.model_dump_json()
    )
    db.add(job_rec)
    db.add(scan_rec)
    db.commit()

    if res.is_deepfake:
        alert_engine.synthesize(media_res=res)

    return res

@app.post("/api/scan/media-file", response_model=MediaScanResponse, dependencies=[Depends(rate_limit_dependency)])
async def scan_media_file(
    file: UploadFile = File(...),
    sampling_fps: float = Form(2.0),
    extract_audio: bool = Form(True),
    db: Session = Depends(get_db)
):
    """
    Accepts uploaded video/audio file using bounded disk-backed streaming.
    Enforces upload-size limits, checks magic bytes, and cleans up temporary files safely.
    """
    with disk_stream_upload(file) as (temp_path, total_bytes):
        # Read bounded sample bytes (first 64KB) for heuristic validation
        with open(temp_path, "rb") as f_in:
            sample_bytes = f_in.read(65536)

        req = MediaScanRequest(sampling_fps=sampling_fps, extract_audio=extract_audio)
        res = deepfake_engine.scan_media(req, file_bytes=sample_bytes, filename=file.filename)

        # Persist JobState and ScanRecord
        job_rec = JobState(job_id=res.job_id, status="COMPLETED", result_payload=res.model_dump_json())
        scan_rec = ScanRecord(
            scan_id=res.job_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            scan_type="media",
            input_target=file.filename,
            risk_score=max(res.visual_manipulation_score, res.acoustic_manipulation_score),
            verdict="CRITICAL_FRAUD" if res.is_deepfake else "SAFE",
            processing_time_ms=res.processing_time_ms,
            raw_payload=res.model_dump_json()
        )
        db.add(job_rec)
        db.add(scan_rec)
        db.commit()

        if res.is_deepfake:
            alert_engine.synthesize(media_res=res)

        return res

@app.post("/api/scan/message", response_model=MessageScanResponse, dependencies=[Depends(rate_limit_dependency)])
async def scan_message(request: MessageScanRequest, db: Session = Depends(get_db)):
    """Scans chat/social advisory message for unregistered advice and impersonation with persistence."""
    res = impersonation_engine.scan_message(request)
    scan_id = str(uuid.uuid4())

    job_rec = JobState(job_id=scan_id, status="COMPLETED", result_payload=res.model_dump_json())
    scan_rec = ScanRecord(
        scan_id=scan_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        scan_type="message",
        input_target=f"{request.platform}: {request.message_text[:60]}...",
        risk_score=res.composite_risk_score,
        verdict="CRITICAL_FRAUD" if res.composite_risk_score >= 0.70 else ("SUSPICIOUS" if res.is_suspicious_advisory else "SAFE"),
        processing_time_ms=10.0,
        raw_payload=res.model_dump_json()
    )
    db.add(job_rec)
    db.add(scan_rec)
    db.commit()

    if res.is_suspicious_advisory:
        alert_engine.synthesize(advisory_res=res)
    return res

@app.post("/api/scan/app", response_model=AppScanResponse, dependencies=[Depends(rate_limit_dependency)])
async def scan_app(request: AppScanRequest, db: Session = Depends(get_db)):
    """Scans domain URL or Android APK permissions for broker spoofing with persistence."""
    res = trading_app_engine.scan_infrastructure(request)
    scan_id = str(uuid.uuid4())

    job_rec = JobState(job_id=scan_id, status="COMPLETED", result_payload=res.model_dump_json())
    scan_rec = ScanRecord(
        scan_id=scan_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        scan_type="app",
        input_target=res.scan_target,
        risk_score=res.risk_score,
        verdict="CRITICAL_FRAUD" if res.is_fraudulent else "SAFE",
        processing_time_ms=5.0,
        raw_payload=res.model_dump_json()
    )
    db.add(job_rec)
    db.add(scan_rec)
    db.commit()

    if res.is_fraudulent:
        alert_engine.synthesize(infra_res=res)
    return res

@app.post("/api/scan/comprehensive", response_model=ComprehensiveScanResponse, dependencies=[Depends(rate_limit_dependency)])
async def scan_comprehensive(request: ComprehensiveScanRequest, db: Session = Depends(get_db)):
    """Executes multi-vector scan synthesizing findings across media, text, and infrastructure with persistence."""
    media_res = None
    advisory_res = None
    infra_res = None

    if request.media_url:
        media_res = deepfake_engine.scan_media(MediaScanRequest(media_url=request.media_url))

    if request.message_text:
        advisory_res = impersonation_engine.scan_message(
            MessageScanRequest(
                message_text=request.message_text,
                platform=request.platform or "generic"
            )
        )

    if request.target_url or request.package_name:
        infra_res = trading_app_engine.scan_infrastructure(
            AppScanRequest(
                target_url=request.target_url,
                package_name=request.package_name,
                app_label=request.app_label
            )
        )

    comp = alert_engine.synthesize(
        media_res=media_res,
        advisory_res=advisory_res,
        infra_res=infra_res
    )

    job_rec = JobState(job_id=comp.scan_id, status="COMPLETED", result_payload=comp.model_dump_json())
    scan_rec = ScanRecord(
        scan_id=comp.scan_id,
        timestamp=comp.timestamp,
        scan_type="comprehensive",
        input_target=request.target_url or request.package_name or request.app_label or "Multi-vector",
        risk_score=comp.composite_risk_score,
        verdict=comp.final_verdict,
        processing_time_ms=25.0,
        raw_payload=comp.model_dump_json()
    )
    db.add(job_rec)
    db.add(scan_rec)
    db.commit()

    return comp

@app.get("/api/jobs/{job_id}")
async def get_job_state(job_id: str, db: Session = Depends(get_db)):
    """Queries asynchronous or persisted job state from SQLite."""
    job = db.query(JobState).filter(JobState.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    
    result = None
    if job.result_payload:
        try:
            result = json.loads(job.result_payload)
        except Exception:
            result = job.result_payload

    return {
        "job_id": job.job_id,
        "status": job.status,
        "error_message": job.error_message,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "updated_at": job.updated_at.isoformat() if job.updated_at else None,
        "result": result
    }

@app.get("/api/scans/recent")
async def get_recent_scans(limit: int = 20, db: Session = Depends(get_db)):
    """Queries audit log of scan records from SQLite."""
    scans = db.query(ScanRecord).order_by(ScanRecord.id.desc()).limit(limit).all()
    return [
        {
            "scan_id": s.scan_id,
            "timestamp": s.timestamp,
            "scan_type": s.scan_type,
            "input_target": s.input_target,
            "risk_score": s.risk_score,
            "verdict": s.verdict,
            "processing_time_ms": s.processing_time_ms
        }
        for s in scans
    ]

@app.get("/api/alerts/recent", response_model=List[AlertNotification])
async def get_recent_alerts(limit: int = 20):
    """Returns real-time risk alert stream queried from SQLite."""
    return alert_engine.get_recent_alerts(limit=limit)

@app.post("/api/report/generate-dossier", response_model=SebiScoresDossier, dependencies=[Depends(rate_limit_dependency)])
async def generate_dossier(request: ComprehensiveScanRequest, db: Session = Depends(get_db)):
    """Generates an official pre-formatted SEBI SCORES Complaint Dossier."""
    comp = await scan_comprehensive(request, db=db)
    if comp.scores_dossier:
        return comp.scores_dossier
    return SebiScoresDossier(
        dossier_id="SCORES-CLEAN-AUDIT",
        generated_at=comp.timestamp,
        alleged_entity_or_handle=request.app_label or request.package_name or "Audit-Subject",
        regulatory_acts_cited=["SEBI Investor Protection Guidelines"],
        factual_summary="Forensic evaluation completed. No regulatory violations detected.",
        evidence_points=["Asset matches verified registry credentials or within safe thresholds."],
        recommended_sebi_action="No regulatory enforcement action needed."
    )
