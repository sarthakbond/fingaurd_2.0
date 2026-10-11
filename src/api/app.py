import os
import sys
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import config
from src.core.logging import logger
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

# Initialize FastAPI App
app = FastAPI(
    title="FinGuard 2.0 — Real-Time Investor Fraud & Impersonation Detection System",
    description="SEBI Investor-Protection Aligned System detecting Financial Deepfakes, Impersonation Messaging, and Fake Trading Infrastructure.",
    version="2.0-phase1"
)

# Configure CORS
origins = config.server.get("cors_origins", ["*"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static directories exist
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Instantiate detection engines
deepfake_engine = DeepfakeEngine()
impersonation_engine = ImpersonationEngine()
trading_app_engine = TradingAppEngine()
alert_engine = AlertEngine()

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
        "version": "2.0-phase1",
        "phase": "Phase 1: Working Application Foundation",
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
        "engines": config.engines
    }

@app.get("/api/benchmarks/info")
async def get_benchmarks_info():
    """Returns information about deepfake evaluation benchmarks (FaceForensics++ & DFDC)."""
    return {
        "evaluation_guidelines": "Official academic datasets used for deepfake model validation prior to deployment.",
        "benchmarks": config.benchmarks
    }

@app.post("/api/scan/media", response_model=MediaScanResponse)
async def scan_media_json(request: MediaScanRequest):
    """Scans media file by URL or parameter request."""
    return deepfake_engine.scan_media(request)

@app.post("/api/scan/media-file", response_model=MediaScanResponse)
async def scan_media_file(
    file: UploadFile = File(...),
    sampling_fps: float = Form(2.0),
    extract_audio: bool = Form(True)
):
    """Accepts uploaded video/audio file and performs deepfake forensic inspection."""
    file_bytes = await file.read()
    max_mb = config.server.get("max_upload_mb", 100)
    if len(file_bytes) > (max_mb * 1024 * 1024):
        raise HTTPException(status_code=413, detail=f"File exceeds maximum upload size of {max_mb}MB.")

    req = MediaScanRequest(sampling_fps=sampling_fps, extract_audio=extract_audio)
    res = deepfake_engine.scan_media(req, file_bytes=file_bytes, filename=file.filename)
    
    # Broadcast alert if manipulated
    if res.is_deepfake:
        alert_engine.synthesize(media_res=res)
        
    return res

@app.post("/api/scan/message", response_model=MessageScanResponse)
async def scan_message(request: MessageScanRequest):
    """Scans chat/social advisory message for unregistered advice and impersonation."""
    res = impersonation_engine.scan_message(request)
    if res.is_suspicious_advisory:
        alert_engine.synthesize(advisory_res=res)
    return res

@app.post("/api/scan/app", response_model=AppScanResponse)
async def scan_app(request: AppScanRequest):
    """Scans domain URL or Android APK permissions for broker spoofing or credential theft."""
    res = trading_app_engine.scan_infrastructure(request)
    if res.is_fraudulent:
        alert_engine.synthesize(infra_res=res)
    return res

@app.post("/api/scan/comprehensive", response_model=ComprehensiveScanResponse)
async def scan_comprehensive(request: ComprehensiveScanRequest):
    """Executes multi-vector scan synthesizing findings across media, text, and infrastructure."""
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

    return alert_engine.synthesize(
        media_res=media_res,
        advisory_res=advisory_res,
        infra_res=infra_res
    )

@app.get("/api/alerts/recent", response_model=List[AlertNotification])
async def get_recent_alerts(limit: int = 20):
    """Returns real-time risk alert stream."""
    return alert_engine.get_recent_alerts(limit=limit)

@app.post("/api/report/generate-dossier", response_model=SebiScoresDossier)
async def generate_dossier(request: ComprehensiveScanRequest):
    """Generates an official pre-formatted SEBI SCORES Complaint Dossier."""
    comp = await scan_comprehensive(request)
    if comp.scores_dossier:
        return comp.scores_dossier
    # If clean, generate standard informative verification record
    return SebiScoresDossier(
        dossier_id="SCORES-CLEAN-AUDIT",
        generated_at=comp.timestamp,
        alleged_entity_or_handle=request.app_label or request.package_name or "Audit-Subject",
        regulatory_acts_cited=["SEBI Investor Protection Guidelines"],
        factual_summary="Forensic evaluation completed. No regulatory violations detected.",
        evidence_points=["Asset matches verified registry credentials or within safe thresholds."],
        recommended_sebi_action="No regulatory enforcement action needed."
    )
