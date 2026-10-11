from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from src.models.media import MediaScanResponse
from src.models.advisory import MessageScanResponse
from src.models.infrastructure import AppScanResponse

class AlertNotification(BaseModel):
    alert_id: str
    timestamp: str
    severity: str = Field(..., description="INFO, WARNING, or CRITICAL")
    category: str = Field(..., description="DEEPFAKE_MEDIA, SUSPICIOUS_ADVISORY, FAKE_TRADING_APP, PHISHING_DOMAIN")
    summary: str
    risk_score: float = Field(..., ge=0.0, le=1.0)
    source: str
    action_required: str

class SebiScoresDossier(BaseModel):
    dossier_id: str
    generated_at: str
    alleged_entity_or_handle: str
    regulatory_acts_cited: List[str]
    factual_summary: str
    evidence_points: List[str]
    recommended_sebi_action: str

class ComprehensiveScanRequest(BaseModel):
    media_url: Optional[str] = None
    message_text: Optional[str] = None
    platform: Optional[str] = "unknown"
    target_url: Optional[str] = None
    package_name: Optional[str] = None
    app_label: Optional[str] = None

class ComprehensiveScanResponse(BaseModel):
    scan_id: str
    timestamp: str
    final_verdict: str = Field(..., description="SAFE, SUSPICIOUS, or CRITICAL_FRAUD")
    composite_risk_score: float = Field(..., ge=0.0, le=1.0)
    media_findings: Optional[MediaScanResponse] = None
    advisory_findings: Optional[MessageScanResponse] = None
    infrastructure_findings: Optional[AppScanResponse] = None
    synthesized_alerts: List[AlertNotification] = Field(default_factory=list)
    scores_dossier: Optional[SebiScoresDossier] = None
