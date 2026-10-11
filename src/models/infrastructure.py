from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class DomainScanResult(BaseModel):
    input_domain: str
    is_phishing_suspect: bool
    impersonated_broker: Optional[str] = None
    levenshtein_distance: int = Field(default=-1)
    matched_official_domain: Optional[str] = None
    risk_score: float = Field(..., ge=0.0, le=1.0)
    reasons: List[str] = Field(default_factory=list)

class ApkPermissionAnalysis(BaseModel):
    declared_permissions_count: int
    high_risk_permissions: List[str] = Field(default_factory=list)
    has_sms_interception: bool = False
    has_accessibility_hijacking: bool = False
    has_overlay_permission: bool = False

class AppScanRequest(BaseModel):
    target_url: Optional[str] = Field(None, description="Domain or URL to scan for broker impersonation")
    package_name: Optional[str] = Field(None, description="Android APK package name")
    app_label: Optional[str] = Field(None, description="Claimed application display name")
    declared_permissions: Optional[List[str]] = Field(default_factory=list, description="List of Android permissions")

class AppScanResponse(BaseModel):
    scan_target: str
    is_fraudulent: bool
    risk_score: float = Field(..., ge=0.0, le=1.0)
    domain_result: Optional[DomainScanResult] = None
    apk_result: Optional[ApkPermissionAnalysis] = None
    whitelist_status: str = Field(..., description="OFFICIAL_WHITELISTED, UNLISTED_THIRD_PARTY, or IMPERSONATION_ALERT")
    indicators: List[str] = Field(default_factory=list)
    warning_message: str
