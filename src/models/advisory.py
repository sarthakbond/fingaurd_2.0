from pydantic import BaseModel, Field
from typing import List, Optional

class AdvisorClaim(BaseModel):
    claimed_name: Optional[str] = None
    claimed_reg_number: Optional[str] = None
    matched_entity_name: Optional[str] = None
    matched_reg_number: Optional[str] = None
    match_score: float = Field(default=0.0, ge=0.0, le=100.0)
    registry_status: str = Field(
        default="NOT_CLAIMED",
        description="VERIFIED, UNREGISTERED, MISMATCH, NOT_CLAIMED, or INVALID_FORMAT"
    )

class MessageScanRequest(BaseModel):
    message_text: str = Field(..., min_length=2, description="Raw message text from WhatsApp, Telegram, etc.")
    platform: str = Field(default="generic", description="whatsapp, telegram, sms, instagram, youtube")
    sender_handle: Optional[str] = Field(None, description="Sender username, channel name, or phone number")

class MessageScanResponse(BaseModel):
    original_length: int
    sanitized_text: str
    redacted_pii: List[str] = Field(default_factory=list)
    platform: str
    is_suspicious_advisory: bool
    composite_risk_score: float = Field(..., ge=0.0, le=1.0)
    urgency_score: float = Field(..., ge=0.0, le=1.0)
    guarantee_score: float = Field(..., ge=0.0, le=1.0)
    unregistered_solicitation_score: float = Field(..., ge=0.0, le=1.0)
    advisor_claim: AdvisorClaim
    detected_phrases: List[str] = Field(default_factory=list)
    detected_hinglish_slang: List[str] = Field(default_factory=list)
    violations: List[str] = Field(default_factory=list)
    recommendation: str
