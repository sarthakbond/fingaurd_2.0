import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from src.core.config import config
from src.core.logging import logger
from src.core.security import SecurityShield
from src.services.base import BaseImpersonationDetector
from src.models.advisory import MessageScanRequest, MessageScanResponse, AdvisorClaim

class ImpersonationEngine(BaseImpersonationDetector):
    """
    Impersonation Accounts and Suspicious Advisory Message Detection Engine.
    Detects high-pressure solicitation, guaranteed returns, Hinglish fraud slang,
    and cross-references claimed credentials against the SEBI Registered Intermediary Roster.
    """

    GUARANTEED_RETURN_PATTERNS = [
        r"guaranteed\s+(?:\d+%\s+)?returns?",
        r"100%\s+(?:profit|sure|loss\s+recovery)",
        r"zero\s+(?:risk|loss)",
        r"paisa\s+double",
        r"fixed\s+monthly\s+income",
        r"double\s+your\s+capital",
        r"no\s+loss\s+strategy",
        r"risk[\-\s]free\s+trading"
    ]

    URGENCY_PATTERNS = [
        r"only\s+\d+\s+slots?\s+left",
        r"join\s+(?:vip\s+)?group\s+now",
        r"limited\s+time\s+offer",
        r"hurry\s+up",
        r"offer\s+ends\s+tonight",
        r"act\s+fast",
        r"exclusive\s+access"
    ]

    UNREGISTERED_SOLICITATION_PATTERNS = [
        r"dm\s+(?:to\s+)?(?:mirror|copy)",
        r"copy\s+trading",
        r"automated\s+(?:bot|algo)\s+execution",
        r"hand\s+over\s+your\s+account",
        r"profit\s+sharing\s+basis",
        r"send\s+funds\s+directly",
        r"crypto\s+usdt\s+deposit",
        r"telegram\s+vip\s+channel"
    ]

    HINGLISH_SLANG = [
        "pakka jackpot",
        "sure shot call",
        "rocket call",
        "paisa double",
        "loss recover hoga",
        "operator call",
        "internal leak",
        "super blast trade"
    ]

    SEBI_REG_REGEX = re.compile(r"\b(IN[A-Z]{1,3}\d{6,12})\b", re.IGNORECASE)

    def __init__(self):
        self.registry_data: List[Dict[str, Any]] = self._load_registry()
        self.advisory_threshold = float(config.thresholds.get("advisory_risk_threshold", 0.60))
        self.urgency_threshold = float(config.thresholds.get("urgency_threshold", 0.50))
        self.guarantee_threshold = float(config.thresholds.get("guarantee_threshold", 0.70))
        logger.info(f"Initialized ImpersonationEngine with {len(self.registry_data)} SEBI registry records.")

    def _load_registry(self) -> List[Dict[str, Any]]:
        reg_path = Path(config.engines.get("impersonation_engine", {}).get("registry_path", "data/sebi_registry.json"))
        if not reg_path.exists():
            logger.warning(f"SEBI registry file not found at {reg_path}")
            return []
        try:
            with open(reg_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load SEBI registry: {e}")
            return []

    def _calculate_pattern_score(self, text: str, patterns: List[str]) -> Tuple[float, List[str]]:
        matches = []
        lower = text.lower()
        for pat in patterns:
            found = re.findall(pat, lower)
            if found:
                matches.extend(found)
        score = min(len(matches) * 0.35, 1.0)
        return round(score, 2), matches

    def _detect_hinglish(self, text: str) -> List[str]:
        lower = text.lower()
        found = [s for s in self.HINGLISH_SLANG if s in lower]
        return found

    def _extract_and_verify_advisor(self, text: str) -> AdvisorClaim:
        reg_match = self.SEBI_REG_REGEX.search(text)
        claimed_num = reg_match.group(1).upper() if reg_match else None

        # Look for entity mentions against registry
        matched_rec = None
        highest_score = 0.0

        for rec in self.registry_data:
            rec_name = rec.get("name", "").lower()
            rec_aliases = [a.lower() for a in rec.get("aliases", [])]
            rec_num = rec.get("registration_number", "").upper()

            # Direct number match
            if claimed_num and claimed_num == rec_num:
                matched_rec = rec
                highest_score = 100.0
                break

            # Keyword match
            for alias in rec_aliases:
                if alias in text.lower():
                    matched_rec = rec
                    highest_score = 85.0
                    break

        if not claimed_num and not matched_rec:
            return AdvisorClaim(registry_status="NOT_CLAIMED")

        if claimed_num and matched_rec:
            if claimed_num == matched_rec.get("registration_number", "").upper():
                return AdvisorClaim(
                    claimed_name=matched_rec.get("name"),
                    claimed_reg_number=claimed_num,
                    matched_entity_name=matched_rec.get("name"),
                    matched_reg_number=claimed_num,
                    match_score=highest_score,
                    registry_status="VERIFIED"
                )
            else:
                return AdvisorClaim(
                    claimed_reg_number=claimed_num,
                    matched_entity_name=matched_rec.get("name"),
                    matched_reg_number=matched_rec.get("registration_number"),
                    match_score=highest_score,
                    registry_status="MISMATCH"
                )

        if claimed_num and not matched_rec:
            return AdvisorClaim(
                claimed_reg_number=claimed_num,
                registry_status="UNREGISTERED",
                match_score=0.0
            )

        return AdvisorClaim(
            claimed_name=matched_rec.get("name"),
            matched_entity_name=matched_rec.get("name"),
            matched_reg_number=matched_rec.get("registration_number"),
            registry_status="VERIFIED",
            match_score=highest_score
        )

    def scan_message(self, request: MessageScanRequest) -> MessageScanResponse:
        raw = request.message_text
        sanitized, redacted_types = SecurityShield.redact_pii(raw)

        g_score, g_phrases = self._calculate_pattern_score(sanitized, self.GUARANTEED_RETURN_PATTERNS)
        u_score, u_phrases = self._calculate_pattern_score(sanitized, self.URGENCY_PATTERNS)
        s_score, s_phrases = self._calculate_pattern_score(sanitized, self.UNREGISTERED_SOLICITATION_PATTERNS)
        hinglish = self._detect_hinglish(sanitized)

        # Composite risk: Noisy-OR blend of financial scam indicators
        composite = 1.0 - ((1.0 - g_score * 0.9) * (1.0 - u_score * 0.4) * (1.0 - s_score * 0.85))
        if hinglish:
            composite = min(composite + 0.20, 1.0)

        advisor_claim = self._extract_and_verify_advisor(raw)
        if advisor_claim.registry_status in ["UNREGISTERED", "MISMATCH"]:
            composite = max(composite, 0.85)

        is_suspicious = composite >= self.advisory_threshold

        violations = []
        if g_score > 0:
            violations.append("SEBI (Investment Advisers) Regulations 2013: Prohibition of guaranteed or risk-free returns.")
        if s_score > 0:
            violations.append("SEBI (Portfolio Managers) Regulations 2020: Unregistered copy-trading / discretionary execution solicitation.")
        if advisor_claim.registry_status == "UNREGISTERED":
            violations.append("SEBI Act 1992, Section 12(1): Operating investment advisory services without SEBI certification.")
        elif advisor_claim.registry_status == "MISMATCH":
            violations.append("SEBI PFUTP Regulations 2003: Impersonation / fraudulent use of a third-party license number.")

        rec = (
            "CRITICAL: Do not transfer capital or follow trading tips. Report to SEBI SCORES immediately."
            if is_suspicious else
            "Informational / low-risk content. Always verify SEBI credentials before investing."
        )

        return MessageScanResponse(
            original_length=len(raw),
            sanitized_text=sanitized,
            redacted_pii=redacted_types,
            platform=request.platform,
            is_suspicious_advisory=is_suspicious,
            composite_risk_score=round(composite, 2),
            urgency_score=u_score,
            guarantee_score=g_score,
            unregistered_solicitation_score=s_score,
            advisor_claim=advisor_claim,
            detected_phrases=g_phrases + u_phrases + s_phrases,
            detected_hinglish_slang=hinglish,
            violations=violations,
            recommendation=rec
        )
