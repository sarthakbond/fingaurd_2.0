import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import urlparse
from src.core.config import config
from src.core.logging import logger
from src.services.base import BaseTradingAppDetector
from src.models.infrastructure import (
    AppScanRequest,
    AppScanResponse,
    DomainScanResult,
    ApkPermissionAnalysis
)

def compute_levenshtein(s1: str, s2: str) -> int:
    """Computes Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return compute_levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

class TradingAppEngine(BaseTradingAppDetector):
    """
    Fake Trading App & Phishing Infrastructure Scanner.
    Detects brand typo-squatting against official SEBI broker domains
    and inspects Android APK permissions for malicious credential & OTP theft capabilities.
    """

    SUSPICIOUS_TLDS = {".xyz", ".top", ".club", ".vip", ".click", ".live", ".work", ".link", ".apk"}

    HIGH_RISK_PERMISSIONS = {
        "READ_SMS": "Intercepts incoming 2FA authentication SMS OTPs.",
        "RECEIVE_SMS": "Silently reads incoming banking confirmation messages.",
        "BIND_ACCESSIBILITY_SERVICE": "Can click buttons, read screen contents, and bypass user consent.",
        "PACKAGE_USAGE_STATS": "Tracks when legitimate banking/brokerage apps are opened.",
        "SYSTEM_ALERT_WINDOW": "Displays deceptive full-screen login overlays over genuine apps.",
        "READ_CONTACTS": "Harvests contact lists for extortion and aggressive social engineering."
    }

    def __init__(self):
        self.whitelist: List[Dict[str, Any]] = self._load_whitelist()
        self.max_lev_dist = int(config.thresholds.get("domain_max_levenshtein_distance", 3))
        logger.info(f"Initialized TradingAppEngine with {len(self.whitelist)} whitelisted brokers.")

    def _load_whitelist(self) -> List[Dict[str, Any]]:
        path = Path(config.engines.get("trading_app_engine", {}).get("whitelist_path", "data/broker_whitelist.json"))
        if not path.exists():
            logger.warning(f"Broker whitelist file not found at {path}")
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load broker whitelist: {e}")
            return []

    def _extract_domain(self, target: str) -> str:
        cleaned = target.strip()
        if not cleaned.startswith(("http://", "https://")):
            cleaned = "https://" + cleaned
        parsed = urlparse(cleaned)
        domain = parsed.netloc.lower() or parsed.path.lower()
        if ":" in domain:
            domain = domain.split(":")[0]
        if domain.startswith("www."):
            domain = domain[4:]
        return domain

    def scan_domain(self, target_url: str) -> DomainScanResult:
        domain = self._extract_domain(target_url)
        reasons = []

        # Check exact whitelist
        for b in self.whitelist:
            for official_dom in b.get("domains", []):
                if domain == official_dom or domain.endswith("." + official_dom):
                    return DomainScanResult(
                        input_domain=domain,
                        is_phishing_suspect=False,
                        impersonated_broker=b.get("broker"),
                        matched_official_domain=official_dom,
                        levenshtein_distance=0,
                        risk_score=0.0,
                        reasons=["Exact match with verified official broker domain."]
                    )

        # Typo-squatting & Brand spoofing analysis
        closest_broker = None
        closest_official = None
        min_dist = 999
        is_spoof = False

        for b in self.whitelist:
            b_name = b.get("broker", "").lower()
            for official_dom in b.get("domains", []):
                # Check Levenshtein distance on base domain name
                base_domain = domain.split(".")[0]
                base_official = official_dom.split(".")[0]

                dist = compute_levenshtein(base_domain, base_official)
                if dist < min_dist:
                    min_dist = dist
                    closest_broker = b.get("broker")
                    closest_official = official_dom

                # Check compound keyword fraud: "zerodha-login", "groww-invest-bonus"
                if base_official in domain and domain != official_dom:
                    is_spoof = True
                    reasons.append(f"Contains registered broker trademark '{base_official}' in an unauthorized hostname.")

        # Check suspicious TLD
        for tld in self.SUSPICIOUS_TLDS:
            if domain.endswith(tld):
                reasons.append(f"Uses high-risk fraud top-level domain ({tld}).")
                break

        if min_dist <= self.max_lev_dist and min_dist > 0:
            is_spoof = True
            reasons.append(f"Typo-squatting detected: {min_dist} character edits from '{closest_official}'.")

        risk_score = 0.90 if is_spoof else (0.45 if reasons else 0.10)

        return DomainScanResult(
            input_domain=domain,
            is_phishing_suspect=is_spoof,
            impersonated_broker=closest_broker if is_spoof else None,
            matched_official_domain=closest_official if is_spoof else None,
            levenshtein_distance=min_dist if is_spoof else -1,
            risk_score=round(risk_score, 2),
            reasons=reasons or ["No known broker impersonation identified."]
        )

    def scan_apk_permissions(self, pkg_name: Optional[str], perms: List[str]) -> Tuple[ApkPermissionAnalysis, str]:
        normalized_perms = [p.split(".")[-1].upper() for p in (perms or [])]
        found_high_risk = []

        for p_norm, explanation in self.HIGH_RISK_PERMISSIONS.items():
            if any(p_norm in user_p for user_p in normalized_perms):
                found_high_risk.append(f"{p_norm}: {explanation}")

        has_sms = any("SMS" in p for p in normalized_perms)
        has_acc = any("ACCESSIBILITY" in p for p in normalized_perms)
        has_overlay = any("SYSTEM_ALERT" in p or "OVERLAY" in p for p in normalized_perms)

        # Check package name against whitelist
        status = "UNLISTED_THIRD_PARTY"
        if pkg_name:
            for b in self.whitelist:
                if pkg_name in b.get("package_names", []):
                    status = "OFFICIAL_WHITELISTED"
                    break

        if status != "OFFICIAL_WHITELISTED" and (has_sms or has_acc):
            status = "IMPERSONATION_ALERT"

        analysis = ApkPermissionAnalysis(
            declared_permissions_count=len(perms),
            high_risk_permissions=found_high_risk,
            has_sms_interception=has_sms,
            has_accessibility_hijacking=has_acc,
            has_overlay_permission=has_overlay
        )
        return analysis, status

    def scan_infrastructure(self, request: AppScanRequest) -> AppScanResponse:
        domain_res = None
        apk_res = None
        indicators = []
        whitelist_status = "UNLISTED_THIRD_PARTY"
        max_risk = 0.0

        target_identifier = request.target_url or request.package_name or request.app_label or "unknown_target"

        if request.target_url:
            domain_res = self.scan_domain(request.target_url)
            max_risk = max(max_risk, domain_res.risk_score)
            if domain_res.is_phishing_suspect:
                indicators.append(f"Domain impersonates {domain_res.impersonated_broker}: {', '.join(domain_res.reasons)}")
                whitelist_status = "IMPERSONATION_ALERT"
            elif domain_res.risk_score == 0.0:
                whitelist_status = "OFFICIAL_WHITELISTED"

        if request.declared_permissions or request.package_name:
            apk_res, apk_status = self.scan_apk_permissions(request.package_name, request.declared_permissions or [])
            if apk_status == "IMPERSONATION_ALERT":
                whitelist_status = "IMPERSONATION_ALERT"
                max_risk = max(max_risk, 0.88)
                indicators.append("App binary requests critical surveillance/OTP interception permissions without official broker certification.")
            elif apk_status == "OFFICIAL_WHITELISTED":
                if whitelist_status != "IMPERSONATION_ALERT":
                    whitelist_status = "OFFICIAL_WHITELISTED"

        is_fraud = (max_risk >= 0.70) or (whitelist_status == "IMPERSONATION_ALERT")
        warning = (
            "DANGER: Unverified financial application/domain with potential for credential harvesting and OTP theft."
            if is_fraud else
            "Standard / verified broker asset. Verify app signatures before authorizing funds."
        )

        return AppScanResponse(
            scan_target=target_identifier,
            is_fraudulent=is_fraud,
            risk_score=round(max_risk, 2),
            domain_result=domain_res,
            apk_result=apk_res,
            whitelist_status=whitelist_status,
            indicators=indicators,
            warning_message=warning
        )
