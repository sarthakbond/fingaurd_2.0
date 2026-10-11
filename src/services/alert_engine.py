import uuid
from datetime import datetime, timezone
from typing import List, Optional
from collections import deque
from src.core.config import config
from src.core.logging import logger
from src.services.base import BaseAlertSynthesizer
from src.models.media import MediaScanResponse
from src.models.advisory import MessageScanResponse
from src.models.infrastructure import AppScanResponse
from src.models.verdict import (
    ComprehensiveScanResponse,
    AlertNotification,
    SebiScoresDossier
)

class AlertEngine(BaseAlertSynthesizer):
    """
    Multi-Factor Decision Matrix, Real-Time Alert Dispatcher,
    and Regulatory SEBI SCORES Complaint Dossier Generator.
    """

    def __init__(self):
        max_alerts = int(config.engines.get("alert_engine", {}).get("max_recent_alerts", 100))
        self.recent_alerts = deque(maxlen=max_alerts)
        self.crit_threshold = float(config.thresholds.get("verdict_critical_threshold", 0.70))
        self.susp_threshold = float(config.thresholds.get("verdict_suspicious_threshold", 0.40))
        logger.info(f"Initialized AlertEngine (ring buffer size: {max_alerts}).")

    def _generate_scores_dossier(
        self,
        media_res: Optional[MediaScanResponse],
        advisory_res: Optional[MessageScanResponse],
        infra_res: Optional[AppScanResponse],
        composite_risk: float
    ) -> SebiScoresDossier:
        dossier_id = f"SCORES-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
        ts = datetime.now(timezone.utc).isoformat()

        acts = []
        evidence = []
        target = "Unknown Fraudulent Entity"

        if media_res and media_res.is_deepfake:
            acts.append("SEBI PFUTP Regulations, 2003 (Prohibition of Fraudulent and Unfair Trade Practices) — Deepfake impersonation")
            evidence.append(f"Synthetic media detected with visual score {media_res.visual_manipulation_score} and audio score {media_res.acoustic_manipulation_score}.")
            if media_res.filename:
                target = media_res.filename

        if advisory_res:
            if advisory_res.is_suspicious_advisory or advisory_res.composite_risk_score >= self.susp_threshold:
                acts.append("SEBI (Investment Advisers) Regulations, 2013, Reg 3(1) — Unregistered financial advisory")
                if advisory_res.guarantee_score > 0 or advisory_res.unregistered_solicitation_score > 0:
                    acts.append("SEBI Code of Conduct for Intermediaries — Illegal promise of guaranteed / risk-free returns")
                if advisory_res.advisor_claim and advisory_res.advisor_claim.claimed_name:
                    target = advisory_res.advisor_claim.claimed_name
                evidence.extend(advisory_res.violations)
                if advisory_res.detected_phrases:
                    evidence.append(f"High-pressure deceptive phrases flagged: {', '.join(advisory_res.detected_phrases[:3])}")
                if advisory_res.detected_hinglish_slang:
                    evidence.append(f"Coded colloquial fraud slang flagged: {', '.join(advisory_res.detected_hinglish_slang)}")

        if infra_res and infra_res.is_fraudulent:
            acts.append("SEBI Act, 1992, Section 12(1) — Operating unauthorized securities execution channel")
            evidence.extend(infra_res.indicators)
            target = infra_res.scan_target

        summary = (
            f"Forensic investigation flagged digital securities fraud with composite risk {composite_risk:.2f}. "
            f"Target entity '{target}' operates deceptive materials violating SEBI investor protection mandates."
        )

        recommended = (
            "1. Issue cease-and-desist order under Section 11D of SEBI Act.\n"
            "2. Direct telecommunication and hosting intermediaries to take down malicious channels/domains.\n"
            "3. Notify cyber crime authorities for financial asset freezing."
        )

        return SebiScoresDossier(
            dossier_id=dossier_id,
            generated_at=ts,
            alleged_entity_or_handle=target,
            regulatory_acts_cited=list(set(acts)),
            factual_summary=summary,
            evidence_points=evidence,
            recommended_sebi_action=recommended
        )

    def synthesize(
        self,
        media_res: Optional[MediaScanResponse] = None,
        advisory_res: Optional[MessageScanResponse] = None,
        infra_res: Optional[AppScanResponse] = None
    ) -> ComprehensiveScanResponse:
        scan_id = str(uuid.uuid4())
        ts = datetime.now(timezone.utc).isoformat()
        alerts: List[AlertNotification] = []

        scores: List[float] = []
        if media_res:
            m_score = max(media_res.visual_manipulation_score, media_res.acoustic_manipulation_score)
            scores.append(m_score)
            if media_res.is_deepfake:
                alert = AlertNotification(
                    alert_id=str(uuid.uuid4()),
                    timestamp=ts,
                    severity="CRITICAL",
                    category="DEEPFAKE_MEDIA",
                    summary=f"Deepfake media detected ({media_res.media_type})",
                    risk_score=m_score,
                    source=media_res.filename or "Media Upload",
                    action_required="Block media dissemination and warn investor audience."
                )
                alerts.append(alert)
                self.recent_alerts.append(alert)

        if advisory_res:
            scores.append(advisory_res.composite_risk_score)
            if advisory_res.is_suspicious_advisory:
                severity = "CRITICAL" if advisory_res.composite_risk_score >= 0.75 else "WARNING"
                alert = AlertNotification(
                    alert_id=str(uuid.uuid4()),
                    timestamp=ts,
                    severity=severity,
                    category="SUSPICIOUS_ADVISORY",
                    summary=f"High-risk advisory message detected on {advisory_res.platform}",
                    risk_score=advisory_res.composite_risk_score,
                    source=advisory_res.platform,
                    action_required="Do not authorize capital transfer. Report channel to SEBI SCORES."
                )
                alerts.append(alert)
                self.recent_alerts.append(alert)

        if infra_res:
            scores.append(infra_res.risk_score)
            if infra_res.is_fraudulent:
                alert = AlertNotification(
                    alert_id=str(uuid.uuid4()),
                    timestamp=ts,
                    severity="CRITICAL",
                    category="FAKE_TRADING_APP" if infra_res.apk_result else "PHISHING_DOMAIN",
                    summary=f"Fraudulent infrastructure detected: {infra_res.scan_target}",
                    risk_score=infra_res.risk_score,
                    source=infra_res.scan_target,
                    action_required="Block domain / uninstall APK and revoke SMS/accessibility permissions."
                )
                alerts.append(alert)
                self.recent_alerts.append(alert)

        composite_risk = max(scores) if scores else 0.0

        if composite_risk >= self.crit_threshold:
            verdict = "CRITICAL_FRAUD"
        elif composite_risk >= self.susp_threshold:
            verdict = "SUSPICIOUS"
        else:
            verdict = "SAFE"

        dossier = None
        if verdict in ["CRITICAL_FRAUD", "SUSPICIOUS"]:
            dossier = self._generate_scores_dossier(media_res, advisory_res, infra_res, composite_risk)

        return ComprehensiveScanResponse(
            scan_id=scan_id,
            timestamp=ts,
            final_verdict=verdict,
            composite_risk_score=round(composite_risk, 2),
            media_findings=media_res,
            advisory_findings=advisory_res,
            infrastructure_findings=infra_res,
            synthesized_alerts=alerts,
            scores_dossier=dossier
        )

    def get_recent_alerts(self, limit: int = 20) -> List[AlertNotification]:
        return list(self.recent_alerts)[-limit:]
