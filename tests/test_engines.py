import pytest
from src.services.deepfake_engine import DeepfakeEngine
from src.services.impersonation_engine import ImpersonationEngine
from src.services.infrastructure_engine import TradingAppEngine, compute_levenshtein
from src.services.alert_engine import AlertEngine
from src.models.media import MediaScanRequest
from src.models.advisory import MessageScanRequest
from src.models.infrastructure import AppScanRequest

def test_deepfake_engine_baseline():
    engine = DeepfakeEngine()
    req = MediaScanRequest(media_url="https://samples.test/video.mp4")
    res = engine.scan_media(req, filename="authentic_video.mp4")
    assert res.status == "COMPLETED"
    assert res.media_type == "video"
    assert len(res.benchmarks_evaluated) >= 2
    assert res.benchmarks_evaluated[0].benchmark_name == "FaceForensics++"

def test_impersonation_engine_guaranteed_returns():
    engine = ImpersonationEngine()
    req = MessageScanRequest(
        message_text="Pakka jackpot! Guaranteed 100% loss recovery call. DM to mirror automated algo trades. Contact +91 9876543210 immediately!",
        platform="whatsapp"
    )
    res = engine.scan_message(req)
    assert res.is_suspicious_advisory is True
    assert res.composite_risk_score >= 0.60
    assert "PHONE_NUMBER" in res.redacted_pii
    assert len(res.detected_hinglish_slang) > 0
    assert len(res.violations) > 0

def test_impersonation_engine_verified_broker():
    engine = ImpersonationEngine()
    req = MessageScanRequest(
        message_text="Official update from Zerodha Broking Limited (INZ000031633). Markets will remain closed on trading holiday.",
        platform="sms"
    )
    res = engine.scan_message(req)
    assert res.advisor_claim.registry_status == "VERIFIED"
    assert res.is_suspicious_advisory is False

def test_levenshtein_distance():
    assert compute_levenshtein("groww", "groww") == 0
    assert compute_levenshtein("groww", "grow") == 1
    assert compute_levenshtein("zerodha", "zerodhaa") == 1

def test_infrastructure_typosquatting():
    engine = TradingAppEngine()
    req = AppScanRequest(target_url="https://groww-bonus-reward.xyz/login")
    res = engine.scan_infrastructure(req)
    assert res.is_fraudulent is True
    assert res.whitelist_status == "IMPERSONATION_ALERT"
    assert res.domain_result.is_phishing_suspect is True

def test_infrastructure_official_broker():
    engine = TradingAppEngine()
    req = AppScanRequest(target_url="https://zerodha.com/kite", package_name="com.zerodha.kite3")
    res = engine.scan_infrastructure(req)
    assert res.is_fraudulent is False
    assert res.whitelist_status == "OFFICIAL_WHITELISTED"

def test_alert_engine_synthesis():
    alert_engine = AlertEngine()
    imp_engine = ImpersonationEngine()
    
    advisory_res = imp_engine.scan_message(
        MessageScanRequest(
            message_text="100% profit guaranteed rocket call! DM to mirror.",
            platform="telegram"
        )
    )
    comp = alert_engine.synthesize(advisory_res=advisory_res)
    assert comp.final_verdict in ["CRITICAL_FRAUD", "SUSPICIOUS"]
    assert comp.scores_dossier is not None
    assert "SEBI (Investment Advisers) Regulations, 2013" in str(comp.scores_dossier.regulatory_acts_cited)
