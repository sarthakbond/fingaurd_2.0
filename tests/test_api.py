from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

def test_api_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "HEALTHY"
    assert "deepfake_engine" in data["engines"]
    assert "impersonation_engine" in data["engines"]
    assert "trading_app_engine" in data["engines"]

def test_api_config():
    res = client.get("/api/config")
    assert res.status_code == 200
    data = res.json()
    assert "thresholds" in data
    assert "benchmarks" in data

def test_api_benchmarks_info():
    res = client.get("/api/benchmarks/info")
    assert res.status_code == 200
    data = res.json()
    assert "FaceForensics++" in str(data)
    assert "DFDC" in str(data)

def test_api_scan_message():
    payload = {
        "message_text": "Guaranteed 100% daily profit! DM to mirror trades. Reg: INA999999",
        "platform": "whatsapp"
    }
    res = client.post("/api/scan/message", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["is_suspicious_advisory"] is True
    assert data["composite_risk_score"] > 0.5

def test_api_scan_app():
    payload = {
        "target_url": "https://groww-bonus-reward.xyz",
        "declared_permissions": ["READ_SMS", "RECEIVE_SMS", "BIND_ACCESSIBILITY_SERVICE"]
    }
    res = client.post("/api/scan/app", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["is_fraudulent"] is True
    assert data["whitelist_status"] == "IMPERSONATION_ALERT"

def test_api_scan_comprehensive():
    payload = {
        "message_text": "Risk-free return on trading tips! Join VIP Telegram channel.",
        "target_url": "https://zerodha-bonus.xyz"
    }
    res = client.post("/api/scan/comprehensive", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["final_verdict"] in ["CRITICAL_FRAUD", "SUSPICIOUS"]
    assert len(data["synthesized_alerts"]) > 0

def test_api_alerts_recent():
    res = client.get("/api/alerts/recent")
    assert res.status_code == 200
    assert isinstance(res.json(), list)

def test_api_generate_dossier():
    payload = {
        "message_text": "Guaranteed 100% profit rocket calls! Call +91 9999999999",
        "app_label": "Scam Broker App"
    }
    res = client.post("/api/report/generate-dossier", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "SCORES" in data["dossier_id"]
    assert len(data["regulatory_acts_cited"]) > 0
