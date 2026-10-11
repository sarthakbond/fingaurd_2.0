# FinGuard 2.0 — Real-Time Investor Fraud & Impersonation Detection System

> **Primary Source of Truth**: Hackathon 2026 — Engineering Track Problem Statement #7 (Domain: Fintech)  
> **Mandate**: Build a real-time detection and alert system for securities-market fraud — flagging deepfake media, impersonation accounts, and fake trading-app indicators — aligned with SEBI's investor-protection mandate.  
> **Current Status**: **Phase 1: Working Application Foundation**

---

## 1. Architectural Mapping to Official Requirements

FinGuard 2.0 maps directly to the three core requirements specified in Problem Statement #7:

```
                                  +-------------------------------------------------------------+
                                  |                 FinGuard 2.0 Ingestion Bus                  |
                                  +-------------------------------------------------------------+
                                                                 |
              +----------------------------------+---------------+----------------------------------+
              |                                  |                                                  |
              v                                  v                                                  v
+-------------------------------+  +-------------------------------+  +-------------------------------+
|  Requirement 1: Media Engine  |  | Requirement 2: Advisory Engine|  | Requirement 3: App/Infra Eng |
|  - Video/Audio Preprocessing  |  |  - DPDP Act 2023 PII Shield   |  |  - Domain Typo-Squatting Scan |
|  - Face Extraction / Acoustic |  |  - Guaranteed Returns Detector|  |  - Levenshtein Broker Match   |
|  - Timeline Risk Scrubber     |  |  - Hinglish Colloquialisms    |  |  - Android APK Permission Chk |
|  - Benchmarked on FF++ / DFDC |  |  - SEBI Intermediary Registry |  |  - Surveillance / OTP Threat  |
+-------------------------------+  +-------------------------------+  +-------------------------------+
              |                                  |                                                  |
              +----------------------------------+---------------+----------------------------------+
                                                                 |
                                                                 v
                                  +-------------------------------------------------------------+
                                  |           Alert & Multi-Factor Decision Synthesizer         |
                                  |  - Calibrated 3-Tier Verdict (Safe / Suspicious / Fraud)    |
                                  |  - Real-Time Push Event Feed & WebSocket/SSE Stream         |
                                  |  - Pre-Formatted SEBI SCORES Complaint Dossier Generator   |
                                  +-------------------------------------------------------------+
```

### Requirement 1: Deepfake Video & Audio Detection
- **Architectural Mapping**: [`src/services/deepfake_engine.py`](file:///e:/fingaurd2.0/src/services/deepfake_engine.py) & [`src/models/media.py`](file:///e:/fingaurd2.0/src/models/media.py).
- **Functionality**: Evaluates visual and acoustic manipulation probability across media files with an interactive timeline scrubber.
- **Academic Benchmark Alignment**: Built to integrate and evaluate models benchmarked against **FaceForensics++** (methods: Deepfakes, Face2Face, FaceSwap, NeuralTextures across c23/c40 compression) and the **DFDC (Deepfake Detection Challenge)** dataset. Candidate backbones (`EfficientNet-B4`, `Xception`, `Wav2Vec2`) are evaluated before weight integration.

### Requirement 2: Impersonation Accounts & Advisory Messages (WhatsApp / Telegram)
- **Architectural Mapping**: [`src/services/impersonation_engine.py`](file:///e:/fingaurd2.0/src/services/impersonation_engine.py) & [`src/models/advisory.py`](file:///e:/fingaurd2.0/src/models/advisory.py).
- **Functionality**: Analyzes social messaging text for illegal investment advice patterns, promised guaranteed returns (*"100% loss recovery"*, *"paisa double"*), artificial urgency (*"only 5 slots left"*), and coded Hinglish colloquialisms (*"pakka jackpot"*, *"rocket call"*).
- **Regulatory Verification**: Sanitizes sensitive data under the **Digital Personal Data Protection (DPDP) Act 2023** and verifies claimed registration numbers against the official **SEBI Registered Intermediary Roster** ([`data/sebi_registry.json`](file:///e:/fingaurd2.0/data/sebi_registry.json)).

### Requirement 3: Fake Trading Applications & Fraudulent Intermediaries
- **Architectural Mapping**: [`src/services/infrastructure_engine.py`](file:///e:/fingaurd2.0/src/services/infrastructure_engine.py) & [`src/models/infrastructure.py`](file:///e:/fingaurd2.0/src/models/infrastructure.py).
- **Functionality**: Detects typo-squatted phishing domains spoofing legitimate brokers (e.g., Zerodha, Groww, AngelOne) using Levenshtein distance calculations. Inspects Android APK permissions for dangerous surveillance capabilities such as 2FA OTP interception (`READ_SMS`, `RECEIVE_SMS`) and screen manipulation (`BIND_ACCESSIBILITY_SERVICE`, `SYSTEM_ALERT_WINDOW`).

---

## 2. Phase 1 Implementation Scope

In strict accordance with guidelines:
- **Core Focus**: Clean, production-grade application foundation.
- **Modular Contracts**: Abstract interfaces (`BaseDeepfakeDetector`, `BaseImpersonationDetector`, `BaseTradingAppDetector`, `BaseAlertSynthesizer`) allow seamless plugging of heavy deep learning weights in later phases without refactoring the application.
- **Intellectual Integrity**: No fabricated accuracy claims or unverified metrics. Benchmark evaluation standards (FaceForensics++ and DFDC) are documented in [`benchmarks/BENCHMARKS.md`](file:///e:/fingaurd2.0/benchmarks/BENCHMARKS.md).
- **Working Foundation**: Full FastAPI backend, complete Pydantic data schemas, reference data files, unit test suite, and a modern glassmorphic web dashboard.

---

## 3. Directory Structure

```
fingaurd2.0/
├── config.yaml                     # System configuration & thresholds
├── requirements.txt                # Python environment dependencies
├── pytest.ini                      # Test runner configuration
├── main.py                         # Application entrypoint runner
├── data/
│   ├── sebi_registry.json          # SEBI registered intermediary roster
│   └── broker_whitelist.json       # Whitelisted broker domains & package IDs
├── benchmarks/
│   └── BENCHMARKS.md               # FaceForensics++ & DFDC evaluation protocol
├── src/
│   ├── core/
│   │   ├── config.py               # Typed YAML configuration manager
│   │   ├── logging.py              # Structured logging utility
│   │   └── security.py             # DPDP Act PII redactor & input sanitizer
│   ├── models/
│   │   ├── media.py                # Video/audio deepfake data contracts
│   │   ├── advisory.py             # Chat & impersonation data contracts
│   │   ├── infrastructure.py       # Domain & APK analysis data contracts
│   │   └── verdict.py              # Multi-factor verdict & alert models
│   ├── services/
│   │   ├── base.py                 # Abstract engine interfaces
│   │   ├── deepfake_engine.py      # Video & audio forensic baseline engine
│   │   ├── impersonation_engine.py # Chat & advisory compliance engine
│   │   ├── infrastructure_engine.py# Domain & APK threat auditor
│   │   └── alert_engine.py         # Verdict calibration & SEBI SCORES dossier
│   └── api/
│       └── app.py                  # FastAPI application & REST endpoints
├── static/
│   ├── index.html                  # FinGuard command center dashboard
│   ├── css/
│   │   └── styles.css              # Glassmorphic dark design system
│   └── js/
│       └── app.js                  # Dynamic client-side reactive controller
└── tests/
    ├── test_config.py              # Configuration & benchmark validation
    ├── test_engines.py             # Modular forensic engines validation
    └── test_api.py                 # REST API endpoints & data contract tests
```

---

## 4. Quickstart Guide

### 1. Activate Environment & Run Tests
```powershell
# Run the test suite
.\.venv\Scripts\python.exe -m pytest -v
```

### 2. Start FinGuard 2.0 Server
```powershell
# Launch the ASGI server
.\.venv\Scripts\python.exe main.py
```

### 3. Access Web Dashboard & API Docs
- **Web Command Center**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Interactive Swagger Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
