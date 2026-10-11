# FinGuard 2.0 — Real-Time Investor Fraud & Impersonation Detection System

> **Primary Source of Truth**: Hackathon 2026 — Engineering Track Problem Statement #7 (Domain: Fintech)  
> **Mandate**: Build a real-time detection and alert system for securities-market fraud — flagging deepfake media, impersonation accounts, and fake trading-app indicators — aligned with SEBI's investor-protection mandate.  
> **Current Milestone**: **Phase 2A: Audio Deepfake Detection** (Completed & Verified)

---

## 1. Architectural Mapping to Official Requirements

FinGuard 2.0 addresses the three core requirements specified in Problem Statement #7:

```
                                  +-------------------------------------------------------------+
                                  |                 FinGuard Ingestion Gateway                  |
                                  +-------------------------------------------------------------+
                                                                 |
              +----------------------------------+---------------+----------------------------------+
              |                                  |                                                  |
              v                                  v                                                  v
+-------------------------------+  +-------------------------------+  +-------------------------------+
|  Requirement 1: Media Engine  |  | Requirement 2: Advisory Engine|  | Requirement 3: App/Infra Eng |
|  - Neural Wav2Vec2 Audio Model|  |  - DPDP Act 2023 PII Shield   |  |  - Domain Typo-Squatting Scan |
|  - RTX 4050 GPU Acceleration  |  |  - Guaranteed Returns Detector|  |  - Levenshtein Distance Match |
|  - Sliding-Window Chunking    |  |  - Hinglish Colloquialisms    |  |  - Android APK Permission Chk |
|  - Bounded Disk-Stream Upload |  |  - SEBI Intermediary Registry |  |  - Surveillance / OTP Threat  |
|  - Magic Byte Signature Check |  |                               |  |                               |
|  - Video: Phase 2B Baseline   |  |                               |  |                               |
+-------------------------------+  +-------------------------------+  +-------------------------------+
              |                                  |                                                  |
              +----------------------------------+---------------+----------------------------------+
                                                                 |
                                                                 v
                                  +-------------------------------------------------------------+
                                  |           Alert & Multi-Factor Decision Synthesizer         |
                                  |  - Calibrated 3-Tier Verdict (Safe / Suspicious / Fraud)    |
                                  |  - SQLite Persistence (SQLAlchemy: Scans, Jobs, Alerts)     |
                                  |  - Sliding-Window IP Rate Limiting (Configurable)           |
                                  |  - Pre-Formatted SEBI SCORES Complaint Dossier Generator   |
                                  +-------------------------------------------------------------+
```

### Requirement 1: Deepfake Video & Audio Detection
- **Architectural Mapping**: [`src/services/deepfake_engine.py`](file:///e:/fingaurd2.0/src/services/deepfake_engine.py), [`src/services/audio_deepfake_detector.py`](file:///e:/fingaurd2.0/src/services/audio_deepfake_detector.py), [`src/services/audio_preprocessor.py`](file:///e:/fingaurd2.0/src/services/audio_preprocessor.py) & [`src/models/media.py`](file:///e:/fingaurd2.0/src/models/media.py).
- **Phase 2A Implementation (Audio Track)**:
  - **Genuine Neural Model**: Integrates `Vansh180/deepfake-audio-wav2vec2` (`facebook/wav2vec2-base` backbone, ~95M parameters, `model.safetensors`).
  - **Hardware Execution**: Hardware-accelerated on **NVIDIA GeForce RTX 4050 Laptop GPU** (`cuda:0`) in FP16 precision (~1.1 GB VRAM, ~120–150 ms per chunk) with automatic CPU fallback.
  - **Signal Preprocessing & Chunking**: Automatic mono conversion and 16,000 Hz resampling via [`src/services/audio_preprocessor.py`](file:///e:/fingaurd2.0/src/services/audio_preprocessor.py). Processes audio using configurable 4.0-second sliding windows with 1.0-second overlap, bounded to 180 seconds to prevent resource exhaustion.
  - **Transparent Aggregation**: Returns both peak chunk spoof probability and mean score ($0.70 \times \text{peak} + 0.30 \times \text{mean}$) along with per-chunk timeline segments.
  - **Fail-Safe Integrity**: If PyTorch, model weights, or audio decoding fail, the engine explicitly returns `status="INFERENCE_UNAVAILABLE"` or `"INFERENCE_ERROR"` with `score=None` and `confidence="UNAVAILABLE"`. It **never** silently returns `0.0` or marks failed audio as safe.
  - **Training Basis & Documented Limitations**: The candidate model was fine-tuned on a balanced subset of **ASVspoof 2021 PA** (Physical Access / Replay & Synthetic voice attacks). It has not been domain-calibrated for modern commercial zero-shot neural vocoders (e.g. Tortoise, ElevenLabs) or compressed social media audio (e.g. WhatsApp Opus voice notes). All scores represent an initial research baseline.
- **Video Track (Pending Phase 2B)**: Preserved in Phase 1 baseline foundation mode with candidate benchmark tracking (FaceForensics++ c23/c40 and DFDC). Video neural models remain completely out of scope for Phase 2A.

### Requirement 2: Impersonation Accounts & Advisory Messages (WhatsApp / Telegram)
- **Architectural Mapping**: [`src/services/impersonation_engine.py`](file:///e:/fingaurd2.0/src/services/impersonation_engine.py) & [`src/models/advisory.py`](file:///e:/fingaurd2.0/src/models/advisory.py).
- **Current Implementation**: Social chat analysis for illegal investment advice patterns, promised guaranteed returns (*"100% loss recovery"*, *"paisa double"*), artificial urgency (*"only 5 slots left"*), and coded Hinglish colloquialisms (*"pakka jackpot"*, *"rocket call"*).
- **Regulatory Verification**: Sanitizes sensitive data under the **Digital Personal Data Protection (DPDP) Act 2023** and verifies claimed registration numbers against the curated sample **SEBI Registered Intermediary Roster** ([`data/sebi_registry.json`](file:///e:/fingaurd2.0/data/sebi_registry.json)).

### Requirement 3: Fake Trading Applications & Fraudulent Intermediaries
- **Architectural Mapping**: [`src/services/infrastructure_engine.py`](file:///e:/fingaurd2.0/src/services/infrastructure_engine.py) & [`src/models/infrastructure.py`](file:///e:/fingaurd2.0/src/models/infrastructure.py).
- **Current Implementation**: Pure-Python Levenshtein distance typo-squatting scanner detecting brand impersonation (e.g., `groww-bonus-reward.xyz`, `zerodha-terminal.top`) against authorized broker domains in [`data/broker_whitelist.json`](file:///e:/fingaurd2.0/data/broker_whitelist.json). Evaluates high-risk Android permissions (`READ_SMS`, `RECEIVE_SMS`, `BIND_ACCESSIBILITY_SERVICE`, `SYSTEM_ALERT_WINDOW`, `PACKAGE_USAGE_STATS`).

---

## 2. Hardening Features Implemented

1. **SQLite Database Persistence via SQLAlchemy**:
   - `ScanRecord` table: audit log of all scans, input targets, timestamps, verdicts, and risk scores.
   - `JobState` table: asynchronous and completed job execution tracking with result payloads.
   - `AlertHistory` table: persistent log of all critical, warning, and info security alerts.
   - Query routes: `GET /api/jobs/{job_id}`, `GET /api/scans/recent`, and persistent `GET /api/alerts/recent`.

2. **Bounded Disk-Backed Upload Streaming**:
   - Media uploads in `POST /api/scan/media-file` stream 64KB chunks directly to temporary files on disk, avoiding RAM exhaustion from buffering large files.
   - Enforces configurable `max_upload_mb` (default 50MB) and immediately aborts with HTTP 413 if the limit is exceeded.

3. **File Extension & Magic Byte Verification**:
   - Inspects binary headers against known magic bytes signatures (MP4 `ftyp`, MKV `\x1a\x45\xdf\xa3`, AVI `RIFF..AVI `, WAV `RIFF..WAVE`, MP3 `ID3`, FLAC `fLaC`, OGG `OggS`).
   - Rejects forged or non-media files with HTTP 400.

4. **Guaranteed Temporary Storage Cleanup**:
   - Managed upload context manager removes disk-buffered files in a `finally` block in all scenarios (success, validation failure, oversized limits, or unhandled exceptions).
   - Startup cleanup routine automatically evicts stale `.tmp` files older than 1 hour.

5. **Sliding-Window IP Rate Limiting**:
   - Thread-safe sliding window rate limiter tracking requests per rolling 60-second window per client IP.
   - Returns HTTP 429 Too Many Requests with `Retry-After` headers when limits are exceeded.

6. **Environment-Configured CORS Protection**:
   - Replaced wildcard `*` with explicit allowed origins (`http://localhost:8000`, `http://127.0.0.1:8000`), overridable via `ALLOWED_ORIGINS` environment variable.

---

## 3. Directory Structure

```
fingaurd2.0/
├── config.yaml                                       # System configuration, security, database & upload settings
├── requirements.txt                                  # Pinned Python package dependencies
├── pytest.ini                                        # Pytest configuration with pythonpath = .
├── main.py                                           # ASGI server bootstrap runner
├── README.md                                         # Project orientation and quickstart guide
├── PHASE_1_ARCHITECTURE_AND_CODEBASE_MANUAL.md      # Comprehensive technical manual
├── AUDIT_REPORT.md                                   # Empirical technical audit report
├── benchmarks/
│   └── BENCHMARKS.md                                 # Academic benchmark evaluation specifications (FF++ & DFDC)
├── data/
│   ├── finguard.db                                   # SQLite persistence database (generated at runtime)
│   ├── sebi_registry.json                            # Curated sample SEBI intermediary roster (24 records)
│   └── broker_whitelist.json                         # Curated sample broker domains & package IDs (18 records)
├── src/
│   ├── __init__.py                                   # Source package marker
│   ├── core/
│   │   ├── config.py                                 # Typed AppConfig YAML & environment loader
│   │   ├── database.py                               # SQLAlchemy engine, session provider & init_db
│   │   ├── logging.py                                # Structured console logger
│   │   ├── rate_limit.py                             # Sliding-window IP rate limiter
│   │   ├── security.py                               # DPDP Act PII redactor & prompt injection firewall
│   │   └── upload.py                                 # Disk-backed streaming, magic bytes & safe cleanup
│   ├── models/
│   │   ├── __init__.py                               # Models barrel export
│   │   ├── db.py                                     # SQLAlchemy models: ScanRecord, JobState, AlertHistory
│   │   ├── media.py                                  # Video/audio deepfake data contracts
│   │   ├── advisory.py                               # Chat & impersonation data contracts
│   │   ├── infrastructure.py                         # Domain & APK analysis data contracts
│   │   └── verdict.py                                # Multi-factor verdict & alert models
│   ├── services/
│   │   ├── __init__.py                               # Services barrel export
│   │   ├── base.py                                   # Abstract engine interfaces
│   │   ├── deepfake_engine.py                        # Video & audio baseline forensic engine
│   │   ├── impersonation_engine.py                   # Chat & advisory compliance engine
│   │   ├── infrastructure_engine.py                  # Domain & APK threat auditor
│   │   └── alert_engine.py                           # SQLite-persisted alert & SCORES dossier engine
│   └── api/
│       └── app.py                                    # FastAPI application exposing 13 HTTP endpoints
├── static/
│   ├── index.html                                    # Modern glassmorphic command center dashboard
│   ├── css/
│   │   └── styles.css                                # Native CSS3 glassmorphic design system
│   └── js/
│       └── app.js                                    # Vanilla ES6 reactive client controller
└── tests/
    ├── test_api.py                                   # API routes & schema tests
    ├── test_config.py                                # Configuration & benchmark metadata tests
    ├── test_engines.py                               # Forensic engine logic tests
    └── test_hardening.py                             # SQLite persistence, rate limit, upload & cleanup tests
```

---

## 4. Quickstart Guide

### 1. Run Automated Test Suite
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```
All **26 tests** pass with 0 failures.

### 2. Start Application Server
```powershell
.\.venv\Scripts\python.exe main.py
```

### 3. Access Web Dashboard & API Docs
- **Web Command Center**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Interactive Swagger Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
