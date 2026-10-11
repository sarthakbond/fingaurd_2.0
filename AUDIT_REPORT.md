# FinGuard 2.0 — Phase 1 Technical Audit Report

> **Target**: `e:\fingaurd2.0`  
> **Evaluation Base**: Problem Statement #7 & `PHASE_1_ARCHITECTURE_AND_CODEBASE_MANUAL.md`  
> **Audit Timestamp**: 2026-10-11  

---

## 1. Project Files, Placeholder Implementations & Discrepancies

### Codebase Inventory (Excluding `.venv` and Git metadata)
- **Core backend modules**: 4 files (`__init__.py`, `config.py`, `logging.py`, `security.py`)
- **Pydantic contracts**: 5 files (`__init__.py`, `media.py`, `advisory.py`, `infrastructure.py`, `verdict.py`)
- **Forensic services**: 6 files (`__init__.py`, `base.py`, `deepfake_engine.py`, `impersonation_engine.py`, `infrastructure_engine.py`, `alert_engine.py`)
- **API routes**: 2 files (`__init__.py`, `app.py`)
- **Static assets**: 3 files (`index.html`, `styles.css`, `app.js`)
- **Reference data**: 2 files (`sebi_registry.json`, `broker_whitelist.json`)
- **Tests**: 3 files (`test_api.py`, `test_config.py`, `test_engines.py`)
- **Configuration & Root**: 6 files (`config.yaml`, `.env.example`, `.gitignore`, `pytest.ini`, `requirements.txt`, `main.py`)
- **Documentation**: 4 files (`README.md`, `PHASE_1_ARCHITECTURE_AND_CODEBASE_MANUAL.md`, `AUDIT_REPORT.md`, `benchmarks/BENCHMARKS.md`)

### Identified Discrepancies & Incomplete Implementations
1. **Real-time Alert Transmission Method**:
   - *Manual Claim*: Mentions "WebSocket/SSE Stream" in diagrams.
   - *Actual Code*: No WebSocket (`@app.websocket`) or SSE (`EventSource`) endpoints exist in `src/api/app.py`. The frontend (`static/js/app.js:377`) polls `GET /api/alerts/recent` every 10 seconds via HTTP.
2. **APK Binary Extraction**:
   - *Manual Claim*: References static binary parsing.
   - *Actual Code*: `TradingAppEngine.scan_apk_permissions` in `src/services/infrastructure_engine.py` accepts a pre-extracted list of permission strings and a package name. It does not parse raw `.apk` zip archives or extract `AndroidManifest.xml` binaries.
3. **LLM Reasoning**:
   - *Manual Claim*: Mentions Ollama/LLaMA 3.1.
   - *Actual Code*: Phase 1 uses regex and rule-based heuristics in `src/services/impersonation_engine.py`. No local or remote LLM calls are executed.
4. **Media Processing Dependencies**:
   - *Actual Code*: Neither `ffmpeg`, `moviepy`, nor `opencv-python` are imported in Phase 1. The media engine estimates duration from file byte length rather than demuxing AV streams.

---

## 2. Test Suite Execution & Output

Executed `.\.venv\Scripts\python.exe -m pytest -v --tb=short` in the workspace virtual environment:

```
============================= test session starts =============================
platform win32 -- Python 3.11.17, pytest-9.1.1, pluggy-1.6.0 -- E:\fingaurd2.0\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: E:\fingaurd2.0
configfile: pytest.ini
testpaths: tests
plugins: anyio-4.15.1
collecting ... collected 18 items

tests/test_api.py::test_api_health PASSED                                [  5%]
tests/test_api.py::test_api_config PASSED                                [ 11%]
tests/test_api.py::test_api_benchmarks_info PASSED                       [ 16%]
tests/test_api.py::test_api_scan_message PASSED                          [ 22%]
tests/test_api.py::test_api_scan_app PASSED                              [ 27%]
tests/test_api.py::test_api_scan_comprehensive PASSED                    [ 33%]
tests/test_api.py::test_api_alerts_recent PASSED                         [ 38%]
tests/test_api.py::test_api_generate_dossier PASSED                      [ 44%]
tests/test_config.py::test_config_loaded PASSED                          [ 50%]
tests/test_config.py::test_benchmarks_metadata PASSED                    [ 55%]
tests/test_config.py::test_thresholds PASSED                             [ 61%]
tests/test_engines.py::test_deepfake_engine_baseline PASSED              [ 66%]
tests/test_engines.py::test_impersonation_engine_guaranteed_returns PASSED [ 72%]
tests/test_engines.py::test_impersonation_engine_verified_broker PASSED  [ 77%]
tests/test_engines.py::test_levenshtein_distance PASSED                  [ 83%]
tests/test_engines.py::test_infrastructure_typosquatting PASSED          [ 88%]
tests/test_engines.py::test_infrastructure_official_broker PASSED        [ 94%]
tests/test_engines.py::test_alert_engine_synthesis PASSED                [100%]

======================== 18 passed, 1 warning in 0.34s ========================
```

---

## 3. Server Startup, Health & Frontend Verification

The ASGI server is running via Uvicorn on `http://127.0.0.1:8000`. Actual query responses:

### `GET /` (Frontend UI)
```http
HTTP/1.1 200 OK
Content-Type: text/html; charset=utf-8
Content-Length: 18647 bytes
```

### `GET /api/health`
```json
{
  "status": "HEALTHY",
  "version": "2.0-phase1",
  "phase": "Phase 1: Working Application Foundation",
  "engines": {
    "deepfake_engine": {
      "status": "READY",
      "mode": "baseline_foundation",
      "benchmark_targets": [
        "FaceForensics++",
        "DFDC"
      ]
    },
    "impersonation_engine": {
      "status": "READY",
      "sebi_registry_records": 24
    },
    "trading_app_engine": {
      "status": "READY",
      "whitelisted_brokers": 18
    },
    "alert_engine": {
      "status": "READY",
      "active_alerts_count": 2
    }
  }
}
```

---

## 4. API Input Validation & Error Handling

All endpoints were tested with valid and invalid payloads. Pydantic v2 schemas intercepted invalid types and lengths without 500 crashes:

| Endpoint | Test Case | Status | Server Behavior |
| :--- | :--- | :--- | :--- |
| `GET /` | Standard page request | **200 OK** | Serves complete HTML dashboard |
| `GET /api/health` | Status probe | **200 OK** | Returns active engine metrics |
| `GET /api/config` | Config probe | **200 OK** | Returns threshold dictionary |
| `GET /api/benchmarks/info` | Benchmark probe | **200 OK** | Returns FF++ and DFDC metadata |
| `POST /api/scan/media` | Valid JSON URL | **200 OK** | Returns `MediaScanResponse` |
| `POST /api/scan/media` | Invalid `sampling_fps` type (`"text"`) | **422 Unprocessable** | Caught by Pydantic parser |
| `POST /api/scan/media-file` | Multipart audio/video upload | **200 OK** | Ingests file and returns timeline |
| `POST /api/scan/media-file` | Empty payload (missing `file`) | **422 Unprocessable** | Caught by FastAPI form validator |
| `POST /api/scan/message` | Valid message text | **200 OK** | Returns advisory risk score |
| `POST /api/scan/message` | Empty string (`""`) | **422 Unprocessable** | Rejected (`min_length=2` constraint) |
| `POST /api/scan/message` | Missing `message_text` field | **422 Unprocessable** | Rejected (`missing field`) |
| `POST /api/scan/app` | Valid domain URL | **200 OK** | Returns typo-squatting evaluation |
| `POST /api/scan/app` | Empty JSON `{}` | **200 OK** | Handled gracefully with fallback defaults |
| `POST /api/scan/comprehensive` | Multi-vector payload | **200 OK** | Returns synthesized verdict & alerts |
| `GET /api/alerts/recent` | Valid request | **200 OK** | Returns JSON list of recent alerts |
| `GET /api/alerts/recent` | Invalid query param (`limit="text"`) | **422 Unprocessable** | Caught by integer parser |
| `POST /api/report/generate-dossier`| Valid complaint request | **200 OK** | Returns pre-formatted SEBI dossier |

---

## 5. Media Detector Inference Verification

> **Explicit Finding**: The media detector does **NOT** run actual AI deep learning inference in Phase 1.

- **Inspection of `src/services/deepfake_engine.py:62-67`**:
  ```python
  is_simulated_fake = any(k in fname.lower() for k in ["deepfake", "manipulated", "synthetic", "cloned", "fake"])
  visual_score = 0.85 if is_simulated_fake and media_type == "video" else 0.12
  acoustic_score = 0.82 if is_simulated_fake else 0.08
  ```
- No PyTorch (`torch`), TorchVision, Torchaudio, Timm, or ONNX runtimes are loaded in Phase 1.
- No model weights are unpacked or queried.
- Scores are generated deterministically based on file naming heuristics and simulated temporal steps for the timeline scrubber.

---

## 6. Frontend Framework Verification

> **Explicit Finding**: The frontend is implemented strictly in **Vanilla JavaScript, Vanilla CSS3, and HTML5**.

- **No React**: No JSX, React DOM, or React runtime scripts.
- **No Vite**: No bundling pipeline, node development server, or `vite.config.js`.
- **No TypeScript**: All browser code is written in `.js`.
- **No TailwindCSS**: Styles are in `static/css/styles.css` utilizing native CSS custom properties (`:root`), Flexbox, CSS Grid, and custom animations.
- The web interface is served directly by FastAPI via Starlette's `StaticFiles` mount at `/static`.

---

## 7. Registry & Broker Whitelist Provenance

> **Explicit Finding**: The registry and whitelist datasets are **static, curated sample snapshots**, not live or officially verified registry syncs.

- **`data/sebi_registry.json`**:
  - Total records: **24 entries**.
  - Content: Curated sample of major retail brokers (Zerodha, Groww, Angel One, ICICI Direct, HDFC Securities, Kotak Securities, Motilal Oswal, etc.).
  - Provenance: Manually compiled sample. It is **not** connected to the official SEBI public search portal, nor does it contain the thousands of actively registered Indian RIAs (`INA`), Research Analysts (`INH`), or Portfolio Managers (`INP`).
- **`data/broker_whitelist.json`**:
  - Total records: **18 entries**.
  - Content: Manually compiled list of official domain hostnames and Android package IDs.
  - Provenance: Sample reference list; it does not verify cryptographic APK signing certificates or Google Play Store developer metadata.

---

## 8. Summary Audit Findings

### Verified Working Components
1. **FastAPI Application Framework**: ASGI runtime running on Uvicorn without crashes.
2. **Data Contracts**: Pydantic v2 schemas validating all request/response models.
3. **Impersonation & Advisory Chat Engine**: Regex detection of guaranteed returns, urgency language, coded Hinglish colloquialisms (*"pakka jackpot"*, *"rocket call"*), and fuzzy license matching against the local 24-record registry.
4. **Data Privacy Shield**: DPDP Act 2023 regex filters redacting Indian mobile numbers, PAN cards, Aadhaar numbers, and UPI handles before processing.
5. **Infrastructure Engine**: Pure-Python Levenshtein distance matching detecting typo-squatted broker domains, plus evaluation of high-risk Android permissions.
6. **Multi-Factor Decision Matrix & SCORES Dossier**: Synthesizes cross-vector results into a 3-tier verdict (`SAFE`, `SUSPICIOUS`, `CRITICAL_FRAUD`) and formats pre-filing complaint dossiers citing Indian statutory frameworks.
7. **Frontend Dashboard**: Glassmorphic dark UI with live tabs, functional presets, and real-time alert polling.
8. **Automated Test Suite**: 18 passing pytest tests validating core logic.

### Failed Tests and Errors
- **None**: All 18 tests pass with 0 failures (`18 passed, 1 warning in 0.34s`).
- All 11 API routes return expected status codes (200 on valid inputs, 422 on invalid inputs).

### Missing or Incomplete Functionality
- **Actual Deep Learning Models**: Video and audio deepfake detection relies on baseline heuristics without loaded neural weights.
- **WebSocket/SSE Streaming**: Real-time alerts operate via HTTP polling instead of true push streaming.
- **Binary APK Decompiler**: Static APK analysis evaluates declared permission strings rather than extracting files from raw `.apk` packages.
- **Dynamic SEBI Registry Sync**: The registry is limited to a local static sample of 24 records.

### Security Issues Requiring Resolution Before Public Deployment
1. **Unrestricted CORS**: `config.yaml` sets `cors_origins: ["*"]`. Must be locked down to specific frontend origins before public exposure.
2. **In-Memory Volatility**: Alerts and scan jobs are stored in an in-memory `deque`. Restarts or multi-process workers will wipe active state. A persistent database (e.g. SQLite/PostgreSQL) is required for production.
3. **In-Memory File Uploads**: `POST /api/scan/media-file` loads the full file into memory via `await file.read()`. High concurrency with 100MB files will exhaust server RAM. Needs spooling to disk.
4. **Lack of Rate Limiting**: No rate-limiting middleware is currently attached to public scan endpoints.

---

### The Single Highest-Priority Next Action
> **Connect a persistent database (e.g., SQLite via SQLAlchemy) for job states and alert logs, and implement disk-spooled streaming for media uploads** so the application foundation is robust against memory exhaustion before introducing Phase 2 deep learning dependencies.
