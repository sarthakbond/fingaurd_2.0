# FinGuard 2.0 — Technical Audit Report (Phase 1 & Foundation Hardening)

> **Target**: `e:\fingaurd2.0`  
> **Evaluation Base**: Problem Statement #7 & `PHASE_1_ARCHITECTURE_AND_CODEBASE_MANUAL.md`  
> **Initial Audit Timestamp**: 2026-10-11  
> **Hardening Milestone Verification**: 2026-10-11 (Completed)

---

## 1. Project Files, Codebase Inventory & Current Discrepancies

### Codebase Inventory (Excluding `.venv` and Git metadata)
- **Core backend modules**: 7 files (`__init__.py`, `config.py`, `database.py`, `logging.py`, `rate_limit.py`, `security.py`, `upload.py`)
- **Data models & schemas**: 6 files (`__init__.py`, `db.py`, `media.py`, `advisory.py`, `infrastructure.py`, `verdict.py`)
- **Forensic services**: 6 files (`__init__.py`, `base.py`, `deepfake_engine.py`, `impersonation_engine.py`, `infrastructure_engine.py`, `alert_engine.py`)
- **API routes**: 2 files (`__init__.py`, `app.py`)
- **Static assets**: 3 files (`index.html`, `styles.css`, `app.js`)
- **Reference data**: 2 files (`sebi_registry.json`, `broker_whitelist.json`)
- **Persistence layer**: SQLite database initialized at `data/finguard.db` with SQLAlchemy models
- **Tests**: 4 files (`test_api.py`, `test_config.py`, `test_engines.py`, `test_hardening.py`)
- **Configuration & Root**: 6 files (`config.yaml`, `.env.example`, `.gitignore`, `pytest.ini`, `requirements.txt`, `main.py`)
- **Documentation**: 4 files (`README.md`, `PHASE_1_ARCHITECTURE_AND_CODEBASE_MANUAL.md`, `AUDIT_REPORT.md`, `benchmarks/BENCHMARKS.md`)

### Status of Architectural Nuances
1. **Real-time Alert Transmission Method**:
   - The frontend polls `GET /api/alerts/recent` every 10 seconds via HTTP. In the hardening milestone, recent alerts are additionally persisted in and retrieved from SQLite (`AlertHistory`). No WebSocket/SSE stream is currently enabled.
2. **APK Binary Extraction**:
   - `TradingAppEngine.scan_apk_permissions` in `src/services/infrastructure_engine.py` evaluates permission string arrays against the dangerous permissions index. Raw `.apk` decompression is slated for future expansion.
3. **LLM Reasoning**:
   - Phase 1 & Hardening use deterministic regex, rule-based heuristics, and Levenshtein distance matching. No external or local LLMs are invoked.
4. **Media Processing Dependencies**:
   - Disk-spooled uploads (`src/core/upload.py`) now perform binary magic byte signature verification (MP4, MKV, AVI, WAV, MP3, FLAC, OGG). The media detection engine currently operates in `baseline_foundation` heuristic mode. Neural model weights (PyTorch/Timm/Wav2Vec2) will be introduced in the upcoming Deepfake Detection milestone.

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
collecting ... collected 26 items

tests/test_api.py::test_api_health PASSED                                [  3%]
tests/test_api.py::test_api_config PASSED                                [  7%]
tests/test_api.py::test_api_benchmarks_info PASSED                       [ 11%]
tests/test_api.py::test_api_scan_message PASSED                          [ 15%]
tests/test_api.py::test_api_scan_app PASSED                              [ 19%]
tests/test_api.py::test_api_scan_comprehensive PASSED                    [ 23%]
tests/test_api.py::test_api_alerts_recent PASSED                         [ 26%]
tests/test_api.py::test_api_generate_dossier PASSED                      [ 30%]
tests/test_config.py::test_config_loaded PASSED                          [ 34%]
tests/test_config.py::test_benchmarks_metadata PASSED                    [ 38%]
tests/test_config.py::test_thresholds PASSED                             [ 42%]
tests/test_engines.py::test_deepfake_engine_baseline PASSED              [ 46%]
tests/test_engines.py::test_impersonation_engine_guaranteed_returns PASSED [ 50%]
tests/test_engines.py::test_impersonation_engine_verified_broker PASSED  [ 53%]
tests/test_engines.py::test_levenshtein_distance PASSED                  [ 57%]
tests/test_engines.py::test_infrastructure_typosquatting PASSED          [ 61%]
tests/test_engines.py::test_infrastructure_official_broker PASSED        [ 65%]
tests/test_engines.py::test_alert_engine_synthesis PASSED                [ 69%]
tests/test_hardening.py::test_sqlite_persistence_via_api PASSED          [ 73%]
tests/test_hardening.py::test_alert_history_persistence PASSED           [ 76%]
tests/test_hardening.py::test_media_upload_valid_disk_stream PASSED      [ 80%]
tests/test_hardening.py::test_media_upload_invalid_extension PASSED      [ 84%]
tests/test_hardening.py::test_media_upload_invalid_magic_bytes PASSED    [ 88%]
tests/test_hardening.py::test_media_upload_oversized_limit PASSED        [ 92%]
tests/test_hardening.py::test_rate_limiting_enforcement PASSED           [ 96%]
tests/test_hardening.py::test_cors_explicit_configuration PASSED         [100%]

======================== 26 passed, 1 warning in 0.63s ========================
```

---

## 3. Server Startup, Health & Frontend Verification

The ASGI server is running via Uvicorn on `http://127.0.0.1:8000` with lifespan resource management. Live query responses:

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
  "version": "2.0-foundation-hardened",
  "phase": "Milestone: Foundation Hardening",
  "security": {
    "cors_origins": ["http://localhost:8000", "http://127.0.0.1:8000"],
    "rate_limit_per_minute": 60,
    "max_upload_mb": 50
  },
  "persistence": {
    "database_url": "sqlite:///./data/finguard.db",
    "driver": "SQLAlchemy / SQLite"
  },
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
      "active_alerts_count": 1
    }
  }
}
```

---

## 4. API Input Validation, Rate Limiting & Error Handling

All endpoints were tested with valid and invalid payloads. Pydantic v2 schemas and hardening guards intercepted invalid requests cleanly:

| Endpoint | Test Case | Status | Server Behavior |
| :--- | :--- | :--- | :--- |
| `GET /` | Standard page request | **200 OK** | Serves dashboard UI |
| `GET /api/health` | Status probe | **200 OK** | Returns active engine, security, and persistence status |
| `GET /api/config` | Config probe | **200 OK** | Returns active threshold & security settings |
| `GET /api/benchmarks/info` | Benchmark probe | **200 OK** | Returns FF++ and DFDC metadata |
| `GET /api/jobs/{job_id}` | Existing SQLite job query | **200 OK** | Returns persisted job status and metadata |
| `GET /api/scans/recent` | Recent scan query | **200 OK** | Returns persisted `ScanRecord` history |
| `POST /api/scan/media` | Valid JSON URL | **200 OK** | Returns `MediaScanResponse` & persists record |
| `POST /api/scan/media-file` | Valid streamed media upload | **200 OK** | Spools to disk, checks magic bytes, processes & cleans up |
| `POST /api/scan/media-file` | Invalid extension (`.exe`) | **400 Bad Request** | Rejected by file extension validator |
| `POST /api/scan/media-file` | Forged MIME / Invalid magic bytes | **400 Bad Request** | Rejected by binary magic byte inspection |
| `POST /api/scan/media-file` | Oversized payload (> max upload MB) | **413 Content Too Large**| Bounded stream stops and unlinks temp file |
| `POST /api/scan/message` | Valid message text | **200 OK** | Evaluates risk & persists record to SQLite |
| `POST /api/scan/message` | Empty string (`""`) | **422 Unprocessable** | Rejected (`min_length=2` constraint) |
| `POST /api/scan/*` | Rapid scan requests (> rate limit) | **429 Too Many Requests** | Enforces rate limit with `Retry-After` header |
| `POST /api/scan/app` | Valid domain URL | **200 OK** | Evaluates typo-squatting & persists record |
| `POST /api/scan/comprehensive` | Multi-vector payload | **200 OK** | Synthesizes verdict, creates alert, saves to SQLite |
| `GET /api/alerts/recent` | Valid request | **200 OK** | Fetches recent alerts from `AlertHistory` table |
| `POST /api/report/generate-dossier`| Valid complaint request | **200 OK** | Generates pre-formatted SEBI dossier |

---

## 5. Media Detector Inference Verification

> **Explicit Finding**: The media detector operates in **baseline heuristic mode** (`baseline_foundation`) for Phase 1 / Foundation Hardening.

- Neural weights and model inference (e.g., PyTorch, Timm, Wav2Vec2) are **not** loaded yet.
- Synthetic risk values are calculated from file characteristics and simulated temporal tracks for timeline verification.
- **Next Milestone**: Real deepfake detection with verified FaceForensics++ and DFDC benchmark architectures will be implemented in the next milestone.

---

## 6. Frontend Framework Verification

> **Explicit Finding**: The frontend is implemented strictly in **Vanilla JavaScript, Vanilla CSS3, and HTML5**.

- **No React / Vite / TypeScript / Tailwind**: Native ES modules, standard DOM APIs, CSS custom properties, and Starlette static mount.
- Real-time alerts interface with SQLite-backed `/api/alerts/recent` through background polling.

---

## 7. Registry & Broker Whitelist Provenance

> **Explicit Finding**: The registry and whitelist datasets are **static, curated reference snapshots**.

- **`data/sebi_registry.json`**: 24 curated sample entries of major retail brokers (Zerodha, Groww, Angel One, ICICI Direct, etc.).
- **`data/broker_whitelist.json`**: 18 curated domain hostnames and Android package IDs.

---

## 8. Security Audit & Hardening Resolution Status

In the initial Phase 1 audit, 4 critical security/architecture gaps were identified prior to public release. Here is their current status after the **Foundation Hardening** milestone:

| Initial Issue | Pre-Hardening State | Hardened State | Status |
| :--- | :--- | :--- | :--- |
| **1. Unrestricted CORS** | Wildcard `cors_origins: ["*"]` allowed arbitrary origin access. | Restricted to explicit origins (`http://localhost:8000`, `http://127.0.0.1:8000`) or environment variable `FINGUARD_CORS_ORIGINS`. | **RESOLVED** |
| **2. In-Memory Volatility** | Scans and alerts stored in volatile `collections.deque` wiped on restart. | Persistent SQLite database via SQLAlchemy (`data/finguard.db`) storing `ScanRecord`, `JobState`, and `AlertHistory`. | **RESOLVED** |
| **3. In-Memory File Uploads** | `await file.read()` buffered entire media files in RAM, posing memory exhaustion risks. | Bounded disk-backed streaming uploads (`64 KB` chunks) to `temp/uploads/` with size cap enforcement and guaranteed cleanup. | **RESOLVED** |
| **4. Lack of Rate Limiting** | Public scan endpoints had no throttling. | Sliding-window client IP rate limiter returning HTTP `429` with `Retry-After` header applied to `/api/scan/*`. | **RESOLVED** |

---

### The Single Highest-Priority Next Action
> **Implement Real Deepfake Detection (Phase 2)**: Integrate real deepfake detection models (visual spatial-temporal artifact models and acoustic synthesis detectors) evaluated against the FaceForensics++ and DFDC benchmark frameworks without fabricating metrics.

