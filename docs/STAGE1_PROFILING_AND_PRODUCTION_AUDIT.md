# FinGuard 2.0 — Stage 1 Inference Profiling & Production-Readiness Audit Report

All profiling benchmarks and production-readiness audits for the fine-tuned **Pellav2 Stage 1** detector have been completed. Measurements were recorded on the **NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM)** and Intel CPU.

Both reports have been saved to disk under:
- Full Profiling Telemetry: `E:\datasets\experiments\pellav2_stage1\reports\inference_profiling_report.json`
- Production-Readiness Audit: `E:\datasets\experiments\pellav2_stage1\reports\production_readiness_audit.md`

---

### A. Checkpoint Protection & Verification

Both checkpoint hashes were verified via SHA-256 before profiling:

| Checkpoint | Path | SHA-256 Checksum | Status |
| :--- | :--- | :--- | :---: |
| **Baseline (Original)** | `E:\project models\pellav2_detector.pt` | `8214705ad784f66bca0736300e3432d5bfc325f1fc211a3aea5f9a13c790b023` | **Untouched (Read-Only)** |
| **Stage 1 (Fine-Tuned)** | `E:\datasets\experiments\pellav2_stage1\best_model.pt` | `ea3cad0cdc107c574542f651852050225274b0ddcd1ee93d563fa6efb0cc2ad8` | **Verified & Locked** |

*No retraining was performed, no thresholds were modified, and the locked test set was not re-accessed.*

---

### B. Hardware Profiling Results (RTX 4050 Laptop GPU vs. CPU)

Timing was measured using `torch.cuda.Event(enable_timing=True)` with explicit `torch.cuda.synchronize()` barriers around each pass to capture true device execution time without host-queue skew.

#### 1. Latency & Initialization Breakdown (Representative 4.0-Second Audio Chunk)

| Component / Pipeline Stage | Device | Median Latency | p95 Latency | p99 Latency | Throughput | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cold Model Loading** | GPU Cold Start | **3.199 s** (3,199 ms) | — | — | — | **1,203 MB** |
| **Audio Preprocessing (isolated)** | CPU (200 runs) | **6.13 ms** | **11.08 ms** | **12.53 ms** | ~163 clips/s | — |
| **Warm Single-Clip Forward (BS=1)** | GPU (250 runs, fp16) | **25.76 ms** | **26.19 ms** | **26.48 ms** | **38.82 clips/s** | **3,077 MB** |
| **CPU Fallback Forward (BS=1)** | CPU (30 runs, fp32) | **430.45 ms** | **462.78 ms** | **466.80 ms** | **2.32 clips/s** | Host RAM |

* **GPU Speedup Factor:** **16.71× faster** than CPU fallback.
* **Per-Chunk Operational SLA:** For a standard 12-second inbound voice call (3 sliding chunks at 1s step), neural inference takes **~77 ms total on GPU**, leaving ample headroom for Whisper transcription and registry lookups.
* **Numerical Stability:** **0 NaNs or Infs** detected across all 250 warm passes; zero CUDA errors or memory leaks.

#### 2. GPU Batch Throughput Scaling

| Batch Size ($B$) | Median Batch Latency | Latency per 4s Clip | Throughput (Clips/Sec) | Peak VRAM Allocated |
| :---: | :---: | :---: | :---: | :---: |
| **1** | 25.52 ms | 25.52 ms | **39.18 clips/s** | 3,077 MB (50.1% of 6GB) |
| **2** | 37.70 ms | 18.85 ms | **53.06 clips/s** | 3,120 MB (50.8% of 6GB) |
| **4** | 60.07 ms | 15.02 ms | **66.59 clips/s** | 3,203 MB (52.1% of 6GB) |

---

### C. Diagnostic Data Audit & Missing-File Investigation

The locked-test report noted 2 evaluated diagnostic probes and 18 missing cases:

1. **Evaluated Controls (2/20):**
   - `diag_01` (`temp/synthetic_tts_speech.wav`): Windows SAPI financial prompt (173 KB). Evaluated ($P = 0.9999$).
   - `diag_02` (`ElevenLabs_Roger`): Resides in `C:\Users\sarthak\Downloads\ElevenLabs_...mp3.mpeg` (131 KB). Evaluated ($P = 0.7428$).
2. **Missing Placeholders (18/20):**
   - `diag_03` through `diag_10` (`temp/diag_fake_*.wav`) and `diag_real_01` through `diag_real_10` (`temp/diag_real_*.wav`).
   - **Root Cause:** When `data/manifest_audit.json` was authored during Phase 1 planning, 20 probe slots were reserved in the schema as targets for custom synthetic fraud prompts and authentic briefing audio. However, files 3–20 were never synthesized or recorded into `temp/`.
3. **Recovery Plan (For Approval Before Action):**
   - Relocate the existing ElevenLabs Roger audio into `temp/diagnostic/ElevenLabs_Roger.mpeg` to match manifest expectations directly.
   - For the 18 empty slots, write a non-destructive generation script (`scripts/generate_diagnostic_controls.py`) to generate local Windows SAPI prompts and pull authentic public speeches only when authorized.

---

### D. Production-Readiness Audit (Architecture Inspection Only)

Inspection of `src/services/audio_deepfake_detector.py` confirms that Stage 1 integrates cleanly into the production architecture without design overhauls:

1. **Score Semantics & Multi-Signal Principle:**  
   The Pellav2 Stage 1 model is an **acoustic authenticity estimator**, outputting the probability ($P \in [0, 1]$) of neural synthesis. Per FinGuard 2.0 design requirements, it must **never be treated as a definitive standalone fraud verdict**, but fused alongside Whisper transcription, SEBI advisor registry validation, and behavioral urgency signals.
2. **State Dictionary Compatibility:**  
   The training checkpoint `best_model.pt` is saved as a dictionary bundle containing metadata (`"model_state_dict"`, `"epoch"`, `"val_metrics"`). To integrate into `AudioDeepfakeDetector._lazy_initialize()`, the cleanest zero-breakage path is either:
   - Exporting a clean production weights file `E:\project models\pellav2_stage1_production.pt` containing exclusively the raw `state_dict`, OR:
   - Updating `_lazy_initialize()` with a one-line unpacking helper: `state = state.get("model_state_dict", state)`.
3. **Threshold Behavior:**  
   Default classification threshold is anchored at **0.50** (`thresholds.audio_fake_threshold`). Sliding-window aggregation uses peak weighting (70%) and mean weighting (30%).
4. **Failure Modes & Fallback:**  
   - **CUDA OOM / Missing GPU:** The service catches CUDA exceptions and gracefully falls back to CPU mode (430 ms latency).
   - **Invalid / Truncated Audio:** Rejected by `AudioPreprocessor` with explicit `status="ERROR"` — never a fabricated safe score.
   - **Timeouts (>5s):** Async worker aborts and logs an audit alert.
5. **Safe Instant Rollback:**  
   Because the input dimensions (16 kHz, 64k samples) and model architecture are 100% bit-compatible, reverting to the original baseline at any time requires only switching the weights path back to `E:\project models\pellav2_detector.pt`.

---

### E. Status & Wrap-Up

- Profiling complete and reports saved under `E:\datasets\experiments\pellav2_stage1\reports\`.
- Checkpoints verified and locked.
- Production code and weights remain completely untouched.
- Background tasks terminated cleanly.

We are stopped and ready to pick up tomorrow with production staging, diagnostic controls generation, or Stage 2 exploration whenever you are ready!
