# FinGuard 2.0 — Pellav2 Stage 1 Fine-Tuning Pipeline

This document details the architecture, safety constraints, configuration, and execution workflow for **Stage 1 fine-tuning** of the Pellav2 audio deepfake detector (`Sadanie/pellav2-audio-deepfake-detector`).

---

## 1. Stage 1 Architectural Design

In Stage 1, we tune the detector's classification head and multi-layer attention fusion weights while keeping the heavy acoustic foundation model completely frozen:

```
Raw Audio (WAV / FLAC)
  │
  ▼
Preprocessing: 16 kHz Mono │ 4.0s Center-Crop/Pad (64,000 samples) │ Per-Clip Mean-Std Normalization
  │
  ▼
Wav2Vec2-XLS-R-300M Backbone (315,438,720 parameters) ──► [FROZEN / EVAL MODE]
  │ Extracts 25 hidden states: [hs_0, hs_1, ..., hs_24] (shape: 25 × B × 199 × 1024)
  ▼
Learned Softmax Layer Weights (25 parameters) ──────────► [TRAINABLE]
  │ h = Σ softmax(w_i) · hs_i averaged across temporal frames (shape: B × 1024)
  ▼
Linear Classification Head (1,024 weights + 1 bias) ─────► [TRAINABLE]
  │ Head output: raw logit z (shape: B)
  ▼
Loss: BCEWithLogitsLoss(z, y) ──► Target: 0.0 (Real), 1.0 (Fake)
```

### Parameter Breakdown
| Component | Layer Type | Parameters | Trainable in Stage 1 |
|---|---|:---:|:---:|
| **Backbone** | `Wav2Vec2Model` (XLS-R-300M) | 315,438,720 | **No (Frozen)** |
| **Layer Fusion** | `nn.Parameter(torch.zeros(25))` | 25 | **Yes** |
| **Linear Head** | `nn.Linear(1024, 1)` | 1,025 | **Yes** |
| **Total** | Full Detector | **315,439,770** | **1,050 (0.00033%)** |

---

## 2. Hardware Profile & VRAM Feasibility

- **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6,141 MB total VRAM)
- **Batch Size:** 8 clips (4.0 s audio per clip)
- **Precision:** Mixed Precision (`torch.amp.autocast(dtype=torch.float16)`)
- **Peak VRAM Allocated:** **~1,906 MB (~1.9 GB)**
- **Headroom:** **>4.2 GB VRAM free**, guaranteeing zero Out-of-Memory (OOM) errors during execution.

---

## 3. Dataset Partitions & Isolation Guarantees

All files and manifests reside strictly on drive `E:`:

1. **Training Partition (`3,800` clips):** [`E:\datasets\splits\train_manifest.json`](file:///E:/datasets/splits/train_manifest.json)
   - 1,900 Genuine Human Speech (Gary Stafford YouTube + In-the-Wild Group A)
   - 1,900 Synthetic Clones (Amazon Polly, ElevenLabs, Luvvoice, In-the-Wild Clones)
2. **Validation Partition (`1,000` clips):** [`E:\datasets\splits\val_manifest.json`](file:///E:/datasets/splits/val_manifest.json)
   - 500 Genuine Human Speech (Independent YouTube channels + In-the-Wild Group B)
   - 500 Synthetic Clones (Amazon Polly, ElevenLabs, Luvvoice, In-the-Wild Clones)
   - Used exclusively for epoch validation and early stopping.
3. **Locked Test Partition (`1,200` clips):** [`E:\datasets\splits\locked_test_manifest.json`](file:///E:/datasets/splits/locked_test_manifest.json)
   - 600 Genuine Human Speech (Independent channels + In-the-Wild Group C)
   - 600 Synthetic Clones (Held-out generators + In-the-Wild Clones)
   - **Holdout Guarantee:** Hexgrad Kokoro (`68`), Hume AI (`116`), and Speechify (`211`) are 100% excluded from training and validation.
   - **Quarantine Guarantee:** NEVER loaded, accessed, or evaluated during Stage 1 training.
4. **Diagnostic Probe Controls (`20` clips):** [`E:\datasets\splits\diagnostic_manifest.json`](file:///E:/datasets/splits/diagnostic_manifest.json)
   - Contains targeted probes (Windows SAPI, ElevenLabs Roger, authentic financial briefings) quarantined from core splits.

---

## 4. Pipeline Scripts & Commands

All commands run inside the project virtual environment `.\.venv\Scripts\python.exe`.

### Step 1: Preflight Verification & Smoke Test
Validates manifest schemas, file integrity on disk, strict split isolation, model architecture loading, and executes a minimal 2-sample forward pass without backward steps:
```powershell
.\.venv\Scripts\python.exe scripts\validate_training_setup.py
```

### Step 2: Stage 1 Training Execution (When Authorized)
Trains the 1,050 parameters using AdamW (lr=1e-4) with early stopping based on validation ROC-AUC:
```powershell
.\.venv\Scripts\python.exe scripts\train_pellav2.py --config configs\train_pellav2_stage1.yaml
```
Output artifacts are saved to:
- Checkpoint: `E:\datasets\experiments\pellav2_stage1\best_model.pt`
- Experiment Metadata: `E:\datasets\experiments\pellav2_stage1\experiment_metadata.json`

### Step 3: Side-by-Side Model Evaluation
Compares the original Pellav2 checkpoint against the fine-tuned checkpoint on the validation set at the standard 0.50 threshold:
```powershell
.\.venv\Scripts\python.exe scripts\evaluate_pellav2.py --fine-tuned-checkpoint E:\datasets\experiments\pellav2_stage1\best_model.pt
```

To run targeted diagnostic probe evaluation:
```powershell
.\.venv\Scripts\python.exe scripts\evaluate_pellav2.py --fine-tuned-checkpoint E:\datasets\experiments\pellav2_stage1\best_model.pt --diagnostic
```

---

## 5. Asset Protection & Safety Rules

- **Original Weights:** [`E:\project models\pellav2_detector.pt`](file:///E:/project%20models/pellav2_detector.pt) is strictly opened in read-only mode (`torch.load`) and never overwritten.
- **Production Code:** Files in `src/`, `app/`, and `main.py` remain completely untouched.
- **Storage Target:** All experiments and data are confined to `E:\datasets\`.
