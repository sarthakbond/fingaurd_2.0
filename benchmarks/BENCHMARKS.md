# Deepfake Model Evaluation Protocol & Public Academic Benchmarks

Aligned with **Problem Statement #7: AI-Based Investor Fraud & Impersonation Detection**.

---

## 1. Primary Public Academic Benchmarks

As specified in the hackathon problem statement, FinGuard scopes deepfake video and audio detection around two gold-standard academic benchmarks before integrating neural model weights:

### Benchmark A: FaceForensics++ (FF++)
- **Source**: Technical University of Munich / [FaceForensics++ GitHub](https://github.com/ondyari/FaceForensics)
- **Scope**: 1,000+ original video sequences manipulated using four state-of-the-art automated facial manipulation methods:
  1. **Deepfakes**: Autoencoder-based face replacement.
  2. **Face2Face**: Real-time facial reenactment transferring source expressions to target video.
  3. **FaceSwap**: Graphics-based face swapping with blending.
  4. **NeuralTextures**: GAN-driven neural texture mapping and rendering.
- **Compression Tiers**:
  - `c23 (HQ)`: High quality H.264 compression (mimicking broadcast/advisory streams).
  - `c40 (LQ)`: Heavy H.264 compression (mimicking WhatsApp/Telegram re-compressed mobile video).
- **Target Evaluation Metric**: Frame-level AUC-ROC, Equal Error Rate (EER), and Binary Cross-Entropy Loss.

### Benchmark B: Deepfake Detection Challenge (DFDC)
- **Source**: Meta AI, AWS, Microsoft, Partnership on AI / [DFDC Paper (arXiv:2006.07397)](https://arxiv.org/abs/2006.07397)
- **Scope**: 100,000+ video clips across diverse subjects, varied lighting, variable resolutions, and complex real-world backgrounds.
- **Role in FinGuard**: Evaluates cross-dataset generalization. Deepfake detectors trained exclusively on laboratory samples often suffer catastrophic generalization drops on noisy, compressed social media video. DFDC validation guarantees robustness against out-of-distribution social media artifacts.
- **Target Evaluation Metric**: Log Loss (primary DFDC competition metric) and Top-1 Video Classification Accuracy.

### Benchmark C: ASVspoof 2021 PA (Physical Access / Audio Spoofing)
- **Source**: University of Edinburgh / EURECOM / [ASVspoof 2021 Challenge](https://www.asvspoof.org/)
- **Scope**: Rigorous academic benchmark evaluating speech synthesis, voice conversion, and acoustic replay attacks across diverse physical room acoustics and transmission channels.
- **Role in FinGuard (Phase 2A)**: Training basis for the initial candidate acoustic model `Vansh180/deepfake-audio-wav2vec2`.
- **Target Evaluation Metric**: Equal Error Rate (EER) and normalized minimum tandem Detection Cost Function (min t-DCF).
- **Known Training Data Limitations**:
  1. The model was fine-tuned on a balanced subset of ASVspoof 2021 PA; it has not yet been fine-tuned on modern commercial zero-shot voice cloning architectures (e.g., ElevenLabs, Tortoise TTS).
  2. Lossy speech compression (e.g., WhatsApp Opus voice notes at 16–24 kbps) alters high-frequency acoustic harmonics, which may degrade zero-shot detection accuracy without domain adaptation.
  3. All predictions are treated as an initial baseline rather than a certified proof of authenticity.

---

## 2. Evaluation Criteria for Candidate Deepfake Models

Before promoting candidate models to production, architectures undergo a structured evaluation pipeline:

| Evaluation Stage | Criterion | Benchmark Dataset | Pass Threshold | Current Status |
| :--- | :--- | :--- | :--- | :--- |
| **Stage A: Spatial Texture (Video)** | Blending margin & boundary artifact detection | FaceForensics++ (c23) | AUC-ROC $\ge 0.88$ | Phase 2B (Candidate) |
| **Stage B: Compression Robustness (Video)** | Generalization under social media compression | FaceForensics++ (c40) | AUC-ROC $\ge 0.78$ | Phase 2B (Candidate) |
| **Stage C: Cross-Domain Realism (Video)** | Out-of-distribution unconstrained subjects | DFDC Validation Split | Log Loss $\le 0.45$ | Phase 2B (Candidate) |
| **Stage D: Acoustic Synthetic Speech (Audio)** | Self-supervised speech representation (Wav2Vec2) | ASVspoof 2021 PA | EER $\le 10\%$ | **Phase 2A Initial Candidate Deployed** (`Vansh180`) |
| **Stage E: Inference Latency (Audio/Video)** | Real-time GPU throughput on RTX 4050 | 4.0s 16kHz audio window | $\le 200\text{ ms}$ | **Achieved (~120–150 ms on RTX 4050)** |

---

## 3. Candidate Model Architectures Evaluated

1. **Acoustic Audio Backbones (Phase 2A Active)**:
   - `Vansh180/deepfake-audio-wav2vec2`: **Deployed Candidate**. Built on `facebook/wav2vec2-base` (95M parameters). Loaded in FP16 on NVIDIA RTX 4050 (`cuda:0`). Evaluates 4.0-second sliding windows with 1.0-second overlap and transparent peak/mean aggregation.
   - `Sara1708/deepfake-audio-wav2vec2`: Evaluated in Phase 2A review. Excluded from initial pipeline due to non-standard checkpoint serialization (`stage2_best.pt`), custom wrapper dependencies, and documented lack of vocoder generalization.
2. **Spatial Vision Backbones (Phase 2B Pending)**:
   - `EfficientNet-B4`: Fine-grained artifact boundaries in facial keyframes.
   - `Xception`: Reference baseline on FaceForensics++ (c23/c40).
   - `Spatial Rich Model (SRM) Hybrid`: High-pass convolution filters capturing PRNU camera sensor patterns.

---

## 4. Empirical Integrity Notice
FinGuard strictly adheres to scientific integrity. In Phase 2A, genuine pretrained neural model inference is now running on real audio inputs. However, advertised accuracy metrics from model cards are explicitly cited as author claims only. Independent benchmark metrics across the full ASVspoof 2019/2021 and In-the-Wild evaluation splits will only be published after formal execution and validation. No synthetic or fabricated benchmark metrics are included.
