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

---

## 2. Evaluation Criteria for Candidate Deepfake Models

Before integrating neural weights into Phase 2, candidate architectures must undergo a structured evaluation pipeline:

| Evaluation Stage | Criterion | Benchmark Dataset | Pass Threshold |
| :--- | :--- | :--- | :--- |
| **Stage A: Spatial Texture** | High-frequency blending margin & boundary artifact detection | FaceForensics++ (c23) | AUC-ROC $\ge 0.88$ |
| **Stage B: Compression Robustness** | Generalization under social media compression | FaceForensics++ (c40) | AUC-ROC $\ge 0.78$ |
| **Stage C: Cross-Domain Realism** | Out-of-distribution evaluation across unconstrained subjects | DFDC Public Validation Split | Log Loss $\le 0.45$ |
| **Stage D: Inference Latency** | Per-frame GPU/CPU throughput for real-time alerting | 1080p 30fps sample clip | $\le 45\text{ ms}$ per keyframe |

---

## 3. Candidate Model Architectures to Evaluate

1. **Spatial Vision Backbones**:
   - `EfficientNet-B4`: Strong parameter-to-accuracy efficiency for fine-grained feature boundaries.
   - `Xception`: The reference baseline on FaceForensics++ for artifact identification in color space residuals.
   - `Spatial Rich Model (SRM) Hybrid`: High-pass convolution filters capturing sensor PRNU noise patterns.
2. **Acoustic Audio Backbones**:
   - `Wav2Vec2 (Self-Supervised Speech)`: Evaluates phase jitter, harmonic spectral coherence, and neural vocoder footprints in 16 kHz audio.
   - `Mel-Spectrogram Inception CNN`: Detects synthetic frequency discontinuities in synthesized voice clones.

---

## 4. Empirical Integrity Notice
FinGuard strictly adheres to scientific integrity. In Phase 1 (Application Foundation), candidate model hooks and benchmark metadata structures are established. Neural model checkpoints will only have accuracy scores recorded after empirical execution against the validation test sets. No fabricated accuracy claims or unverified metrics are permitted.
