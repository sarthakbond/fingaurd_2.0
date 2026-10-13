"""
FinGuard 2.0 — Side-by-Side Audio Deepfake Evaluation
======================================================
Compares the production baseline (Vansh180/deepfake-audio-wav2vec2) against
the Pellav2 candidate (Sadanie/pellav2-audio-deepfake-detector).

Usage
-----
  # Download weights automatically to E:\\project models (default)
  python scripts/evaluate_audio_models.py

  # Use an already-downloaded local weights file
  python scripts/evaluate_audio_models.py --pellav2-weights "E:\\project models\\pellav2_detector.pt"

Output
------
  - Console table: per-sample p_fake scores, decisions, correctness, winner
  - JSON report:   temp/eval_results_YYYYMMDD_HHMMSS.json

Requirements: torch transformers soundfile numpy scipy huggingface_hub
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ── Project root on sys.path so we can import AudioPreprocessor ───────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.services.audio_preprocessor import AudioPreprocessor

# ── Paths & constants ─────────────────────────────────────────────────────────
MODEL_DIR    = Path(r"E:\project models")          # user-specified download location
PELLAV2_REPO = "Sadanie/pellav2-audio-deepfake-detector"
BASELINE_REPO = "Vansh180/deepfake-audio-wav2vec2"
SR           = 16_000
PELLAV2_CROP = 4 * SR                              # 4-second window used by official infer script

TEMP = PROJECT_ROOT / "temp"

# ── Test corpus ───────────────────────────────────────────────────────────────
TEST_SAMPLES = [
    {
        "id":          "librispeech_legitimate",
        "path":        TEMP / "speech_sample_legit.flac",
        "label":       "HUMAN",
        "description": "LibriSpeech read speech (clean, legitimate human recording)",
    },
    {
        "id":          "armstrong_ogg_human",
        "path":        TEMP / "armstrong_moon_authentic.ogg",
        "label":       "HUMAN",
        "description": "Historical broadcast speech (Armstrong moon landing clip)",
    },
    {
        "id":          "windows_sapi_tts",
        "path":        TEMP / "synthetic_tts_speech.wav",
        "label":       "AI",
        "description": "Windows SAPI synthetic TTS (known AI voice)",
    },
    {
        "id":          "long_speech_30s",
        "path":        TEMP / "long_speech_recording.wav",
        "label":       "UNKNOWN",
        "description": "30-second multi-chunk recording (boundary / aggregation test)",
    },
]

# ElevenLabs — optional, only included if file is present
_ELEVENLABS = Path(
    r"C:\Users\sarthak\Downloads"
    r"\ElevenLabs_2026-10-11T14_57_02_Roger - Laid-Back, Casual, Resonant_pre_sp100_s50_sb75_v4.mp3.mpeg"
)
if _ELEVENLABS.exists():
    TEST_SAMPLES.append(
        {
            "id":          "elevenlabs_roger_ai",
            "path":        _ELEVENLABS,
            "label":       "AI",
            "description": "ElevenLabs Roger voice (commercial neural TTS)",
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pellav2 model — architecture copied verbatim from official pellav2_infer.py
# ─────────────────────────────────────────────────────────────────────────────
def _build_pellav2_detector_class():
    import torch
    import torch.nn as nn
    from transformers import Wav2Vec2Model

    class Detector(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m")
            self.fusion = True
            self.layer_weights = nn.Parameter(
                torch.zeros(self.backbone.config.num_hidden_layers + 1)
            )
            self.head = nn.Linear(self.backbone.config.hidden_size, 1)

        def forward(self, x):
            hs = self.backbone(x, output_hidden_states=True).hidden_states
            w  = torch.softmax(self.layer_weights, dim=0)
            h  = torch.stack(hs).mul(w[:, None, None, None]).sum(0).mean(dim=1)
            return self.head(h).squeeze(-1)

    return Detector


# ─────────────────────────────────────────────────────────────────────────────
# Pellav2 preprocessing — verbatim from official pellav2_infer.py
# ─────────────────────────────────────────────────────────────────────────────
def _pellav2_load_clip(path: str) -> np.ndarray:
    """Load audio, centre-crop/pad to 4 s at 16 kHz, mean-std normalise."""
    import soundfile as sf

    wav, sr = sf.read(str(path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    # Resample if needed (reuses our existing scipy backend)
    if sr != SR:
        proc = AudioPreprocessor(target_sample_rate=SR)
        wav  = proc.resample_waveform(wav, sr, SR)

    # Centre-crop or zero-pad to exactly PELLAV2_CROP samples (official logic)
    if len(wav) >= PELLAV2_CROP:
        off = (len(wav) - PELLAV2_CROP) // 2
        wav = wav[off : off + PELLAV2_CROP]
    else:
        wav = np.pad(wav, (0, PELLAV2_CROP - len(wav)))

    # Mean-std normalisation (official)
    wav = (wav - wav.mean()) / (wav.std() + 1e-7)
    return wav.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Pellav2 evaluator
# ─────────────────────────────────────────────────────────────────────────────
class Pellav2Evaluator:
    """Loads Pellav2 and scores individual audio files."""

    def __init__(self, weights_path: str | None = None):
        import torch
        from huggingface_hub import hf_hub_download

        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        print(f"\n[Pellav2] Device : {self.device}")

        # ── Resolve weights ────────────────────────────────────────────────
        if weights_path and Path(weights_path).is_file():
            pt_path = weights_path
            print(f"[Pellav2] Weights : {pt_path} (local)")
        else:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            pt_path = MODEL_DIR / "pellav2_detector.pt"

            if pt_path.exists():
                print(f"[Pellav2] Weights already present: {pt_path}")
            else:
                print(f"[Pellav2] Downloading config …")
                hf_hub_download(
                    repo_id=PELLAV2_REPO,
                    filename="config.json",
                    local_dir=str(MODEL_DIR),
                )
                print(f"[Pellav2] Downloading pellav2_detector.pt (~1.2 GB) → {MODEL_DIR}")
                hf_hub_download(
                    repo_id=PELLAV2_REPO,
                    filename="pellav2_detector.pt",
                    local_dir=str(MODEL_DIR),
                )
                print(f"[Pellav2] Download complete.")

        # ── Build & load model ─────────────────────────────────────────────
        Detector = _build_pellav2_detector_class()
        print("[Pellav2] Building backbone (facebook/wav2vec2-xls-r-300m) …")
        self.model = Detector().to(self.device)

        print("[Pellav2] Loading fine-tuned weights …")
        state = torch.load(str(pt_path), map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        print("[Pellav2] Model ready [OK]")

        if self.device.type == "cuda":
            vram_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            print(f"[Pellav2] VRAM used after load: {vram_mb:.0f} MB")

    def score(self, audio_path: str) -> dict:
        """Return p_fake in [0,1] for one file. Higher = more likely AI/synthetic."""
        import torch

        t0  = time.perf_counter()
        wav = _pellav2_load_clip(audio_path)
        x   = torch.from_numpy(wav)[None].to(self.device)

        with torch.inference_mode():
            p_fake = torch.sigmoid(self.model(x)).item()

        return {
            "p_fake":      round(p_fake, 4),
            "decision":    "AI" if p_fake >= 0.5 else "HUMAN",
            "latency_ms":  round((time.perf_counter() - t0) * 1000, 1),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Baseline (Vansh180) evaluator
# ─────────────────────────────────────────────────────────────────────────────
class BaselineEvaluator:
    """Replicates the production AudioDeepfakeDetector pipeline exactly."""

    def __init__(self):
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        print(f"\n[Baseline] Device : {self.device}")
        print(f"[Baseline] Loading {BASELINE_REPO} …")

        self.fe    = AutoFeatureExtractor.from_pretrained(BASELINE_REPO)
        self.model = AutoModelForAudioClassification.from_pretrained(BASELINE_REPO)
        if self.device.type == "cuda":
            self.model = self.model.half()
        self.model.to(self.device).eval()

        id2label = getattr(self.model.config, "id2label", {0: "bonafide", 1: "spoof"})
        self._spoof_idx = 1
        for idx, lbl in id2label.items():
            if any(k in str(lbl).lower() for k in ("spoof", "fake", "synth")):
                self._spoof_idx = int(idx)
                break
        print(f"[Baseline] Spoof label index: {self._spoof_idx}  ({id2label.get(self._spoof_idx)})")
        print("[Baseline] Model ready [OK]")

        self.preprocessor = AudioPreprocessor(
            target_sample_rate=16_000,
            window_duration_sec=4.0,
            overlap_duration_sec=1.0,
            max_duration_sec=180.0,
        )

    def score(self, audio_path: str) -> dict:
        import torch

        t0 = time.perf_counter()
        waveform, sr, duration = self.preprocessor.load_audio(audio_path)
        chunks = self.preprocessor.chunk_waveform(waveform, sample_rate=sr)

        scores = []
        for chunk in chunks:
            if chunk.is_silence:
                scores.append(0.0)
                continue
            inputs = self.fe(chunk.waveform, sampling_rate=16_000, return_tensors="pt")
            if self.device.type == "cuda":
                inputs = {k: v.to(self.device).half() for k, v in inputs.items()}
            else:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                logits = self.model(**inputs).logits
                probs  = torch.softmax(logits, dim=-1).squeeze(0)
                scores.append(float(probs[self._spoof_idx].item()))

        peak = round(max(scores), 4)
        mean = round(sum(scores) / len(scores), 4)
        # Production aggregation weights: 70 % peak + 30 % mean
        agg  = round(min(max(0.70 * peak + 0.30 * mean, 0.0), 1.0), 4)

        return {
            "p_fake":       agg,
            "peak":         peak,
            "mean":         mean,
            "chunks":       len(chunks),
            "duration_sec": round(duration, 2),
            "decision":     "AI" if agg >= 0.5 else "HUMAN",
            "latency_ms":   round((time.perf_counter() - t0) * 1000, 1),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="FinGuard 2.0 Audio Model Evaluation")
    parser.add_argument(
        "--pellav2-weights",
        type=str,
        default=None,
        help=(
            "Path to an already-downloaded pellav2_detector.pt. "
            "If omitted the file is downloaded to E:\\project models"
        ),
    )
    args = parser.parse_args()

    print("\n" + "=" * 75)
    print("  FinGuard 2.0 - Audio Deepfake Model Evaluation")
    print(f"  Pellav2 weights directory: {MODEL_DIR}")
    print("=" * 75)

    # ── Load models ───────────────────────────────────────────────────────────
    print("\n[STEP 1/3] Loading Pellav2 candidate …")
    pellav2_ok, pellav2_err, pellav2_eval = False, None, None
    try:
        pellav2_eval = Pellav2Evaluator(weights_path=args.pellav2_weights)
        pellav2_ok   = True
    except Exception as e:
        pellav2_err = str(e)
        print(f"  [FAIL] Pellav2 failed to load: {e}")

    print("\n[STEP 2/3] Loading Baseline (Vansh180) …")
    baseline_ok, baseline_err, baseline_eval = False, None, None
    try:
        baseline_eval = BaselineEvaluator()
        baseline_ok   = True
    except Exception as e:
        baseline_err = str(e)
        print(f"  [FAIL] Baseline failed to load: {e}")

    # ── Run inference ─────────────────────────────────────────────────────────
    print("\n[STEP 3/3] Running inference on test corpus …\n")
    COL = 35
    hdr = (
        f"{'Sample':<{COL}} {'True':>5}  "
        f"{'Pellav2':>8} {'Dec':>5}  "
        f"{'Baseline':>8} {'Dec':>5}  "
        f"{'Winner':>7}"
    )
    print(hdr)
    print("-" * len(hdr))

    results = []
    for s in TEST_SAMPLES:
        path = Path(str(s["path"]))
        if not path.exists():
            print(f"  [SKIP] {s['id']} — not found")
            continue

        true_lbl = s["label"]

        p2_res = {"p_fake": None, "decision": "ERROR", "latency_ms": 0.0}
        bl_res = {"p_fake": None, "decision": "ERROR", "latency_ms": 0.0}

        if pellav2_ok:
            try:
                p2_res = pellav2_eval.score(str(path))
            except Exception as e:
                p2_res["error"] = str(e)
                print(f"    Pellav2 error on {s['id']}: {e}")

        if baseline_ok:
            try:
                bl_res = baseline_eval.score(str(path))
            except Exception as e:
                bl_res["error"] = str(e)
                print(f"    Baseline error on {s['id']}: {e}")

        p2_correct = (p2_res["decision"] == true_lbl) if true_lbl != "UNKNOWN" else None
        bl_correct = (bl_res["decision"] == true_lbl) if true_lbl != "UNKNOWN" else None

        if true_lbl == "UNKNOWN":
            winner = "N/A"
        elif p2_correct and not bl_correct:
            winner = "Pellav2"
        elif bl_correct and not p2_correct:
            winner = "Baseline"
        elif p2_correct and bl_correct:
            winner = "Both"
        else:
            winner = "Neither"

        p2_s = f"{p2_res['p_fake']:.4f}" if p2_res["p_fake"] is not None else "  ERR"
        bl_s = f"{bl_res['p_fake']:.4f}" if bl_res["p_fake"] is not None else "  ERR"

        print(
            f"{s['id']:<{COL}} {true_lbl:>5}  "
            f"{p2_s:>8} {p2_res['decision']:>5}  "
            f"{bl_s:>8} {bl_res['decision']:>5}  "
            f"{winner:>7}"
        )

        results.append({
            "sample_id":        s["id"],
            "description":      s["description"],
            "true_label":       true_lbl,
            "pellav2":          p2_res,
            "pellav2_correct":  p2_correct,
            "baseline":         bl_res,
            "baseline_correct": bl_correct,
            "winner":           winner,
        })

    print("─" * len(hdr))

    # ── Summary ───────────────────────────────────────────────────────────────
    labeled = [r for r in results if r["true_label"] != "UNKNOWN"]
    if labeled:
        p2_acc = sum(1 for r in labeled if r["pellav2_correct"])  / len(labeled)
        bl_acc = sum(1 for r in labeled if r["baseline_correct"]) / len(labeled)
        print(f"\n  Accuracy on {len(labeled)} labeled samples:")
        print(f"    Pellav2  : {p2_acc:.0%}")
        print(f"    Baseline : {bl_acc:.0%}")

    p2_lats = [r["pellav2"]["latency_ms"]  for r in results if r["pellav2"].get("p_fake")  is not None]
    bl_lats = [r["baseline"]["latency_ms"] for r in results if r["baseline"].get("p_fake") is not None]
    if p2_lats:
        print(f"  Pellav2  mean latency : {sum(p2_lats)/len(p2_lats):.0f} ms/sample")
    if bl_lats:
        print(f"  Baseline mean latency : {sum(bl_lats)/len(bl_lats):.0f} ms/sample")

    # ── Save JSON report ──────────────────────────────────────────────────────
    TEMP.mkdir(exist_ok=True)
    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = TEMP / f"eval_results_{ts}.json"
    report      = {
        "run_timestamp":    ts,
        "model_dir":        str(MODEL_DIR),
        "pellav2_loaded":   pellav2_ok,
        "pellav2_error":    pellav2_err,
        "baseline_loaded":  baseline_ok,
        "baseline_error":   baseline_err,
        "samples":          results,
    }
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(f"\n  Report saved → {report_path}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
