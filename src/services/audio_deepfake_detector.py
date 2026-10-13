import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np

from src.core.config import config
from src.core.logging import logger
from src.models.media import TimelineSegment, BenchmarkEvaluationMetadata
from src.services.audio_preprocessor import AudioPreprocessor, AudioChunk

# ---------------------------------------------------------------------------
# Local weights path — downloaded to E:\project models during evaluation
# ---------------------------------------------------------------------------
_LOCAL_WEIGHTS = Path(r"E:\project models\pellav2_detector.pt")
_PELLAV2_REPO  = "Sadanie/pellav2-audio-deepfake-detector"
_SR            = 16_000
_CROP          = 4 * _SR   # 4-second centre-crop (official pellav2_infer.py)


# ---------------------------------------------------------------------------
# Pellav2 model definition — copied verbatim from pellav2_infer.py
# ---------------------------------------------------------------------------
def _build_pellav2_detector():
    """Returns the Pellav2 Detector class (deferred import so torch is optional)."""
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


def _pellav2_preprocess(waveform: np.ndarray, sr: int, preprocessor: AudioPreprocessor) -> np.ndarray:
    """
    Pellav2 single-clip preprocessing (verbatim from official pellav2_infer.py):
      1. Resample to 16 kHz if needed
      2. Centre-crop or zero-pad to exactly 4 s
      3. Mean-std normalise
    Returns float32 numpy array of length _CROP.
    """
    if sr != _SR:
        waveform = preprocessor.resample_waveform(waveform, sr, _SR)

    if len(waveform) >= _CROP:
        off      = (len(waveform) - _CROP) // 2
        waveform = waveform[off: off + _CROP]
    else:
        waveform = np.pad(waveform, (0, _CROP - len(waveform)))

    waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-7)
    return waveform.astype(np.float32)


# ===========================================================================
# Public result container — API contract unchanged
# ===========================================================================
class AudioInferenceResult:
    """Detailed result container from genuine neural audio deepfake detection."""

    def __init__(
        self,
        status: str,                            # "SUCCESS" | "UNAVAILABLE" | "ERROR"
        model_name: str,
        inference_device: str,
        acoustic_manipulation_score: Optional[float],
        peak_chunk_score: Optional[float],
        mean_chunk_score: Optional[float],
        is_deepfake: Optional[bool],
        confidence_level: str,
        duration_sec: float,
        chunk_count: int,
        timeline: List[TimelineSegment],
        indicators: List[str],
        limitations: List[str],
        error_message: Optional[str] = None,
        processing_time_ms: float = 0.0
    ):
        self.status                      = status
        self.model_name                  = model_name
        self.inference_device            = inference_device
        self.acoustic_manipulation_score = acoustic_manipulation_score
        self.peak_chunk_score            = peak_chunk_score
        self.mean_chunk_score            = mean_chunk_score
        self.is_deepfake                 = is_deepfake
        self.confidence_level            = confidence_level
        self.duration_sec                = duration_sec
        self.chunk_count                 = chunk_count
        self.timeline                    = timeline
        self.indicators                  = indicators
        self.limitations                 = limitations
        self.error_message               = error_message
        self.processing_time_ms          = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status":                      self.status,
            "model_name":                  self.model_name,
            "inference_device":            self.inference_device,
            "acoustic_manipulation_score": self.acoustic_manipulation_score,
            "peak_chunk_score":            self.peak_chunk_score,
            "mean_chunk_score":            self.mean_chunk_score,
            "is_deepfake":                 self.is_deepfake,
            "confidence_level":            self.confidence_level,
            "duration_sec":                self.duration_sec,
            "chunk_count":                 self.chunk_count,
            "timeline":                    [t.model_dump() for t in self.timeline],
            "indicators":                  self.indicators,
            "limitations":                 self.limitations,
            "error_message":               self.error_message,
            "processing_time_ms":          self.processing_time_ms,
        }


# ===========================================================================
# Pellav2 Audio Deepfake Detector
# ===========================================================================
class AudioDeepfakeDetector:
    """
    Pellav2 Audio Deepfake Detection Service.
    Model:  Sadanie/pellav2-audio-deepfake-detector
    Backbone: facebook/wav2vec2-xls-r-300m + learned layer-attention + linear head
    Weights: E:\\project models\\pellav2_detector.pt (HF Hub fallback if missing)

    - GPU acceleration (RTX 4050 cu124) with CPU fallback
    - Sliding-window chunking: 4 s window / 1 s step, up to 180 s
    - Per-chunk Pellav2 sigmoid score; peak-weighted aggregation
    - Explicit UNAVAILABLE / ERROR returned on any load or inference failure
    """

    MODEL_NAME = "Sadanie/pellav2-audio-deepfake-detector"
    model_id = "Sadanie/pellav2-audio-deepfake-detector"

    KNOWN_LIMITATIONS = [
        "Pellav2 is fine-tuned on ASVspoof 2019 LA + in-the-wild neural TTS data.",
        "Performance may vary on highly compressed codecs (MP3 <64 kbps, Opus <24 kbps).",
        "Evaluation scores above reflect a single 4-second centre-crop per chunk; "
        "very short clips (<1 s) are zero-padded to meet the minimum input length.",
        "Independent domain calibration is recommended for production deployment.",
    ]

    def __init__(self):
        self.preferred_device  = config.get("hardware.preferred_device", "cuda")
        self.force_cpu         = config.get("hardware.force_cpu", False)

        # Chunking parameters
        self.window_sec        = float(config.get("audio_detection.chunk_duration_sec",      4.0))
        self.overlap_sec       = float(config.get("audio_detection.chunk_overlap_sec",        1.0))
        self.max_duration_sec  = float(config.get("audio_detection.max_audio_duration_sec", 180.0))

        # Aggregation parameters
        self.audio_threshold   = float(config.get("thresholds.audio_fake_threshold",           0.50))
        self.flag_threshold    = float(config.get("thresholds.audio_segment_flag_threshold",   0.45))
        self.peak_weight       = float(config.get("audio_detection.aggregation_peak_weight",   0.70))
        self.mean_weight       = float(config.get("audio_detection.aggregation_mean_weight",   0.30))

        self.preprocessor = AudioPreprocessor(
            target_sample_rate=_SR,
            window_duration_sec=self.window_sec,
            overlap_duration_sec=self.overlap_sec,
            max_duration_sec=self.max_duration_sec,
        )

        # Lazy-loaded internals
        self._model            = None
        self._device           = None
        self._load_error: Optional[str] = None
        self._is_initialized   = False

    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        """Returns True if the Pellav2 model is loaded and ready."""
        if not self._is_initialized:
            self._lazy_initialize()
        return self._model is not None and self._load_error is None

    def get_device(self) -> str:
        """Returns the active inference device string."""
        return str(self._device) if self._device else "unallocated"

    # ------------------------------------------------------------------
    def _lazy_initialize(self):
        """Loads Pellav2 weights lazily on first access."""
        if self._is_initialized:
            return
        self._is_initialized = True

        try:
            import torch

            # Select device
            if not self.force_cpu and torch.cuda.is_available() and self.preferred_device == "cuda":
                self._device = torch.device("cuda:0")
                logger.info(f"Pellav2Detector: NVIDIA GPU — {torch.cuda.get_device_name(0)}")
            else:
                self._device = torch.device("cpu")
                logger.info("Pellav2Detector: CPU mode.")

            # Resolve weights path
            if _LOCAL_WEIGHTS.is_file():
                pt_path = _LOCAL_WEIGHTS
                logger.info(f"Pellav2Detector: Loading weights from local path: {pt_path}")
            else:
                logger.info(
                    f"Pellav2Detector: Local weights not found at {_LOCAL_WEIGHTS}. "
                    "Downloading from HF Hub …"
                )
                from huggingface_hub import hf_hub_download
                hf_hub_download(_PELLAV2_REPO, "config.json")
                pt_path = Path(hf_hub_download(_PELLAV2_REPO, "pellav2_detector.pt"))
                logger.info(f"Pellav2Detector: Downloaded to {pt_path}")

            # Build Detector and load state dict
            logger.info("Pellav2Detector: Building backbone (wav2vec2-xls-r-300m) …")
            Detector    = _build_pellav2_detector()
            self._model = Detector().to(self._device)

            logger.info("Pellav2Detector: Loading fine-tuned weights …")
            state = torch.load(str(pt_path), map_location=self._device)
            self._model.load_state_dict(state)
            self._model.eval()

            if self._device.type == "cuda":
                vram_mb = torch.cuda.memory_allocated() / (1024 ** 2)
                logger.info(f"Pellav2Detector: Ready on {self._device} | VRAM: {vram_mb:.0f} MB")
            else:
                logger.info("Pellav2Detector: Ready on CPU.")

        except ImportError as ie:
            self._load_error = f"Missing neural inference dependency: {ie}"
            logger.warning(self._load_error)
        except Exception as e:
            self._load_error = f"Failed to load Pellav2 weights: {e}"
            logger.error(self._load_error, exc_info=True)

    # ------------------------------------------------------------------
    def analyze_audio(self, audio_source: Any, filename: Optional[str] = None) -> AudioInferenceResult:
        """
        Runs Pellav2 neural audio deepfake detection over audio_source.
        If the model is unavailable or inference fails, returns an explicit
        UNAVAILABLE or ERROR result — never a fabricated safe score.
        """
        start_time = time.time()

        if not self.is_available():
            err = self._load_error or "Pellav2 model is not loaded."
            return AudioInferenceResult(
                status="UNAVAILABLE",
                model_name=self.MODEL_NAME,
                inference_device="none",
                acoustic_manipulation_score=None,
                peak_chunk_score=None,
                mean_chunk_score=None,
                is_deepfake=None,
                confidence_level="UNAVAILABLE",
                duration_sec=0.0,
                chunk_count=0,
                timeline=[],
                indicators=[f"Inference unavailable: {err}"],
                limitations=self.KNOWN_LIMITATIONS,
                error_message=err,
                processing_time_ms=round((time.time() - start_time) * 1000, 2),
            )

        try:
            import torch

            # 1. Decode, mono-mix, resample to 16 kHz
            waveform, sr, original_duration = self.preprocessor.load_audio(audio_source)

            # 2. Sliding-window chunking (4 s / 1 s step)
            chunks = self.preprocessor.chunk_waveform(waveform, sample_rate=sr)
            if not chunks:
                return AudioInferenceResult(
                    status="ERROR",
                    model_name=self.MODEL_NAME,
                    inference_device=str(self._device),
                    acoustic_manipulation_score=None,
                    peak_chunk_score=None,
                    mean_chunk_score=None,
                    is_deepfake=None,
                    confidence_level="UNAVAILABLE",
                    duration_sec=original_duration,
                    chunk_count=0,
                    timeline=[],
                    indicators=["No valid audio chunks could be extracted."],
                    limitations=self.KNOWN_LIMITATIONS,
                    error_message="Audio stream produced zero chunks.",
                    processing_time_ms=round((time.time() - start_time) * 1000, 2),
                )

            # 3. Per-chunk Pellav2 inference
            timeline: List[TimelineSegment] = []
            chunk_scores: List[float]       = []

            for chunk in chunks:
                if chunk.is_silence:
                    # Silence has no meaningful deepfake features; score as 0
                    timeline.append(TimelineSegment(
                        timestamp_start=chunk.start_sec,
                        timestamp_end=chunk.end_sec,
                        score=0.0,
                        flagged=False,
                        segment_type="audio",
                    ))
                    chunk_scores.append(0.0)
                    continue

                # Pellav2 preprocessing: centre-crop + mean-std normalise
                clip = _pellav2_preprocess(chunk.waveform, sr, self.preprocessor)
                x    = torch.from_numpy(clip)[None].to(self._device)

                with torch.inference_mode():
                    p_fake = torch.sigmoid(self._model(x)).item()

                chunk_score = round(min(max(p_fake, 0.0), 1.0), 4)
                chunk_scores.append(chunk_score)

                timeline.append(TimelineSegment(
                    timestamp_start=chunk.start_sec,
                    timestamp_end=chunk.end_sec,
                    score=chunk_score,
                    flagged=(chunk_score >= self.flag_threshold),
                    segment_type="audio",
                ))

            # 4. Aggregation: 70 % peak + 30 % mean (catches short spliced clips)
            peak_score = round(max(chunk_scores), 4)
            mean_score = round(sum(chunk_scores) / len(chunk_scores), 4)
            aggregate_score = round(
                min(max(self.peak_weight * peak_score + self.mean_weight * mean_score, 0.0), 1.0), 4
            )

            is_deepfake = aggregate_score >= self.audio_threshold

            # 5. Confidence calibration
            if peak_score > 0.85 or aggregate_score > 0.75:
                confidence = "HIGH"
            elif aggregate_score < 0.20 and peak_score < 0.35:
                confidence = "HIGH"
            else:
                confidence = "MEDIUM"

            # 6. Human-readable indicators
            indicators = []
            if is_deepfake:
                indicators.append(
                    f"Synthetic voice anomaly detected (Peak: {peak_score:.2f}, "
                    f"Aggregate: {aggregate_score:.2f}). Pellav2 layer-attention "
                    f"signatures are inconsistent with authentic human speech."
                )
                if peak_score > 0.80 and mean_score < 0.50:
                    indicators.append(
                        "Localised spoof spike detected — possible audio splice or "
                        "phrase-level voice insertion."
                    )
                else:
                    indicators.append(
                        "Broad neural vocoder or voice-conversion artefacts present "
                        "across the recording."
                    )
            else:
                indicators.append(
                    f"Acoustic features consistent with authentic human speech "
                    f"(Aggregate: {aggregate_score:.2f}, Peak: {peak_score:.2f})."
                )

            return AudioInferenceResult(
                status="SUCCESS",
                model_name=self.MODEL_NAME,
                inference_device=str(self._device),
                acoustic_manipulation_score=aggregate_score,
                peak_chunk_score=peak_score,
                mean_chunk_score=mean_score,
                is_deepfake=is_deepfake,
                confidence_level=confidence,
                duration_sec=round(original_duration, 2),
                chunk_count=len(chunks),
                timeline=timeline,
                indicators=indicators,
                limitations=self.KNOWN_LIMITATIONS,
                processing_time_ms=round((time.time() - start_time) * 1000, 2),
            )

        except Exception as ex:
            logger.error(f"Pellav2 inference error: {ex}", exc_info=True)
            return AudioInferenceResult(
                status="ERROR",
                model_name=self.MODEL_NAME,
                inference_device=str(self._device) if self._device else "none",
                acoustic_manipulation_score=None,
                peak_chunk_score=None,
                mean_chunk_score=None,
                is_deepfake=None,
                confidence_level="UNAVAILABLE",
                duration_sec=0.0,
                chunk_count=0,
                timeline=[],
                indicators=[f"Inference execution failed: {str(ex)}"],
                limitations=self.KNOWN_LIMITATIONS,
                error_message=str(ex),
                processing_time_ms=round((time.time() - start_time) * 1000, 2),
            )
