import os
import time
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from src.core.config import config
from src.core.logging import logger
from src.models.media import TimelineSegment, BenchmarkEvaluationMetadata
from src.services.audio_preprocessor import AudioPreprocessor, AudioChunk

class AudioInferenceResult:
    """Detailed result container from genuine neural audio deepfake detection."""
    def __init__(
        self,
        status: str,  # "SUCCESS", "UNAVAILABLE", "ERROR"
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
        self.status = status
        self.model_name = model_name
        self.inference_device = inference_device
        self.acoustic_manipulation_score = acoustic_manipulation_score
        self.peak_chunk_score = peak_chunk_score
        self.mean_chunk_score = mean_chunk_score
        self.is_deepfake = is_deepfake
        self.confidence_level = confidence_level
        self.duration_sec = duration_sec
        self.chunk_count = chunk_count
        self.timeline = timeline
        self.indicators = indicators
        self.limitations = limitations
        self.error_message = error_message
        self.processing_time_ms = processing_time_ms

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "model_name": self.model_name,
            "inference_device": self.inference_device,
            "acoustic_manipulation_score": self.acoustic_manipulation_score,
            "peak_chunk_score": self.peak_chunk_score,
            "mean_chunk_score": self.mean_chunk_score,
            "is_deepfake": self.is_deepfake,
            "confidence_level": self.confidence_level,
            "duration_sec": self.duration_sec,
            "chunk_count": self.chunk_count,
            "timeline": [t.model_dump() for t in self.timeline],
            "indicators": self.indicators,
            "limitations": self.limitations,
            "error_message": self.error_message,
            "processing_time_ms": self.processing_time_ms
        }

class AudioDeepfakeDetector:
    """
    Genuine Wav2Vec2 Audio Deepfake Detection Service.
    Wraps candidate model Vansh180/deepfake-audio-wav2vec2 with:
    - GPU acceleration (RTX 4050 cu124) and CPU fallback
    - Configurable sliding-window temporal chunking
    - Transparent peak + mean score aggregation
    - Explicit unverified / unavailable status if model is missing or fails
    """

    KNOWN_LIMITATIONS = [
        "Trained primarily on ASVspoof 2021 PA physical replay and synthetic speech benchmarks.",
        "May exhibit performance degradation on unseen neural vocoders (e.g. HiFi-GAN, Tortoise, ElevenLabs).",
        "Lossy compression codecs (e.g. WhatsApp voice notes, low-bitrate MP3/Opus) can alter high-frequency artifacts.",
        "Model represents an initial research baseline; independent domain calibration is required."
    ]

    def __init__(self):
        self.model_id = config.get("audio_detection.model_id", "Vansh180/deepfake-audio-wav2vec2")
        self.preferred_device = config.get("hardware.preferred_device", "cuda")
        self.force_cpu = config.get("hardware.force_cpu", False)
        
        # Chunking parameters
        self.window_sec = float(config.get("audio_detection.chunk_duration_sec", 4.0))
        self.overlap_sec = float(config.get("audio_detection.chunk_overlap_sec", 1.0))
        self.max_duration_sec = float(config.get("audio_detection.max_audio_duration_sec", 180.0))
        
        # Aggregation parameters
        self.audio_threshold = float(config.get("thresholds.audio_fake_threshold", 0.50))
        self.flag_threshold = float(config.get("thresholds.audio_segment_flag_threshold", 0.45))
        self.peak_weight = float(config.get("audio_detection.aggregation_peak_weight", 0.70))
        self.mean_weight = float(config.get("audio_detection.aggregation_mean_weight", 0.30))

        self.preprocessor = AudioPreprocessor(
            target_sample_rate=16000,
            window_duration_sec=self.window_sec,
            overlap_duration_sec=self.overlap_sec,
            max_duration_sec=self.max_duration_sec
        )

        self._model = None
        self._feature_extractor = None
        self._device = None
        self._spoof_label_idx: int = 1
        self._load_error: Optional[str] = None
        self._is_initialized = False

    def is_available(self) -> bool:
        """Returns True if PyTorch, transformers, and model weights are ready."""
        if not self._is_initialized:
            self._lazy_initialize()
        return self._model is not None and self._load_error is None

    def get_device(self) -> str:
        """Returns current device ('cuda:0' or 'cpu')."""
        if self._device is not None:
            return str(self._device)
        return "unallocated"

    def _lazy_initialize(self):
        """Loads PyTorch, sets device, and loads model weights lazily on first access."""
        if self._is_initialized:
            return
        self._is_initialized = True

        try:
            import torch
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

            # Select device
            if not self.force_cpu and torch.cuda.is_available() and self.preferred_device == "cuda":
                self._device = torch.device("cuda:0")
                logger.info(f"AudioDeepfakeDetector using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
            else:
                self._device = torch.device("cpu")
                logger.info("AudioDeepfakeDetector using CPU.")

            logger.info(f"Loading pretrained audio model '{self.model_id}'...")
            self._feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_id)
            self._model = AutoModelForAudioClassification.from_pretrained(self.model_id)
            
            # Use FP16 on GPU for reduced VRAM and high throughput
            if self._device.type == "cuda":
                self._model = self._model.half()

            self._model.to(self._device)
            self._model.eval()

            # Inspect label mapping
            id2label = getattr(self._model.config, "id2label", {0: "bonafide", 1: "spoof"})
            logger.info(f"Model id2label mapping: {id2label}")
            
            # Determine spoof label index dynamically
            self._spoof_label_idx = 1
            for idx, label in id2label.items():
                l_lower = str(label).lower()
                if "spoof" in l_lower or "fake" in l_lower or "synthetic" in l_lower:
                    self._spoof_label_idx = int(idx)
                    break
            
            logger.info(f"Resolved spoof label index: {self._spoof_label_idx} ('{id2label.get(self._spoof_label_idx)}')")

        except ImportError as ie:
            self._load_error = f"Missing neural inference dependencies: {ie}"
            logger.warning(self._load_error)
        except Exception as e:
            self._load_error = f"Failed to load model weights for {self.model_id}: {e}"
            logger.error(self._load_error)

    def analyze_audio(self, audio_source: Any, filename: Optional[str] = None) -> AudioInferenceResult:
        """
        Runs genuine neural audio deepfake detection over audio_source.
        If the model is unavailable or fails, returns an explicit UNAVAILABLE or ERROR result.
        NEVER returns 0.0 or marks a recording safe when inference failed.
        """
        start_time = time.time()

        if not self.is_available():
            err = self._load_error or "Audio neural model is not loaded."
            return AudioInferenceResult(
                status="UNAVAILABLE",
                model_name=self.model_id,
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
                processing_time_ms=round((time.time() - start_time) * 1000, 2)
            )

        try:
            import torch

            # 1. Decode, mono-mix, and resample
            waveform, sr, original_duration = self.preprocessor.load_audio(audio_source)

            # 2. Chunk into sliding windows
            chunks = self.preprocessor.chunk_waveform(waveform, sample_rate=sr)
            if not chunks:
                return AudioInferenceResult(
                    status="ERROR",
                    model_name=self.model_id,
                    inference_device=str(self._device),
                    acoustic_manipulation_score=None,
                    peak_chunk_score=None,
                    mean_chunk_score=None,
                    is_deepfake=None,
                    confidence_level="UNAVAILABLE",
                    duration_sec=original_duration,
                    chunk_count=0,
                    timeline=[],
                    indicators=["No valid audio chunks extracted."],
                    limitations=self.KNOWN_LIMITATIONS,
                    error_message="Audio stream could not be chunked.",
                    processing_time_ms=round((time.time() - start_time) * 1000, 2)
                )

            # 3. Perform genuine inference on each chunk
            timeline: List[TimelineSegment] = []
            chunk_scores: List[float] = []

            for chunk in chunks:
                # Handle pure silence chunks: silence has ambiguous synthetic features
                if chunk.is_silence:
                    silence_score = 0.0
                    timeline.append(
                        TimelineSegment(
                            timestamp_start=chunk.start_sec,
                            timestamp_end=chunk.end_sec,
                            score=silence_score,
                            flagged=False,
                            segment_type="audio"
                        )
                    )
                    chunk_scores.append(silence_score)
                    continue

                # Feature extraction
                inputs = self._feature_extractor(
                    chunk.waveform,
                    sampling_rate=16000,
                    return_tensors="pt"
                )

                # Move to device and cast to half if on GPU
                if self._device.type == "cuda":
                    inputs = {k: v.to(self._device).half() for k, v in inputs.items()}
                else:
                    inputs = {k: v.to(self._device) for k, v in inputs.items()}

                with torch.inference_mode():
                    outputs = self._model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1).squeeze(0)
                    spoof_prob = float(probs[self._spoof_label_idx].item())

                chunk_score = round(min(max(spoof_prob, 0.0), 1.0), 4)
                chunk_scores.append(chunk_score)

                timeline.append(
                    TimelineSegment(
                        timestamp_start=chunk.start_sec,
                        timestamp_end=chunk.end_sec,
                        score=chunk_score,
                        flagged=(chunk_score >= self.flag_threshold),
                        segment_type="audio"
                    )
                )

            # 4. Aggregation
            peak_score = round(max(chunk_scores), 4)
            mean_score = round(sum(chunk_scores) / len(chunk_scores), 4)
            
            # Calibrated blend: peak-weighted to catch short spliced clips
            aggregate_score = round(
                min(max(self.peak_weight * peak_score + self.mean_weight * mean_score, 0.0), 1.0),
                4
            )

            is_deepfake = aggregate_score >= self.audio_threshold
            
            # Confidence calibration
            if peak_score > 0.85 or aggregate_score > 0.75:
                confidence = "HIGH"
            elif aggregate_score < 0.20 and peak_score < 0.35:
                confidence = "HIGH"
            else:
                confidence = "MEDIUM"

            # Interpretive indicators
            indicators = []
            if is_deepfake:
                indicators.append(
                    f"Acoustic synthetic anomaly identified (Peak spoof probability: {peak_score:.2f}, Aggregate: {aggregate_score:.2f})."
                )
                if peak_score > 0.80 and mean_score < 0.50:
                    indicators.append("Localized temporal spoof spike detected; possible audio splicing or phrase insertion.")
                else:
                    indicators.append("Broad synthetic vocoder or voice conversion artifacts present across recording.")
            else:
                indicators.append(
                    f"Acoustic features consistent with authentic human speech (Aggregate score: {aggregate_score:.2f}, Peak: {peak_score:.2f})."
                )

            processing_ms = round((time.time() - start_time) * 1000, 2)

            return AudioInferenceResult(
                status="SUCCESS",
                model_name=self.model_id,
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
                processing_time_ms=processing_ms
            )

        except Exception as ex:
            logger.error(f"Error during audio inference: {ex}", exc_info=True)
            return AudioInferenceResult(
                status="ERROR",
                model_name=self.model_id,
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
                processing_time_ms=round((time.time() - start_time) * 1000, 2)
            )
