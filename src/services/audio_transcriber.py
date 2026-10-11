import time
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import numpy as np

from src.core.config import config
from src.core.logging import logger
from src.services.audio_preprocessor import AudioPreprocessor

class AudioTranscriber:
    """
    Speech-to-Text & Captioning Service for FinGuard 2.0.
    Uses openai/whisper-tiny (GPU-accelerated on RTX 4050 with CPU fallback)
    to transcribe spoken audio into text captions, enabling:
    1. Human readability of spoken dialogue in the UI.
    2. Cross-vector fraud inspection (feeding transcribed text into ImpersonationEngine).
    """

    def __init__(self):
        self.enabled = config.get("audio_detection.enable_transcription", True)
        self.model_id = config.get("audio_detection.transcription_model", "openai/whisper-tiny")
        self.preferred_device = config.get("hardware.preferred_device", "cuda")
        self.force_cpu = config.get("hardware.force_cpu", False)
        
        self.preprocessor = AudioPreprocessor(target_sample_rate=16000)
        self._pipeline = None
        self._device = None
        self._is_initialized = False
        self._load_error = None

    def is_available(self) -> bool:
        if not self._is_initialized:
            self._lazy_initialize()
        return self._pipeline is not None and self._load_error is None

    def get_device(self) -> str:
        if not self._is_initialized:
            self._lazy_initialize()
        return self._device or "unavailable"

    def _lazy_initialize(self):
        if self._is_initialized:
            return
        self._is_initialized = True

        if not self.enabled:
            logger.info("Audio transcription is disabled in configuration.")
            return

        try:
            import torch
            from transformers import pipeline

            if not self.force_cpu and torch.cuda.is_available() and self.preferred_device == "cuda":
                self._device = "cuda:0"
                dtype = torch.float16
            else:
                self._device = "cpu"
                dtype = torch.float32

            logger.info(f"Loading transcription model '{self.model_id}' on {self._device}...")
            self._pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.model_id,
                device=self._device,
                dtype=dtype
            )
            logger.info(f"AudioTranscriber successfully initialized with '{self.model_id}'.")

        except Exception as e:
            self._load_error = f"Failed to initialize transcriber: {e}"
            logger.warning(self._load_error)

    def transcribe(self, audio_source: Union[str, Path, bytes, np.ndarray]) -> Dict[str, Any]:
        """
        Transcribes audio into text captions with timestamps.
        Returns: {
            "text": str or None,
            "segments": List[Dict[str, Any]],
            "language": str,
            "is_speech": bool,
            "processing_time_ms": float,
            "error": Optional[str]
        }
        """
        start_time = time.time()

        if not self.is_available():
            return {
                "text": None,
                "segments": [],
                "language": "unknown",
                "is_speech": False,
                "processing_time_ms": round((time.time() - start_time) * 1000, 2),
                "error": self._load_error or "Transcriber unavailable or disabled."
            }

        try:
            # 1. Load, mono-downmix, and resample to 16kHz
            if isinstance(audio_source, np.ndarray):
                waveform = audio_source
                sr = 16000
            else:
                waveform, sr, _ = self.preprocessor.load_audio(audio_source)

            # Check if audio is near-silence
            rms = float(np.sqrt(np.mean(waveform ** 2))) if len(waveform) > 0 else 0.0
            if rms < 1e-4 or len(waveform) < 1600:
                return {
                    "text": "(No audible speech detected)",
                    "segments": [],
                    "language": "en",
                    "is_speech": False,
                    "processing_time_ms": round((time.time() - start_time) * 1000, 2),
                    "error": None
                }

            # 2. Run Whisper inference with timestamps
            result = self._pipeline(
                waveform,
                return_timestamps=True,
                generate_kwargs={"task": "transcribe"}
            )

            raw_text = result.get("text", "").strip()
            raw_chunks = result.get("chunks", [])

            cleaned_segments = []
            for chunk in raw_chunks:
                ts = chunk.get("timestamp", (0.0, 0.0))
                start_sec = ts[0] if ts and ts[0] is not None else 0.0
                end_sec = ts[1] if ts and len(ts) > 1 and ts[1] is not None else start_sec + 2.0
                cleaned_segments.append({
                    "start": round(float(start_sec), 2),
                    "end": round(float(end_sec), 2),
                    "text": chunk.get("text", "").strip()
                })

            is_speech = len(raw_text) > 0 and raw_text != "(No audible speech detected)"
            proc_ms = round((time.time() - start_time) * 1000, 2)

            return {
                "text": raw_text if raw_text else "(No speech recognized)",
                "segments": cleaned_segments,
                "language": "en",
                "is_speech": is_speech,
                "processing_time_ms": proc_ms,
                "error": None
            }

        except Exception as e:
            logger.error(f"Error during audio transcription: {e}")
            return {
                "text": None,
                "segments": [],
                "language": "unknown",
                "is_speech": False,
                "processing_time_ms": round((time.time() - start_time) * 1000, 2),
                "error": str(e)
            }
