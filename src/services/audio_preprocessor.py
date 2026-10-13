import math
from pathlib import Path
from typing import Tuple, List, Optional, Union
import numpy as np

class AudioChunk:
    """Represents a bounded temporal segment extracted from an audio track."""
    def __init__(
        self,
        chunk_index: int,
        start_sec: float,
        end_sec: float,
        waveform: np.ndarray,
        sample_rate: int,
        is_silence: bool = False,
        rms_energy: float = 0.0
    ):
        self.chunk_index = chunk_index
        self.start_sec = round(start_sec, 3)
        self.end_sec = round(end_sec, 3)
        self.waveform = waveform  # 1D float32 array
        self.sample_rate = sample_rate
        self.is_silence = is_silence
        self.rms_energy = rms_energy

    def duration(self) -> float:
        return self.end_sec - self.start_sec

class AudioPreprocessor:
    """
    Handles robust audio ingestion, mono conversion, 16kHz resampling,
    and sliding-window temporal chunking for Wav2Vec2 inference.
    """

    def __init__(
        self,
        target_sample_rate: int = 16000,
        window_duration_sec: float = 4.0,
        overlap_duration_sec: float = 1.0,
        max_duration_sec: float = 180.0,
        silence_rms_threshold: float = 1e-4
    ):
        self.target_sample_rate = target_sample_rate
        self.window_duration_sec = window_duration_sec
        self.overlap_duration_sec = overlap_duration_sec
        self.max_duration_sec = max_duration_sec
        self.silence_rms_threshold = silence_rms_threshold

    def load_audio(self, source: Union[str, Path, bytes]) -> Tuple[np.ndarray, int, float]:
        """
        Loads an audio file or raw bytes, converts to mono float32, and resamples to target rate.
        Returns: (resampled_mono_waveform, sample_rate, original_duration_sec)
        """
        waveform: Optional[np.ndarray] = None
        sr: int = 0

        # Method 1: Try soundfile if available
        try:
            import soundfile as sf
            import io
            if isinstance(source, bytes):
                data, sr = sf.read(io.BytesIO(source), dtype="float32")
            else:
                data, sr = sf.read(str(source), dtype="float32")
            waveform = data
        except Exception:
            waveform = None

        # Method 2: Try standard library wave module for WAV files
        if waveform is None:
            f = None
            try:
                import wave
                import io
                f = io.BytesIO(source) if isinstance(source, bytes) else open(source, "rb")
                with wave.open(f, "rb") as wf:
                    channels = wf.getnchannels()
                    sr = wf.getframerate()
                    sampwidth = wf.getsampwidth()
                    frames = wf.readframes(wf.getnframes())
                    
                    if sampwidth == 2:
                        raw = np.frombuffer(frames, dtype=np.int16)
                        waveform = (raw.astype(np.float32) / 32768.0)
                    elif sampwidth == 1:
                        raw = np.frombuffer(frames, dtype=np.uint8)
                        waveform = ((raw.astype(np.float32) - 128.0) / 128.0)
                    elif sampwidth == 4:
                        raw = np.frombuffer(frames, dtype=np.int32)
                        waveform = (raw.astype(np.float32) / 2147483648.0)
                    else:
                        raise ValueError(f"Unsupported sample width: {sampwidth}")

                    if channels > 1:
                        waveform = waveform.reshape(-1, channels)
            except Exception as e:
                raise ValueError(f"Failed to decode audio stream: {e}")
            finally:
                if f is not None and not isinstance(source, bytes):
                    try:
                        f.close()
                    except Exception:
                        pass

        if waveform is None or len(waveform) == 0:
            raise ValueError("Decoded audio stream is empty (0 samples).")

        # Convert multi-channel to mono
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1)

        original_duration = len(waveform) / float(sr)

        # Resample to target_sample_rate if needed
        if sr != self.target_sample_rate:
            waveform = self.resample_waveform(waveform, sr, self.target_sample_rate)
            sr = self.target_sample_rate

        # Normalize waveform amplitudes if max exceeds 1.0
        max_abs = np.max(np.abs(waveform))
        if max_abs > 1.0:
            waveform = waveform / max_abs

        return waveform.astype(np.float32), sr, original_duration

    def resample_waveform(self, waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resamples 1D numpy waveform from orig_sr to target_sr."""
        if orig_sr == target_sr:
            return waveform

        # Try scipy resample if available
        try:
            from scipy import signal
            target_len = int(len(waveform) * target_sr / orig_sr)
            return signal.resample(waveform, target_len).astype(np.float32)
        except ImportError:
            pass

        # Fallback: High-precision linear interpolation
        old_indices = np.linspace(0, len(waveform) - 1, num=len(waveform))
        target_len = int(len(waveform) * target_sr / orig_sr)
        new_indices = np.linspace(0, len(waveform) - 1, num=target_len)
        return np.interp(new_indices, old_indices, waveform).astype(np.float32)

    def chunk_waveform(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
        window_duration_sec: Optional[float] = None,
        overlap_duration_sec: Optional[float] = None,
        max_duration_sec: Optional[float] = None
    ) -> List[AudioChunk]:
        """
        Splits waveform into overlapping temporal chunks.
        Applies bounding, minimum padding, and silence detection.
        """
        w_dur = window_duration_sec or self.window_duration_sec
        o_dur = overlap_duration_sec or self.overlap_duration_sec
        m_dur = max_duration_sec or self.max_duration_sec

        window_samples = int(w_dur * sample_rate)
        step_duration_sec = max(w_dur - o_dur, 0.5)
        step_samples = int(step_duration_sec * sample_rate)
        max_samples = int(m_dur * sample_rate)

        # Truncate if audio exceeds maximum allowed duration
        if len(waveform) > max_samples:
            waveform = waveform[:max_samples]

        total_samples = len(waveform)
        total_duration = total_samples / float(sample_rate)

        # Handle audio less than or equal to single window
        if total_samples <= window_samples:
            pad_needed = window_samples - total_samples
            padded = np.pad(waveform, (0, pad_needed), mode="symmetric" if total_samples > 1 else "constant")
            rms = float(np.sqrt(np.mean(waveform ** 2))) if total_samples > 0 else 0.0
            is_silence = rms < self.silence_rms_threshold
            return [
                AudioChunk(
                    chunk_index=0,
                    start_sec=0.0,
                    end_sec=round(total_duration, 3),
                    waveform=padded.astype(np.float32),
                    sample_rate=sample_rate,
                    is_silence=is_silence,
                    rms_energy=rms
                )
            ]

        chunks: List[AudioChunk] = []
        chunk_idx = 0
        curr_sample = 0

        while curr_sample < total_samples:
            end_sample = min(curr_sample + window_samples, total_samples)
            chunk_slice = waveform[curr_sample:end_sample]

            # If the last slice is smaller than window_samples, pad it symmetrically
            if len(chunk_slice) < window_samples:
                pad_needed = window_samples - len(chunk_slice)
                chunk_slice = np.pad(chunk_slice, (0, pad_needed), mode="symmetric")

            start_t = curr_sample / float(sample_rate)
            end_t = min((curr_sample + window_samples) / float(sample_rate), total_duration)

            rms = float(np.sqrt(np.mean(chunk_slice ** 2)))
            is_silence = rms < self.silence_rms_threshold

            chunks.append(
                AudioChunk(
                    chunk_index=chunk_idx,
                    start_sec=start_t,
                    end_sec=end_t,
                    waveform=chunk_slice.astype(np.float32),
                    sample_rate=sample_rate,
                    is_silence=is_silence,
                    rms_energy=rms
                )
            )

            chunk_idx += 1
            if end_sample >= total_samples:
                break
            curr_sample += step_samples

        return chunks
