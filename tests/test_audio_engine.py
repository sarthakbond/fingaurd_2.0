import io
import wave
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.services.audio_preprocessor import AudioPreprocessor
from src.services.audio_deepfake_detector import AudioDeepfakeDetector
from src.services.deepfake_engine import DeepfakeEngine
from src.models.media import MediaScanRequest

def create_synthetic_wav(duration_sec: float = 3.0, sample_rate: int = 16000, freq_hz: float = 440.0, channels: int = 1) -> bytes:
    """Helper to generate in-memory PCM 16-bit WAV audio for automated unit testing."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    waveform = (0.5 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
    if channels > 1:
        waveform = np.stack([waveform] * channels, axis=-1)

    raw_int16 = (waveform * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(raw_int16.tobytes())
    return buf.getvalue()

# ============================================================================
# PART 1: SYNTHETIC UNIT TESTS (PREPROCESSOR & SIGNAL NORMALIZATION)
# ============================================================================

def test_audio_preprocessor_resampling_and_mono():
    """Unit Test: Verifies stereo 44.1kHz audio downmixes to mono and resamples to 16kHz."""
    preprocessor = AudioPreprocessor(target_sample_rate=16000)
    stereo_wav = create_synthetic_wav(duration_sec=2.0, sample_rate=44100, channels=2)

    waveform, sr, duration = preprocessor.load_audio(stereo_wav)
    assert sr == 16000
    assert waveform.ndim == 1  # Mono
    assert round(duration, 1) == 2.0
    assert abs(len(waveform) - (16000 * 2.0)) < 100

def test_audio_preprocessor_chunking_sliding_window():
    """Unit Test: Verifies temporal sliding-window boundaries and overlap."""
    preprocessor = AudioPreprocessor(
        target_sample_rate=16000,
        window_duration_sec=4.0,
        overlap_duration_sec=1.0,
        max_duration_sec=180.0
    )
    long_wav = create_synthetic_wav(duration_sec=10.0, sample_rate=16000)
    waveform, sr, _ = preprocessor.load_audio(long_wav)
    chunks = preprocessor.chunk_waveform(waveform, sample_rate=sr)

    # 10s audio with window=4s and step=3s:
    # Chunk 0: 0.0 to 4.0s
    # Chunk 1: 3.0 to 7.0s
    # Chunk 2: 6.0 to 10.0s (covers to end of stream)
    assert len(chunks) == 3
    assert chunks[0].start_sec == 0.0
    assert chunks[0].end_sec == 4.0
    assert chunks[1].start_sec == 3.0
    assert chunks[1].end_sec == 7.0
    assert chunks[2].start_sec == 6.0
    assert chunks[2].end_sec == 10.0

def test_audio_preprocessor_silence_detection():
    """Unit Test: Verifies RMS silence detector flags zero-signal buffers."""
    preprocessor = AudioPreprocessor(target_sample_rate=16000, silence_rms_threshold=1e-4)
    silent_waveform = np.zeros(16000 * 4, dtype=np.float32)
    chunks = preprocessor.chunk_waveform(silent_waveform, sample_rate=16000)

    assert len(chunks) == 1
    assert chunks[0].is_silence is True
    assert chunks[0].rms_energy == 0.0

def test_audio_preprocessor_short_clip_padding():
    """Unit Test: Verifies clips shorter than window length are padded safely."""
    preprocessor = AudioPreprocessor(target_sample_rate=16000, window_duration_sec=4.0)
    short_wav = create_synthetic_wav(duration_sec=0.8, sample_rate=16000)
    waveform, sr, _ = preprocessor.load_audio(short_wav)
    chunks = preprocessor.chunk_waveform(waveform, sample_rate=sr)

    assert len(chunks) == 1
    assert len(chunks[0].waveform) == 16000 * 4  # Padded to 4s window
    assert chunks[0].end_sec == 0.8  # Original end timestamp preserved

# ============================================================================
# PART 2: ERROR HANDLING & TRANSPARENCY TESTS
# ============================================================================

def test_audio_detector_unavailable_mode_transparency():
    """Unit Test: Verifies detector returns explicit UNAVAILABLE status and None score when not ready."""
    detector = AudioDeepfakeDetector()
    detector._model = None
    detector._load_error = "Mock uninitialized state"

    res = detector.analyze_audio(b"dummy_bytes", filename="test.wav")
    assert res.status == "UNAVAILABLE"
    assert res.acoustic_manipulation_score is None  # Must NOT return 0.0
    assert res.confidence_level == "UNAVAILABLE"
    assert res.is_deepfake is None

def test_audio_detector_corrupt_file_handling():
    """Unit Test: Verifies invalid/corrupted audio bytes produce explicit ERROR result."""
    detector = AudioDeepfakeDetector()
    # Provide corrupted non-audio bytes
    res = detector.analyze_audio(b"NOT_A_VALID_WAV_HEADER_CORRUPT_DATA", filename="corrupt.wav")
    assert res.status in ["ERROR", "UNAVAILABLE"]
    assert res.acoustic_manipulation_score is None

# ============================================================================
# PART 3: REAL / INTEGRATED MODEL INFERENCE TESTS
# ============================================================================

def test_deepfake_engine_audio_pipeline_execution():
    """Integration Test: Verifies DeepfakeEngine processes audio with real model (or graceful fallback)."""
    engine = DeepfakeEngine()
    test_wav = create_synthetic_wav(duration_sec=4.5, sample_rate=16000)
    req = MediaScanRequest()

    res = engine.scan_media(req, file_bytes=test_wav, filename="test_sample.wav")
    assert res.media_type == "audio"
    assert len(res.benchmarks_evaluated) >= 3

    if res.status == "COMPLETED":
        assert res.audio_inference_mode == "genuine_neural_wav2vec2"
        assert res.audio_model_identity == "Vansh180/deepfake-audio-wav2vec2"
        assert res.audio_chunk_count >= 1
        assert res.acoustic_manipulation_score is not None
        assert 0.0 <= res.acoustic_manipulation_score <= 1.0
        assert len(res.audio_limitations) > 0
    else:
        # If running in environment where weights couldn't download
        assert res.status.startswith("INFERENCE_")
        assert res.acoustic_manipulation_score is None
        assert res.confidence_level == "UNAVAILABLE"

def test_api_scan_audio_file_upload_endpoint():
    """API Integration Test: Tests POST /api/scan/media-file with valid WAV upload."""
    client = TestClient(app)
    test_wav = create_synthetic_wav(duration_sec=2.5, sample_rate=16000)
    files = {"file": ("test_advisor_audio.wav", test_wav, "audio/wav")}

    res = client.post("/api/scan/media-file", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["filename"] == "test_advisor_audio.wav"
    assert data["media_type"] == "audio"

    if data["status"] == "COMPLETED":
        assert data["audio_inference_mode"] == "genuine_neural_wav2vec2"
        assert data["acoustic_manipulation_score"] is not None
        assert len(data["timeline"]) >= 1
    else:
        assert data["status"].startswith("INFERENCE_")
        assert data["acoustic_manipulation_score"] is None
