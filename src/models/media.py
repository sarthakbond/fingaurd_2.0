from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class TimelineSegment(BaseModel):
    timestamp_start: float = Field(..., description="Start offset in seconds")
    timestamp_end: float = Field(..., description="End offset in seconds")
    score: float = Field(..., ge=0.0, le=1.0, description="Manipulation probability for this segment")
    flagged: bool = Field(..., description="Whether this segment exceeds flag threshold")
    segment_type: str = Field(default="audio", description="Type: audio or video_frame")

class BenchmarkEvaluationMetadata(BaseModel):
    benchmark_name: str = Field(..., description="E.g. FaceForensics++ or DFDC")
    evaluation_metric: str = Field(..., description="E.g. AUC-ROC or Log Loss")
    evaluation_status: str = Field(..., description="Model evaluation or verification status")
    notes: str = Field(..., description="Notes on benchmark alignment and validation")

class MediaScanRequest(BaseModel):
    media_url: Optional[str] = Field(None, description="Optional public or cloud URL of media file")
    sampling_fps: float = Field(default=2.0, ge=0.5, le=10.0, description="Visual frame extraction rate")
    extract_audio: bool = Field(default=True, description="Whether to isolate and analyze audio track")

class MediaScanResponse(BaseModel):
    job_id: str
    status: str
    filename: Optional[str] = None
    media_type: str = Field(..., description="video, audio, or image")
    visual_manipulation_score: float = Field(default=0.0, ge=0.0, le=1.0)
    acoustic_manipulation_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    is_deepfake: bool
    confidence_level: str = Field(..., description="LOW, MEDIUM, HIGH, or UNAVAILABLE")
    timeline: List[TimelineSegment] = Field(default_factory=list)
    key_indicators: List[str] = Field(default_factory=list)
    benchmarks_evaluated: List[BenchmarkEvaluationMetadata] = Field(default_factory=list)
    processing_time_ms: float

    # Phase 2A Neural Audio Metadata (Backward-compatible additions)
    audio_model_identity: Optional[str] = None
    audio_inference_mode: str = Field(
        default="baseline_foundation",
        description="genuine_neural_wav2vec2, baseline_foundation, or unavailable"
    )
    audio_duration_seconds: Optional[float] = None
    audio_chunk_count: int = 0
    audio_peak_chunk_score: Optional[float] = None
    audio_mean_chunk_score: Optional[float] = None
    audio_limitations: List[str] = Field(default_factory=list)

    # Captioning & Spoken Content Inspection
    transcript: Optional[str] = Field(default=None, description="Transcribed audio caption text")
    transcript_segments: List[Dict[str, Any]] = Field(default_factory=list, description="Timestamped speech subtitle chunks")
    spoken_advisory_risk: Optional[float] = Field(default=None, description="Risk score if spoken speech violates advisory regulations")
    spoken_advisory_verdict: Optional[str] = Field(default=None, description="Advisory risk category: SAFE, SUSPICIOUS, or CRITICAL_FRAUD")
    spoken_advisory_indicators: List[str] = Field(default_factory=list, description="Specific illegal advisory signals identified in spoken speech")
