import os
import uuid
import time
from pathlib import Path
from typing import Tuple, Generator
from contextlib import contextmanager
from fastapi import UploadFile, HTTPException

from src.core.config import config
from src.core.logging import logger

# Recognized file signatures (magic bytes)
MAGIC_SIGNATURES = {
    # Video
    "mp4": [b"ftyp", b"moov", b"mdat"],  # typically at offset 4
    "mkv": [b"\x1a\x45\xdf\xa3"],
    "webm": [b"\x1a\x45\xdf\xa3"],
    "avi": [b"RIFF"],  # with 'AVI ' at offset 8
    "mov": [b"ftyp", b"moov", b"mdat", b"wide", b"skip"],
    "mpg": [b"\x00\x00\x01\xba", b"\x00\x00\x01\xb3"],
    # Audio
    "wav": [b"RIFF"],  # with 'WAVE' at offset 8
    "mp3": [b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2", b"\xff\xfa"],
    "mpeg": [b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2", b"\xff\xfa", b"\x00\x00\x01\xba", b"\x00\x00\x01\xb3"],
    "mpga": [b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2", b"\xff\xfa"],
    "aac": [b"\xff\xf1", b"\xff\xf9", b"ID3"],
    "ogg": [b"OggS"],
    "flac": [b"fLaC"],
    "m4a": [b"ftyp"],
}

def validate_file_signature(header: bytes, ext: str) -> bool:
    """Validates the beginning of a file against known magic bytes."""
    ext_clean = ext.lower().lstrip(".")
    if len(header) < 4:
        return False

    if ext_clean in ["mp4", "m4a", "mov"]:
        # Check offset 4..12 or header start
        sub = header[4:12]
        return any(sig in sub or sig in header[:8] for sig in MAGIC_SIGNATURES.get(ext_clean, []))

    if ext_clean == "avi":
        return header[:4] == b"RIFF" and len(header) >= 12 and header[8:12] == b"AVI "

    if ext_clean == "wav":
        return header[:4] == b"RIFF" and len(header) >= 12 and header[8:12] == b"WAVE"

    expected = MAGIC_SIGNATURES.get(ext_clean, [])
    if not expected:
        return True  # Fallback if unconfigured signature

    return any(header.startswith(sig) for sig in expected)

def get_temp_upload_dir() -> Path:
    raw_dir = config.get("uploads.temp_dir", "temp/uploads")
    path = Path(raw_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path

def cleanup_stale_uploads(max_age_seconds: int = 3600) -> int:
    """Cleans up leftover upload temp files older than max_age_seconds."""
    temp_dir = get_temp_upload_dir()
    count = 0
    now = time.time()
    try:
        for p in temp_dir.glob("*.tmp"):
            if p.is_file() and (now - p.stat().st_mtime > max_age_seconds):
                p.unlink(missing_ok=True)
                count += 1
    except Exception as e:
        logger.warning(f"Error during stale upload cleanup: {e}")
    return count

@contextmanager
def disk_stream_upload(upload_file: UploadFile) -> Generator[Tuple[Path, int], None, None]:
    """
    Streams an uploaded file in bounded chunks to a temporary disk file.
    Validates file extension, upload size limits, and magic bytes.
    Guarantees file cleanup in all success and failure conditions.
    """
    filename = upload_file.filename or "unknown.mp4"
    ext = Path(filename).suffix.lower()

    allowed_video = config.get("uploads.allowed_video_extensions", [".mp4", ".mkv", ".avi", ".mov", ".webm"])
    allowed_audio = config.get("uploads.allowed_audio_extensions", [".wav", ".mp3", ".ogg", ".flac", ".m4a"])
    allowed_all = set(allowed_video + allowed_audio)

    if ext not in allowed_all:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{ext}'. Allowed extensions: {', '.join(sorted(allowed_all))}"
        )

    max_mb = config.max_upload_mb
    max_bytes = max_mb * 1024 * 1024
    chunk_size = 64 * 1024  # 64 KB

    temp_dir = get_temp_upload_dir()
    temp_path = temp_dir / f"upload_{uuid.uuid4().hex}.tmp"
    total_bytes = 0
    first_chunk = True

    try:
        with open(temp_path, "wb") as f_out:
            while True:
                chunk = upload_file.file.read(chunk_size)
                if not chunk:
                    break

                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Uploaded file exceeds maximum limit of {max_mb}MB."
                    )

                if first_chunk:
                    enforce_sig = config.get("uploads.enforce_file_signatures", True)
                    if enforce_sig and not validate_file_signature(chunk, ext):
                        raise HTTPException(
                            status_code=400,
                            detail=f"File content does not match declared extension '{ext}' (invalid magic bytes signature)."
                        )
                    first_chunk = False

                f_out.write(chunk)

        if total_bytes == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes).")

        yield temp_path, total_bytes

    finally:
        # Guarantee cleanup even on errors or client disconnects
        if temp_path.exists():
            try:
                temp_path.unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to remove temp upload file {temp_path}: {e}")
