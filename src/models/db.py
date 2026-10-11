from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, Text, DateTime
from src.core.database import Base

class ScanRecord(Base):
    __tablename__ = "scan_records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    scan_id = Column(String(64), unique=True, index=True, nullable=False)
    timestamp = Column(String(64), nullable=False)
    scan_type = Column(String(32), index=True, nullable=False)  # media, message, app, comprehensive
    input_target = Column(String(512), nullable=True)
    raw_payload = Column(Text, nullable=True)  # JSON serialized
    risk_score = Column(Float, nullable=False, default=0.0)
    verdict = Column(String(64), nullable=False)
    processing_time_ms = Column(Float, default=0.0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class JobState(Base):
    __tablename__ = "job_states"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    job_id = Column(String(64), unique=True, index=True, nullable=False)
    status = Column(String(32), index=True, nullable=False)  # PENDING, PROCESSING, COMPLETED, FAILED
    error_message = Column(Text, nullable=True)
    result_payload = Column(Text, nullable=True)  # JSON serialized result
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

class AlertHistory(Base):
    __tablename__ = "alert_history"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    alert_id = Column(String(64), unique=True, index=True, nullable=False)
    timestamp = Column(String(64), nullable=False)
    severity = Column(String(32), index=True, nullable=False)  # INFO, WARNING, CRITICAL
    category = Column(String(64), index=True, nullable=False)
    summary = Column(Text, nullable=False)
    risk_score = Column(Float, nullable=False, default=0.0)
    source = Column(String(255), nullable=True)
    action_required = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
