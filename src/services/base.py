from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from src.models.media import MediaScanRequest, MediaScanResponse
from src.models.advisory import MessageScanRequest, MessageScanResponse
from src.models.infrastructure import AppScanRequest, AppScanResponse
from src.models.verdict import ComprehensiveScanRequest, ComprehensiveScanResponse, AlertNotification

class BaseDeepfakeDetector(ABC):
    """Abstract interface for Video and Audio Deepfake Detection."""

    @abstractmethod
    def scan_media(self, request: MediaScanRequest, file_bytes: Optional[bytes] = None, filename: Optional[str] = None) -> MediaScanResponse:
        """Analyzes media file for visual or acoustic manipulation."""
        pass

class BaseImpersonationDetector(ABC):
    """Abstract interface for Impersonation and Unregistered Advisory Message Detection."""

    @abstractmethod
    def scan_message(self, request: MessageScanRequest) -> MessageScanResponse:
        """Analyzes chat or text message for impersonation and illegal financial solicitation."""
        pass

class BaseTradingAppDetector(ABC):
    """Abstract interface for Fake Trading App and Domain Typo-squatting Detection."""

    @abstractmethod
    def scan_infrastructure(self, request: AppScanRequest) -> AppScanResponse:
        """Analyzes domain URL or APK permissions for broker impersonation or malware traits."""
        pass

class BaseAlertSynthesizer(ABC):
    """Abstract interface for Multi-Factor Risk Synthesis and Alert Generation."""

    @abstractmethod
    def synthesize(self, media_res: Optional[MediaScanResponse], advisory_res: Optional[MessageScanResponse], infra_res: Optional[AppScanResponse]) -> ComprehensiveScanResponse:
        """Synthesizes cross-vector results into a unified verdict and alert feed."""
        pass
