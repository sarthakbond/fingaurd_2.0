import re
from typing import Tuple, List

class SecurityShield:
    """
    DPDP Act 2023 Compliance & Input Sanitization.
    Redacts sensitive personal identifiable information before processing
    and detects adversarial evasion attempts.
    """

    RE_MOBILE = re.compile(r"(?:\+91[\-\s]?)?[6-9]\d{9}\b")
    RE_PAN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
    RE_AADHAAR = re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b")
    RE_UPI = re.compile(r"\b[a-zA-Z0-9.\-_]{2,64}@[a-zA-Z]{2,32}\b")
    RE_TELEGRAM_LINK = re.compile(r"(?:https?://)?(?:t\.me|telegram\.me)/[a-zA-Z0-9_]+")
    RE_WHATSAPP_LINK = re.compile(r"(?:https?://)?chat\.whatsapp\.com/[a-zA-Z0-9]+")

    ADVERSARIAL_INJECTION_PATTERNS = [
        r"ignore\s+(?:all\s+)?previous\s+instructions",
        r"ignore\s+prior\s+guidelines",
        r"mark\s+(?:this\s+as\s+)?safe",
        r"override\s+fraud\s+verdict",
        r"you\s+are\s+now\s+in\s+developer\s+mode",
    ]

    @classmethod
    def redact_pii(cls, text: str) -> Tuple[str, List[str]]:
        """Redacts PII tokens from input text and returns redacted text + list of redacted types."""
        redacted_types = []
        result = text

        if cls.RE_MOBILE.search(result):
            result = cls.RE_MOBILE.sub("[REDACTED_PHONE]", result)
            redacted_types.append("PHONE_NUMBER")

        if cls.RE_PAN.search(result):
            result = cls.RE_PAN.sub("[REDACTED_PAN]", result)
            redacted_types.append("PAN_CARD")

        if cls.RE_AADHAAR.search(result):
            result = cls.RE_AADHAAR.sub("[REDACTED_AADHAAR]", result)
            redacted_types.append("AADHAAR_NUMBER")

        if cls.RE_UPI.search(result):
            result = cls.RE_UPI.sub("[REDACTED_UPI]", result)
            redacted_types.append("UPI_HANDLE")

        return result, redacted_types

    @classmethod
    def detect_adversarial_injection(cls, text: str) -> bool:
        """Detects prompt-injection phrases attempting to evade fraud detection."""
        lower = text.lower()
        for pat in cls.ADVERSARIAL_INJECTION_PATTERNS:
            if re.search(pat, lower):
                return True
        return False
