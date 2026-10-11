import time
import threading
from collections import defaultdict
from typing import Dict, List, Optional
from fastapi import Request, HTTPException, status
from src.core.config import config
from src.core.logging import logger

class SlidingWindowRateLimiter:
    """
    Thread-safe in-memory sliding window rate limiter per client IP.
    """

    def __init__(self):
        self._requests: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def _get_client_ip(self, request: Optional[Request]) -> str:
        if not request:
            return "127.0.0.1"
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        client = request.client
        return client.host if client else "127.0.0.1"

    def check(self, request: Optional[Request] = None, custom_limit: Optional[int] = None):
        if not config.rate_limit_enabled:
            return

        limit = custom_limit or config.rate_limit_per_minute
        ip = self._get_client_ip(request)
        now = time.time()
        window = 60.0  # 1 minute

        with self._lock:
            # Purge timestamps outside the rolling 60s window
            cutoff = now - window
            self._requests[ip] = [t for t in self._requests[ip] if t > cutoff]

            if len(self._requests[ip]) >= limit:
                retry_after = int(window - (now - self._requests[ip][0])) + 1
                logger.warning(f"Rate limit exceeded for IP {ip}: {len(self._requests[ip])}/{limit} in 60s.")
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded. Maximum {limit} requests per minute allowed.",
                    headers={"Retry-After": str(max(retry_after, 1))}
                )

            self._requests[ip].append(now)

    def reset(self):
        """Resets all tracking counters (used in tests)."""
        with self._lock:
            self._requests.clear()

rate_limiter = SlidingWindowRateLimiter()

def rate_limit_dependency(request: Request):
    """FastAPI dependency for rate-limiting scan routes."""
    rate_limiter.check(request)
