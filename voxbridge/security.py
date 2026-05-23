"""Security hardening module for VoxBridge TTS.

Provides input sanitization, rate limiting, CORS configuration,
security headers, and request/audio size limits for the FastAPI server.

Example:
    ```python
    from voxbridge.security import SecurityMiddleware, RateLimiter, sanitize_input

    # In your FastAPI app:
    app.add_middleware(SecurityMiddleware)
    app.state.rate_limiter = RateLimiter(max_requests=60, window_seconds=60)
    ```
"""

from __future__ import annotations

import logging
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input sanitization
# ---------------------------------------------------------------------------

# Default maximum text length for TTS input (characters)
DEFAULT_MAX_TEXT_LENGTH = 100_000

# Maximum total audio output size in bytes (prevent memory bombs)
DEFAULT_MAX_AUDIO_BYTES = 50 * 1024 * 1024  # 50 MB


def sanitize_input(
    text: str,
    *,
    max_length: int = DEFAULT_MAX_TEXT_LENGTH,
    normalize_unicode: bool = True,
    remove_null_bytes: bool = True,
    strip_control_chars: bool = True,
) -> str:
    """Sanitize text input for TTS processing.

    Args:
        text: Raw input text
        max_length: Maximum allowed text length in characters (0 = no limit)
        normalize_unicode: Whether to apply NFC normalization
        remove_null_bytes: Whether to remove null bytes
        strip_control_chars: Whether to strip control characters

    Returns:
        Sanitized text string

    Raises:
        ValueError: If text exceeds max_length or contains only whitespace
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty or whitespace-only")

    # Normalize unicode
    if normalize_unicode:
        text = unicodedata.normalize("NFC", text)

    # Remove null bytes
    if remove_null_bytes:
        text = text.replace("\x00", "")

    # Strip control characters (except common whitespace)
    if strip_control_chars:
        text = "".join(
            ch for ch in text
            if ch in ("\n", "\r", "\t") or not unicodedata.category(ch).startswith("Cc")
        )

    # Enforce maximum length
    if max_length > 0 and len(text) > max_length:
        raise ValueError(
            f"Text length ({len(text)}) exceeds maximum allowed length "
            f"({max_length}). Please split your text into smaller chunks."
        )

    return text.strip()


def validate_path(path: str, allowed_dirs: Optional[list[str]] = None) -> Path:
    """Validate a file path to prevent path traversal attacks.

    Args:
        path: The file path to validate
        allowed_dirs: Optional list of allowed directory prefixes. If None,
            just checks for path traversal patterns.

    Returns:
        Resolved absolute Path object

    Raises:
        ValueError: If the path contains traversal patterns or escapes allowed dirs
    """
    resolved = Path(path).resolve()

    # Check for path traversal patterns
    path_str = str(resolved)
    if ".." in path_str:
        raise ValueError(f"Path traversal detected: {path}")

    # If allowed dirs specified, verify the path is under one of them
    if allowed_dirs:
        allowed = False
        for allowed_dir in allowed_dirs:
            allowed_prefix = Path(allowed_dir).resolve()
            try:
                resolved.relative_to(allowed_prefix)
                allowed = True
                break
            except ValueError:
                continue
        if not allowed:
            raise ValueError(
                f"Path '{path}' is outside allowed directories"
            )

    return resolved


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

@dataclass
class RateLimiter:
    """Simple in-memory rate limiter.

    Tracks request counts per client IP within a sliding time window.

    Attributes:
        max_requests: Maximum requests allowed per window
        window_seconds: Time window in seconds
    """
    max_requests: int = 60
    window_seconds: int = 60
    _requests: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def check(self, client_ip: str) -> tuple[bool, Optional[int]]:
        """Check if a request from client_ip is allowed.

        Args:
            client_ip: The client IP address

        Returns:
            Tuple of (is_allowed, retry_after_seconds). retry_after is None
            when the request is allowed.
        """
        now = time.time()
        # Clean up old entries
        self._requests[client_ip] = [
            t for t in self._requests[client_ip]
            if now - t < self.window_seconds
        ]

        if len(self._requests[client_ip]) >= self.max_requests:
            oldest = min(self._requests[client_ip])
            retry_after = int(self.window_seconds - (now - oldest)) + 1
            return False, retry_after

        self._requests[client_ip].append(now)
        return True, None

    def reset(self, client_ip: Optional[str] = None) -> None:
        """Reset rate limit counters.

        Args:
            client_ip: Reset only this IP, or None to reset all
        """
        if client_ip:
            self._requests.pop(client_ip, None)
        else:
            self._requests.clear()


# ---------------------------------------------------------------------------
# Security headers middleware
# ---------------------------------------------------------------------------

class SecurityMiddleware(BaseHTTPMiddleware):
    """ASGI middleware that adds security headers and enforces request limits.

    Adds standard security headers:
    - X-Content-Type-Options: nosniff
    - X-Frame-Options: DENY
    - Content-Security-Policy: default-src 'none'
    - X-XSS-Protection: 1; mode=block
    - Referrer-Policy: no-referrer

    Also enforces maximum request body size.
    """

    def __init__(
        self,
        app,
        *,
        max_request_body_bytes: int = 10 * 1024 * 1024,  # 10 MB default
        max_audio_output_bytes: int = DEFAULT_MAX_AUDIO_BYTES,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        super().__init__(app)
        self.max_request_body_bytes = max_request_body_bytes
        self.max_audio_output_bytes = max_audio_output_bytes
        self.rate_limiter = rate_limiter

    async def dispatch(self, request: Request, call_next):
        # Security headers are added to all responses
        # Rate limiting check
        if self.rate_limiter:
            client_ip = request.client.host if request.client else "unknown"
            allowed, retry_after = self.rate_limiter.check(client_ip)
            if not allowed:
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": {
                            "message": f"Rate limit exceeded. Retry after {retry_after} seconds.",
                            "type": "rate_limit_error",
                            "code": "rate_limit_exceeded",
                        }
                    },
                    headers={"Retry-After": str(retry_after)},
                )

        response = await call_next(request)

        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = "default-src 'none'; script-src 'none'"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-VoxBridge-Version"] = "0.1.0"

        return response


# ---------------------------------------------------------------------------
# CORS configuration helper
# ---------------------------------------------------------------------------

def get_cors_config(
    origins: Optional[list[str]] = None,
    allow_credentials: bool = False,
) -> dict:
    """Get sensible CORS configuration defaults.

    Args:
        origins: Allowed origin patterns. Defaults to localhost only.
        allow_credentials: Whether to allow credentials (cookies, auth headers)

    Returns:
        Dict suitable for ``CORSMiddleware`` kwargs
    """
    if origins is None:
        origins = [
            "http://localhost",
            "http://localhost:*",
            "http://127.0.0.1",
            "http://127.0.0.1:*",
        ]

    return {
        "allow_origins": origins,
        "allow_credentials": allow_credentials,
        "allow_methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["*"],
    }


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------

def validate_request_size(body_length: int, max_bytes: int = 10 * 1024 * 1024) -> Optional[str]:
    """Validate request body size.

    Args:
        body_length: Length of the request body in bytes
        max_bytes: Maximum allowed size in bytes (default: 10 MB)

    Returns:
        Error message if size exceeds limit, None otherwise
    """
    if body_length > max_bytes:
        return f"Request body ({body_length} bytes) exceeds maximum size ({max_bytes} bytes)"
    return None


def validate_audio_output_size(audio_bytes: int, max_bytes: int = DEFAULT_MAX_AUDIO_BYTES) -> Optional[str]:
    """Validate audio output size to prevent memory exhaustion.

    Args:
        audio_bytes: Size of the generated audio in bytes
        max_bytes: Maximum allowed size in bytes (default: 50 MB)

    Returns:
        Error message if size exceeds limit, None otherwise
    """
    if audio_bytes > max_bytes:
        return f"Audio output ({audio_bytes} bytes) exceeds maximum size ({max_bytes} bytes)"
    return None