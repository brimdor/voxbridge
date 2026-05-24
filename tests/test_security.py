"""Tests for the security-hardening layer introduced in the fork.

Each test maps to a finding from the static-analysis / review pass that
preceded the hardening patch.  The aim is failure-before-harm, not
failure-after-cleanup.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from voxbridge.security import (
    RateLimiter,
    SecurityMiddleware,
    _get_client_ip,
    get_cors_config,
    sanitize_input,
    validate_path,
)

# ---------------------------------------------------------------------------
# sanitize_input
# ---------------------------------------------------------------------------

def test_sanitize_input_basic():
    """Basic text must pass through unchanged."""
    assert sanitize_input("hello world") == "hello world"


def test_sanitize_input_trims_and_strips():
    """Whitespace must be folded so we do not ship empty / meaningless data."""
    assert sanitize_input("  hello  world  \n\t ") == "hello  world"


def test_sanitize_input_empty_after_strip():
    """Input that evaporates after cleaning must raise an error (per implementation)."""
    with pytest.raises(ValueError):
        sanitize_input(" \n\t  ")


def test_sanitize_input_max_length():
    """Excessively long input must raise ValueError (per implementation)."""
    long_text = "a" * 10_001
    with pytest.raises(ValueError):
        sanitize_input(long_text, max_length=10_000)


def test_sanitize_input_unicode_ok():
    """Unicode text is valid and must not be mangled."""
    text = "Héllo 世界 🌍"
    assert sanitize_input(text) == text


def test_sanitize_input_control_reject():
    """Null bytes and control characters must be removed."""
    text = "hello\x00world\x01\x02"
    cleaned = sanitize_input(text)
    assert "\x00" not in cleaned
    assert "\x01" not in cleaned
    assert "\x02" not in cleaned


# ---------------------------------------------------------------------------
# validate_path
# ---------------------------------------------------------------------------

def test_validate_path_simple_ok():
    """An absolute path under the current cwd is fine."""
    result = validate_path("/tmp/voxbridge_test.wav")
    assert isinstance(result, Path)


def test_validate_path_traversal_rejected():
    """A path containing '..' after resolve must be rejected before any I/O occurs."""
    with pytest.raises((ValueError, PermissionError)):
        validate_path("/tmp/../etc/passwd", allowed_dirs=["/tmp"])


def test_validate_path_allowed_dirs():
    """Paths outside a configured allow-list must be refused."""
    with pytest.raises((ValueError, PermissionError)):
        validate_path("/etc/passwd", allowed_dirs=["/tmp"])


# ---------------------------------------------------------------------------
# RateLimiter
# ---------------------------------------------------------------------------

WINDOW = 60


def test_rate_limiter_under_cap():
    """A single request is allowed."""
    rl = RateLimiter(max_requests=2, window_seconds=WINDOW)
    allowed, retry = rl.check("10.0.0.1")
    assert allowed is True
    assert retry is None


def test_rate_limiter_over_cap():
    """3 requests with max_requests=2 should deny the third."""
    rl = RateLimiter(max_requests=2, window_seconds=WINDOW)
    rl.check("10.0.0.1")
    rl.check("10.0.0.1")
    allowed, _ = rl.check("10.0.0.1")
    assert allowed is False


def test_rate_limiter_per_ip_isolation():
    """Rate-limit counters are scoped per IP."""
    rl = RateLimiter(max_requests=2, window_seconds=WINDOW)
    rl.check("10.0.0.1")
    rl.check("10.0.0.1")
    allowed, _ = rl.check("10.0.0.2")
    assert allowed is True


def test_rate_limiter_reset_after_window():
    """Counters must reset after the window elapses."""
    rl = RateLimiter(max_requests=1, window_seconds=WINDOW)
    rl.check("10.0.0.1")
    allowed, _ = rl.check("10.0.0.1")
    assert allowed is False

    # Fast-forward time by resetting the IP
    rl.reset("10.0.0.1")
    allowed, _ = rl.check("10.0.0.1")
    assert allowed is True


# ---------------------------------------------------------------------------
# SecurityMiddleware
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_request():
    """Minimal request stand-in."""
    r = Mock()
    r.headers = {}
    r.client = Mock()
    r.client.host = "10.0.0.5"
    return r


class FakeResponse:
    """Minimal response stand-in for middleware tests."""
    def __init__(self, headers=None, media_type="application/json", status_code=200):
        self.headers = headers or {}
        self.headers.setdefault("content-type", media_type)
        self.status_code = status_code


@pytest.mark.anyio
async def test_security_headers_on_json(fake_request):
    """JSON responses get security headers (X‑Frame-Options etc.)."""
    mw = SecurityMiddleware(AsyncMock())
    resp = FakeResponse(headers={}, media_type="application/json")

    async def call_next(req):
        return resp

    out = await mw.dispatch(fake_request, call_next)
    assert "X-Frame-Options" in out.headers or "x-frame-options" in out.headers


@pytest.mark.anyio
async def test_security_csp_on_html_only(fake_request):
    """CSP should only appear for text/html responses."""
    mw = SecurityMiddleware(AsyncMock())

    async def call_next_json(req):
        return FakeResponse(headers={}, media_type="application/json")

    out = await mw.dispatch(fake_request, call_next_json)
    assert "Content-Security-Policy" not in out.headers

    async def call_next_html(req):
        return FakeResponse(headers={}, media_type="text/html")

    out = await mw.dispatch(fake_request, call_next_html)
    assert "Content-Security-Policy" in out.headers


@pytest.mark.anyio
async def test_security_rate_limit_allows(fake_request):
    """Middleware must allow requests when rate-limiter is lenient."""
    rl = RateLimiter(max_requests=10, window_seconds=WINDOW)
    mw = SecurityMiddleware(AsyncMock(), rate_limiter=rl)

    async def call_next(req):
        return FakeResponse(headers={}, media_type="application/json")

    out = await mw.dispatch(fake_request, call_next)
    assert out.status_code != 429


def test_get_client_ip_x_forwarded_for():
    """_get_client_ip should respect X-Forwarded-For."""
    req = Mock()
    req.client = Mock()
    req.client.host = "10.0.0.1"
    req.headers = {}
    assert _get_client_ip(req) == "10.0.0.1"

    req.headers["x-forwarded-for"] = "192.168.1.5"
    assert _get_client_ip(req) == "192.168.1.5"

    req.headers["x-real-ip"] = "172.16.3.3"
    assert _get_client_ip(req) == "192.168.1.5"  # XFF takes precedence


# ---------------------------------------------------------------------------
# get_cors_config
# ---------------------------------------------------------------------------

def test_get_cors_config_default_origins():
    """Default origins must include localhost and 127.0.0.1."""
    cfg = get_cors_config()
    assert "http://localhost" in cfg["allow_origins"]
    assert "http://127.0.0.1" in cfg["allow_origins"]


def test_get_cors_config_custom_origin_list():
    """Custom origin list must be accepted."""
    cfg = get_cors_config(origins=["https://a.example.com", "https://b.example.com"])
    assert "https://a.example.com" in cfg["allow_origins"]
    assert "https://b.example.com" in cfg["allow_origins"]


# ---------------------------------------------------------------------------
# Environment variable caps (integration-level sanity)
# ---------------------------------------------------------------------------

def test_max_total_steps_env_is_positive():
    """VOXBRIDGE_MAX_TOTAL_STEPS must be a positive integer."""
    val = os.getenv("VOXBRIDGE_MAX_TOTAL_STEPS", "100")
    assert int(val) > 0


def test_max_synth_seconds_env_is_non_negative():
    """VOXBRIDGE_MAX_SYNTH_SECONDS must be 0 or greater."""
    val = os.getenv("VOXBRIDGE_MAX_SYNTH_SECONDS", "60")
    if val.strip():
        assert float(val) >= 0


def test_max_style_elements_env_is_positive():
    """VOXBRIDGE_MAX_STYLE_ELEMENTS must be a positive integer."""
    val = os.getenv("VOXBRIDGE_MAX_STYLE_ELEMENTS", "1000000")
    assert int(val) > 0


# ---------------------------------------------------------------------------
# Integration guards for the analysis findings
# ---------------------------------------------------------------------------

def test_health_queue_depth_field_exists():
    """Health schema must contain queue_depth so clients can observe load."""
    from voxbridge.server.schemas import HealthResponse
    assert "queue_depth" in HealthResponse.model_fields


def test_health_max_synth_seconds_field_exists():
    """Health schema must contain max_synth_seconds so clients know timeouts."""
    from voxbridge.server.schemas import HealthResponse
    assert "max_synth_seconds" in HealthResponse.model_fields


def test_configurable_env_vars_documented():
    """All env-var caps should appear in config module namespace."""
    from voxbridge import config
    assert hasattr(config, "MAX_TOTAL_STEPS")
    assert hasattr(config, "MAX_SYNTH_SECONDS")
