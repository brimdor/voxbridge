"""Tests for regression / new-feature guards."""

from __future__ import annotations

from voxbridge.server.schemas import HealthResponse


def test_health_queue_depth_field_exists():
    """Health schema must contain queue_depth so clients can observe load."""
    assert "queue_depth" in HealthResponse.model_fields


def test_health_max_synth_seconds_field_exists():
    """Health schema must contain max_synth_seconds so clients know timeouts."""
    assert "max_synth_seconds" in HealthResponse.model_fields


# ---------------------------------------------------------------------------
# Config sanity checks
# ---------------------------------------------------------------------------

def test_max_total_steps_is_positive():
    """MAX_TOTAL_STEPS must be a positive integer."""
    from voxbridge import config
    assert config.MAX_TOTAL_STEPS > 0


# ---------------------------------------------------------------------------
# CLI argument safety
# ---------------------------------------------------------------------------

def test_cli_parser_allows_safe_output_path():
    """The --output flag exists on "synthesize" and accepts a clean absolute path."""
    from voxbridge.cli import create_parser
    parser = create_parser()
    args = parser.parse_args(["synthesize", "--output", "/tmp/test.wav", "hello"])
    assert args.output == "/tmp/test.wav"


# ---------------------------------------------------------------------------
# Normalizer sanity checks
# ---------------------------------------------------------------------------

def test_normalizer_callable():
    """Normalizer must expose a .normalize() method."""
    from voxbridge.normalizer import Normalizer
    n = Normalizer()
    assert hasattr(n, "normalize")


# ---------------------------------------------------------------------------
# Routes module sanity
# ---------------------------------------------------------------------------

def test_synth_timeout_importable():
    """The synthesis timeout should be importable from the config module."""
    from voxbridge import config
    assert hasattr(config, "MAX_SYNTH_SECONDS")
    assert config.MAX_SYNTH_SECONDS is None or config.MAX_SYNTH_SECONDS >= 0


# ---------------------------------------------------------------------------
# Schema sanity
# ---------------------------------------------------------------------------

def test_pydantic_steps_has_boundaries():
    """The TTSRequest steps field must expose minimum/maximum in its schema."""
    from voxbridge.server.schemas import TTSRequest
    info = TTSRequest.model_fields["steps"]
    # Pydantic v2 stores constraints in info.metadata as Annotated constraints
    constraints = {type(m).__name__: m for m in info.metadata}
    assert "Ge" in constraints
    assert "Le" in constraints
    assert constraints["Ge"].ge >= 1
    assert constraints["Le"].le > constraints["Ge"].ge


# ---------------------------------------------------------------------------
# Integration: all new env vars are documented
# ---------------------------------------------------------------------------

def test_env_vars_exist():
    """All env-var caps should be present in at least one module."""
    from voxbridge import config, loader
    assert hasattr(config, "MAX_TOTAL_STEPS")
    assert hasattr(config, "MAX_SYNTH_SECONDS")
    assert hasattr(loader, "_MAX_STYLE_ELEMENTS")
