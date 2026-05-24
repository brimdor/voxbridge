"""Tests for model loading and voice style management."""

from pathlib import Path

import pytest

from voxbridge.config import MODEL_CONFIGS
from voxbridge.loader import (
    get_cache_dir,
    has_all_onnx_modules,
    list_available_voice_style_names,
    load_voice_style_from_name,
)


def test_get_cache_dir():
    """Test cache directory creation."""
    cache_dir = get_cache_dir()
    assert isinstance(cache_dir, Path)
    assert cache_dir.exists()
    assert cache_dir.is_dir()


# ---------------------------------------------------------------------------
# VOXBRIDGE_CACHE_DIR env var must be honored on every code path — including
# get_cache_dir(model_name=...) which used to silently fall through to
# ~/.cache/<model>. Regression guard for the issue fixed in 1.3.1.
# ---------------------------------------------------------------------------


def test_get_cache_dir_honors_env_var_with_model_name(monkeypatch, tmp_path):
    """``get_cache_dir("supertonic-3")`` must respect ``VOXBRIDGE_CACHE_DIR``.

    This is the exact reproduction from the bug report: prior to the fix the
    env var was only consulted on the ``model_name is None`` branch, so the
    default ``TTS(model="supertonic-3")`` call shape silently ignored it.
    """
    monkeypatch.setenv("VOXBRIDGE_CACHE_DIR", str(tmp_path))
    assert get_cache_dir("supertonic-3") == tmp_path
    assert get_cache_dir("supertonic-2") == tmp_path
    assert get_cache_dir("supertonic") == tmp_path


def test_get_cache_dir_honors_env_var_without_model_name(monkeypatch, tmp_path):
    monkeypatch.setenv("VOXBRIDGE_CACHE_DIR", str(tmp_path))
    assert get_cache_dir() == tmp_path
    assert get_cache_dir(None) == tmp_path


def test_get_cache_dir_default_paths_per_model(monkeypatch):
    """Env var unset: each model resolves to its own ``~/.cache/<cache_dir>``.

    Regression guard so the fix doesn't accidentally collapse every model
    onto a single path when the env var is *not* set.
    """
    monkeypatch.delenv("VOXBRIDGE_CACHE_DIR", raising=False)
    home_cache = Path.home() / ".cache"
    for model_name, cfg in MODEL_CONFIGS.items():
        assert get_cache_dir(model_name) == home_cache / cfg["cache_dir"]


def test_get_cache_dir_resolution_is_lazy(monkeypatch, tmp_path):
    """The env var is resolved on every call, not snapshotted at import.

    Locks in lazy evaluation against any future regression to the previous
    module-level ``DEFAULT_CACHE_DIR = os.getenv(...)`` pattern.
    """
    monkeypatch.delenv("VOXBRIDGE_CACHE_DIR", raising=False)
    first = get_cache_dir("supertonic-3")
    assert first != tmp_path  # baseline: still the default

    monkeypatch.setenv("VOXBRIDGE_CACHE_DIR", str(tmp_path))
    second = get_cache_dir("supertonic-3")
    assert second == tmp_path  # env var picked up *after* the first call


def test_has_all_onnx_modules():
    """Test ONNX module checking."""
    cache_dir = get_cache_dir()
    # Should return True if models are downloaded, False otherwise
    result = has_all_onnx_modules(cache_dir)
    assert isinstance(result, bool)


def test_list_voice_styles():
    """Test listing voice styles."""
    cache_dir = get_cache_dir()
    try:
        styles = list_available_voice_style_names(cache_dir)
        assert isinstance(styles, list)
        # If models are downloaded, should have styles
        if styles:
            assert all(isinstance(s, str) for s in styles)
    except FileNotFoundError:
        # OK if models not downloaded yet
        pass


def test_load_voice_style_nonexistent():
    """Test loading non-existent voice style raises error."""
    cache_dir = get_cache_dir()

    with pytest.raises(FileNotFoundError):
        load_voice_style_from_name(cache_dir, "NonExistentVoice12345")
