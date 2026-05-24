"""Tests for the pluggable backend system (TTSBackend protocol, Kokoro, Supertone).

These cover the new architecture introduced in Issue #2.  Supertone tests
exercise the *real* ONNX engine (same as other test modules).  Kokoro tests
use a minimal mock where possible to avoid requiring the ~311 MiB model file
during CI.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from voxbridge.backends import (
    KokoroBackend,
    SupertoneBackend,
    TTSBackend,
    VoiceInfo,
    build_backend,
    get_provider,
    list_providers,
    register_provider,
)
from voxbridge.backends.kokoro import KOKORO_VOICE_MAP, _resolve_kokoro_lang


# -----------------------------------------------------------------------------
# Voice name mapping helpers (no ONNX needed)
# -----------------------------------------------------------------------------

def test_resolve_kokoro_lang_american():
    assert _resolve_kokoro_lang("af_bella") == "en-us"
    assert _resolve_kokoro_lang("am_echo") == "en-us"


def test_resolve_kokoro_lang_british():
    assert _resolve_kokoro_lang("bf_alice") == "en-gb"
    assert _resolve_kokoro_lang("bm_daniel") == "en-gb"


def test_resolve_kokoro_lang_other():
    assert _resolve_kokoro_lang("ef_dora") == "es"
    assert _resolve_kokoro_lang("jf_nezumi") == "ja"
    assert _resolve_kokoro_lang("zf_xiaoxiao") == "zh"


def test_voice_map_complete():
    """Every entry in the map must be unique and non-empty."""
    assert len(set(KOKORO_VOICE_MAP.values())) == len(KOKORO_VOICE_MAP)
    for name, internal in KOKORO_VOICE_MAP.items():
        assert name
        assert internal
        assert internal.startswith(("af_", "am_", "bf_", "bm_", "ef_", "em_",
                                     "ff_", "hf_", "hm_", "if_", "im_", "jf_",
                                     "jm_", "pf_", "pm_", "zf_", "zm_"))


# -----------------------------------------------------------------------------
# Registry unit tests (no ONNX needed)
# -----------------------------------------------------------------------------

def test_list_providers_builtin():
    providers = list_providers()
    assert "supertone" in providers
    assert "kokoro" in providers


def test_get_provider_supertone():
    cls = get_provider("supertone")
    assert cls is SupertoneBackend


def test_get_provider_kokoro():
    cls = get_provider("kokoro")
    assert cls is KokoroBackend


def test_get_provider_unknown():
    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider("nonexistent")


def test_register_provider_roundtrip():
    """Custom providers can be registered at runtime."""

    class Dummy(TTSBackend):
        @property
        def name(self) -> str:
            return "dummy"

        @property
        def sample_rate(self) -> int:
            return 16000

        @property
        def voice_style_names(self) -> list[str]:
            return ["alpha"]

        def load(self) -> None:
            pass

        def synthesize(self, text, voice, *, speed=1.0, lang="en") -> np.ndarray:
            return np.zeros((1, 16000), dtype=np.float32)

        def list_voices(self) -> list[VoiceInfo]:
            return [VoiceInfo(name="alpha", language="en")]

        def get_voice(self, name: str) -> VoiceInfo:
            return VoiceInfo(name=name, language="en")

    register_provider("dummy", Dummy)
    assert "dummy" in list_providers()

    inst = build_backend("dummy")
    assert inst.name == "dummy"
    assert inst.sample_rate == 16000
    wav = inst.synthesize("hello", "alpha")
    assert wav.shape == (1, 16000)


# -----------------------------------------------------------------------------
# KokoroBackend unit tests (mocking kokoro_onnx)
# -----------------------------------------------------------------------------

def test_kokoro_voice_info_from_dict():
    from voxbridge.backends.kokoro import _voice_info_from_id as vi
    info = vi("af_bella")
    assert info.name == "bella"
    assert info.language == "en-us"
    assert info.gender == "female"
    assert info.internal_id == "af_bella"


def test_kokoro_list_voices_static():
    """list_voices() works without loading ONNX."""
    backend = KokoroBackend()
    voices = backend.list_voices()
    assert len(voices) == len(KOKORO_VOICE_MAP)
    names = {v.name for v in voices}
    assert "bella" in names
    assert "echo" in names
    assert "adam" in names


def test_kokoro_get_voice():
    backend = KokoroBackend()
    v = backend.get_voice("bella")
    assert v.internal_id == "af_bella"


def test_kokoro_get_voice_unknown():
    backend = KokoroBackend()
    with pytest.raises(ValueError, match="Unknown Kokoro voice"):
        backend.get_voice("nonexistent")


def test_kokoro_supports_lang():
    backend = KokoroBackend()
    assert backend.supports_lang("en-us")
    assert backend.supports_lang("en")
    assert backend.supports_lang("ja")
    assert not backend.supports_lang("de")


@pytest.mark.skipif(True, reason="requires real ONNX model file (~311 MiB)")
def test_kokoro_load_real(self):
    """Optional: run with model file present to exercise real load."""
    backend = KokoroBackend()
    backend.load()
    assert backend._kokoro is not None
    assert backend.name == "kokoro"
    assert backend.sample_rate == 24000


# -----------------------------------------------------------------------------
# SupertoneBackend (lightweight integration tests)
# -----------------------------------------------------------------------------

@pytest.mark.slow  # downloads models from HuggingFace if missing
@pytest.mark.timeout(120)
def test_supertone_backend_properties():
    """SupertoneBackend default state before loading."""
    backend = SupertoneBackend()
    assert backend.name == "supertone"
    assert backend.sample_rate == 44100  # preset; will update on load


@pytest.mark.slow
@pytest.mark.timeout(120)
def test_supertone_backend_load():
    backend = SupertoneBackend(model="supertonic-3")
    backend.load()
    assert backend._engine is not None
    assert backend.sample_rate == 44100  # Supertonic-3 sample rate


@pytest.mark.slow
@pytest.mark.timeout(120)
def test_supertone_backend_list_voices():
    backend = SupertoneBackend(model="supertonic-3")
    backend.load()
    voices = backend.list_voices()
    names = [v.name for v in voices]
    assert "M1" in names
    assert "F1" in names


@pytest.mark.slow
@pytest.mark.timeout(120)
def test_supertone_backend_get_voice():
    backend = SupertoneBackend(model="supertonic-3")
    backend.load()
    v = backend.get_voice("M1")
    assert v.name == "M1"
    assert v.provider == "supertone"


def test_supertone_backend_get_voice_unknown():
    backend = SupertoneBackend(model="supertonic-3")
    with pytest.raises(ValueError, match="Unknown Supertone voice"):
        backend.get_voice("nonexistent")


@pytest.mark.slow
@pytest.mark.timeout(120)
def test_supertone_backend_synthesize():
    backend = SupertoneBackend(model="supertonic-3")
    backend.load()
    wav = backend.synthesize("Hello world.", voice="M1", speed=1.0, lang="na", total_steps=4)
    assert wav.ndim == 2
    assert wav.shape[0] == 1
    assert wav.shape[1] > 0


# -----------------------------------------------------------------------------
# TTS class dispatch tests — ensure provider switching works end-to-end
# -----------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.timeout(120)
def test_tts_supertone_dispatch():
    from voxbridge.pipeline import TTS
    tts = TTS(model="supertonic-3", provider="supertone")
    assert tts.provider == "supertone"
    style = tts.get_voice_style("M1")
    wav, dur = tts.synthesize("Hello world", voice_style=style, total_steps=4, verbose=False)
    assert wav.shape[0] == 1
    assert dur > 0


def test_tts_kokoro_mock():
    """Mock kokoro_onnx so no real model download/load is required."""
    from voxbridge.pipeline import TTS

    mock_backend = MagicMock()
    mock_backend.sample_rate = 24000
    mock_backend.voice_style_names = ["bella", "echo"]
    mock_backend.synthesize.return_value = np.zeros((1, 24000), dtype=np.float32)

    with patch("voxbridge.backends.build_backend", return_value=mock_backend):
        tts = TTS(provider="kokoro")
        assert tts.provider == "kokoro"
        assert tts.sample_rate == 24000
        wav, dur = tts.synthesize("Hello", voice_style="bella")
        assert wav.shape == (1, 44100)  # resampled to 44100
        assert pytest.approx(dur, 0.01) == 1.0
        mock_backend.synthesize.assert_called_once()


def test_tts_unknown_provider():
    from voxbridge.pipeline import TTS
    with pytest.raises(ValueError, match="Unknown provider"):
        TTS(provider="not_a_provider")
