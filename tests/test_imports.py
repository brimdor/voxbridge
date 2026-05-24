"""Test that all public APIs are importable."""


def test_import_tts():
    """Test main TTS class import."""
    from voxbridge import TTS

    assert TTS is not None


def test_import_version():
    """Test __version__ import."""
    from voxbridge import __version__

    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_all_exports():
    """Test that __all__ contains expected exports."""
    import voxbridge

    assert hasattr(voxbridge, "__all__")
    assert "TTS" in voxbridge.__all__
    assert "__version__" in voxbridge.__all__
    assert "AVAILABLE_LANGUAGES" in voxbridge.__all__
    assert "SUPPORTED_LANGUAGES" in voxbridge.__all__
    assert "UNKNOWN_LANGUAGE" in voxbridge.__all__


def test_default_model_is_voxbridge_3():
    """Default model should be voxbridge-3 (multilingual, 31 languages)."""
    from voxbridge import AVAILABLE_MODELS, DEFAULT_MODEL

    assert DEFAULT_MODEL == "supertonic-3"
    assert "supertonic-3" in AVAILABLE_MODELS
    assert "supertonic-2" in AVAILABLE_MODELS
    assert "supertonic" in AVAILABLE_MODELS


def test_available_languages_contains_31_codes_plus_na():
    """AVAILABLE_LANGUAGES should expose 31 ISO codes plus the 'na' fallback."""
    from voxbridge import AVAILABLE_LANGUAGES, SUPPORTED_LANGUAGES, UNKNOWN_LANGUAGE

    assert UNKNOWN_LANGUAGE == "na"
    assert len(SUPPORTED_LANGUAGES) == 31
    assert UNKNOWN_LANGUAGE in AVAILABLE_LANGUAGES
    assert AVAILABLE_LANGUAGES == [*SUPPORTED_LANGUAGES, UNKNOWN_LANGUAGE]
    # Spot-check coverage of original v2 languages
    for code in ("en", "ko", "es", "pt", "fr"):
        assert code in SUPPORTED_LANGUAGES
    # Spot-check newly added languages
    for code in ("ja", "de", "ru", "vi", "hi"):
        assert code in SUPPORTED_LANGUAGES
