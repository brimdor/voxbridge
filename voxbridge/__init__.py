"""VoxBridge — Open TTS engine: local, private, expressiveness unlocked.

VoxBridge is a high-performance, on-device text-to-speech system powered by
ONNX Runtime. It delivers state-of-the-art speech synthesis with unprecedented
speed and efficiency. Forked from the MIT-licensed Supertonic SDK with open
expression support and text normalization.

VoxBridge-3 (default) supports multilingual synthesis across 31 languages:

    en, ko, ja, ar, bg, cs, da, de, el, es, et, fi, fr, hi, hr, hu, id,
    it, lt, lv, nl, pl, pt, ro, ru, sk, sl, sv, tr, uk, vi

For text whose language is unknown or outside this set, pass ``lang="na"``
to fall back to the language-agnostic ``<na>`` token.

Example:
    ```python
    from voxbridge import TTS

    tts = TTS()
    style = tts.get_voice_style("M1")  # 10 built-in voices: M1–M5, F1–F5

    # English — pass an ISO code to opt into language-specific handling
    wav, sr = tts.synthesize("Welcome to VoxBridge!", voice_style=style, lang="en")

    # Korean
    wav_ko, _ = tts.synthesize("안녕하세요!", voice_style=style, lang="ko")

    # Default: `lang=None` resolves to the "na" fallback for VoxBridge-3,
    # so unknown text just works without picking a code.
    wav_na, _ = tts.synthesize("Some text", voice_style=style)

    tts.save_audio(wav, "output.wav")
    ```

One-shot convenience (creates/disposes a temporary TTS instance):
    ```python
    from voxbridge import synthesize, save_audio
    wav, sr = synthesize("Hello world!", voice="bella", provider="kokoro")
    save_audio(wav, "hello.wav", sample_rate=sr)
    ```
"""

from __future__ import annotations

import logging

import numpy as np

from .backends import build_backend, get_provider, list_providers
from .config import (
    AVAILABLE_LANGUAGES,
    AVAILABLE_MODELS,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    SUPPORTED_LANGUAGES,
    UNKNOWN_LANGUAGE,
)
from .core import Style, UnicodeProcessor
from .expressions import ExpressionProcessor
from .normalizer import Normalizer
from .pipeline import TTS


def synthesize(
    text: str,
    voice: str = "bella",
    provider: str = "kokoro",
    *,
    speed: float = 1.0,
    lang: str | None = "en",
    model: str = DEFAULT_MODEL,
    total_steps: int = 8,
    expressions: bool = True,
    normalizer: bool = True,
) -> tuple[np.ndarray, int]:
    """One-shot speech synthesis. Creates a temporary TTS instance.

    Args:
        text: Text to synthesize.
        voice: Voice name (e.g. ``'bella'``, ``'sky'``, ``'M1'``).
        provider: ``'kokoro'`` or ``'supertone'``.
        speed: Speech-rate multiplier (default 1.0).
        lang: Language code. ``None`` for auto / ``'na'`` fallback.
        model: Supertone model name (only used when provider='supertone').
        total_steps: Quality steps (Supertone only).
        expressions: Enable expression tag processing.
        normalizer: Enable text normalization.

    Returns:
        Tuple of ``(waveform, sample_rate_hz)``.
    """
    tts = TTS(
        model=model,
        provider=provider,
        normalizer=normalizer,
        expressions=expressions,
    )
    voice_style = voice if provider == "kokoro" else tts.get_voice_style(voice)
    return tts.synthesize(
        text,
        voice_style=voice_style,
        speed=speed,
        lang=lang,
        total_steps=total_steps,
    )


def save_audio(
    wav: np.ndarray,
    output_path: str,
    sample_rate: int = 44100,
) -> None:
    """Save a waveform to a WAV file.

    Args:
        wav: Audio array of shape ``(1, N)`` or ``(N,)``.
        output_path: Destination file path.
        sample_rate: Sample rate in Hz (default 44100).
    """
    try:
        import soundfile as sf  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "soundfile library is required to save audio. "
            "Install it with: pip install soundfile"
        ) from e

    from pathlib import Path

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), wav.squeeze(), sample_rate)


__version__ = "0.2.3"

__all__ = [
    "TTS",
    "Style",
    "UnicodeProcessor",
    "Normalizer",
    "ExpressionProcessor",
    "AVAILABLE_LANGUAGES",
    "AVAILABLE_MODELS",
    "DEFAULT_LANGUAGE",
    "DEFAULT_MODEL",
    "SUPPORTED_LANGUAGES",
    "UNKNOWN_LANGUAGE",
    "__version__",
    "list_providers",
    "get_provider",
    "build_backend",
    "synthesize",
    "save_audio",
]

# Configure logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
