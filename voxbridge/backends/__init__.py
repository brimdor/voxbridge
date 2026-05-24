"""TTSBackend protocol and provider registry for VoxBridge.

All TTS providers must implement the ``TTSBackend`` protocol.  The registry
maps human-readable provider names to the concrete implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from voxbridge.expressions import ExpressionProcessor
    from voxbridge.normalizer import Normalizer


@dataclass
class VoiceInfo:
    """Metadata for a single voice."""

    name: str
    language: str
    gender: str | None = None
    provider: str = ""
    tags: list[str] = field(default_factory=list)
    internal_id: str = ""


class TTSBackend(ABC):
    """Abstract base class every TTS provider must satisfy."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider slug — e.g. ``'kokoro'``, ``'supertone'``."""
        ...

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Audio sample rate in Hz."""
        ...

    @property
    @abstractmethod
    def voice_style_names(self) -> list[str]:
        """Human-readable voice names available on this provider."""
        ...

    @abstractmethod
    def load(self) -> None:
        """One-time setup (ONNX session, voice files, tokenizer, etc.)."""
        ...

    @abstractmethod
    def synthesize(
        self,
        text: str,
        voice: str,
        *,
        speed: float = 1.0,
        lang: str | None = "en",
    ) -> tuple[np.ndarray, int]:
        """Return audio waveform + sample rate.

        Args:
            text: Clean text after normalization/expression extraction.
            voice: Voice name as understood by the provider.
            speed: Speech-rate multiplier (default 1.0).
            lang: Language code, or ``None`` if not applicable.

        Returns:
            Tuple of ``(audio_array, sample_rate)`` where audio is a 1-D
            ``np.ndarray`` of shape ``(num_samples,)`` and sample_rate is in Hz.
        """
        ...

    @abstractmethod
    def list_voices(self) -> list[VoiceInfo]:
        """Return all voices this provider supports."""
        ...

    @abstractmethod
    def get_voice(self, name: str) -> VoiceInfo:
        """Look up a voice by human-readable name.  Raise on unknown."""
        ...

    # -- optional hooks for advanced features --

    def get_voice_style(self, name: str):
        """Return a duck-typed dict holding ``provider`` and ``name``."""
        return {"provider": self.name, "name": name}

    def supports_lang(self, lang: str) -> bool:
        """Whether this provider can synthesise in ``lang``.  Default ``True``."""
        return True


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[TTSBackend]] = {}


def register_provider(name: str, cls: type[TTSBackend]) -> None:
    """Register a concrete backend class under ``name``."""
    _REGISTRY[name] = cls


def get_provider(name: str) -> type[TTSBackend]:
    """Return the provider class for ``name``."""
    if name not in _REGISTRY:
        available = ", ".join(_REGISTRY)
        raise ValueError(
            f"Unknown provider '{name}'.  Available providers: {available}"
        )
    return _REGISTRY[name]


def list_providers() -> list[str]:
    """Return all registered provider names."""
    return list(_REGISTRY)


def build_backend(name: str) -> TTSBackend:
    """Instantiate and ``.load()`` a backend by name."""
    cls = get_provider(name)
    instance = cls()
    instance.load()
    return instance


# ---------------------------------------------------------------------------
# Auto-register built-in backends
# ---------------------------------------------------------------------------

from .kokoro import KokoroBackend  # noqa: E402
from .supertone import SupertoneBackend  # noqa: E402

register_provider("kokoro", KokoroBackend)
register_provider("supertone", SupertoneBackend)
