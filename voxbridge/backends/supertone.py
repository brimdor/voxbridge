"""Supertone ONNX backend for VoxBridge.

Extracts the existing VoxBridge inference pipeline into a clean
TTSBackend implementation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

# mypy: disable-error-code="attr-defined"

from . import TTSBackend, VoiceInfo

logger = logging.getLogger(__name__)


class SupertoneBackend(TTSBackend):
    """Supertone ONNX backend wrapping the existing VoxBridge engine."""

    _name: str = "supertone"
    _sample_rate: int = 44100

    def __init__(
        self,
        model: str = "supertonic-3",
        model_dir: Optional[Path | str] = None,
        auto_download: bool = True,
        intra_op_num_threads: Optional[int] = None,
        inter_op_num_threads: Optional[int] = None,
    ) -> None:
        self._model_name = model
        self._model_dir = model_dir
        self._auto_download = auto_download
        self._intra_op = intra_op_num_threads
        self._inter_op = inter_op_num_threads
        self._engine: Optional[object] = None
        self._loaded = False

    @property
    def name(self) -> str:  # type: ignore[override]
        return self._name

    @property
    def sample_rate(self) -> int:  # type: ignore[override]
        return self._sample_rate

    @property
    def model_name(self) -> str:
        return self._model_name

    def load(self) -> None:
        from ..loader import load_model, get_cache_dir

        if self._model_dir is None:
            model_dir = get_cache_dir(self._model_name)
        else:
            model_dir = Path(self._model_dir)

        logger.info("Loading Supertone model %s from %s", self._model_name, model_dir)
        self._engine = load_model(
            model_dir=model_dir,
            auto_download=self._auto_download,
            intra_op_num_threads=self._intra_op,
            inter_op_num_threads=self._inter_op,
            model_name=self._model_name,
        )
        self._sample_rate = self._engine.sample_rate  # type: ignore[attr-defined]
        self._loaded = True
        logger.info("Supertone backend ready: %s @ %d Hz", self._model_name, self.sample_rate)

    @property
    def engine(self):
        """Raw VoxBridge engine reference (for advanced use)."""
        return self._engine

    def synthesize(
        self,
        text: str,
        voice: str,
        *,
        speed: float = 1.0,
        lang: str | None = "en",
        total_steps: int = 8,
    ) -> np.ndarray:
        if self._engine is None:
            raise RuntimeError("SupertoneBackend not loaded. Call .load() first.")

        from ..loader import load_voice_style_from_name, get_cache_dir

        md = get_cache_dir(self._model_name)
        if self._model_dir is not None:
            md = Path(self._model_dir)
        voice_style = load_voice_style_from_name(md, voice)

        if lang == "na":
            lang = None

        wav, _ = self._engine(
            text_list=[text],
            style=voice_style,
            total_step=total_steps,
            speed=speed,
            lang=lang,
        )
        return wav

    def list_voices(self) -> list[VoiceInfo]:
        from ..loader import get_cache_dir, list_available_voice_style_names

        model_dir = get_cache_dir(self._model_name)
        if self._model_dir is not None:
            model_dir = Path(self._model_dir)
        names = list_available_voice_style_names(model_dir)
        return [
            VoiceInfo(name=n, language="en", gender="unknown", provider="supertone", tags=[],
                      internal_id=n)
            for n in names
        ]

    def get_voice(self, name: str) -> VoiceInfo:
        for v in self.list_voices():
            if v.name == name:
                return v
        raise ValueError(f"Unknown Supertone voice: {name!r}")

    def supports_lang(self, lang: str) -> bool:
        from ..config import AVAILABLE_LANGUAGES
        return lang in AVAILABLE_LANGUAGES + ["en-gb", "en-us"]

    @property
    def voice_style_names(self) -> list[str]:
        if self._engine is not None:
            return list(self._engine.voice_style_names)  # type: ignore[attr-defined]
        from ..loader import get_cache_dir, list_available_voice_style_names
        return list_available_voice_style_names(get_cache_dir(self._model_name))
