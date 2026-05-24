"""Kokoro ONNX backend for VoxBridge.

Wraps ``kokoro-onnx`` which loads a light-weight ONNX model and
voice-style data from NPZ files.

Voice naming convention: internal Kokoro ids like ``"af_bella"``
are aliased to human names in VoxBridge (``"bella"`` → ``"af_bella"``).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

from . import TTSBackend, VoiceInfo

logger = logging.getLogger(__name__)

# Default voice mapping: human name → kokoro internal id.
# Prefix key: a=American, b=British, e=Spanish, f=French, h=Hindi,
#             i=Italian, j=Japanese, p=Portuguese, z=Chinese.
KOKORO_VOICE_MAP: dict[str, str] = {
    # American English (en-us) — female
    "bella":    "af_bella",  "sarah":    "af_sarah",
    "nicole":   "af_nicole", "sky":      "af_sky",
    "jessica":  "af_jessica","river":    "af_river",
    "alloy":    "af_alloy",  "nova":     "af_nova",
    "heart":    "af_heart",  "kore":     "af_kore",
    "aoede":    "af_aoede",
    # American English (en-us) — male
    "adam":     "am_adam",   "echo":     "am_echo",
    "puck":     "am_puck",   "fenrir":   "am_fenrir",
    "michael":  "am_michael","eric":     "am_eric",
    "liam":     "am_liam",   "onyx":     "am_onyx",
    # British English (en-gb) — female
    "alice":    "bf_alice",  "emma":     "bf_emma",
    "isabella": "bf_isabella","lily":    "bf_lily",
    # British English (en-gb) — male
    "daniel":   "bm_daniel", "fable":    "bm_fable",
    "george":   "bm_george", "lewis":    "bm_lewis",
    # Spanish
    "dora_es":  "ef_dora",   "alex_es":  "em_alex",
    "santa_es": "em_santa",
    # French
    "siwis":    "ff_siwis",
    # Hindi
    "alpha_hi": "hf_alpha",  "beta_hi":  "hf_beta",
    "omega_hi": "hm_omega",  "psi_hi":   "hm_psi",
    # Italian
    "sara":     "if_sara",   "nicola":   "im_nicola",
    # Japanese
    "alpha_ja": "jf_alpha",  "gongitsune": "jf_gongitsune",
    "nezumi":   "jf_nezumi", "tebukuro": "jf_tebukuro",
    "kumo":     "jm_kumo",
    # Portuguese (Brazil)
    "dora_pt":  "pf_dora",   "alex_pt":  "pm_alex",
    "santa_pt": "pm_santa",
    # Chinese (Mandarin)
    "xiaobei":  "zf_xiaobei","xiaoni":   "zf_xiaoni",
    "xiaoxiao": "zf_xiaoxiao","xiaoyi":  "zf_xiaoyi",
    "yunjian":  "zm_yunjian","yunxi":    "zm_yunxi",
    "yunxia":   "zm_yunxia", "yunyang":  "zm_yunyang",
}

_INTERNAL_TO_HUMAN: dict[str, str] = {v: k for k, v in KOKORO_VOICE_MAP.items()}

_VOICE_LANG: dict[str, str] = {
    "ef_": "es", "em_": "es", "ff_": "fr",
    "hf_": "hi", "hm_": "hi", "if_": "it",
    "im_": "it", "jf_": "ja", "jm_": "ja",
    "pf_": "pt-br", "pm_": "pt-br",
    "zf_": "zh", "zm_": "zh",
}


def _resolve_kokoro_lang(internal_id: str) -> str:
    """Return the best-fit ``lang`` argument for Kokoro.create()."""
    for prefix, lang in _VOICE_LANG.items():
        if internal_id.startswith(prefix):
            return lang
    if internal_id.startswith(("bf_", "bm_")):
        return "en-gb"
    return "en-us"


def _get_cache_dir() -> Path:
    env = os.environ.get("VOXBRIDGE_CACHE_DIR")
    if env:
        return Path(env).expanduser() / "kokoro"
    return Path.home() / ".cache" / "voxbridge" / "kokoro"


def _get_model_path() -> str:
    env = os.environ.get("VOXBRIDGE_KOKORO_MODEL")
    if env:
        return env
    return str(_get_cache_dir() / "kokoro-v1.0.onnx")


def _get_voices_path() -> str:
    env = os.environ.get("VOXBRIDGE_KOKORO_VOICES")
    if env:
        return env
    return str(_get_cache_dir() / "voices-v1.0.bin")


def _static_voice_list() -> list[VoiceInfo]:
    return [_voice_info_from_id(v) for v in sorted(KOKORO_VOICE_MAP.values())]


def _voice_info_from_id(internal_id: str) -> VoiceInfo:
    human = _INTERNAL_TO_HUMAN.get(internal_id, internal_id)
    lang = _resolve_kokoro_lang(internal_id)
    gender = "unknown"
    if len(internal_id) >= 2:
        gender_code = internal_id[1]
        if gender_code == "f":
            gender = "female"
        elif gender_code == "m":
            gender = "male"
    return VoiceInfo(
        name=human, language=lang, gender=gender,
        provider="kokoro", tags=[], internal_id=internal_id,
    )


def _try_download() -> None:
    """Download Kokoro model and voice files if missing."""
    cache = _get_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    model = cache / "kokoro-v1.0.onnx"
    voices = cache / "voices-v1.0.bin"
    if model.exists() and voices.exists():
        return
    import urllib.request
    urls = {
        model: (
            "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
            "model-files-v1.0/kokoro-v1.0.onnx"
        ),
        voices: (
            "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
            "model-files-v1.0/voices-v1.0.bin"
        ),
    }
    for target, url in urls.items():
        if target.exists():
            continue
        logger.info("Downloading Kokoro asset -> %s", target)
        urllib.request.urlretrieve(url, str(target))  # noqa: S310
        logger.info(
            "Downloaded %s (%.1f MiB)", target,
            target.stat().st_size / (1024 * 1024),
        )


def _apply_fade_out(wav: np.ndarray, sr: int, duration_ms: int) -> np.ndarray:
    """Apply a short linear fade-out to the last ``duration_ms`` of a mono waveform.

    This masks Kokoro's phrase-ending energy dip / rebound artifact.
    """
    if wav.ndim != 2 or wav.shape[0] != 1:
        return wav  # guard against unexpected shapes
    fade_samples = int(sr * duration_ms / 1000)
    if fade_samples >= wav.shape[1]:
        return wav  # too short to fade
    tail = wav[:, -fade_samples:]
    taper = np.linspace(1.0, 0.82, fade_samples, dtype=wav.dtype)
    tail *= taper[np.newaxis, :]  # match (1, N)
    wav = np.concatenate([wav[:, :-fade_samples], tail], axis=1)
    return wav


class KokoroBackend(TTSBackend):
    """Kokoro-ONNX backend.

    Environment variables:
        * ``VOXBRIDGE_KOKORO_MODEL`` — path to ``kokoro-v1.0.onnx``
        * ``VOXBRIDGE_KOKORO_VOICES`` — path to ``voices-v1.0.bin``
        * ``VOXBRIDGE_CACHE_DIR`` overrides the default cache directory
    """

    _name: str = "kokoro"
    _sample_rate: int = 24000
    _kokoro: Optional[object] = None

    # Properties that mirror the abstract protocol expectations without
    # conflicting with Pydantic dataclass fields in schemas.
    @property
    def name(self) -> str:  # type: ignore[override]
        return self._name

    @property
    def sample_rate(self) -> int:  # type: ignore[override]
        return self._sample_rate

    def load(self) -> None:
        from kokoro_onnx import Kokoro  # type: ignore[import-untyped]
        model_path = _get_model_path()
        voices_path = _get_voices_path()
        if not Path(model_path).exists() or not Path(voices_path).exists():
            _try_download()
        logger.info("Loading Kokoro ONNX from %s", model_path)
        self._kokoro = Kokoro(model_path=model_path, voices_path=voices_path)
        logger.info("Kokoro backend ready (%d voices)", len(self.list_voices()))

    def synthesize(
        self,
        text: str,
        voice: str,
        *,
        speed: float = 1.0,
        lang: str | None = "en",
        fade_ending: bool = True,
    ) -> np.ndarray:
        if self._kokoro is None:
            raise RuntimeError("KokoroBackend not loaded. Call .load() first.")
        internal = KOKORO_VOICE_MAP.get(voice, voice)
        kokoro_lang = _resolve_kokoro_lang(internal)
        if lang and lang in {"en-gb", "en-us", "ja", "es", "fr", "hi", "it", "pt-br", "zh"}:
            kokoro_lang = lang
        kokoro = self._kokoro  # type: ignore[no-untyped-call]
        audio, _sr = kokoro.create(text=text, voice=internal, speed=speed, lang=kokoro_lang)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        if fade_ending:
            audio = _apply_fade_out(audio, sr=self.sample_rate, duration_ms=120)
        return audio

    def list_voices(self) -> list[VoiceInfo]:
        if self._kokoro is None:
            return _static_voice_list()
        kokoro = self._kokoro  # type: ignore[no-untyped-call]
        return [_voice_info_from_id(key) for key in sorted(kokoro.voices.files)]

    def get_voice(self, name: str) -> VoiceInfo:
        internal = KOKORO_VOICE_MAP.get(name, name)
        if internal not in KOKORO_VOICE_MAP.values():
            raise ValueError(f"Unknown Kokoro voice: {name!r}")
        if self._kokoro is not None:
            kokoro = self._kokoro  # type: ignore[no-untyped-call]
            if internal not in kokoro.voices.files:
                raise ValueError(f"Unknown Kokoro voice: {name!r}")
        return _voice_info_from_id(internal)

    def supports_lang(self, lang: str) -> bool:
        return lang in {"en", "ja", "zh", "es", "fr", "hi", "it", "pt", "pt-br",
                        "na", "en-us", "en-gb"}

    @property
    def voice_style_names(self) -> list[str]:
        return sorted(KOKORO_VOICE_MAP)
