"""Open expression/prosody system for VoxBridge TTS.

Parses expression tags from text and applies audio modifications post-synthesis.
All processing uses numpy — no external API calls needed.

Supported tags:
    ``<laugh>``, ``<chuckle>``    → brief amplitude modulation + pitch jitter
    ``<breath>``, ``<sigh>``, ``<gasp>`` → insert short silence + noise-shaped breath
    ``<pause>``                    → insert configurable silence duration
    ``<whisper>``, ``<shout>``     → amplitude scaling + spectral tilt adjustment
    ``<cough>``, ``<groan>``       → insert brief noise burst

Users can also register custom expression definitions.

Example:
    ```python
    from voxbridge.expressions import ExpressionProcessor

    processor = ExpressionProcessor()
    text, tags = processor.extract("Hello <laugh> that's funny <pause/>")
    # Later, after TTS synthesis:
    audio = processor.apply(wav, tags, sample_rate=44100)
    ```
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExpressionTag:
    """A parsed expression tag with timing information.

    Attributes:
        kind: Expression type (e.g., 'laugh', 'breath', 'pause')
        content: Optional attribute string from the tag
        start_sample: Approximate start sample in the audio waveform
        duration_samples: How many samples this expression should span
        params: Additional parameters parsed from tag attributes
    """
    kind: str
    content: str = ""
    start_sample: int = 0
    duration_samples: int = 0
    params: dict = field(default_factory=dict)


@dataclass
class ExpressionDefinition:
    """A user-defined expression with custom audio processing.

    Attributes:
        name: Expression name (used in tags as ``<name>``)
        default_duration_s: Default duration in seconds
        process_fn: Callable(wav_segment, sr, **params) → wav_segment
    """
    name: str
    default_duration_s: float = 0.3
    process_fn: Optional[Callable] = None


# ---------------------------------------------------------------------------
# Built-in audio effects
# ---------------------------------------------------------------------------

def _amplitude_modulate(wav: np.ndarray, sr: int, rate: float = 8.0, depth: float = 0.3) -> np.ndarray:
    """Apply amplitude modulation (tremolo effect) for laugh/chuckle."""
    t = np.arange(wav.shape[-1], dtype=np.float32) / sr
    mod = 1.0 + depth * np.sin(2 * np.pi * rate * t)
    return wav * mod


def _pitch_jitter(wav: np.ndarray, sr: int, jitter_hz: float = 3.0, jitter_depth: float = 0.05) -> np.ndarray:
    """Apply subtle pitch variation for laugh/chuckle naturalness."""
    t = np.arange(wav.shape[-1], dtype=np.float32) / sr
    phase_mod = np.sin(2 * np.pi * jitter_hz * t) * jitter_depth
    # Apply as amplitude variation to approximate pitch jitter
    return wav * (1.0 + phase_mod)


def _generate_breath_noise(duration_samples: int, sr: int, amplitude: float = 0.05) -> np.ndarray:
    """Generate noise-shaped breath sound."""
    noise = np.random.randn(duration_samples).astype(np.float32) * amplitude
    # Apply a simple envelope: attack, sustain, release
    env = np.ones(duration_samples, dtype=np.float32)
    attack = min(int(0.05 * sr), duration_samples // 4)
    release = min(int(0.1 * sr), duration_samples // 4)
    if attack > 0:
        env[:attack] = np.linspace(0, 1, attack, dtype=np.float32)
    if release > 0:
        env[-release:] = np.linspace(1, 0, release, dtype=np.float32)
    return noise * env


def _generate_noise_burst(duration_samples: int, sr: int, amplitude: float = 0.15) -> np.ndarray:
    """Generate a brief noise burst for cough/groan."""
    noise = np.random.randn(duration_samples).astype(np.float32) * amplitude
    # Quick attack, quick decay envelope
    env = np.ones(duration_samples, dtype=np.float32)
    attack = min(int(0.01 * sr), duration_samples // 8)
    release = min(int(0.05 * sr), duration_samples // 4)
    if attack > 0:
        env[:attack] = np.linspace(0, 1, attack, dtype=np.float32)
    if release > 0:
        env[-release:] = np.linspace(1, 0, release, dtype=np.float32)
    return noise * env


def _apply_spectral_tilt(wav: np.ndarray, tilt_db_per_octave: float = -3.0) -> np.ndarray:
    """Apply spectral tilt to audio (negative = brighter for shout, positive = darker for whisper)."""
    n = wav.shape[-1]
    if n < 2:
        return wav
    # Apply FFT-based spectral shaping
    spectrum = np.fft.rfft(wav)
    freqs = np.fft.rfftfreq(n, d=1.0)
    # Avoid division by zero at DC
    freqs[0] = 1.0
    # Create tilt: multiply high frequencies relative to low
    # Normalize so 1kHz is unity gain
    ref_freq = 1000.0
    tilt_factor = np.ones_like(freqs, dtype=np.float32)
    nonzero_mask = freqs > 0
    octaves_from_ref = np.log2(np.where(nonzero_mask, freqs, 1.0) / ref_freq)
    tilt_factor[nonzero_mask] = 10.0 ** (tilt_db_per_octave * octaves_from_ref[nonzero_mask] / 20.0)
    tilt_factor[0] = 1.0  # DC unchanged
    spectrum *= tilt_factor
    return np.fft.irfft(spectrum, n=n).astype(np.float32)


# ---------------------------------------------------------------------------
# Main processor
# ---------------------------------------------------------------------------

_DEFAULT_PAUSE_DURATION_S = 0.5
_DEFAULT_BREATH_DURATION_S = 0.25
_DEFAULT_LAUGH_DURATION_S = 0.4
_DEFAULT_COUCH_DURATION_S = 0.2
_DEFAULT_GASP_DURATION_S = 0.15

_TAG_PATTERN = re.compile(
    r"<(laugh|breath|sigh|cough|gasp|groan|chuckle|whisper|shout|pause)"
    r"(?:\s+([^>]*))?\s*/?>",
    re.IGNORECASE,
)


class ExpressionProcessor:
    """Open expression/prosody system for post-synthesis audio modification.

    Parses expression tags from text, then applies numpy-based audio
    modifications to the synthesized waveform. No API calls needed.

    Built-in expressions:
        laugh, chuckle, breath, sigh, gasp, pause, whisper, shout, cough, groan

    Users can register custom expressions via :meth:`register_expression`.

    Example:
        ```python
        from voxbridge.expressions import ExpressionProcessor

        processor = ExpressionProcessor()
        text = "Hello <laugh> that's great <pause duration='1.0'/>"
        clean_text, tags = processor.extract(text)
        # ... run TTS on clean_text ...
        wav = processor.apply(raw_wav, tags, sample_rate=44100)
        ```
    """

    def __init__(self, default_pause_duration: float = _DEFAULT_PAUSE_DURATION_S, *, provider: str = ""):
        """Initialize the ExpressionProcessor.

        Args:
            default_pause_duration: Default pause duration in seconds for ``<pause>`` tags
            provider: TTS backend provider name — affects which expression implementations are used
        """
        self.default_pause_duration = default_pause_duration
        self.provider = provider
        self._custom_expressions: dict[str, ExpressionDefinition] = {}

        # Built-in expression durations (seconds)
        self._durations: dict[str, float] = {
            "laugh": _DEFAULT_LAUGH_DURATION_S,
            "chuckle": 0.25,
            "breath": _DEFAULT_BREATH_DURATION_S,
            "sigh": 0.5,
            "gasp": _DEFAULT_GASP_DURATION_S,
            "pause": default_pause_duration,
            "whisper": 0.0,  # whisper/shout affect the whole segment, no insert
            "shout": 0.0,
            "cough": _DEFAULT_COUCH_DURATION_S,
            "groan": 0.3,
        }

    def register_expression(self, definition: ExpressionDefinition) -> None:
        """Register a custom expression definition.

        Args:
            definition: The custom expression definition to register
        """
        self._custom_expressions[definition.name.lower()] = definition

    def extract(self, text: str) -> tuple[str, list[ExpressionTag]]:
        """Extract expression tags from text, returning cleaned text and tag list.

        Tags are removed from the text but their character positions are noted
        for approximate timing during audio post-processing.

        Args:
            text: Input text potentially containing expression tags

        Returns:
            Tuple of (cleaned_text, list_of_tags)
        """
        tags: list[ExpressionTag] = []
        # Track cumulative character offset from removed tags
        offset = 0

        for m in _TAG_PATTERN.finditer(text):
            kind = m.group(1).lower()
            attrs_str = m.group(2) or ""

            # Parse attributes (simple key=value or key='value' parsing)
            params: dict = {}
            if attrs_str:
                for kv in re.findall(r"(\w+)\s*=\s*['\"]?([^'\">\s]+)['\"]?", attrs_str):
                    key, val = kv
                    try:
                        params[key] = float(val)
                    except ValueError:
                        params[key] = val

            tag = ExpressionTag(
                kind=kind,
                content=attrs_str,
                start_sample=0,  # Filled in during apply()
                duration_samples=0,
                params=params,
            )
            tags.append(tag)

        # Remove tags from text
        clean_text = _TAG_PATTERN.sub("", text)
        # Clean up multiple spaces left behind
        clean_text = re.sub(r"\s+", " ", clean_text).strip()

        return clean_text, tags

    def apply(self, wav: np.ndarray, tags: list[ExpressionTag], sample_rate: int) -> np.ndarray:
        """Apply expression processing to synthesized audio.

        Expression tags are placed proportionally along the audio timeline.
        Inserts (breath, pause, cough, etc.) extend the audio. Modifiers
        (whisper, shout, laugh, etc.) affect existing audio at their position.

        Args:
            wav: Audio waveform of shape ``(1, num_samples)`` or ``(num_samples,)``
            tags: List of ExpressionTag objects from :meth:`extract`
            sample_rate: Audio sample rate

        Returns:
            Modified audio waveform (same shape as input)
        """
        if not tags:
            return wav

        was_1d = wav.ndim == 1
        if was_1d:
            wav = wav[np.newaxis, :]
        result = wav.copy()
        total_samples = result.shape[-1]

        # Calculate total insert duration to know where tags fall
        total_insert_duration = 0.0
        for tag in tags:
            dur = self._get_duration(tag)
            if self._is_insert(tag.kind):
                total_insert_duration += dur

        # Assign start_sample proportionally
        if len(tags) > 1:
            for i, tag in enumerate(tags):
                tag.start_sample = int((i + 1) / (len(tags) + 1) * total_samples)
        elif len(tags) == 1:
            tag = tags[0]
            # For single tags, place them in the middle or where they appear
            tag.start_sample = total_samples // 2

        # Apply modifications
        for tag in tags:
            result = self._apply_expression(result, tag, sample_rate)

        if was_1d:
            result = result.squeeze(0)

        return result

    def _get_duration(self, tag: ExpressionTag) -> float:
        """Get duration in seconds for a tag, checking params then defaults."""
        # Check for duration attribute in params
        if "duration" in tag.params:
            try:
                return float(tag.params["duration"])
            except (ValueError, TypeError):
                pass
        # Check custom expression
        if tag.kind in self._custom_expressions:
            return self._custom_expressions[tag.kind].default_duration_s
        # Check built-in defaults
        return self._durations.get(tag.kind, 0.3)

    def _is_insert(self, kind: str) -> bool:
        """Whether this expression type inserts audio (vs modifying existing)."""
        return kind in ("breath", "sigh", "gasp", "pause", "cough", "groan")

    def _apply_expression(self, wav: np.ndarray, tag: ExpressionTag, sr: int) -> np.ndarray:
        """Apply a single expression tag to the waveform."""
        kind = tag.kind

        # Custom expression?
        if kind in self._custom_expressions:
            custom = self._custom_expressions[kind]
            if custom.process_fn is not None:
                duration = self._get_duration(tag)
                duration_samples = int(duration * sr)
                start = max(0, min(tag.start_sample, wav.shape[-1] - 1))
                end = min(start + duration_samples, wav.shape[-1])
                segment = wav[:, start:end]
                processed = custom.process_fn(segment, sr, **tag.params)
                # Resize if needed
                if processed.shape[-1] != end - start:
                    if start + processed.shape[-1] <= wav.shape[-1]:
                        wav[:, start:start + processed.shape[-1]] = processed
                else:
                    wav[:, start:end] = processed
            return wav

        # Built-in expressions
        if kind in ("laugh", "chuckle"):
            return self._apply_laugh(wav, tag, sr)
        elif kind in ("breath", "sigh", "gasp"):
            return self._apply_breath(wav, tag, sr)
        elif kind == "pause":
            return self._apply_pause(wav, tag, sr)
        elif kind == "whisper":
            return self._apply_whisper(wav, tag, sr)
        elif kind == "shout":
            return self._apply_shout(wav, tag, sr)
        elif kind in ("cough", "groan"):
            return self._apply_cough(wav, tag, sr)
        else:
            logger.warning(f"Unknown expression kind: {kind!r}, skipping")
            return wav

    # --- Built-in implementations ---

    def _apply_laugh(self, wav: np.ndarray, tag: ExpressionTag, sr: int) -> np.ndarray:
        """Apply laugh/chuckle: provider-aware.'''
        Supertone: amplitude modulation + pitch jitter (tremolo works well).
        Kokoro: gentle volume swell (no pitch modulation — avoids harmonic tearing).
        """
        duration = self._get_duration(tag)
        duration_samples = min(int(duration * sr), wav.shape[-1])
        start = max(0, min(tag.start_sample, wav.shape[-1] - duration_samples))
        end = start + duration_samples
        segment = wav[:, start:end]

        if self.provider == "kokoro":
            # Volume swell: rise 30%, hold, fall 30% — emulates breathy chuckle
            segment = self._apply_volume_swell(segment, sr, duration_s=duration)
        else:
            # Supertone: tremolo + jitter works well on pitch-variable output
            segment = _amplitude_modulate(
                segment, sr, rate=8.0 if tag.kind == "laugh" else 12.0,
                depth=0.3 if tag.kind == "laugh" else 0.15
            )
            segment = _pitch_jitter(
                segment, sr, jitter_hz=4.0 if tag.kind == "laugh" else 6.0,
                jitter_depth=0.03
            )
        wav[:, start:end] = segment
        return wav

    def _apply_volume_swell(self, wav: np.ndarray, sr: int, duration_s: float = 0.4) -> np.ndarray:
        """Gentle volume envelope: fade in, hold, fade out — mimics breathy chuckle."""
        n = wav.shape[-1]
        if n < 2:
            return wav
        envelope = np.ones(n, dtype=wav.dtype)
        # Rise over 30% of sample count, fall over last 30%
        rise = int(n * 0.3)
        fall = int(n * 0.3)
        if rise > 1:
            # Hann-cosine rise from 0 → 1
            envelope[:rise] = 0.5 * (1.0 - np.cos(np.pi * np.arange(rise) / rise))
        if fall > 1:
            envelope[-fall:] = np.linspace(1.0, 0.4, fall, dtype=wav.dtype)
        return wav * envelope[np.newaxis, :]

    def _apply_breath(self, wav: np.ndarray, tag: ExpressionTag, sr: int) -> np.ndarray:
        """Apply breath/sigh/gasp: insert silence + noise-shaped breath."""
        duration = self._get_duration(tag)
        silence_before = int(0.05 * sr)
        duration_samples = int(duration * sr)

        # Generate breath noise
        amplitude = 0.05
        if tag.kind == "sigh":
            amplitude = 0.04
            duration_samples = int(duration * sr)
        elif tag.kind == "gasp":
            amplitude = 0.07
            silence_before = int(0.02 * sr)

        breath_noise = _generate_breath_noise(duration_samples, sr, amplitude)
        silence = np.zeros((1, silence_before), dtype=np.float32)
        breath_stereo = breath_noise[np.newaxis, :]

        # Insert at position
        start = max(0, min(tag.start_sample, wav.shape[-1] - 1))
        insert = np.concatenate([silence, breath_stereo], axis=1)

        # Insert into waveform by splicing
        result = np.concatenate([
            wav[:, :start],
            insert,
            wav[:, start:]
        ], axis=1)
        return result

    def _apply_pause(self, wav: np.ndarray, tag: ExpressionTag, sr: int) -> np.ndarray:
        """Apply pause: insert configurable silence duration."""
        duration = self._get_duration(tag)
        duration_samples = int(duration * sr)
        silence = np.zeros((1, duration_samples), dtype=np.float32)

        start = max(0, min(tag.start_sample, wav.shape[-1] - 1))
        result = np.concatenate([
            wav[:, :start],
            silence,
            wav[:, start:]
        ], axis=1)
        return result

    def _apply_whisper(self, wav: np.ndarray, tag: ExpressionTag, sr: int) -> np.ndarray:
        """Apply whisper: reduce amplitude + darken spectral tilt."""
        result = wav.copy()
        # Whisper is roughly 40-60% amplitude
        scale = float(tag.params.get("scale", 0.5))
        tilt = float(tag.params.get("tilt", 3.0))  # positive = darker
        result = result * scale
        result = _apply_spectral_tilt(result, tilt_db_per_octave=tilt)
        return result

    def _apply_shout(self, wav: np.ndarray, tag: ExpressionTag, sr: int) -> np.ndarray:
        """Apply shout: increase amplitude + brighten spectral tilt."""
        result = wav.copy()
        # Shout is louder and brighter
        scale = float(tag.params.get("scale", 1.5))
        tilt = float(tag.params.get("tilt", -3.0))  # negative = brighter
        result = result * scale
        result = _apply_spectral_tilt(result, tilt_db_per_octave=tilt)
        # Clip to prevent distortion
        result = np.clip(result, -1.0, 1.0)
        return result

    def _apply_cough(self, wav: np.ndarray, tag: ExpressionTag, sr: int) -> np.ndarray:
        """Apply cough/groan: insert brief noise burst."""
        duration = self._get_duration(tag)
        duration_samples = int(duration * sr)

        amplitude = 0.15 if tag.kind == "cough" else 0.1
        noise = _generate_noise_burst(duration_samples, sr, amplitude)
        noise_stereo = noise[np.newaxis, :]

        start = max(0, min(tag.start_sample, wav.shape[-1] - 1))

        # Insert noise burst
        result = np.concatenate([
            wav[:, :start],
            noise_stereo,
            wav[:, start:]
        ], axis=1)
        return result