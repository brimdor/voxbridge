# VoxBridge

**Open TTS engine — local, private, expressiveness unlocked.**

> VoxBridge is a fork of the MIT-licensed [Supertonic Python SDK](https://github.com/supertone-inc/supertonic-py) by Supertone Inc. — see [FORK.md](./FORK.md) for the complete fork history and attribution.

VoxBridge is a high-performance, on-device text-to-speech system that runs entirely on your CPU via ONNX Runtime. No cloud calls. No API keys. No paywalled features. No bait and switch.

Forked from the MIT-licensed Supertonic SDK, VoxBridge adds what should have been there from the start: **open expression support**, **production-grade text normalization**, and **pluggable backends** — all fully local, all free.

## Why VoxBridge?

| Feature | Supertonic (original) | VoxBridge |
|---|---|---|
| Local inference | ✅ | ✅ |
| 31 languages | ✅ | ✅ (Supertone) |
| High-quality English voices | ✅ | ✅ (Kokoro — 53 voices) |
| No GPU required | ✅ | ✅ |
| OpenAI-compatible API | ✅ | ✅ |
| Pluggable backends | ❌ | ✅ **Supertone + Kokoro** |
| Expression tags (`<laugh>`, `<breath>`, etc.) | ❌ Paywalled behind API key | ✅ **Free, local, numpy-based** |
| Number/currency/date normalization | ❌ Breaks on real-world text | ✅ **Built-in Normalizer** |
| Text preprocessing | Basic | **Full pipeline** (currency, time, phone, ordinals) |
| Security hardening | None | ✅ Rate limiting, input sanitization, path traversal protection |
| License | MIT code, OpenRAIL-M model | MIT code, OpenRAIL-M model |

## Install

```bash
# Standard install (Supertone backend + server deps)
pip install voxbridge

# With Kokoro backend (recommended — best quality English voices)
pip install voxbridge kokoro-onnx

# With audio playback support
pip install voxbridge[playback]

# With HTTP server
pip install voxbridge[serve]
```

First run automatically downloads model weights:
- **Supertone**: ~200MB from HuggingFace `Supertone/supertonic-3`
- **Kokoro**: ~311MB ONNX + voice pack (auto-downloaded on first use)

## Quick Start

### Supertone (default — 31 languages, M1–F5 voices)

```python
from voxbridge import TTS

# Default: Supertone backend with 31-language support
tts = TTS()
style = tts.get_voice_style("M1")

# Simple synthesis
wav, duration = tts.synthesize("Welcome to VoxBridge!", voice_style=style, lang="en")
tts.save_audio(wav, "output.wav")

# Multilingual: Korean, English, French, etc.
wav_ko, _ = tts.synthesize("안녕하세요!", voice_style=style, lang="ko")
wav_fr, _ = tts.synthesize("Bonjour le monde!", voice_style=style, lang="fr")
```

### Kokoro (best English — 53 voices, faster, more natural)

```python
from voxbridge import TTS

# Use Kokoro backend for best-quality English synthesis
tts = TTS(provider="kokoro")

# Human-readable voice names (no JSON loading needed)
wav, dur = tts.synthesize("Hello world!", voice_style="bella")
wav, dur = tts.synthesize("Good afternoon.", voice_style="adam")
wav, dur = tts.synthesize("Nice to meet you.", voice_style="echo")

# Speed control (0.7–2.0, default 1.0)
wav, dur = tts.synthesize(
    "This is a longer sentence with natural prosody.",
    voice_style="bella",
    speed=0.95,  # Slightly slower for maximum naturalness
)
```

### With Text Normalization

```python
from voxbridge import TTS

tts = TTS(provider="kokoro")

# Automatically expands: $12,458.75 → "twelve thousand four hundred fifty eight dollars and seventy five cents"
wav, dur = tts.synthesize(
    "Your balance is $12,458.75, due on June 15, 2026.",
    voice_style="bella",
)

# Automatically expands: 5:30 p.m. → "five thirty PM"
wav, dur = tts.synthesize(
    "Your appointment is at 5:30 p.m.",
    voice_style="bella",
)

# Works with both providers — same Normalizer interface
```

### Expression Tags

VoxBridge supports 10 expression tags — all processed locally with numpy, no external API:

| Tag | Effect | Post-processing |
|---|---|---|
| `<breath/>` | Intake of breath | Insert silence + noise-shaped breath |
| `<sigh/>` | Exhale sigh | Long breath noise insert |
| `<gasp/>` | Sharp gasp | Short sharp breath |
| `<pause/>` | Silence | Configurable pause (default 0.5s) |
| `<pause duration="1.0"/>` | Timed pause | Custom duration in seconds |
| `<whisper/>` | Whispered speech | Reduced amplitude + spectral tilt |
| `<shout/>` | Loud speech | Increased amplitude + brightness |
| `<cough/>` | Cough sound | Brief noise burst |
| `<laugh/>` | Brief laugh | **Kokoro**: volume swell (gentle chuckle) |
| `<laugh/>` | Brief laugh | **Supertone**: amplitude modulation + pitch jitter |

Expressions are **provider-aware**: what sounds good on Supertone's pitch-variable output may not on Kokoro's steady output, so different implementations are used per backend.

```python
tts = TTS(provider="kokoro")
wav, dur = tts.synthesize(
    "Hello <breath/> how are you? <pause/> I'm doing great today.",
    voice_style="bella",
)
```

Custom expressions via `ExpressionProcessor.register_expression()`.

## Backends

VoxBridge uses a **pluggable backend architecture**. Choose the provider that fits your use case:

| | **Supertone** (default) | **Kokoro** |
|---|---|---|
| **Best for** | Multilingual, 31 languages | English quality, speed |
| **Sample rate** | 44.1 kHz | 44.1 kHz (auto-resampled from 24 kHz) |
| **Voices** | M1–M5, F1–F5 (10 built-in) | 53 voices (bella, adam, echo, etc.) |
| **Speed** | ~0.3x realtime | ~4.5x realtime |
| **Languages** | 31 + `na` fallback | English (+ 7 other langs via voice selection) |
| **Expression tags** | Tremolo + jitter (`<laugh/>`) | Volume swell (`<laugh/>`) |
| **Phrase ending** | Natural fade | Applied 120ms taper |
| **Model size** | ~200MB | ~338MB |

### Kokoro Voice Quick Reference

**American English (female)**: bella, sarah, nicole, sky, jessica, river, alloy, nova, heart, kore, aoede
**American English (male)**: adam, echo, puck, fenrir, michael, eric, liam, onyx
**British English (female)**: alice, emma, isabella, lily
**British English (male)**: daniel, fable, george, lewis
**Spanish**: dora_es, alex_es, santa_es
**French**: siwis
**Hindi**: alpha_hi, beta_hi, omega_hi, psi_hi
**Italian**: sara, nicola
**Japanese**: alpha_ja, gongitsune, nezumi, tebukuro, kumo
**Portuguese (Brazil)**: dora_pt, alex_pt, santa_pt
**Chinese (Mandarin)**: xiaobei, xiaoni, xiaoxiao, xiaoyi, yunjian, yunxi, yunxia, yunyang

## CLI

```bash
# Supertone — default, 31 languages
voxbridge tts "Hello world" -o hello.wav --voice M1

# Kokoro — best English, 53 voices
voxbridge tts "Hello world" -o hello.wav --provider kokoro --voice bella

# Start server with Kokoro
voxbridge serve --provider kokoro --host 127.0.0.1 --port 7788

# List voices for a specific provider
voxbridge list-voices --provider kokoro

# Show backend info
voxbridge info --provider kokoro
```

### Local HTTP Server

```bash
pip install 'voxbridge[serve]'
voxbridge serve --provider kokoro --host 127.0.0.1 --port 7788
```

**Note on concurrency:** The HTTP server runs in a FastAPI threadpool. Because ONNX Runtime inference sessions are not thread-safe, multiple providers are **serialized** within a single process. If you need both Supertone and Kokoro live simultaneously, run two server instances:

```bash
# Terminal 1 — Kokoro
voxbridge serve --provider kokoro --port 7788

# Terminal 2 — Supertone
voxbridge serve --provider supertone --port 7789
```

**OpenAI-compatible endpoint** — swap your base URL, done:

```bash
# Drop-in replacement for OpenAI TTS
curl http://127.0.0.1:7788/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"kokoro","input":"Hello world","voice":"bella"}' \
  --output speech.mp3
```

Interactive API docs at `http://127.0.0.1:7788/docs`.

## Security

VoxBridge includes built-in security hardening for the HTTP server:

- **Input sanitization**: Max text length, unicode normalization, null byte removal, control character stripping
- **Rate limiting**: Configurable per-IP rate limiting (default: 60 requests/minute)
- **Security headers**: X-Content-Type-Options, X-Frame-Options, CSP, X-XSS-Protection, Referrer-Policy
- **Path traversal protection**: All file operations validated against allowed directories
- **Request size limits**: Configurable max body size (default: 10MB)
- **Audio output limits**: Prevent memory exhaustion from huge text inputs (default: 50MB)

```python
from voxbridge.security import SecurityMiddleware, RateLimiter, sanitize_input

# Input sanitization
text = sanitize_input(user_input, max_length=50000)

# Rate-limited server
limiter = RateLimiter(max_requests=100, window_seconds=60)
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VOXBRIDGE_CACHE_DIR` | `~/.cache/voxbridge3` | Model cache directory |
| `VOXBRIDGE_MODEL_REPO` | `Supertone/supertonic-3` | HuggingFace model repository |
| `VOXBRIDGE_MODEL_REVISION` | Pinned SHA | Specific model revision |
| `VOXBRIDGE_INTRA_OP_THREADS` | Auto | ONNX Runtime intra-op threads |
| `VOXBRIDGE_INTER_OP_THREADS` | Auto | ONNX Runtime inter-op threads |
| `VOXBRIDGE_LOG_LEVEL` | `INFO` | Logging level |
| `VOXBRIDGE_KOKORO_MODEL` | `~/.cache/voxbridge/kokoro/kokoro-v1.0.onnx` | Kokoro ONNX model path |
| `VOXBRIDGE_KOKORO_VOICES` | `~/.cache/voxbridge/kokoro/voices-v1.0.bin` | Kokoro voice pack path |

## How It Differs from Supertonic

VoxBridge is a fork of the [Supertonic Python SDK](https://github.com/supertone-inc/supertonic-py) (MIT license). Key differences:

1. **Pluggable backends**: Supertone (31 languages) + Kokoro (best English) via `TTS(provider="...")`
2. **Open expressions**: `<laugh>`, `<breath>`, `<sigh>`, etc. — processed locally with numpy, no API key, no paywall
3. **Text normalization**: Currency, dates, times, phone numbers, ordinals, abbreviations — all expanded before synthesis
4. **Security**: Input sanitization, rate limiting, path traversal protection, request size limits
5. **Rebranding**: Package `supertonic` → `voxbridge`, class `Supertonic` → `VoxBridge`, env vars `SUPERTONIC_*` → `VOXBRIDGE_*`
6. **Community-first**: No paywalled features. No hidden costs. Everything runs locally.

## Legal

VoxBridge code is **MIT licensed**. Original Supertonic SDK code is © Supertone Inc. (MIT). New code © Brimdor (MIT).

Model weights:
- `Supertone/supertonic-3`: **Big**Science OpenRAIL-M license (allows commercial use, modification, and distribution)
- `Kokoro-82M`: Models from `onnx-community/Kokoro-82M-v1.0-ONNX` under Apache 2.0

See [FORK.md](./FORK.md) for the complete fork history and [LICENSE]（LICENSE) for the full MIT license text.

## Contributing

PRs welcome. This project exists because the community deserves TTS that doesn't bait and switch.

1. Fork the repo
2. Create a feature branch
3. Run `pytest` to verify existing tests pass (188 tests as of v0.2.x)
4. Add tests for new code
5. Submit a PR

## Roadmap

- [ ] Fine-tune expression quality with open emotion datasets (IEMOCAP, RAVDESS)
- [ ] Add NeMo text processing for production-grade normalization
- [ ] Benchmark suite against Edge TTS, Kokoro-82M, and F5-TTS
- [ ] Streaming audio output for real-time applications
- [ ] GPU inference option via CUDAExecutionProvider
- [ ] Voice cloning from reference audio (local, private)
- [ ] SSML `<prosody>` support (rate, pitch, volume per word)
- [ ] Neural spectral extension for Kokoro (post-EQ above 8 kHz)
