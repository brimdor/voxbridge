# VoxBridge

**Open TTS engine — local, private, expressiveness unlocked.**

> VoxBridge is a fork of the MIT-licensed [Supertonic Python SDK](https://github.com/supertone-inc/supertonic-py) by Supertone Inc. — see [FORK.md](./FORK.md) for the complete fork history and attribution.

VoxBridge is a high-performance, on-device text-to-speech system that runs entirely on your CPU via ONNX Runtime. No cloud calls. No API keys. No paywalled features. No bait and switch.

Forked from the MIT-licensed Supertonic SDK, VoxBridge adds what should have been there from the start: **open expression support** and **production-grade text normalization** — both fully local, both free.

## Why VoxBridge?

| Feature | Supertonic (original) | VoxBridge |
|---|---|---|
| Local inference | ✅ | ✅ |
| 31 languages | ✅ | ✅ |
| 99M param model | ✅ | ✅ |
| No GPU required | ✅ | ✅ |
| OpenAI-compatible API | ✅ | ✅ |
| Expression tags (`<laugh>`, `<breath>`, etc.) | ❌ Paywalled behind API key | ✅ **Free, local, numpy-based** |
| Number/currency/date normalization | ❌ Breaks on real-world text | ✅ **Built-in Normalizer** |
| Text preprocessing | Basic | **Full pipeline** (currency, time, phone, ordinals) |
| Security hardening | None | ✅ Rate limiting, input sanitization, path traversal protection |
| License | MIT code, OpenRAIL-M model | MIT code, OpenRAIL-M model |

## Install

```bash
pip install voxbridge
```

First run automatically downloads model weights from HuggingFace (~200MB).

## Quick Start

### Python SDK

```python
from voxbridge import TTS

tts = TTS()
style = tts.get_voice_style("M1")

# Simple synthesis
wav, duration = tts.synthesize("Welcome to VoxBridge!", voice_style=style, lang="en")
tts.save_audio(wav, "output.wav")

# With text normalization (handles money, dates, phone numbers)
wav, dur = tts.synthesize(
    "Your balance is $12,458.75, due on June 15, 2026.",
    voice_style=style,
    lang="en",
    normalize=True,  # Auto-expands: "twelve thousand four hundred fifty eight dollars and seventy five cents..."
)

# With expressions — fully local, no API key
wav, dur = tts.synthesize(
    "Hello <laugh/> that's really funny <pause/> but seriously though.",
    voice_style=style,
    lang="en",
    expressions=True,  # Processes <laugh/>, <pause/> tags post-synthesis
)
tts.save_audio(wav, "output.wav")
```

### Expression Tags

VoxBridge supports 10 expression tags — all processed locally with numpy, no external API:

| Tag | Effect | Post-processing |
|---|---|---|
| `<laugh/>` | Brief laugh | Amplitude modulation + pitch jitter |
| `<chuckle/>` | Subtle chuckle | Gentle amplitude modulation |
| `<breath/>` | Intake of breath | Insert silence + noise-shaped breath |
| `<sigh/>` | Exhale sigh | Long breath noise insert |
| `<gasp/>` | Sharp gasp | Short sharp breath |
| `<pause/>` | Silence | Configurable pause (default 0.5s) |
| `<pause duration="1.0"/>` | Timed pause | Custom duration in seconds |
| `<whisper/>` | Whispered speech | Reduced amplitude + spectral tilt |
| `<shout/>` | Loud speech | Increased amplitude + brightness |
| `<cough/>` | Cough sound | Brief noise burst |

Custom expressions via `ExpressionProcessor.register_expression()`.

### Text Normalization

The `Normalizer` class pre-processes text before synthesis — critical for production apps:

```python
from voxbridge.normalizer import Normalizer

norm = Normalizer()
norm.normalize("$12,458.75")
# → "twelve thousand four hundred fifty eight dollars and seventy five cents"

norm.normalize("June 15, 2026")
# → "June fifteenth, twenty twenty six"

norm.normalize("5:30 p.m.")
# → "five thirty PM"

norm.normalize("1-800-555-0199")
# → "one eight hundred five five five zero one nine nine"

norm.normalize("1st, 2nd, 3rd")
# → "first, second, third"
```

All normalization categories are independently toggleable:

```python
norm = Normalizer(currency=True, dates=True, phone_numbers=False, time=True)
```

### Local HTTP Server

```bash
pip install 'voxbridge[serve]'
voxbridge serve --host 127.0.0.1 --port 7788
```

**Note on concurrency:** The HTTP server runs in a FastAPI threadpool. Because ONNX Runtime inference sessions are not thread-safe, synthesis is serialized behind a single lock. This means one request at a time per process — slow high-step requests can create queueing. Check `/v1/health` for `queue_depth` and `max_synth_seconds`. If you need more throughput, run multiple single-process instances behind a reverse proxy (nginx/haproxy) and load-balance.

**OpenAI-compatible endpoint** — swap your base URL, done:

```bash
# Drop-in replacement for OpenAI TTS
curl http://127.0.0.1:7788/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"supertonic-3","input":"Hello world","voice":"M1"}' \
  --output speech.mp3
```

**Native endpoint** with normalization and expressions:

```bash
curl http://127.0.0.1:7788/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"Your balance is $12,458.75","voice":"M1","normalize":true,"expressions":true}' \
  --output speech.wav
```

Interactive API docs at `http://127.0.0.1:7788/docs`.

### CLI

```bash
# Synthesize to file
voxbridge say "Hello world" --voice M1 --lang en --output hello.wav

# Start server
voxbridge serve --host 0.0.0.0 --port 7788

# List voices
voxbridge list-voices

# Show model info
voxbridge info
```

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

## Supported Languages

31 languages + language-agnostic fallback:

`ar`, `bg`, `cs`, `da`, `de`, `el`, `en`, `es`, `et`, `fi`, `fr`, `hi`, `hr`, `hu`, `id`, `it`, `ja`, `ko`, `lt`, `lv`, `nl`, `pl`, `pt`, `ro`, `ru`, `sk`, `sl`, `sv`, `tr`, `uk`, `vi`

Pass `lang="na"` for unknown/mixed-language text.

## Voice Styles

10 built-in voices: **M1–M5** (male) and **F1–F5** (female).

Custom voice profiles can be imported from JSON files via the Voice Builder.

## Model Weights

VoxBridge uses the open-weight Supertonic-3 model (OpenRAIL-M license) from HuggingFace:

- **Model**: [`Supertone/supertonic-3`](https://huggingface.co/Supertone/supertonic-3)
- **License**: BigScience OpenRAIL-M (allows commercial use, modification, and distribution)
- **Size**: ~200MB download, runs on CPU
- **No API key required** for basic synthesis

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VOXBRIDGE_CACHE_DIR` | `~/.cache/voxbridge3` | Model cache directory |
| `VOXBRIDGE_MODEL_REPO` | `Supertone/supertonic-3` | HuggingFace model repository |
| `VOXBRIDGE_MODEL_REVISION` | Pinned SHA | Specific model revision |
| `VOXBRIDGE_INTRA_OP_THREADS` | Auto | ONNX Runtime intra-op threads |
| `VOXBRIDGE_INTER_OP_THREADS` | Auto | ONNX Runtime inter-op threads |
| `VOXBRIDGE_LOG_LEVEL` | `INFO` | Logging level |

## How It Differs from Supertonic

VoxBridge is a fork of the [Supertonic Python SDK](https://github.com/supertone-inc/supertonic-py) (MIT license). Key differences:

1. **Open expressions**: `<laugh>`, `<breath>`, `<sigh>`, etc. are processed locally with numpy — no API key, no paywall
2. **Text normalization**: Currency, dates, times, phone numbers, ordinals, abbreviations — all expanded before synthesis
3. **Security**: Input sanitization, rate limiting, path traversal protection, request size limits
4. **Rebranding**: Package `supertonic` → `voxbridge`, class `Supertonic` → `VoxBridge`, env vars `SUPERTONIC_*` → `VOXBRIDGE_*`
5. **Community-first**: No paywalled features. No hidden costs. Everything runs locally.

## Legal

VoxBridge code is **MIT licensed**. Original Supertonic SDK code is © Supertone Inc. (MIT). New code © Brimdor (MIT).

Model weights (`Supertone/supertonic-3`) are under the **BigScience OpenRAIL-M license**, which permits commercial use, modification, and distribution with use-based restrictions (no criminal/abusive applications). See the [model license](https://huggingface.co/Supertone/supertonic-3/blob/main/LICENSE) for details.

See [FORK.md](./FORK.md) for the complete fork history and [LICENSE](./LICENSE) for the full MIT license text.

## Contributing

PRs welcome. This project exists because the community deserves TTS that doesn't bait and switch.

1. Fork the repo
2. Create a feature branch
3. Run `pytest` to verify existing tests pass
4. Add tests for new code
5. Submit a PR

## Roadmap

- [ ] Fine-tune expression quality with open emotion datasets (IEMOCAP, RAVDESS)
- [ ] Add NeMo text processing for production-grade normalization
- [ ] Benchmark suite against Edge TTS, Kokoro-82M, and F5-TTS
- [ ] Streaming audio output for real-time applications
- [ ] GPU inference option via CUDAExecutionProvider
- [ ] Voice cloning from reference audio (local, private)