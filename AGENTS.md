# AGENTS.md — VoxBridge Agent Guide

## Project Overview

VoxBridge is an open-source TTS (text-to-speech) engine forked from the MIT-licensed Supertonic SDK. It adds open expression support, text normalization, and security hardening — all running locally with no API keys required.

## Architecture

```
voxbridge/
├── __init__.py          # Package exports: TTS, Style, Normalizer, ExpressionProcessor
├── cli.py               # CLI entry point: voxbridge say, serve, list-voices, info
├── config.py            # Constants: model names, languages, speeds, limits, env vars
├── core.py              # Core engine: UnicodeProcessor, Style, VoxBridge (ONNX inference)
├── expressions.py       # Open expression system: <laugh>, <breath>, <pause>, etc.
├── loader.py            # Model downloading from HuggingFace, voice style loading
├── normalizer.py         # Text normalization: currency, dates, time, phone, ordinals
├── pipeline.py           # High-level TTS interface with normalize/expressions support
├── security.py           # Input sanitization, rate limiting, CORS, security headers
├── utils.py              # Text chunking, filename sanitization, validation
└── server/
    ├── app.py            # FastAPI app factory with security middleware
    ├── audio.py           # Audio format encoding (WAV, MP3, FLAC, etc.)
    ├── routes.py          # HTTP routes: /v1/tts, /v1/audio/speech (OpenAI compat)
    ├── schemas.py         # Pydantic request/response models
    └── styles_store.py    # Custom voice style import/storage
```

## Key Concepts

### Model Weights
- Weights are downloaded from HuggingFace (`Supertone/supertonic-3`) under OpenRAIL-M license
- Cached locally at `~/.cache/voxbridge3/` (or `VOXBRIDGE_CACHE_DIR`)
- Four ONNX models: duration_predictor, text_encoder, vector_estimator, vocoder
- Model names in config: `"supertonic"`, `"supertonic-2"`, `"supertonic-3"` — these are model weight identifiers, NOT package names

### Text Normalization Pipeline
1. Extract expression tags → preserve as placeholders
2. Currency: `$12,458.75` → `twelve thousand four hundred fifty eight dollars and seventy five cents`
3. Time: `5:30 p.m.` → `five thirty PM`
4. Phone numbers: `1-800-555-0199` → digit-by-digit
5. Dates: `June 15, 2026` → `June fifteenth, twenty twenty six`
6. Ordinals: `1st, 2nd, 3rd` → `first, second, third`
7. Abbreviations: `Dr., Mr., etc.` → expanded
8. Re-insert expression tags

### Expression Processing
- Tags are extracted from text before TTS synthesis
- Positions are calculated proportionally along the audio timeline
- Post-synthesis audio modifications applied via numpy:
  - Insert types (breath, pause, cough): splice audio into waveform
  - Modify types (laugh, whisper, shout): spectral/amplitude operations on existing audio
- Custom expressions can be registered via `ExpressionProcessor.register_expression()`

### Security
- `sanitize_input()`: Unicode NFC normalization, null byte removal, control char stripping, length limit
- `validate_path()`: Path traversal prevention
- `RateLimiter`: Per-IP sliding window rate limiting
- `SecurityMiddleware`: Adds security headers to all responses
- `validate_request_size()` / `validate_audio_output_size()`: Prevent memory exhaustion

## Environment Variables

All use `VOXBRIDGE_` prefix:
- `VOXBRIDGE_CACHE_DIR` — Override model cache directory
- `VOXBRIDGE_MODEL_REPO` — Override HuggingFace model repo
- `VOXBRIDGE_MODEL_REVISION` — Override pinned model revision
- `VOXBRIDGE_INTRA_OP_THREADS` / `VOXBRIDGE_INTER_OP_THREADS` — ONNX thread control
- `VOXBRIDGE_LOG_LEVEL` — Logging verbosity
- `VOXBRIDGE_CUSTOM_STYLES_DIR` — Custom voice styles directory

**Do NOT use `SUPERTONIC_*` env vars** — those are from the upstream package and are not recognized by VoxBridge.

## Testing

```bash
pytest tests/ -v
```

Tests cover: config, core, loader, pipeline, server routes, utils, validation, CLI, security, normalizer, expressions.

## Adding New Expression Types

1. Add the tag name to `_TAG_PATTERN` regex in `expressions.py`
2. Add duration default to `_durations` dict in `ExpressionProcessor.__init__`
3. Add the kind to `_is_insert()` or implement as a modifier
4. Implement the audio effect method `_apply_<kind>(self, wav, tag, sr)`
5. Add tests in `tests/test_expressions.py`

## Adding New Normalization Rules

1. Add regex pattern to the `Normalizer` class in `normalizer.py`
2. Implement the replacement method `_normalize_<type>(self, text)`
3. Add the toggle flag to `__init__` and the `normalize()` pipeline
4. Add tests in `tests/test_normalizer.py`

## Fork Relationship

VoxBridge is forked from `supertone-inc/supertonic-py` (MIT license).
- Original authors: Yu Yechan, Lee Juheon, Kim Hyeongju (Supertone Inc.)
- Fork maintained by: Brimdor
- Our additions: normalizer.py, expressions.py, security.py — all MIT
- Model weights: Supertone/supertonic-3 (OpenRAIL-M)