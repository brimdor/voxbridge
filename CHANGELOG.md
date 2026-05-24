# Changelog

All notable changes to VoxBridge are documented in this file.

## [Unreleased]

### Security
- Added configurable caps on style vector element counts via `VOXBRIDGE_MAX_STYLE_ELEMENTS` (default 1_000_000), preventing memory exhaustion from malicious uploaded styles.
- Added `X-Forwarded-For` / `X-Real-IP` aware client IP extraction in `RateLimiter` (middleware-level change).
- Tightened default CORS origins: no longer includes `http://localhost:*` or `http://127.0.0.1:*` wildcards. Only exact `localhost` and `127.0.0.1` are allowed by default.
- Added path traversal validation (`validate_path`) to CLI `--output` options.
- Removed unnecessary `Content-Security-Policy` header from non-HTML API responses.
- Fixed version header (`X-VoxBridge-Version`) to resolve at runtime instead of hardcoding `0.1.0`.

### Performance
- Replaced on-the-fly abbreviation regex compilation with pre-compiled `_ABBREVIATION_PATTERNS` in `normalizer.py`.

### Correctness
- Added lower-level `total_step` bounds check in `VoxBridge.__call__`, mapped to configurable `VOXBRIDGE_MAX_TOTAL_STEPS` env var.
- Removed static `version = "0.1.0"` from pyproject.toml; version is now fully dynamic from `voxbridge.__version__`.

### Server
- Added `queue_depth` and `max_synth_seconds` to `/v1/health` response, exposing real-time synth queue state.
- Added configurable synthesis timeout via `VOXBRIDGE_MAX_SYNTH_SECONDS` (default 60s; set to 0 to disable).
- DRY-refactored `cmd_say` and `cmd_tts` in CLI.

## [0.1.0] - 2025-??-??

- Initial release — fork of Supertonic SDK with expression support, text normalization, and local HTTP server.
