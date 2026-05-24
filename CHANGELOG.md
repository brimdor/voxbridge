# Changelog

All notable changes to VoxBridge are documented in this file.

## [Unreleased]

No changes yet.

## [0.2.0] - 2026-05-24

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

### Testing
- Added `test_security.py` (24 tests) covering `sanitize_input`, `validate_path`, `RateLimiter`, `SecurityMiddleware`, `_get_client_ip`, `get_cors_config`, and env caps.
- Added `test_loader_safety.py` (5 tests) for element count validation, oversized vector rejection, nested shape validation.
- Added `test_new_features.py` (7 tests) for health schema fields, Pydantic step boundaries, env var existence, normalizer sanity, and CLI path validation.
- Fixed 8 pre-existing test failures caused by hardcoded `"voxbridge-3"`/`"voxbridge-2"` model names; corrected to `"supertonic-3"`/`"supertonic-2"`.

### Documentation
- Added `CONTRIBUTING.md`, `CHANGELOG.md`, `SECURITY.md`.
- Added `.github/ISSUE_TEMPLATE/bug_report.md` and `feature_request.md`.
- Added `examples/` directory with `synthesize.py`, `normalize.py`, `expressions.py`, `server.py`.
- Added `.github/workflows/ci.yml` (pytest + ruff on PR/push) and `release.yml` (trusted PyPI publishing on tag).

## [0.1.0] - 2025-??-??

- Initial release — fork of Supertonic SDK with expression support, text normalization, and local HTTP server.
