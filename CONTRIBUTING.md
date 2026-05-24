# Contributing to VoxBridge

Thank you for your interest in making VoxBridge better! This project is a fork of the MIT-licensed Supertonic SDK with community-driven additions: expression tags, text normalization, and security hardening — all running locally.

## Quick Start

1. Fork this repo and clone it locally.
2. Create a virtual environment and install editable with dev dependencies:
   ```bash
   pip install -e '.[dev]'
   ```
3. Run the test suite before making any changes:
   ```bash
   pytest tests/ -v
   ```
4. Make your changes.
5. Add tests for any new code.
6. Run the full test suite again to verify.
7. Submit a pull request.

## Code Style

- We use `black` and `ruff`. Run `black .` and `ruff check .` before committing.
- Line length is 100 characters.
- Type hints on all public APIs.
- Docstrings with `Args`/`Returns`/`Raises` for every public function.

## Tests

- `pytest tests/ -v` covers routes, config, core, CLI, validation, security, normalizer, expressions, loader.
- If you change `pytest` fixtures, also run `pytest tests/ -v --no-cov`.

## Reporting Bugs

When opening an issue, please include:
- Python version (`python --version`)
- VoxBridge version
- Operating system
- Minimal reproduction steps or script

## Security Issues

If you discover a security vulnerability, please open a private issue or email the maintainer instead of a public bug report.

## Fork Attribution

VoxBridge is a fork of `supertone-inc/supertonic-py` (MIT license). Original authors are credited in the package metadata. See `FORK.md` for full history.
