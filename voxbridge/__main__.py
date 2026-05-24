"""Entry point for ``python3 -m voxbridge``.

Suppresses noisy phonemizer warnings before any imports so they
don't leak regardless of import order or logging configuration.
"""
import logging

# Suppress noisy phonemizer/espeak word-count mismatch warnings.
# These fire on virtually every English sentence and provide no value.
# Must happen before kokoro-onnx is imported, which lazily triggers
# phonemizer.
logging.getLogger("phonemizer").setLevel(logging.ERROR)

from voxbridge.cli import main  # noqa: E402

if __name__ == "__main__":
    main()