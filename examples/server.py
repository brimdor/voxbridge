"""Example: run the built-in HTTP server."""
# Usage:  python server.py
# Then:   curl http://localhost:7788/v1/health
# Or:     curl -X POST http://localhost:7788/v1/audio/speech \
#           -H "Content-Type: application/json" \
#           -d '{"input":"Hello world","voice":"M1","model":"supertonic-3"}' \
#           --output hello.wav

import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "voxbridge", "serve", "--host", "127.0.0.1", "--port", "7788"],
    check=True,
)
