# VoxBridge Examples

## Prerequisites

```bash
pip install voxbridge
```

## Files

| Script | What it does |
|--------|-------------|
| `synthesize.py` | Basic TTS — English, save to WAV |
| `oneshot.py` | One-shot synthesis (v0.2.2+) — no TTS instance needed |
| `normalize.py` | Real-world text with numbers, dates, money |
| `expressions.py` | Add `<laugh>`, `<breath>`, `<pause>` tags |
| `server.py` | Start the OpenAI-compatible HTTP server |

## Run

```bash
python synthesize.py
python oneshot.py
python normalize.py
python expressions.py
python server.py   # then hit http://localhost:7788/v1/health
```

## Model Download

First run of `TTS()` downloads ~200MB of ONNX weights from HuggingFace.
Cache is stored at `~/.cache/voxbridge/`.
