# VoxBridge — Fork Notice

VoxBridge is a fork of the [Supertonic Python SDK](https://github.com/supertone-inc/supertonic-py),
originally developed by [Supertone Inc.](https://supertone.ai/) and distributed under the MIT license.

## What Changed

VoxBridge extends the original Supertonic SDK with:

- **Text normalization** (`voxbridge.normalizer`): Pre-processes currencies, dates, times, phone numbers, ordinals, and abbreviations for cleaner TTS output.
- **Open expression system** (`voxbridge.expressions`): Parse and apply prosody/expression tags like `<laugh>`, `<breath>`, `<sigh>`, `<pause>`, `<whisper>`, `<shout>`, `<cough>`, `<groan>`, `<gasp>`, `<chuckle>` — all processed locally with numpy, no external API.
- **Security hardening** (`voxbridge.security`): Input sanitization, rate limiting, CORS defaults, security headers, request size limits, and path traversal prevention for the FastAPI server.
- **Pipeline integration**: `TTS(normalizer=True, expressions=True)` wires normalization and expression processing into the synthesis pipeline.
- **Rebranding**: Package renamed from `supertonic` to `voxbridge`, class `Supertonic` → `VoxBridge`, env vars `SUPERTONIC_*` → `VOXBRIDGE_*`, cache dirs updated, URLs point to `github.com/brimdor/voxbridge`.

## What Stayed the Same

- The core ONNX-based TTS engine is unchanged — same quality, same models.
- Model weights are downloaded from the original [Supertone HuggingFace repos](https://huggingface.co/Supertone/supertonic-3) under the OpenRAIL-M license (which permits commercial use).
- Original authors (Yu Yechan, Lee Juheon, Kim Hyeongju) retain copyright on their code.
- The MIT license applies to the entire project.

## Version History

| Version | Notes |
|---------|-------|
| 1.3.1 | Original Supertonic SDK (upstream) |
| 0.1.0 | First VoxBridge fork release |

## License

Both the original Supertonic code and VoxBridge additions are under the MIT license. See [LICENSE](./LICENSE) for details.