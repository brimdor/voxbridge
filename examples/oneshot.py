"""One-shot synthesis example (voxbridge ≥0.2.2).
No TTS instance management — generate and save in two lines.
"""
from voxbridge import synthesize, save_audio

# Generate with Kokoro (best English) — voice "bella", slower speed for naturalness
wav, sr = synthesize(
    "Hello world! This is a one-shot synthesis with no class setup.",
    voice="bella",
    provider="kokoro",
    speed=0.95,
)
save_audio(wav, "output.wav", sample_rate=44100)
dur = wav.shape[1] / sr
print(f"Saved {dur:.2f}s of audio to output.wav")
