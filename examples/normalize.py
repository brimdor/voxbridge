"""Example: normalize real-world text with numbers, dates, money, etc."""
from voxbridge import TTS

tts = TTS(normalizer=True)
style = tts.get_voice_style("F1")
wav, duration = tts.synthesize(
    "Your balance is $12,458.75, due on June 15, 2026. Call 1-800-555-0123.",
    voice_style=style,
    lang="en",
)
tts.save_audio(wav, "normalized_output.wav")
print(f"Saved {duration:.2f}s of normalized audio to normalized_output.wav")
