"""Quick-start example: synthesize English text and save to WAV."""
from voxbridge import TTS

tts = TTS()
style = tts.get_voice_style("M1")
wav, sr = tts.synthesize(
    "Welcome to VoxBridge — open, local, text to speech.",
    voice_style=style,
    lang="en",
)
tts.save_audio(wav, "output.wav")
duration = wav.shape[1] / sr
print(f"Saved {duration:.2f}s of audio to output.wav")
