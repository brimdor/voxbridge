"""Example: add expression tags (laugh, breath, pause) to speech.

Expression tags are plain text in the string:
  <laugh>...</laugh>  <breath>...</breath>  <pause duration=0.5></pause>

They are processed locally by the ExpressionProcessor when expressions=True.
"""
from voxbridge import TTS

tts = TTS(expressions=True)
style = tts.get_voice_style("M1")
wav, sr = tts.synthesize(
    "<laugh>That was hilarious!</laugh> Let me catch my <breath>breath</breath>.",
    voice_style=style,
)
tts.save_audio(wav, "expressions_output.wav")
duration = wav.shape[1] / sr
print(f"Saved {duration:.2f}s of expressive audio to expressions_output.wav")
