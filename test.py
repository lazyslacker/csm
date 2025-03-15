from huggingface_hub import hf_hub_download
from generator import load_csm_1b, Segment
import torchaudio
import torch
from dotenv import load_dotenv
import os

load_dotenv()

# Choose the best available device for the model
if torch.backends.mps.is_available():
    device = "mps"
    print("Using MPS for model acceleration (watermarking will use CPU)")
elif torch.cuda.is_available():
    device = "cuda"
    print("Using CUDA for model acceleration")
else:
    device = "cpu"
    print("Using CPU for all operations")

model_path = hf_hub_download(repo_id="sesame/csm-1b", filename="ckpt.pt")
generator = load_csm_1b(model_path, device)

# First generation - without context
print("Generating first audio sample...")
audio = generator.generate(
    text="Hello from Sesame. How are you doing tonight?",
    speaker=0,
    context=[],
    max_audio_length_ms=10_000,
)

torchaudio.save("audio1.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
print("Generated first audio saved to audio1.wav")

# Second generation - with context from the first generation
print("Generating second audio sample with context...")

def load_audio(audio_path):
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=generator.sample_rate
    )
    return audio_tensor

# Create a segment using the first generated audio
segment = Segment(
    text="Hello from Sesame. How are you doing tonight?",
    speaker=0,
    audio=load_audio("audio1.wav")
)

# Generate a response using the first audio as context
audio2 = generator.generate(
    text="I'm doing great! It's a beautiful evening.",
    speaker=1,
    context=[segment],
    max_audio_length_ms=10_000,
)

torchaudio.save("audio2.wav", audio2.unsqueeze(0).cpu(), generator.sample_rate)
print("Generated second audio saved to audio2.wav")

# Comment out the second part that requires utterance files
"""
speakers = [0, 1, 0, 0]
transcripts = [
    "Hey how are you doing.",
    "Pretty good, pretty good.",
    "I'm great.",
    "So happy to be speaking to you.",
]
audio_paths = [
    "utterance_0.wav",
    "utterance_1.wav",
    "utterance_2.wav",
    "utterance_3.wav",
]

def load_audio(audio_path):
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=generator.sample_rate
    )
    return audio_tensor

segments = [
    Segment(text=transcript, speaker=speaker, audio=load_audio(audio_path))
    for transcript, speaker, audio_path in zip(transcripts, speakers, audio_paths)
]
audio = generator.generate(
    text="Me too, this is some cool stuff huh?",
    speaker=1,
    context=segments,
    max_audio_length_ms=10_000,
)

torchaudio.save("audio_with_context.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
"""