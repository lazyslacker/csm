from huggingface_hub import hf_hub_download
from generator import load_csm_1b
import torchaudio
import torch
from dotenv import load_dotenv

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
audio = generator.generate(
    text="Hello from Sesame. How are you doing tonight? Its a quiet Friday evening!",
    speaker=1,
    context=[],
    max_audio_length_ms=10_000,
)

torchaudio.save("audio.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)