import torch

# Set safe download location (optional but recommended)
import os
os.environ['TORCH_HOME'] = os.path.join(os.getcwd(), '.torch_cache')

# Download and cache Silero VAD model
torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)

print("Silero VAD downloaded and cached.")
