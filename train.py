import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = WaveNet(R=64, S=128)
