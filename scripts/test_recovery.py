import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

import torch
import numpy as np

from models.timegan import Embedder, Recovery

data = np.load("data/lorenz_windows.npy")

sample = torch.tensor(
    data[:32],
    dtype=torch.float32
)

embedder = Embedder(
    input_dim=3,
    hidden_dim=24
)

recovery = Recovery(
    hidden_dim=24,
    output_dim=3
)

hidden = embedder(sample)
reconstructed = recovery(hidden)

print("Input Shape        :", sample.shape)
print("Hidden Shape       :", hidden.shape)
print("Reconstructed Shape:", reconstructed.shape)