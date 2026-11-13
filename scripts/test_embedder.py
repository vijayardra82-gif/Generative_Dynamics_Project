import torch
import numpy as np

import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

from models.timegan import Embedder

data = np.load("data/lorenz_windows.npy")

sample = torch.tensor(
    data[:32],
    dtype=torch.float32
)

model = Embedder(
    input_dim=3,
    hidden_dim=24
)

output = model(sample)

print("Input Shape :", sample.shape)
print("Output Shape:", output.shape)