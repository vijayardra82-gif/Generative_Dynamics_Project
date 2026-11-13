import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

import torch

from models.timegan import Generator

batch_size = 32
sequence_length = 100
noise_dim = 10

noise = torch.randn(
    batch_size,
    sequence_length,
    noise_dim
)

generator = Generator(
    noise_dim=noise_dim,
    hidden_dim=24
)

generated = generator(noise)

print("Noise Shape     :", noise.shape)
print("Generated Shape :", generated.shape)