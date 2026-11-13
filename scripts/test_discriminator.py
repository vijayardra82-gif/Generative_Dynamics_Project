import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

import torch

from models.timegan import Discriminator

batch_size = 32
sequence_length = 100
hidden_dim = 24

hidden = torch.randn(
    batch_size,
    sequence_length,
    hidden_dim
)

discriminator = Discriminator(
    hidden_dim=hidden_dim
)

output = discriminator(hidden)

print("Input Shape :", hidden.shape)
print("Output Shape:", output.shape)