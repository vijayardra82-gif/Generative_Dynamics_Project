import torch
import torch.nn as nn


# 1. Embedder Network
class Embedder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.rnn = nn.GRU(
            input_dim,
            hidden_dim,
            batch_first=True
        )

    def forward(self, x):
        output, _ = self.rnn(x)
        return output


# 2. Recovery Network
class Recovery(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()

        self.rnn = nn.GRU(
            hidden_dim,
            output_dim,
            batch_first=True
        )

    def forward(self, h):
        output, _ = self.rnn(h)
        return output


# 3. Generator Network
class Generator(nn.Module):
    def __init__(self, noise_dim, hidden_dim):
        super().__init__()

        self.rnn = nn.GRU(
            noise_dim,
            hidden_dim,
            batch_first=True
        )

    def forward(self, z):
        output, _ = self.rnn(z)
        return output


# 4. Supervisor Network
class Supervisor(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.rnn = nn.GRU(
            hidden_dim,
            hidden_dim,
            batch_first=True
        )

    def forward(self, h):
        output, _ = self.rnn(h)
        return output


# 5. Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.rnn = nn.GRU(
            hidden_dim,
            hidden_dim,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_dim,
            1
        )

    def forward(self, h):
        output, _ = self.rnn(h)
        logits = self.fc(output)
        return logits
