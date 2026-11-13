import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.timegan import Recovery, Generator, Supervisor, Discriminator

ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cpu")

DATA_PATH = ROOT / "data" / "lorenz_windows.npy"
CHECKPOINT_PATH = ROOT / "outputs" / "lorenz_timegan_experiment8.pt"

torch.manual_seed(42)

# Load checkpoint and real data
checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False,
)

data = np.load(DATA_PATH).astype(np.float32)

feature_min = checkpoint["feature_min"]
feature_max = checkpoint["feature_max"]
feature_range = feature_max - feature_min

normalized = 2 * (data - feature_min) / feature_range - 1
real_data = torch.tensor(normalized[:64], dtype=torch.float32)

# Reconstruct trained networks
hidden_dim = checkpoint["hidden_dim"]
noise_dim = checkpoint["noise_dim"]

recovery = Recovery(hidden_dim, 3)
generator = Generator(noise_dim, hidden_dim)
supervisor = Supervisor(hidden_dim)
discriminator = Discriminator(hidden_dim)

recovery.load_state_dict(checkpoint["recovery"])
generator.load_state_dict(checkpoint["generator"])
supervisor.load_state_dict(checkpoint["supervisor"])
discriminator.load_state_dict(checkpoint["discriminator"])

recovery.eval()
generator.eval()
supervisor.eval()
discriminator.eval()

# Generate synthetic sequences
noise = torch.randn(64, 100, noise_dim)

generated_hidden = supervisor(generator(noise))
generated_data = recovery(generated_hidden)

mse = nn.MSELoss()
bce = nn.BCEWithLogitsLoss()

# Adversarial loss
generated_logits = discriminator(generated_hidden)

adversarial = bce(
    generated_logits,
    torch.ones_like(generated_logits),
)

# Moment loss
moment = (
    mse(generated_data.mean(dim=(0, 1)),
        real_data.mean(dim=(0, 1)))
    +
    mse(generated_data.std(dim=(0, 1), unbiased=False),
        real_data.std(dim=(0, 1), unbiased=False))
)

# Temporal loss
generated_diff = generated_data[:, 1:] - generated_data[:, :-1]
real_diff = real_data[:, 1:] - real_data[:, :-1]

temporal = (
    mse(generated_diff.mean(dim=(0, 1)),
        real_diff.mean(dim=(0, 1)))
    +
    mse(generated_diff.std(dim=(0, 1), unbiased=False),
        real_diff.std(dim=(0, 1), unbiased=False))
)

# Autocorrelation loss
def autocorrelation(values, lag):
    previous = values[:, :-lag]
    current = values[:, lag:]

    previous = previous - previous.mean(dim=(0, 1), keepdim=True)
    current = current - current.mean(dim=(0, 1), keepdim=True)

    covariance = (previous * current).mean(dim=(0, 1))

    denominator = torch.sqrt(
        previous.square().mean(dim=(0, 1))
        * current.square().mean(dim=(0, 1))
        + 1e-8
    )

    return covariance / denominator


autocorr = torch.tensor(0.0)

for lag in checkpoint["autocorr_lags"]:
    autocorr += mse(
        autocorrelation(generated_data, lag),
        autocorrelation(real_data, lag),
    )

autocorr /= len(checkpoint["autocorr_lags"])

# Physics loss
physical = (
    (generated_data + 1) / 2
) * torch.tensor(feature_range) + torch.tensor(feature_min)

midpoint = (physical[:, 1:] + physical[:, :-1]) / 2

numerical = (
    physical[:, 1:] - physical[:, :-1]
) / checkpoint["dt"]

x = midpoint[..., 0]
y = midpoint[..., 1]
z = midpoint[..., 2]

expected = torch.stack([
    10 * (y - x),
    x * (28 - z) - y,
    x * y - (8 / 3) * z,
], dim=-1)

derivative_scale = torch.tensor(
    checkpoint["physics_derivative_scale"]
)

physics = (((numerical - expected) / derivative_scale) ** 2).mean()

# Weighted contributions
components = {
    "Adversarial": (adversarial, 1.0),
    "Moment": (moment, 10.0),
    "Temporal": (temporal, 5.0),
    "Autocorrelation": (autocorr, checkpoint["autocorr_weight"]),
    "Physics": (physics, checkpoint["physics_weight"]),
}

print("\nEXPERIMENT 8 LOSS DIAGNOSTIC")
print("-" * 55)

for name, (loss, weight) in components.items():
    print(
        f"{name:18s} "
        f"Raw: {loss.item():.6f} "
        f"Weighted: {(loss * weight).item():.6f}"
    )

# Compare gradient magnitudes reaching Generator and Supervisor
parameters = (
    list(generator.parameters())
    + list(supervisor.parameters())
)

print("\nGRADIENT CONTRIBUTIONS")
print("-" * 55)

for name, (loss, weight) in components.items():
    gradients = torch.autograd.grad(
        loss * weight,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )

    norm = torch.sqrt(sum(
        (g.detach() ** 2).sum()
        for g in gradients
        if g is not None
    ))

    print(f"{name:18s} Gradient norm: {norm.item():.6f}")

print("\nDiagnostic completed. No model files were modified.")