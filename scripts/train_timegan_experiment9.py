import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.timegan import (
    Embedder,
    Recovery,
    Generator,
    Supervisor,
    Discriminator,
)

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATA_PATH = PROJECT_ROOT / "data" / "lorenz_windows.npy"
OUTPUT_PATH = (
    PROJECT_ROOT / "outputs" / "lorenz_timegan_experiment9.pt"
)

DEVICE = torch.device("cpu")

INPUT_DIM = 3
HIDDEN_DIM = 24
NOISE_DIM = 10
BATCH_SIZE = 64

RECONSTRUCTION_EPOCHS = 5
SUPERVISOR_EPOCHS = 5
ADVERSARIAL_EPOCHS = 5

LEARNING_RATE = 0.001
MAX_GRAD_NORM = 1.0

AUTOCORR_LAGS = [1, 2, 5, 10, 20]
AUTOCORR_WEIGHT = 5.0

PHYSICS_WEIGHT = 0.25

SIGMA = 10.0
RHO = 28.0
BETA = 8.0 / 3.0

DT = 50.0 / 4999.0

torch.manual_seed(42)
np.random.seed(42)


# --------------------------------------------------
# 2. Load and normalize data
# --------------------------------------------------

print("Loading Lorenz dataset...")

data = np.load(DATA_PATH).astype(np.float32)

if data.ndim != 3 or data.shape[2] != INPUT_DIM:
    raise ValueError(f"Unexpected dataset shape: {data.shape}")

if not np.isfinite(data).all():
    raise ValueError("Dataset contains NaN or Inf.")

feature_min = data.min(axis=(0, 1), keepdims=True)
feature_max = data.max(axis=(0, 1), keepdims=True)

feature_range = feature_max - feature_min
safe_range = np.where(feature_range == 0, 1.0, feature_range)

normalized_data = (
    2.0 * (data - feature_min) / safe_range - 1.0
).astype(np.float32)

tensor_data = torch.from_numpy(normalized_data)

loader = DataLoader(
    TensorDataset(tensor_data),
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=False,
)

# Original physical-unit scaling.
feature_min_t = torch.tensor(
    feature_min.reshape(-1),
    dtype=torch.float32,
    device=DEVICE,
)

feature_range_t = torch.tensor(
    safe_range.reshape(-1),
    dtype=torch.float32,
    device=DEVICE,
)

# Characteristic derivative scales calculated from
# the real Lorenz equations over the training data.
real_tensor = torch.from_numpy(data).to(DEVICE)

real_x = real_tensor[..., 0]
real_y = real_tensor[..., 1]
real_z = real_tensor[..., 2]

real_derivatives = torch.stack(
    [
        SIGMA * (real_y - real_x),
        real_x * (RHO - real_z) - real_y,
        real_x * real_y - BETA * real_z,
    ],
    dim=-1,
)

derivative_scale = torch.sqrt(
    torch.mean(real_derivatives ** 2, dim=(0, 1))
).clamp_min(1e-6)

print("Dataset shape:", data.shape)
print("Physics derivative scales:", derivative_scale.cpu().numpy())


# --------------------------------------------------
# 3. Initialize networks
# --------------------------------------------------

embedder = Embedder(INPUT_DIM, HIDDEN_DIM).to(DEVICE)
recovery = Recovery(HIDDEN_DIM, INPUT_DIM).to(DEVICE)
generator = Generator(NOISE_DIM, HIDDEN_DIM).to(DEVICE)
supervisor = Supervisor(HIDDEN_DIM).to(DEVICE)
discriminator = Discriminator(HIDDEN_DIM).to(DEVICE)

mse_loss = nn.MSELoss()
bce_loss = nn.BCEWithLogitsLoss()

optimizer_er = torch.optim.Adam(
    list(embedder.parameters()) + list(recovery.parameters()),
    lr=LEARNING_RATE,
)


optimizer_g = torch.optim.Adam(
    generator.parameters(),
    lr=LEARNING_RATE,
)

optimizer_s = torch.optim.Adam(
    supervisor.parameters(),
    lr=LEARNING_RATE,
)

optimizer_d = torch.optim.Adam(
    discriminator.parameters(),
    lr=LEARNING_RATE,
)


# --------------------------------------------------
# 4. Helper functions
# --------------------------------------------------

def create_noise(batch_size, sequence_length):
    return torch.randn(
        batch_size,
        sequence_length,
        NOISE_DIM,
        device=DEVICE,
    )


def calculate_autocorrelation(values, lag):

    previous = values[:, :-lag, :]
    current = values[:, lag:, :]

    previous = previous - previous.mean(
        dim=(0, 1), keepdim=True
    )

    current = current - current.mean(
        dim=(0, 1), keepdim=True
    )

    covariance = (previous * current).mean(dim=(0, 1))

    previous_variance = previous.square().mean(dim=(0, 1))
    current_variance = current.square().mean(dim=(0, 1))

    denominator = torch.sqrt(
        previous_variance * current_variance + 1e-8
    )

    return covariance / denominator


def calculate_physics_loss(generated_normalized):

    # Convert generated values to original physical units.
    generated_physical = (
        (generated_normalized + 1.0) / 2.0
    ) * feature_range_t + feature_min_t

    # Midpoint states.
    midpoint = (
        generated_physical[:, 1:, :]
        + generated_physical[:, :-1, :]
    ) / 2.0

    # Numerical derivatives.
    numerical_derivative = (
        generated_physical[:, 1:, :]
        - generated_physical[:, :-1, :]
    ) / DT

    x = midpoint[..., 0]
    y = midpoint[..., 1]
    z = midpoint[..., 2]

    # Lorenz differential equations.
    expected_derivative = torch.stack(
        [
            SIGMA * (y - x),
            x * (RHO - z) - y,
            x * y - BETA * z,
        ],
        dim=-1,
    )

    residual = (
        numerical_derivative - expected_derivative
    )

    normalized_residual = residual / derivative_scale

    return torch.mean(normalized_residual ** 2)


# --------------------------------------------------
# 5. Phase 1: Reconstruction training
# --------------------------------------------------

print("\nPhase 1: Reconstruction training")

for epoch in range(RECONSTRUCTION_EPOCHS):

    total_loss = 0.0

    for (real_data,) in loader:

        real_data = real_data.to(DEVICE)

        hidden = embedder(real_data)
        reconstructed = recovery(hidden)

        loss = mse_loss(reconstructed, real_data)

        optimizer_er.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(embedder.parameters()) + list(recovery.parameters()),
            MAX_GRAD_NORM,
        )

        optimizer_er.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch + 1}/{RECONSTRUCTION_EPOCHS} "
        f"Loss: {total_loss / len(loader):.6f}"
    )


# --------------------------------------------------
# 6. Phase 2: Supervisor training
# --------------------------------------------------

print("\nPhase 2: Supervisor training")

for parameter in embedder.parameters():
    parameter.requires_grad = False

embedder.eval()

for epoch in range(SUPERVISOR_EPOCHS):

    total_loss = 0.0

    for (real_data,) in loader:

        real_data = real_data.to(DEVICE)

        with torch.no_grad():
            hidden = embedder(real_data)

        predicted_hidden = supervisor(hidden)

        loss = mse_loss(
            predicted_hidden[:, :-1, :],
            hidden[:, 1:, :],
        )

        optimizer_s.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            supervisor.parameters(),
            MAX_GRAD_NORM,
        )

        optimizer_s.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch + 1}/{SUPERVISOR_EPOCHS} "
        f"Loss: {total_loss / len(loader):.6f}"
    )


# --------------------------------------------------
# 7. Freeze Embedder and Recovery
# --------------------------------------------------

for parameter in recovery.parameters():
    parameter.requires_grad = False

embedder.eval()
recovery.eval()


# --------------------------------------------------
# 8. Phase 3: Physics-informed adversarial training
# --------------------------------------------------

print("\nPhase 3: Physics-informed adversarial training")

for epoch in range(ADVERSARIAL_EPOCHS):

    total_g_loss = 0.0
    total_d_loss = 0.0
    total_physics_loss = 0.0

    for (real_data,) in loader:

        real_data = real_data.to(DEVICE)

        batch_size = real_data.size(0)
        sequence_length = real_data.size(1)

        with torch.no_grad():
            real_hidden = embedder(real_data)

        # ------------------------------------------
        # Discriminator training
        # ------------------------------------------

        noise = create_noise(batch_size, sequence_length)

        with torch.no_grad():
            fake_hidden = supervisor(generator(noise))

        real_logits = discriminator(real_hidden)
        fake_logits = discriminator(fake_hidden)

        d_loss = (
            bce_loss(real_logits, torch.ones_like(real_logits))
            + bce_loss(fake_logits, torch.zeros_like(fake_logits))
        )

        optimizer_d.zero_grad()
        d_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            discriminator.parameters(),
            MAX_GRAD_NORM,
        )

        optimizer_d.step()

        # ------------------------------------------
        # Generator and Supervisor training
        # ------------------------------------------

        noise = create_noise(batch_size, sequence_length)

        generated_hidden = supervisor(generator(noise))

        generated_logits = discriminator(generated_hidden)

        adversarial_loss = bce_loss(
            generated_logits,
            torch.ones_like(generated_logits),
        )

        generated_data = recovery(generated_hidden)

        # Moment loss.
        moment_loss = (
            mse_loss(
                generated_data.mean(dim=(0, 1)),
                real_data.mean(dim=(0, 1)),
            )
            + mse_loss(
                generated_data.std(dim=(0, 1), unbiased=False),
                real_data.std(dim=(0, 1), unbiased=False),
            )
        )

        # Temporal difference loss.
        generated_diff = (
            generated_data[:, 1:, :]
            - generated_data[:, :-1, :]
        )

        real_diff = (
            real_data[:, 1:, :]
            - real_data[:, :-1, :]
        )

        temporal_loss = (
            mse_loss(
                generated_diff.mean(dim=(0, 1)),
                real_diff.mean(dim=(0, 1)),
            )
            + mse_loss(
                generated_diff.std(dim=(0, 1), unbiased=False),
                real_diff.std(dim=(0, 1), unbiased=False),
            )
        )

        # Multi-lag autocorrelation loss.
        autocorr_loss = torch.tensor(0.0, device=DEVICE)

        for lag in AUTOCORR_LAGS:

            real_corr = calculate_autocorrelation(real_data, lag)

            generated_corr = calculate_autocorrelation(
                generated_data,
                lag,
            )

            autocorr_loss = autocorr_loss + mse_loss(
                generated_corr,
                real_corr,
            )

        autocorr_loss = autocorr_loss / len(AUTOCORR_LAGS)

        # New physics-informed loss.
        physics_loss = calculate_physics_loss(generated_data)

        # Combined objective.
        g_loss = (
            adversarial_loss
            + 10.0 * moment_loss
            + 5.0 * temporal_loss
            + AUTOCORR_WEIGHT * autocorr_loss
            + PHYSICS_WEIGHT * physics_loss
        )

        optimizer_g.zero_grad()
        optimizer_s.zero_grad()

        g_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(generator.parameters()) + list(supervisor.parameters()),
            MAX_GRAD_NORM,
        )

        optimizer_g.step()
        optimizer_s.step()

        total_g_loss += g_loss.item()
        total_d_loss += d_loss.item()
        total_physics_loss += physics_loss.item()

    print(
        f"Epoch {epoch + 1}/{ADVERSARIAL_EPOCHS} | "
        f"G Loss: {total_g_loss / len(loader):.6f} | "
        f"D Loss: {total_d_loss / len(loader):.6f} | "
        f"Physics Loss: {total_physics_loss / len(loader):.6f}"
    )


# --------------------------------------------------
# 9. Save Experiment 8 checkpoint
# --------------------------------------------------

checkpoint = {
    "embedder": embedder.state_dict(),
    "recovery": recovery.state_dict(),
    "generator": generator.state_dict(),
    "supervisor": supervisor.state_dict(),
    "discriminator": discriminator.state_dict(),
    "feature_min": feature_min.reshape(-1),
    "feature_max": feature_max.reshape(-1),
    "input_dim": INPUT_DIM,
    "hidden_dim": HIDDEN_DIM,
    "noise_dim": NOISE_DIM,
    "autocorr_lags": AUTOCORR_LAGS,
    "autocorr_weight": AUTOCORR_WEIGHT,
    "physics_weight": PHYSICS_WEIGHT,
    "physics_derivative_scale": derivative_scale.cpu().numpy(),
    "dt": DT,
}

torch.save(checkpoint, OUTPUT_PATH)

print("\nExperiment 9 training completed!")
print("Checkpoint saved to:", OUTPUT_PATH)