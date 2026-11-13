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
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "lorenz_timegan_experiment7.pt"

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
SUPERVISED_WEIGHT = 1.0

DEVICE = torch.device("cpu")

torch.manual_seed(42)
np.random.seed(42)


# --------------------------------------------------
# 2. Load and normalize data
# --------------------------------------------------

data = np.load(DATA_PATH).astype(np.float32)

feature_min = data.min(axis=(0, 1), keepdims=True)
feature_max = data.max(axis=(0, 1), keepdims=True)

normalized_data = (
    2.0 * (data - feature_min) / (feature_max - feature_min + 1e-8)
) - 1.0

dataset = TensorDataset(torch.tensor(normalized_data))
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
)

print("Dataset shape:", data.shape)
print("Normalized dataset shape:", normalized_data.shape)


# --------------------------------------------------
# 3. Initialize models
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

optimizer_s = torch.optim.Adam(
    supervisor.parameters(),
    lr=LEARNING_RATE,
)

optimizer_g = torch.optim.Adam(
    list(generator.parameters()) + list(supervisor.parameters()),
    lr=LEARNING_RATE,
)

optimizer_d = torch.optim.Adam(
    discriminator.parameters(),
    lr=LEARNING_RATE,
)


# --------------------------------------------------
# 4. Autocorrelation loss
# --------------------------------------------------

def calculate_autocorrelation(x, lag):
    previous = x[:, :-lag, :]
    current = x[:, lag:, :]

    previous = previous - previous.mean(dim=(0, 1), keepdim=True)
    current = current - current.mean(dim=(0, 1), keepdim=True)

    covariance = (previous * current).mean(dim=(0, 1))

    previous_std = previous.square().mean(dim=(0, 1)).sqrt()
    current_std = current.square().mean(dim=(0, 1)).sqrt()

    return covariance / (previous_std * current_std + 1e-8)


def calculate_autocorrelation_loss(real, generated):
    losses = []

    for lag in AUTOCORR_LAGS:
        real_corr = calculate_autocorrelation(real, lag)
        generated_corr = calculate_autocorrelation(generated, lag)

        losses.append(mse_loss(generated_corr, real_corr))

    return torch.stack(losses).mean()


# --------------------------------------------------
# 5. Phase 1: Reconstruction training
# --------------------------------------------------

print("\nPhase 1: Reconstruction training")

for epoch in range(RECONSTRUCTION_EPOCHS):
    epoch_loss = 0.0

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

        epoch_loss += loss.item()

    print(
        f"Epoch {epoch + 1}/{RECONSTRUCTION_EPOCHS} "
        f"Loss: {epoch_loss / len(loader):.6f}"
    )


# --------------------------------------------------
# 6. Phase 2: Supervisor training
# --------------------------------------------------

print("\nPhase 2: Supervisor training")

for epoch in range(SUPERVISOR_EPOCHS):
    epoch_loss = 0.0

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

        epoch_loss += loss.item()

    print(
        f"Epoch {epoch + 1}/{SUPERVISOR_EPOCHS} "
        f"Loss: {epoch_loss / len(loader):.6f}"
    )


# --------------------------------------------------
# 7. Freeze Embedder and Recovery
# --------------------------------------------------

for parameter in embedder.parameters():
    parameter.requires_grad = False

for parameter in recovery.parameters():
    parameter.requires_grad = False


# --------------------------------------------------
# 8. Phase 3: Adversarial training
# --------------------------------------------------

print("\nPhase 3: Adversarial training")

for epoch in range(ADVERSARIAL_EPOCHS):
    g_epoch_loss = 0.0
    d_epoch_loss = 0.0
    supervised_epoch_loss = 0.0

    for (real_data,) in loader:
        real_data = real_data.to(DEVICE)

        batch_size = real_data.size(0)

        # Real hidden representation
        with torch.no_grad():
            real_hidden = embedder(real_data)

        # Generate synthetic hidden representation
        noise = torch.randn(
            batch_size,
            real_data.size(1),
            NOISE_DIM,
            device=DEVICE,
        )

        generated_hidden = supervisor(generator(noise))

        generated_data = recovery(generated_hidden)

        # ------------------------------------------
        # Discriminator training
        # ------------------------------------------

        real_logits = discriminator(real_hidden.detach())
        fake_logits = discriminator(generated_hidden.detach())

        d_real_loss = bce_loss(
            real_logits,
            torch.ones_like(real_logits),
        )

        d_fake_loss = bce_loss(
            fake_logits,
            torch.zeros_like(fake_logits),
        )

        d_loss = d_real_loss + d_fake_loss

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

        noise = torch.randn(
            batch_size,
            real_data.size(1),
            NOISE_DIM,
            device=DEVICE,
        )

        generated_hidden = supervisor(generator(noise))
        generated_data = recovery(generated_hidden)

        generated_logits = discriminator(generated_hidden)

        adversarial_loss = bce_loss(
            generated_logits,
            torch.ones_like(generated_logits),
        )

        # Moment loss
        real_mean = real_data.mean(dim=(0, 1))
        generated_mean = generated_data.mean(dim=(0, 1))

        real_std = real_data.std(dim=(0, 1), unbiased=False)
        generated_std = generated_data.std(dim=(0, 1), unbiased=False)

        moment_loss = (
            mse_loss(generated_mean, real_mean)
            + mse_loss(generated_std, real_std)
        )

        # Temporal difference statistics loss
        real_diff = real_data[:, 1:, :] - real_data[:, :-1, :]
        generated_diff = (
            generated_data[:, 1:, :] - generated_data[:, :-1, :]
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

        # Autocorrelation loss
        autocorr_loss = calculate_autocorrelation_loss(
            real_data,
            generated_data,
        )

        # Supervisor preservation loss on REAL hidden states
        predicted_real_hidden = supervisor(real_hidden.detach())

        supervised_loss = mse_loss(
            predicted_real_hidden[:, :-1, :],
            real_hidden[:, 1:, :].detach(),
        )

        # Combined generator and supervisor loss
        g_loss = (
            adversarial_loss
            + 10.0 * moment_loss
            + 5.0 * temporal_loss
            + AUTOCORR_WEIGHT * autocorr_loss
            + SUPERVISED_WEIGHT * supervised_loss
        )

        optimizer_g.zero_grad()
        g_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(generator.parameters()) + list(supervisor.parameters()),
            MAX_GRAD_NORM,
        )

        optimizer_g.step()

        g_epoch_loss += g_loss.item()
        d_epoch_loss += d_loss.item()
        supervised_epoch_loss += supervised_loss.item()

    print(
        f"Epoch {epoch + 1}/{ADVERSARIAL_EPOCHS} "
        f"G Loss: {g_epoch_loss / len(loader):.6f} | "
        f"D Loss: {d_epoch_loss / len(loader):.6f} | "
        f"Supervisor Loss: "
        f"{supervised_epoch_loss / len(loader):.6f}"
    )


# --------------------------------------------------
# 9. Save checkpoint
# --------------------------------------------------

checkpoint = {
    "embedder": embedder.state_dict(),
    "recovery": recovery.state_dict(),
    "generator": generator.state_dict(),
    "supervisor": supervisor.state_dict(),
    "discriminator": discriminator.state_dict(),
    "feature_min": feature_min,
    "feature_max": feature_max,
    "input_dim": INPUT_DIM,
    "hidden_dim": HIDDEN_DIM,
    "noise_dim": NOISE_DIM,
    "autocorr_lags": AUTOCORR_LAGS,
    "autocorr_weight": AUTOCORR_WEIGHT,
    "supervised_weight": SUPERVISED_WEIGHT,
}

torch.save(checkpoint, OUTPUT_PATH)

print("\nExperiment 7 training completed.")
print("Checkpoint saved to:", OUTPUT_PATH)