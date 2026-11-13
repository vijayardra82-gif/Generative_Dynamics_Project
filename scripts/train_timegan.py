import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.timegan import (
    Embedder,
    Recovery,
    Generator,
    Supervisor,
    Discriminator,
)

DATA_PATH = PROJECT_ROOT / "data" / "lorenz_windows.npy"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")

HIDDEN_DIM = 24
NOISE_DIM = 10
BATCH_SIZE = 64

RECONSTRUCTION_EPOCHS = 5
SUPERVISOR_EPOCHS = 5
ADVERSARIAL_EPOCHS = 5

LEARNING_RATE = 0.001

torch.manual_seed(42)
np.random.seed(42)


# --------------------------------------------------
# 2. Load and normalize the dataset
# --------------------------------------------------

print("Loading Lorenz dataset...")

data = np.load(DATA_PATH)

if data.ndim != 3:
    raise ValueError(
        f"Expected a 3D dataset, but received shape {data.shape}"
    )

if not np.isfinite(data).all():
    raise ValueError("Dataset contains NaN or infinite values.")

print("Original dataset shape:", data.shape)

# Normalize each variable independently to [-1, 1].
feature_min = data.min(axis=(0, 1), keepdims=True)
feature_max = data.max(axis=(0, 1), keepdims=True)

feature_range = feature_max - feature_min
safe_range = np.where(feature_range == 0, 1.0, feature_range)

normalized_data = (
    2.0 * (data - feature_min) / safe_range - 1.0
)

normalized_data = normalized_data.astype(np.float32)

print("Normalized minimum:", normalized_data.min())
print("Normalized maximum:", normalized_data.max())

tensor_data = torch.from_numpy(normalized_data)

dataset = TensorDataset(tensor_data)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=False,
)

INPUT_DIM = data.shape[2]

print("Number of sequences:", len(dataset))
print("Input dimension:", INPUT_DIM)
print("Device:", DEVICE)


# --------------------------------------------------
# 3. Initialize the five networks
# --------------------------------------------------

embedder = Embedder(INPUT_DIM, HIDDEN_DIM).to(DEVICE)

recovery = Recovery(HIDDEN_DIM, INPUT_DIM).to(DEVICE)

generator = Generator(NOISE_DIM, HIDDEN_DIM).to(DEVICE)

supervisor = Supervisor(HIDDEN_DIM).to(DEVICE)

discriminator = Discriminator(HIDDEN_DIM).to(DEVICE)


# --------------------------------------------------
# 4. Loss functions and optimizers
# --------------------------------------------------

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
    generator.parameters(),
    lr=LEARNING_RATE,
)

optimizer_d = torch.optim.Adam(
    discriminator.parameters(),
    lr=LEARNING_RATE,
)


def create_noise(batch_size, sequence_length):
    """Generate random noise for the Generator."""
    return torch.randn(
        batch_size,
        sequence_length,
        NOISE_DIM,
        device=DEVICE,
    )


# --------------------------------------------------
# 5. Phase One: Reconstruction training
# --------------------------------------------------

print("\nPhase 1: Reconstruction training")

for epoch in range(RECONSTRUCTION_EPOCHS):

    embedder.train()
    recovery.train()

    total_loss = 0.0

    for (real_data,) in loader:

        real_data = real_data.to(DEVICE)

        optimizer_er.zero_grad()

        hidden = embedder(real_data)
        reconstructed = recovery(hidden)

        loss = mse_loss(reconstructed, real_data)

        loss.backward()
        optimizer_er.step()

        total_loss += loss.item()

    average_loss = total_loss / len(loader)

    print(
        f"Reconstruction Epoch {epoch + 1}/"
        f"{RECONSTRUCTION_EPOCHS} | "
        f"Loss: {average_loss:.6f}"
    )


# --------------------------------------------------
# 6. Phase Two: Supervisor training
# --------------------------------------------------

print("\nPhase 2: Supervisor training")

embedder.eval()

for parameter in embedder.parameters():
    parameter.requires_grad = False

for epoch in range(SUPERVISOR_EPOCHS):

    supervisor.train()

    total_loss = 0.0

    for (real_data,) in loader:

        real_data = real_data.to(DEVICE)

        with torch.no_grad():
            hidden = embedder(real_data)

        optimizer_s.zero_grad()

        predicted_hidden = supervisor(hidden)

        # Predict the next hidden state.
        loss = mse_loss(
            predicted_hidden[:, :-1, :],
            hidden[:, 1:, :],
        )

        loss.backward()
        optimizer_s.step()

        total_loss += loss.item()

    average_loss = total_loss / len(loader)

    print(
        f"Supervisor Epoch {epoch + 1}/"
        f"{SUPERVISOR_EPOCHS} | "
        f"Loss: {average_loss:.6f}"
    )


# --------------------------------------------------
# 7. Phase Three: Adversarial training
# --------------------------------------------------

print("\nPhase 3: Adversarial training")

for parameter in embedder.parameters():
    parameter.requires_grad = False

for parameter in recovery.parameters():
    parameter.requires_grad = False

embedder.eval()
recovery.eval()

for epoch in range(ADVERSARIAL_EPOCHS):

    generator.train()
    supervisor.train()
    discriminator.train()

    total_g_loss = 0.0
    total_d_loss = 0.0

    for (real_data,) in loader:

        real_data = real_data.to(DEVICE)

        batch_size = real_data.size(0)
        sequence_length = real_data.size(1)

        with torch.no_grad():
            real_hidden = embedder(real_data)

        # ------------------------------------------
        # Train the Discriminator
        # ------------------------------------------

        optimizer_d.zero_grad()

        noise = create_noise(batch_size, sequence_length)

        with torch.no_grad():
            fake_hidden = supervisor(generator(noise))

        real_logits = discriminator(real_hidden.detach())
        fake_logits = discriminator(fake_hidden.detach())

        real_labels = torch.ones_like(real_logits)
        fake_labels = torch.zeros_like(fake_logits)

        loss_real = bce_loss(real_logits, real_labels)
        loss_fake = bce_loss(fake_logits, fake_labels)

        d_loss = loss_real + loss_fake

        d_loss.backward()
        optimizer_d.step()

        # ------------------------------------------
        # Train the Generator and Supervisor
        # ------------------------------------------

        optimizer_g.zero_grad()
        optimizer_s.zero_grad()

        noise = create_noise(batch_size, sequence_length)

        generated_hidden = supervisor(generator(noise))
        generated_logits = discriminator(generated_hidden)

        # Adversarial loss.
        g_adversarial_loss = bce_loss(
            generated_logits,
            torch.ones_like(generated_logits),
        )

        # Convert generated hidden states into data space.
        generated_data = recovery(generated_hidden)

        # Statistical moment losses.
        mean_loss = mse_loss(
            generated_data.mean(dim=(0, 1)),
            real_data.mean(dim=(0, 1)),
        )

        std_loss = mse_loss(
            generated_data.std(dim=(0, 1), unbiased=False),
            real_data.std(dim=(0, 1), unbiased=False),
        )

        moment_loss = mean_loss + std_loss

            # ------------------------------------------
        # Temporal statistics loss
        # ------------------------------------------

        generated_differences = (
            generated_data[:, 1:, :]
            - generated_data[:, :-1, :]
        )

        real_differences = (
            real_data[:, 1:, :]
            - real_data[:, :-1, :]
        )

        # Compare average temporal changes.
        temporal_mean_loss = mse_loss(
            generated_differences.mean(dim=(0, 1)),
            real_differences.mean(dim=(0, 1)),
        )

        # Compare variability in temporal changes.
        temporal_std_loss = mse_loss(
            generated_differences.std(
                dim=(0, 1),
                unbiased=False,
            ),
            real_differences.std(
                dim=(0, 1),
                unbiased=False,
            ),
        )

        temporal_loss = temporal_mean_loss + temporal_std_loss

        # Combined Generator objective.
        g_loss = (
            g_adversarial_loss
            + 10.0 * moment_loss
            + 5.0 * temporal_loss
        )

        g_loss.backward()

        optimizer_g.step()
        optimizer_s.step()

        total_g_loss += g_loss.item()
        total_d_loss += d_loss.item()

    # Calculate average losses for this epoch.
    average_g_loss = total_g_loss / len(loader)
    average_d_loss = total_d_loss / len(loader)

    print(
        f"Adversarial Epoch {epoch + 1}/{ADVERSARIAL_EPOCHS} | "
        f"Generator Loss: {average_g_loss:.6f} | "
        f"Discriminator Loss: {average_d_loss:.6f}"
    )


# --------------------------------------------------
# 8. Save the trained model
# --------------------------------------------------

checkpoint_path = OUTPUT_DIR / "lorenz_timegan.pt"

torch.save(
    {
        "embedder": embedder.state_dict(),
        "recovery": recovery.state_dict(),
        "generator": generator.state_dict(),
        "supervisor": supervisor.state_dict(),
        "discriminator": discriminator.state_dict(),
        "input_dim": INPUT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "noise_dim": NOISE_DIM,
        "feature_min": feature_min.reshape(-1),
        "feature_max": feature_max.reshape(-1),
    },
    checkpoint_path,
)

print("\nTraining completed!")
print("Model checkpoint saved to:", checkpoint_path)
print("Next step: Generate synthetic Lorenz sequences.")