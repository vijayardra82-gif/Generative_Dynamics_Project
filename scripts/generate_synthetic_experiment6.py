import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.timegan import Recovery, Generator, Supervisor


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

CHECKPOINT_PATH = (
    PROJECT_ROOT / "outputs" / "lorenz_timegan_experiment6.pt"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"

NUM_SEQUENCES = 500
SEQUENCE_LENGTH = 100

DEVICE = torch.device("cpu")

torch.manual_seed(42)


# --------------------------------------------------
# 2. Load trained checkpoint
# --------------------------------------------------

print("Loading trained Lorenz model...")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False,
)

input_dim = checkpoint["input_dim"]
hidden_dim = checkpoint["hidden_dim"]
noise_dim = checkpoint["noise_dim"]

print("Input dimension:", input_dim)
print("Hidden dimension:", hidden_dim)
print("Noise dimension:", noise_dim)


# --------------------------------------------------
# 3. Initialize and restore model weights
# --------------------------------------------------

recovery = Recovery(hidden_dim, input_dim).to(DEVICE)
generator = Generator(noise_dim, hidden_dim).to(DEVICE)
supervisor = Supervisor(hidden_dim).to(DEVICE)

recovery.load_state_dict(checkpoint["recovery"])
generator.load_state_dict(checkpoint["generator"])
supervisor.load_state_dict(checkpoint["supervisor"])

recovery.eval()
generator.eval()
supervisor.eval()


# --------------------------------------------------
# 4. Generate synthetic sequences
# --------------------------------------------------

print("\nGenerating synthetic Lorenz sequences...")

with torch.no_grad():

    noise = torch.randn(
        NUM_SEQUENCES,
        SEQUENCE_LENGTH,
        noise_dim,
        device=DEVICE,
    )

    generated_hidden = supervisor(generator(noise))

    generated_normalized = recovery(generated_hidden)

    generated_normalized = generated_normalized.cpu().numpy()


# --------------------------------------------------
# 5. Reverse normalization
# --------------------------------------------------

feature_min = np.asarray(checkpoint["feature_min"])
feature_max = np.asarray(checkpoint["feature_max"])

generated_data = (
    (generated_normalized + 1.0) / 2.0
) * (feature_max - feature_min) + feature_min


# --------------------------------------------------
# 6. Save Experiment 6 generated data
# --------------------------------------------------

npy_path = OUTPUT_DIR / "synthetic_lorenz_experiment6.npy"

csv_path = OUTPUT_DIR / "synthetic_lorenz_experiment6.csv"

np.save(npy_path, generated_data)

flat_data = generated_data.reshape(-1, input_dim)

np.savetxt(
    csv_path,
    flat_data,
    delimiter=",",
    header="x,y,z",
    comments="",
)

print("\nExperiment 6 generation completed!")
print("Generated data shape:", generated_data.shape)
print("NaN values:", np.isnan(generated_data).any())
print("Infinite values:", np.isinf(generated_data).any())
print("Saved NumPy file:", npy_path)
print("Saved CSV file:", csv_path)
