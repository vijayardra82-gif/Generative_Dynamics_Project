import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.timegan import Recovery, Generator, Supervisor


# Configuration
CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "lorenz_timegan_experiment8.pt"

OUTPUT_NPY = PROJECT_ROOT / "outputs" / "synthetic_lorenz_experiment8.npy"
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "synthetic_lorenz_experiment8.csv"

NUM_SAMPLES = 500
SEQ_LENGTH = 100

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cpu")


# Load checkpoint
checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)

HIDDEN_DIM = checkpoint["hidden_dim"]
NOISE_DIM = checkpoint["noise_dim"]
INPUT_DIM = checkpoint["input_dim"]

# Initialize models
recovery = Recovery(HIDDEN_DIM, INPUT_DIM).to(DEVICE)
generator = Generator(NOISE_DIM, HIDDEN_DIM).to(DEVICE)
supervisor = Supervisor(HIDDEN_DIM).to(DEVICE)

recovery.load_state_dict(checkpoint["recovery"])
generator.load_state_dict(checkpoint["generator"])
supervisor.load_state_dict(checkpoint["supervisor"])

recovery.eval()
generator.eval()
supervisor.eval()


# Generate synthetic sequences
with torch.no_grad():

    noise = torch.randn(
        NUM_SAMPLES,
        SEQ_LENGTH,
        NOISE_DIM,
        device=DEVICE
    )

    generated_hidden = supervisor(generator(noise))

    generated_normalized = recovery(generated_hidden)

    generated_normalized = generated_normalized.cpu().numpy()


# Inverse normalization
feature_min = checkpoint["feature_min"]
feature_max = checkpoint["feature_max"]

synthetic_data = (
    ((generated_normalized + 1.0) / 2.0)
    * (feature_max - feature_min)
    + feature_min
)

synthetic_data = synthetic_data.astype(np.float32)


# Validate generated data
if not np.isfinite(synthetic_data).all():
    raise ValueError("Generated data contains NaN or Inf values.")


# Save NumPy file
np.save(OUTPUT_NPY, synthetic_data)


# Save CSV file
flattened_data = synthetic_data.reshape(-1, INPUT_DIM)

df = pd.DataFrame(
    flattened_data,
    columns=["x", "y", "z"]
)

df.to_csv(OUTPUT_CSV, index=False)


print("Experiment 8 synthetic data generated successfully.")
print("Shape:", synthetic_data.shape)
print("NumPy file:", OUTPUT_NPY)
print("CSV file:", OUTPUT_CSV)
print("Contains NaN:", np.isnan(synthetic_data).any())
print("Contains Inf:", np.isinf(synthetic_data).any())
