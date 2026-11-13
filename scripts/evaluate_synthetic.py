import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 1. Load original and synthetic datasets
# --------------------------------------------------

print("Loading original Lorenz data...")

original = np.load(DATA_DIR / "lorenz_windows.npy")

print("Loading synthetic Lorenz data...")

synthetic = np.load(OUTPUT_DIR / "synthetic_lorenz.npy")

print("Original shape:", original.shape)
print("Synthetic shape:", synthetic.shape)

# Use one window from each dataset for trajectory plots.
original_sequence = original[0]
synthetic_sequence = synthetic[0]

# --------------------------------------------------
# 2. Compare statistical properties
# --------------------------------------------------

print("\nStatistical comparison")

variable_names = ["X", "Y", "Z"]

for i, name in enumerate(variable_names):

    original_values = original[:, :, i].reshape(-1)
    synthetic_values = synthetic[:, :, i].reshape(-1)

    print(f"\nVariable {name}")

    print(
        "Original mean:",
        round(float(original_values.mean()), 4),
    )

    print(
        "Synthetic mean:",
        round(float(synthetic_values.mean()), 4),
    )

    print(
        "Original standard deviation:",
        round(float(original_values.std()), 4),
    )

    print(
        "Synthetic standard deviation:",
        round(float(synthetic_values.std()), 4),
    )


# --------------------------------------------------
# 3. Plot time-series comparisons
# --------------------------------------------------

fig, axes = plt.subplots(3, 1, figsize=(12, 9))

for i, name in enumerate(variable_names):

    axes[i].plot(
        original_sequence[:, i],
        label="Original",
    )

    axes[i].plot(
        synthetic_sequence[:, i],
        label="Synthetic",
    )

    axes[i].set_title(f"Lorenz Variable {name}")
    axes[i].set_xlabel("Time step")
    axes[i].set_ylabel(name)
    axes[i].legend()
    axes[i].grid(True)

fig.tight_layout()

time_plot_path = OUTPUT_DIR / "lorenz_time_series_comparison.png"

fig.savefig(time_plot_path, dpi=150)
plt.close(fig)

print("\nSaved time-series plot:", time_plot_path)


# --------------------------------------------------
# 4. Plot three-dimensional trajectories
# --------------------------------------------------

fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(121, projection="3d")

ax1.plot(
    original_sequence[:, 0],
    original_sequence[:, 1],
    original_sequence[:, 2],
)

ax1.set_title("Original Lorenz Trajectory")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")

ax2 = fig.add_subplot(122, projection="3d")

ax2.plot(
    synthetic_sequence[:, 0],
    synthetic_sequence[:, 1],
    synthetic_sequence[:, 2],
)

ax2.set_title("Synthetic Lorenz Trajectory")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Z")

fig.tight_layout()

trajectory_path = OUTPUT_DIR / "lorenz_trajectory_comparison.png"

fig.savefig(trajectory_path, dpi=150)
plt.close(fig)

print("Saved trajectory plot:", trajectory_path)

print("\nEvaluation script completed.")