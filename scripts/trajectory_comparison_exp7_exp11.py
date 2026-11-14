import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

datasets_paths = {
    "Real Lorenz": DATA_DIR / "lorenz_windows.npy",
    "Experiment 7": OUTPUT_DIR / "synthetic_lorenz_experiment7.npy",
    "Experiment 11": OUTPUT_DIR / "synthetic_lorenz_experiment11.npy",
}

NUM_TRAJECTORIES = 5

# --------------------------------------------------
# 2. Load datasets
# --------------------------------------------------

def load_data(path):

    if not path.exists():
        raise FileNotFoundError(path)

    data = np.load(path).astype(np.float64)

    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError(f"Invalid shape: {data.shape}")

    if not np.isfinite(data).all():
        raise ValueError(f"NaN or Inf found: {path}")

    return data


datasets = {}

print("Loading datasets...")

for name, path in datasets_paths.items():

    datasets[name] = load_data(path)

    print(f"{name}: {datasets[name].shape}")

# --------------------------------------------------
# 3. Select trajectories
# --------------------------------------------------

trajectories = {}

for name, data in datasets.items():

    indices = np.linspace(
        0,
        len(data) - 1,
        NUM_TRAJECTORIES,
        dtype=int
    )

    trajectories[name] = data[indices]

    print(f"{name}: selected {len(indices)} trajectories")

# --------------------------------------------------
# 4. Common axis limits
# --------------------------------------------------

all_points = np.concatenate(
    [data.reshape(-1, 3) for data in trajectories.values()],
    axis=0
)

minimum = all_points.min(axis=0)
maximum = all_points.max(axis=0)

padding = (maximum - minimum) * 0.05

minimum -= padding
maximum += padding

# --------------------------------------------------
# 5. Create trajectory comparison
# --------------------------------------------------

fig = plt.figure(figsize=(18, 6))

for index, (name, data) in enumerate(
    trajectories.items(),
    start=1
):

    ax = fig.add_subplot(
        1, 3, index,
        projection="3d"
    )

    for trajectory in data:

        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            trajectory[:, 2],
            linewidth=0.9,
            alpha=0.8
        )

    ax.set_title(name, fontsize=14)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_xlim(minimum[0], maximum[0])
    ax.set_ylim(minimum[1], maximum[1])
    ax.set_zlim(minimum[2], maximum[2])

fig.suptitle(
    "Time-Ordered Lorenz Trajectories: Real vs Experiments 7 and 11",
    fontsize=16
)

fig.tight_layout(rect=[0, 0, 1, 0.93])

# --------------------------------------------------
# 6. Save figure
# --------------------------------------------------

OUTPUT_DIR.mkdir(exist_ok=True)

output_path = OUTPUT_DIR / "trajectory_comparison_real_exp7_exp11.png"

fig.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("\nTrajectory comparison completed!")
print("Saved to:", output_path)