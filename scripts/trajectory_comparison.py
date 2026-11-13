import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

datasets_paths = {
    "Real Lorenz": DATA_DIR / "lorenz_windows.npy",
    "Experiment 6": OUTPUT_DIR / "synthetic_lorenz_experiment6.npy",
    "Experiment 8": OUTPUT_DIR / "synthetic_lorenz_experiment8.npy",
    "Experiment 9": OUTPUT_DIR / "synthetic_lorenz_experiment9.npy",
}

NUM_TRAJECTORIES = 5


# ==========================================
# 2. LOAD DATA
# ==========================================

def load_data(path):

    if not path.exists():
        raise FileNotFoundError(path)

    data = np.load(path).astype(np.float64)

    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError(f"Invalid shape: {data.shape}")

    if not np.isfinite(data).all():
        raise ValueError(f"NaN or Inf found: {path}")

    return data


print("Loading datasets...")

datasets = {}

for name, path in datasets_paths.items():

    datasets[name] = load_data(path)

    print(f"{name}: {datasets[name].shape}")


# ==========================================
# 3. SELECT TRAJECTORIES
# ==========================================

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


# ==========================================
# 4. COMMON AXIS LIMITS
# ==========================================

all_points = np.concatenate(
    [data.reshape(-1, 3) for data in trajectories.values()],
    axis=0
)

minimum = all_points.min(axis=0)
maximum = all_points.max(axis=0)

padding = (maximum - minimum) * 0.05

minimum -= padding
maximum += padding


# ==========================================
# 5. CREATE TRAJECTORY COMPARISON
# ==========================================

fig = plt.figure(figsize=(16, 12))

for index, (name, data) in enumerate(
    trajectories.items(),
    start=1
):

    ax = fig.add_subplot(
        2, 2, index,
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

    ax.set_title(name, fontsize=13)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_xlim(minimum[0], maximum[0])
    ax.set_ylim(minimum[1], maximum[1])
    ax.set_zlim(minimum[2], maximum[2])


fig.suptitle(
    "Time-Ordered Lorenz Trajectories: Real vs Experiments 6, 8 and 9",
    fontsize=16
)

fig.tight_layout(rect=[0, 0, 1, 0.95])


# ==========================================
# 6. SAVE FIGURE
# ==========================================

OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = (
    OUTPUT_DIR /
    "trajectory_comparison_real_exp6_exp8_exp9.png"
)

fig.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)

print("\nTrajectory comparison completed successfully!")
print("Saved to:", OUTPUT_PATH)