import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"

datasets = {
    "Real Lorenz": ROOT / "data" / "lorenz_windows.npy",
    "Experiment 7": OUTPUT_DIR / "synthetic_lorenz_experiment7.npy",
    "Experiment 11": OUTPUT_DIR / "synthetic_lorenz_experiment11.npy",
}

MAX_POINTS = 12000
RNG = np.random.default_rng(42)

# --------------------------------------------------
# 2. Load datasets
# --------------------------------------------------

loaded_data = {}

for name, path in datasets.items():

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    data = np.load(path)

    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError(f"Invalid shape: {name}")

    if not np.isfinite(data).all():
        raise ValueError(f"NaN or Inf found: {name}")

    points = data.reshape(-1, 3)

    if len(points) > MAX_POINTS:
        indices = RNG.choice(
            len(points),
            size=MAX_POINTS,
            replace=False
        )
        points = points[indices]

    loaded_data[name] = points

    print(f"{name}: {data.shape}, displaying {len(points)} points")

# --------------------------------------------------
# 3. Common axis limits
# --------------------------------------------------

all_points = np.concatenate(
    list(loaded_data.values()),
    axis=0
)

axis_min = np.percentile(all_points, 1, axis=0)
axis_max = np.percentile(all_points, 99, axis=0)

# --------------------------------------------------
# 4. Phase-space projections
# --------------------------------------------------

projections = [
    (0, 1, "X", "Y", "X-Y Projection"),
    (0, 2, "X", "Z", "X-Z Projection"),
    (1, 2, "Y", "Z", "Y-Z Projection"),
]

fig, axes = plt.subplots(
    3, 3,
    figsize=(16, 12),
    squeeze=False
)

for row, (name, points) in enumerate(loaded_data.items()):

    for col, (xi, yi, xlabel, ylabel, title) in enumerate(projections):

        ax = axes[row, col]

        ax.scatter(
            points[:, xi],
            points[:, yi],
            s=0.5,
            alpha=0.25
        )

        ax.set_title(f"{name} - {title}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.set_xlim(axis_min[xi], axis_max[xi])
        ax.set_ylim(axis_min[yi], axis_max[yi])

        ax.grid(True, alpha=0.2)

# --------------------------------------------------
# 5. Save figure
# --------------------------------------------------

fig.suptitle(
    "Lorenz Phase-Space Comparison: Real vs Experiments 7 and 11",
    fontsize=17
)

plt.tight_layout(rect=[0, 0, 1, 0.97])

output_path = OUTPUT_DIR / "phase_space_real_exp7_exp11.png"

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nPhase-space comparison completed!")
print("Saved to:", output_path)