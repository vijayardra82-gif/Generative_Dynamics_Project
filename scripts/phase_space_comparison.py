import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

datasets = {
    "Real Lorenz": PROJECT_ROOT / "data" / "lorenz_windows.npy",
    "Experiment 4": OUTPUT_DIR / "synthetic_lorenz_experiment4_previous.npy",
    "Experiment 5": OUTPUT_DIR / "synthetic_lorenz_experiment5.npy",
    "Experiment 6": OUTPUT_DIR / "synthetic_lorenz_experiment6.npy",
    "Experiment 7": OUTPUT_DIR / "synthetic_lorenz_experiment7.npy",
    "Experiment 8": OUTPUT_DIR / "synthetic_lorenz_experiment8.npy",
    "Experiment 9": OUTPUT_DIR / "synthetic_lorenz_experiment9.npy",
}

MAX_POINTS = 12000
RNG = np.random.default_rng(42)


# --------------------------------------------------
# 2. Load datasets
# --------------------------------------------------

loaded_data = {}

for name, file_path in datasets.items():

    if not file_path.exists():
        print(f"File not found: {file_path}")
        continue

    data = np.load(file_path)

    if data.ndim != 3 or data.shape[2] != 3:
        print(f"Invalid shape: {name} - {data.shape}")
        continue

    if not np.isfinite(data).all():
        print(f"Invalid values detected: {name}")
        continue

    points = data.reshape(-1, 3)

    if len(points) > MAX_POINTS:
        indices = RNG.choice(
            len(points),
            size=MAX_POINTS,
            replace=False
        )
        points = points[indices]

    loaded_data[name] = points

    print(f"{name}: {data.shape} -> displaying {len(points)} points")


# --------------------------------------------------
# 3. Calculate common axis limits
# --------------------------------------------------

if not loaded_data:
    raise ValueError("No valid datasets were loaded.")

all_points = np.concatenate(
    list(loaded_data.values()),
    axis=0
)

axis_min = np.percentile(all_points, 1, axis=0)
axis_max = np.percentile(all_points, 99, axis=0)


# --------------------------------------------------
# 4. Create 2D phase-space projections
# --------------------------------------------------

projections = [
    (0, 1, "X", "Y", "X-Y Projection"),
    (0, 2, "X", "Z", "X-Z Projection"),
    (1, 2, "Y", "Z", "Y-Z Projection"),
]

fig, axes = plt.subplots(
    len(loaded_data),
    3,
    figsize=(16, 4 * len(loaded_data)),
    squeeze=False
)

for row, (name, points) in enumerate(loaded_data.items()):

    for col, (x_index, y_index, x_label, y_label, title) in enumerate(projections):

        ax = axes[row, col]

        ax.scatter(
            points[:, x_index],
            points[:, y_index],
            s=0.5,
            alpha=0.25
        )

        ax.set_title(f"{name} - {title}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # Same axis limits for fair comparison
        ax.set_xlim(axis_min[x_index], axis_max[x_index])
        ax.set_ylim(axis_min[y_index], axis_max[y_index])

        ax.grid(True, alpha=0.2)


# --------------------------------------------------
# 5. Save comparison figure
# --------------------------------------------------

plt.suptitle(
    "Lorenz Attractor: Real Data vs Experiments 4-9",
    fontsize=18
)

plt.tight_layout(rect=[0, 0, 1, 0.99])

save_path = OUTPUT_DIR / "phase_space_projections_experiments4_to_9.png"

plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nPhase-space projection comparison completed!")
print("Saved to:", save_path)