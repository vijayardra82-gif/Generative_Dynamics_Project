import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# 1. Project paths
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
}

MAX_POINTS = 25000

# --------------------------------------------------
# 2. Create comparison figure
# --------------------------------------------------

fig = plt.figure(figsize=(18, 12))

for index, (name, file_path) in enumerate(datasets.items(), start=1):

    if not file_path.exists():
        print(f"File not found: {file_path}")
        continue

    data = np.load(file_path)

    if data.ndim != 3 or data.shape[2] != 3:
        print(f"Invalid dataset shape for {name}: {data.shape}")
        continue

    if not np.isfinite(data).all():
        print(f"Invalid values found in {name}")
        continue

    # Combine sequences for occupancy visualization
    points = data.reshape(-1, 3)

    # Select points randomly
    if len(points) > MAX_POINTS:
        rng = np.random.default_rng(42)
        selected_indices = rng.choice(
            len(points),
            size=MAX_POINTS,
            replace=False
        )
        points = points[selected_indices]

    # Create 3D subplot
    ax = fig.add_subplot(2, 3, index, projection="3d")

    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        s=1,
        alpha=0.4
    )

    ax.set_title(name)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    print(f"{name}: {data.shape} -> displaying {len(points)} points")

# --------------------------------------------------
# 3. Save comparison
# --------------------------------------------------

plt.suptitle(
    "Lorenz System - Phase Space Comparison (Experiments 4-8)",
    fontsize=16
)

plt.tight_layout()

save_path = OUTPUT_DIR / "phase_space_comparison_experiment8.png"

plt.savefig(save_path, dpi=300)

plt.close()

print("\nPhase-space comparison completed.")
print("Comparison saved to:", save_path)
