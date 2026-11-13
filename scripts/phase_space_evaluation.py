import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from mpl_toolkits.mplot3d import Axes3D

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Load datasets
datasets = {
    "Real Lorenz": PROJECT_ROOT / "data" / "lorenz_windows.npy",
    "Experiment 4": OUTPUT_DIR / "synthetic_lorenz_experiment4_previous.npy",
    "Experiment 5": OUTPUT_DIR / "synthetic_lorenz_experiment5.npy",
    "Experiment 6": OUTPUT_DIR / "synthetic_lorenz_experiment6.npy",
}

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)

# Plot each dataset
for name, file_path in datasets.items():

    if not file_path.exists():
        print(f"File not found: {file_path}")
        continue

    data = np.load(file_path)

    # Select one sequence
    sequence = data[0]

    x = sequence[:, 0]
    y = sequence[:, 1]
    z = sequence[:, 2]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(x, y, z)

    ax.set_title(f"{name} - 3D Phase Space")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    output_name = name.lower().replace(" ", "_")
    save_path = OUTPUT_DIR / f"{output_name}_phase_space.png"

    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")

print("\nPhase-space evaluation completed!")