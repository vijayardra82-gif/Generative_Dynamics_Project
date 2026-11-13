import numpy as np
from pathlib import Path
from scipy.stats import wasserstein_distance

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Dataset paths
datasets = {
    "Experiment 4": OUTPUT_DIR / "synthetic_lorenz_experiment4_previous.npy",
    "Experiment 5": OUTPUT_DIR / "synthetic_lorenz_experiment5.npy",
    "Experiment 6": OUTPUT_DIR / "synthetic_lorenz_experiment6.npy",
    "Experiment 7": OUTPUT_DIR / "synthetic_lorenz_experiment7.npy",
    "Experiment 8": OUTPUT_DIR / "synthetic_lorenz_experiment8.npy",
    "Experiment 9": OUTPUT_DIR / "synthetic_lorenz_experiment9.npy",
}

REAL_PATH = PROJECT_ROOT / "data" / "lorenz_windows.npy"

# Load real data
real_data = np.load(REAL_PATH)
real_points = real_data.reshape(-1, 3)

print("Real dataset loaded:", real_data.shape)
print("\nWasserstein Distance (lower is better)")
print("-" * 55)

# Compare each synthetic dataset with real data
for name, file_path in datasets.items():

    synthetic_data = np.load(file_path)
    synthetic_points = synthetic_data.reshape(-1, 3)

    print(f"\n{name}")

    for feature_index, feature_name in enumerate(["X", "Y", "Z"]):

        distance = wasserstein_distance(
            real_points[:, feature_index],
            synthetic_points[:, feature_index]
        )

        print(f"{feature_name}: {distance:.6f}")

print("\nDistribution evaluation completed!")
