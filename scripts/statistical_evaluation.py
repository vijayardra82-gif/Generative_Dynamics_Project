import numpy as np
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Dataset paths
datasets = {
    "Real Data": PROJECT_ROOT / "data" / "lorenz_windows.npy",
    "Experiment 4": OUTPUT_DIR / "synthetic_lorenz_experiment4_previous.npy",
    "Experiment 5": OUTPUT_DIR / "synthetic_lorenz_experiment5.npy",
    "Experiment 6": OUTPUT_DIR / "synthetic_lorenz_experiment6.npy",
}

features = ["X", "Y", "Z"]

print("\nSTATISTICAL EVALUATION")
print("=" * 75)

for name, file_path in datasets.items():

    data = np.load(file_path)
    points = data.reshape(-1, 3)

    print(f"\n{name}")
    print("-" * 50)

    for i, feature in enumerate(features):

        values = points[:, i]

        mean = np.mean(values)
        std = np.std(values)
        variance = np.var(values)

        print(f"{feature}:")
        print(f"  Mean:     {mean:.6f}")
        print(f"  Std Dev:  {std:.6f}")
        print(f"  Variance: {variance:.6f}")

print("\nStatistical evaluation completed!")