import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs"

datasets = {
    "REAL DATA": np.load(ROOT / "data" / "lorenz_windows.npy"),

    "EXPERIMENT 4": np.load(OUTPUT / "synthetic_lorenz.npy"),

    "EXPERIMENT 5": np.load(OUTPUT / "synthetic_lorenz_experiment5.npy"),

    "EXPERIMENT 6": np.load(OUTPUT / "synthetic_lorenz_experiment6.npy"),

    "EXPERIMENT 7": np.load(OUTPUT / "synthetic_lorenz_experiment7.npy"),

    "EXPERIMENT 8": np.load(OUTPUT / "synthetic_lorenz_experiment8.npy"),

    "EXPERIMENT 9": np.load(OUTPUT / "synthetic_lorenz_experiment9.npy"),
}

features = ["X", "Y", "Z"]

results = {}

print("\nCROSS-CORRELATION EVALUATION")
print("=" * 65)

for name, data in datasets.items():

    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError(f"Invalid dataset shape: {name}")

    points = data.reshape(-1, 3)

    corr_matrix = np.corrcoef(points, rowvar=False)

    results[name] = corr_matrix

    print(f"\n========== {name} ==========")

    print("\nCorrelation Matrix:")

    print("       X         Y         Z")

    for i, feature in enumerate(features):

        print(
            f"{feature}  "
            f"{corr_matrix[i, 0]:.5f}   "
            f"{corr_matrix[i, 1]:.5f}   "
            f"{corr_matrix[i, 2]:.5f}"
        )


# Compare synthetic correlations with real correlations

print("\n\nCROSS-CORRELATION ERROR COMPARISON")
print("=" * 65)

real_matrix = results["REAL DATA"]

errors = {}

for name in list(datasets.keys())[1:]:

    differences = []

    for i in range(3):

        for j in range(i + 1, 3):

            difference = abs(
                results[name][i, j] - real_matrix[i, j]
            )

            differences.append(difference)

    mae = np.mean(differences)

    errors[name] = mae

    print(f"{name}: MAE = {mae:.6f}")


print("\nBEST EXPERIMENT:", min(errors, key=errors.get))

print("\nCross-correlation evaluation completed!")