import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

real = np.load(PROJECT_ROOT / "data" / "lorenz_windows.npy")

exp4 = np.load(
    PROJECT_ROOT / "outputs" / "synthetic_lorenz.npy"
)

exp5 = np.load(
    PROJECT_ROOT / "outputs" / "synthetic_lorenz_experiment5.npy"
)

datasets = {
    "REAL LORENZ DATA": real,
    "EXPERIMENT 4": exp4,
    "EXPERIMENT 5": exp5,
}

print("\n========== REAL DATA COMPARISON ==========")

for name, data in datasets.items():

    print(f"\n{name}")

    print("Shape:", data.shape)

    print("Mean:",
          np.mean(data, axis=(0, 1)))

    print("Standard deviation:",
          np.std(data, axis=(0, 1)))

    differences = np.diff(data, axis=1)

    print("Temporal difference std:",
          np.std(differences, axis=(0, 1)))

    previous = data[:, :-1, :].reshape(
        -1, data.shape[2]
    )

    current = data[:, 1:, :].reshape(
        -1, data.shape[2]
    )

    correlations = []

    for i in range(data.shape[2]):

        corr = np.corrcoef(
            previous[:, i],
            current[:, i]
        )[0, 1]

        correlations.append(corr)

    print("Lag-1 autocorrelation:", correlations)


print("\n========== EVALUATION COMPLETED ==========")