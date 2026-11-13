import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Load both experiments
exp4 = np.load(OUTPUT_DIR / "synthetic_lorenz.npy")
exp5 = np.load(OUTPUT_DIR / "synthetic_lorenz_experiment5.npy")

datasets = {
    "Experiment 4": exp4,
    "Experiment 5": exp5,
}

print("\n========== DATASET COMPARISON ==========")

for name, data in datasets.items():

    print(f"\n{name}")

    print("Shape:", data.shape)
    print("NaN:", np.isnan(data).any())
    print("Infinite:", np.isinf(data).any())

    print("Mean:", np.mean(data, axis=(0, 1)))
    print("Standard deviation:", np.std(data, axis=(0, 1)))

    differences = np.diff(data, axis=1)

    print("Temporal difference std:",
          np.std(differences, axis=(0, 1)))

    # Lag-1 autocorrelation
    previous = data[:, :-1, :].reshape(-1, data.shape[2])
    current = data[:, 1:, :].reshape(-1, data.shape[2])

    correlations = []

    for feature in range(data.shape[2]):
        correlation = np.corrcoef(
            previous[:, feature],
            current[:, feature]
        )[0, 1]

        correlations.append(correlation)

    print("Lag-1 autocorrelation:", correlations)


print("\n========== EXPERIMENT COMPARISON ==========")

print("Experiment 4 shape:", exp4.shape)
print("Experiment 5 shape:", exp5.shape)

print("\nMean absolute difference between feature means:")

mean4 = np.mean(exp4, axis=(0, 1))
mean5 = np.mean(exp5, axis=(0, 1))

print(np.abs(mean4 - mean5))

print("\nEvaluation completed!")