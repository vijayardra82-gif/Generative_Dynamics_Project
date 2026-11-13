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

lags = [1, 2, 5, 10, 20]
results = {}

for name, data in datasets.items():

    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError(f"Unexpected shape: {name} {data.shape}")

    print(f"\n========== {name} ==========")

    results[name] = {}

    for lag in lags:
        correlations = []

        for feature in range(3):
            window_correlations = []

            for window in data:
                previous = window[:-lag, feature]
                current = window[lag:, feature]

                corr = np.corrcoef(previous, current)[0, 1]

                if np.isfinite(corr):
                    window_correlations.append(corr)

            correlations.append(np.mean(window_correlations))

        results[name][lag] = correlations

        print(
            f"Lag {lag:2d}: "
            f"X={correlations[0]:.5f}, "
            f"Y={correlations[1]:.5f}, "
            f"Z={correlations[2]:.5f}"
        )

print("\nAUTOCORRELATION ERROR COMPARISON")
print("=" * 65)

real_results = results["REAL DATA"]
errors = {}

for name in list(datasets.keys())[1:]:
    differences = []

    for lag in lags:
        differences.extend(
            np.abs(
                np.array(results[name][lag])
                - np.array(real_results[lag])
            )
        )

    mae = np.mean(differences)
    errors[name] = mae

    print(f"{name}: MAE = {mae:.6f}")

print("\nBEST EXPERIMENT:", min(errors, key=errors.get))