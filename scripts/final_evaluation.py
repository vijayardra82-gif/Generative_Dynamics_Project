import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wasserstein_distance

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REAL_PATH = PROJECT_ROOT / "data" / "lorenz_windows.npy"

# Experiments 4–11
EXPERIMENTS = {
    4: "synthetic_lorenz_experiment4_previous.npy",
    5: "synthetic_lorenz_experiment5.npy",
    6: "synthetic_lorenz_experiment6.npy",
    7: "synthetic_lorenz_experiment7.npy",
    8: "synthetic_lorenz_experiment8.npy",
    9: "synthetic_lorenz_experiment9.npy",
    10: "synthetic_lorenz_experiment10.npy",
    11: "synthetic_lorenz_experiment11.npy",
}

FEATURES = ["X", "Y", "Z"]
LAGS = [1, 2, 5, 10, 20]


def autocorrelation(data, lag):
    values = []

    for feature in range(data.shape[2]):
        feature_values = []

        for window in data:
            x = window[:-lag, feature]
            y = window[lag:, feature]

            if np.std(x) > 0 and np.std(y) > 0:
                feature_values.append(np.corrcoef(x, y)[0, 1])

        values.append(np.mean(feature_values))

    return np.array(values)


def cross_correlation(data):
    points = data.reshape(-1, 3)
    matrix = np.corrcoef(points, rowvar=False)

    return np.array([
        matrix[0, 1],
        matrix[0, 2],
        matrix[1, 2]
    ])


# Load real dataset
real = np.load(REAL_PATH)
real_points = real.reshape(-1, 3)

print("Real dataset:", real.shape)

# Real reference metrics
real_diversity = np.std(real, axis=0)

real_autocorr = {
    lag: autocorrelation(real, lag)
    for lag in LAGS
}

real_crosscorr = cross_correlation(real)

results = []

for experiment, filename in EXPERIMENTS.items():

    file_path = OUTPUT_DIR / filename

    if not file_path.exists():
        print(f"WARNING: Missing Experiment {experiment}: {filename}")
        continue

    synthetic = np.load(file_path)

    print(f"Evaluating Experiment {experiment}: {synthetic.shape}")

    if synthetic.ndim != 3 or synthetic.shape[2] != 3:
        print(f"Skipping Experiment {experiment}: Invalid shape")
        continue

    synthetic_points = synthetic.reshape(-1, 3)

    # 1. Wasserstein distance
    wasserstein = np.mean([
        wasserstein_distance(
            real_points[:, i],
            synthetic_points[:, i]
        )
        for i in range(3)
    ])

    # 2. Temporal diversity MAE
    synthetic_diversity = np.std(synthetic, axis=0)

    diversity_mae = np.mean(
        np.abs(real_diversity - synthetic_diversity)
    )

    # 3. Autocorrelation MAE
    autocorr_errors = []

    for lag in LAGS:
        synthetic_autocorr = autocorrelation(synthetic, lag)

        autocorr_errors.extend(
            np.abs(
                real_autocorr[lag] - synthetic_autocorr
            )
        )

    autocorr_mae = np.mean(autocorr_errors)

    # 4. Cross-correlation MAE
    synthetic_crosscorr = cross_correlation(synthetic)

    crosscorr_mae = np.mean(
        np.abs(real_crosscorr - synthetic_crosscorr)
    )

    results.append({
        "Experiment": f"Experiment {experiment}",
        "Wasserstein": wasserstein,
        "Temporal Diversity": diversity_mae,
        "Autocorrelation": autocorr_mae,
        "Cross-correlation": crosscorr_mae
    })


# Create results dataframe
df = pd.DataFrame(results)

# Rank each metric: lower error is better
metric_columns = [
    "Wasserstein",
    "Temporal Diversity",
    "Autocorrelation",
    "Cross-correlation"
]

for metric in metric_columns:

    df[metric + " Rank"] = df[metric].rank(
        method="min",
        ascending=True
    )

rank_columns = [
    metric + " Rank"
    for metric in metric_columns
]

df["Total Rank"] = df[rank_columns].sum(axis=1)

df = df.sort_values("Total Rank")

# Display results
print("\nFINAL EVALUATION RESULTS")
print("=" * 110)

print(df.to_string(index=False))

# Save results
output_file = OUTPUT_DIR / "final_evaluation_results.csv"

df.to_csv(output_file, index=False)

print("\nBest overall experiment:", df.iloc[0]["Experiment"])

print("Results saved to:", output_file)

print("\nFinal evaluation completed!")