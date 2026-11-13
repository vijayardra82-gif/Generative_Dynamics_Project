import numpy as np
from pathlib import Path

# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

REAL_DATA_PATH = DATA_DIR / "lorenz_windows.npy"

EXPERIMENT_FILES = {
    "Experiment 4": OUTPUT_DIR / "synthetic_lorenz_experiment4_previous.npy",
    "Experiment 5": OUTPUT_DIR / "synthetic_lorenz_experiment5.npy",
    "Experiment 6": OUTPUT_DIR / "synthetic_lorenz_experiment6.npy",
    "Experiment 7": OUTPUT_DIR / "synthetic_lorenz_experiment7.npy",
}

LAGS = [1, 2, 5, 10, 20]


# --------------------------------------------------
# 2. Load datasets
# --------------------------------------------------

def load_dataset(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    data = np.load(path)

    if data.ndim != 3:
        raise ValueError(f"Invalid dataset shape: {data.shape}")

    if data.shape[2] != 3:
        raise ValueError(f"Expected 3 features, got: {data.shape[2]}")

    if not np.isfinite(data).all():
        raise ValueError(f"Dataset contains NaN or Inf: {path}")

    return data


print("Loading real Lorenz dataset...")

real_data = load_dataset(REAL_DATA_PATH)

print("Real dataset shape:", real_data.shape)

datasets = {"Real Data": real_data}

for name, path in EXPERIMENT_FILES.items():
    datasets[name] = load_dataset(path)
    print(f"{name} shape:", datasets[name].shape)


# --------------------------------------------------
# 3. Calculate autocorrelation
# --------------------------------------------------

def calculate_autocorrelation(data, lag):

    previous = data[:, :-lag, :]
    current = data[:, lag:, :]

    previous_centered = (
        previous - previous.mean(axis=(0, 1), keepdims=True)
    )

    current_centered = (
        current - current.mean(axis=(0, 1), keepdims=True)
    )

    covariance = (
        previous_centered * current_centered
    ).mean(axis=(0, 1))

    previous_variance = (
        previous_centered ** 2
    ).mean(axis=(0, 1))

    current_variance = (
        current_centered ** 2
    ).mean(axis=(0, 1))

    denominator = np.sqrt(
        previous_variance * current_variance + 1e-8
    )

    return covariance / denominator


# --------------------------------------------------
# 4. Calculate statistical metrics
# --------------------------------------------------

def calculate_metrics(data):

    differences = data[:, 1:, :] - data[:, :-1, :]

    metrics = {
        "mean": data.mean(axis=(0, 1)),
        "std": data.std(axis=(0, 1)),
        "temporal_mean": differences.mean(axis=(0, 1)),
        "temporal_std": differences.std(axis=(0, 1)),
        "autocorrelation": {
            lag: calculate_autocorrelation(data, lag)
            for lag in LAGS
        },
    }

    return metrics


results = {}

for name, data in datasets.items():
    results[name] = calculate_metrics(data)


# --------------------------------------------------
# 5. Display results
# --------------------------------------------------

print("\n" + "=" * 75)
print("EXPERIMENT 4 vs EXPERIMENT 5 vs EXPERIMENT 6 vs EXPERIMENT 7")
print("=" * 75)

for name, metrics in results.items():

    print(f"\n{name}")
    print("-" * 50)

    print("Mean:", np.round(metrics["mean"], 5))
    print("Standard deviation:", np.round(metrics["std"], 5))

    print(
        "Temporal difference mean:",
        np.round(metrics["temporal_mean"], 5),
    )

    print(
        "Temporal difference std:",
        np.round(metrics["temporal_std"], 5),
    )

    print("Autocorrelation:")

    for lag, values in metrics["autocorrelation"].items():
        print(f"  Lag {lag}: {np.round(values, 5)}")


# --------------------------------------------------
# 6. Calculate absolute errors against real data
# --------------------------------------------------

real_metrics = results["Real Data"]

print("\n" + "=" * 75)
print("ABSOLUTE ERROR COMPARED WITH REAL DATA")
print("=" * 75)

for name in EXPERIMENT_FILES:

    metrics = results[name]

    print(f"\n{name}")
    print("-" * 50)

    print(
        "Mean error:",
        np.round(
            np.abs(metrics["mean"] - real_metrics["mean"]),
            5,
        ),
    )

    print(
        "Standard deviation error:",
        np.round(
            np.abs(metrics["std"] - real_metrics["std"]),
            5,
        ),
    )

    print(
        "Temporal difference std error:",
        np.round(
            np.abs(
                metrics["temporal_std"]
                - real_metrics["temporal_std"]
            ),
            5,
        ),
    )

    print("Autocorrelation errors:")

    for lag in LAGS:

        error = np.abs(
            metrics["autocorrelation"][lag]
            - real_metrics["autocorrelation"][lag]
        )

        print(f"  Lag {lag}: {np.round(error, 5)}")


print("\nEvaluation completed successfully!")