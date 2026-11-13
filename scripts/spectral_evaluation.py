import numpy as np
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

REAL_PATH = DATA_DIR / "lorenz_windows.npy"

EXPERIMENTS = {
    "Experiment 4": OUTPUT_DIR / "synthetic_lorenz_experiment4_previous.npy",
    "Experiment 5": OUTPUT_DIR / "synthetic_lorenz_experiment5.npy",
    "Experiment 6": OUTPUT_DIR / "synthetic_lorenz_experiment6.npy",
    "Experiment 7": OUTPUT_DIR / "synthetic_lorenz_experiment7.npy",
    "Experiment 8": OUTPUT_DIR / "synthetic_lorenz_experiment8.npy",
    "Experiment 9": OUTPUT_DIR / "synthetic_lorenz_experiment9.npy",
}

FEATURES = ["X", "Y", "Z"]


# --------------------------------------------------
# 2. Load and validate datasets
# --------------------------------------------------

def load_dataset(path):

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    data = np.load(path)

    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError(f"Invalid dataset shape: {data.shape}")

    if not np.isfinite(data).all():
        raise ValueError(f"NaN or Inf detected in {path}")

    return data


# --------------------------------------------------
# 3. Calculate average normalized power spectrum
# --------------------------------------------------

def calculate_average_spectrum(data):

    # Remove the mean from each window
    data = data - data.mean(axis=1, keepdims=True)

    window_length = data.shape[1]

    # Apply Hann window
    hann = np.hanning(window_length)

    windowed_data = data * hann[None, :, None]

    # Fourier transform along time axis
    fft_values = np.fft.rfft(windowed_data, axis=1)

    # Calculate power
    power = np.abs(fft_values) ** 2

    # Remove zero-frequency component
    power = power[:, 1:, :]

    # Normalize each window's spectrum
    power_sum = power.sum(axis=1, keepdims=True)

    normalized_power = power / (power_sum + 1e-12)

    # Average across all windows
    average_spectrum = normalized_power.mean(axis=0)

    # Normalize average spectrum
    average_spectrum /= (
        average_spectrum.sum(axis=0, keepdims=True) + 1e-12
    )

    return average_spectrum


# --------------------------------------------------
# 4. Calculate spectral comparison errors
# --------------------------------------------------

def calculate_spectral_errors(real_spectrum, synthetic_spectrum):

    # Mean Absolute Error
    mae = np.mean(
        np.abs(real_spectrum - synthetic_spectrum),
        axis=0
    )

    # Spectral Wasserstein distance
    frequency_spacing = 1.0 / 100

    cumulative_difference = np.cumsum(
        real_spectrum - synthetic_spectrum,
        axis=0
    )

    wasserstein = (
        np.sum(np.abs(cumulative_difference), axis=0)
        * frequency_spacing
    )

    return mae, wasserstein


# --------------------------------------------------
# 5. Load real data
# --------------------------------------------------

print("Loading real Lorenz dataset...")

real_data = load_dataset(REAL_PATH)

print("Real dataset shape:", real_data.shape)

real_spectrum = calculate_average_spectrum(real_data)


# --------------------------------------------------
# 6. Evaluate Experiments 4-8
# --------------------------------------------------

results = []

print("\n" + "=" * 75)
print("SPECTRAL ANALYSIS: REAL DATA VS SYNTHETIC EXPERIMENTS")
print("=" * 75)

for name, path in EXPERIMENTS.items():

    print(f"\nEvaluating {name}...")

    synthetic_data = load_dataset(path)

    print("Dataset shape:", synthetic_data.shape)

    synthetic_spectrum = calculate_average_spectrum(
        synthetic_data
    )

    mae, wasserstein = calculate_spectral_errors(
        real_spectrum,
        synthetic_spectrum
    )

    for i, feature in enumerate(FEATURES):

        results.append({
            "Experiment": name,
            "Feature": feature,
            "Spectral_MAE": mae[i],
            "Spectral_Wasserstein": wasserstein[i]
        })

        print(
            f"{feature}: "
            f"MAE = {mae[i]:.6f}, "
            f"Wasserstein = {wasserstein[i]:.6f}"
        )


# --------------------------------------------------
# 7. Display overall comparison
# --------------------------------------------------

df = pd.DataFrame(results)

summary = df.groupby("Experiment")[
    ["Spectral_MAE", "Spectral_Wasserstein"]
].mean()

print("\n" + "=" * 75)
print("OVERALL SPECTRAL ERROR COMPARISON")
print("=" * 75)

print(summary.round(6))


# --------------------------------------------------
# 8. Identify best experiment
# --------------------------------------------------

best_experiment = summary[
    "Spectral_Wasserstein"
].idxmin()

print("\n" + "=" * 75)
print("BEST EXPERIMENT BASED ON SPECTRAL SIMILARITY")
print("=" * 75)

print(best_experiment)

print(
    "Average spectral Wasserstein distance:",
    round(summary.loc[best_experiment, "Spectral_Wasserstein"], 6)
)


# --------------------------------------------------
# 9. Save evaluation results
# --------------------------------------------------

OUTPUT_CSV = OUTPUT_DIR / "spectral_evaluation_results.csv"

df.to_csv(OUTPUT_CSV, index=False)

print("\nDetailed results saved to:")
print(OUTPUT_CSV)

print("\nSpectral evaluation completed successfully!")