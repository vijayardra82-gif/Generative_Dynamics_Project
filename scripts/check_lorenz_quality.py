from pathlib import Path

import numpy as np


# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# File locations
ORIGINAL_FILE = BASE_DIR / "data" / "lorenz_windows.npy"
SYNTHETIC_FILE = BASE_DIR / "outputs" / "synthetic_lorenz.npy"


def load_data(file_path):
    """Load a dataset and check its basic validity."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    data = np.load(file_path)

    if data.ndim != 3:
        raise ValueError(
            f"Expected 3D data, received shape {data.shape}"
        )

    if not np.isfinite(data).all():
        raise ValueError(
            f"Dataset contains NaN or infinite values: {file_path}"
        )

    return data.astype(np.float64)


def describe_data(name, data):
    """Display basic statistics for each variable."""
    print(f"\n{'=' * 45}")
    print(name)
    print("=" * 45)

    print("Shape:", data.shape)
    print("NaN values:", np.isnan(data).any())
    print("Infinite values:", np.isinf(data).any())

    for i, variable in enumerate(["X", "Y", "Z"]):
        values = data[:, :, i].reshape(-1)

        print(f"\nVariable {variable}")
        print(f"Mean: {np.mean(values):.4f}")
        print(f"Standard deviation: {np.std(values):.4f}")
        print(f"Minimum: {np.min(values):.4f}")
        print(f"Maximum: {np.max(values):.4f}")


def compare_statistics(original, synthetic):
    """Compare means and standard deviations."""
    print(f"\n{'=' * 45}")
    print("STATISTICAL COMPARISON")
    print("=" * 45)

    for i, variable in enumerate(["X", "Y", "Z"]):
        original_values = original[:, :, i].reshape(-1)
        synthetic_values = synthetic[:, :, i].reshape(-1)

        original_mean = np.mean(original_values)
        synthetic_mean = np.mean(synthetic_values)

        original_std = np.std(original_values)
        synthetic_std = np.std(synthetic_values)

        print(f"\nVariable {variable}")

        print(
            f"Absolute mean difference: "
            f"{abs(original_mean - synthetic_mean):.4f}"
        )

        print(
            f"Absolute standard deviation difference: "
            f"{abs(original_std - synthetic_std):.4f}"
        )


def main():
    original = load_data(ORIGINAL_FILE)
    synthetic = load_data(SYNTHETIC_FILE)

    describe_data("ORIGINAL LORENZ DATA", original)
    describe_data("SYNTHETIC LORENZ DATA", synthetic)

    compare_statistics(original, synthetic)


if __name__ == "__main__":
    main()