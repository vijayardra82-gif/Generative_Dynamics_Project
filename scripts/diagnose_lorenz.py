from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

original_path = PROJECT_ROOT / "data" / "lorenz_windows.npy"
synthetic_path = PROJECT_ROOT / "outputs" / "synthetic_lorenz.npy"

original = np.load(original_path)
synthetic = np.load(synthetic_path)

print("=" * 50)
print("LORENZ DATA DIAGNOSTICS")
print("=" * 50)

print("\nDataset shapes")
print("Original:", original.shape)
print("Synthetic:", synthetic.shape)


def report_statistics(name, data):
    print(f"\n{name}")

    for i, variable in enumerate(["X", "Y", "Z"]):
        values = data[:, :, i]

        print(f"\nVariable {variable}")
        print(f"Mean: {values.mean():.4f}")
        print(f"Standard deviation: {values.std():.4f}")
        print(f"Minimum: {values.min():.4f}")
        print(f"Maximum: {values.max():.4f}")


def report_temporal_changes(name, data):
    # Differences between consecutive points within each window.
    differences = np.diff(data, axis=1)

    print(f"\n{name}: consecutive-step differences")

    for i, variable in enumerate(["X", "Y", "Z"]):
        values = np.abs(differences[:, :, i])

        print(
            f"{variable} | "
            f"Mean absolute change: {values.mean():.4f} | "
            f"95th percentile: {np.percentile(values, 95):.4f} | "
            f"Maximum: {values.max():.4f}"
        )


report_statistics("ORIGINAL DATA", original)
report_statistics("SYNTHETIC DATA", synthetic)

report_temporal_changes("ORIGINAL DATA", original)
report_temporal_changes("SYNTHETIC DATA", synthetic)

print("\nDiagnostics completed.")