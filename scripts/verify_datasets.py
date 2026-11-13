from pathlib import Path
import numpy as np

# Project data directory
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

datasets = [
    "lorenz_windows.npy",
    "rossler_windows.npy",
    "duffing_windows.npy",
]

print("=" * 55)
print("GENERATIVE DYNAMICS - DATASET VERIFICATION")
print("=" * 55)

for filename in datasets:
    file_path = DATA_DIR / filename

    print(f"\nDataset: {filename}")

    if not file_path.exists():
        print("ERROR: File not found!")
        continue

    data = np.load(file_path)

    print("Shape:", data.shape)
    print("Data type:", data.dtype)
    print("Contains NaN:", np.isnan(data).any())
    print("Contains infinity:", np.isinf(data).any())
    print("Minimum:", data.min())
    print("Maximum:", data.max())

if (
    data.ndim == 3
    and data.shape[1] == 100
    and data.shape[2] in (2, 3)
    and np.isfinite(data).all()
):
    print("Status: PASSED")
else:
    print("Status: CHECK REQUIRED")

print("\nVerification completed.")
