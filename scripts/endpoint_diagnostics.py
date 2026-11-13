import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"

datasets = {
    "Real Lorenz": ROOT / "data" / "lorenz_windows.npy",
    "Experiment 6": OUTPUT_DIR / "synthetic_lorenz_experiment6.npy",
    "Experiment 8": OUTPUT_DIR / "synthetic_lorenz_experiment8.npy",
    "Experiment 9": OUTPUT_DIR / "synthetic_lorenz_experiment9.npy",
}


def calculate_average_distance(points):
    distances = []

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distances.append(np.linalg.norm(points[i] - points[j]))

    return np.mean(distances) if distances else 0.0


for name, path in datasets.items():

    data = np.load(path)

    initial = data[:, 0, :]
    final = data[:, -1, :]

    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)

    print("\nInitial states:")
    print("Mean:", np.mean(initial, axis=0))
    print("Std:", np.std(initial, axis=0))
    print("Min:", np.min(initial, axis=0))
    print("Max:", np.max(initial, axis=0))

    print("\nFinal states:")
    print("Mean:", np.mean(final, axis=0))
    print("Std:", np.std(final, axis=0))
    print("Min:", np.min(final, axis=0))
    print("Max:", np.max(final, axis=0))

    print("\nAverage pairwise distance:")
    print("Initial:", calculate_average_distance(initial))
    print("Final:", calculate_average_distance(final))

print("\nEndpoint diagnostics completed!")