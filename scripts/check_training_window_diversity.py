import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FILES = {
    "Real training windows": ROOT / "data" / "lorenz_windows.npy",
    "Experiment 8": ROOT / "outputs" / "synthetic_lorenz_experiment8.npy",
    "Experiment 9": ROOT / "outputs" / "synthetic_lorenz_experiment9.npy",
}


for name, path in FILES.items():

    data = np.load(path)

    # Extract the first state of every sequence.
    initial_states = data[:, 0, :]

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("Dataset shape:", data.shape)

    print("\nInitial state statistics:")

    print("Mean:", initial_states.mean(axis=0))
    print("Standard deviation:", initial_states.std(axis=0))
    print("Minimum:", initial_states.min(axis=0))
    print("Maximum:", initial_states.max(axis=0))

    print(
        "\nAverage distance from mean:",
        np.linalg.norm(
            initial_states - initial_states.mean(axis=0),
            axis=1
        ).mean()
    )

    print(
        "Unique initial states:",
        np.unique(initial_states, axis=0).shape[0]
    )

print("\nComparison completed!")