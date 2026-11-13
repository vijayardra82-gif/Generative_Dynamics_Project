import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FILES = {
    "Real": ROOT / "data" / "lorenz_windows.npy",
    "Experiment 8": ROOT / "outputs" / "synthetic_lorenz_experiment8.npy",
    "Experiment 9": ROOT / "outputs" / "synthetic_lorenz_experiment9.npy",
}

for name, path in FILES.items():

    data = np.load(path)

    x = data[:, :, 0]

    positive_to_negative = np.sum(
        (x[:, :-1] >= 0) & (x[:, 1:] < 0)
    )

    negative_to_positive = np.sum(
        (x[:, :-1] < 0) & (x[:, 1:] >= 0)
    )

    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)

    print("Positive to negative:", positive_to_negative)
    print("Negative to positive:", negative_to_positive)

    print(
        "Average transitions per sequence:",
        (positive_to_negative + negative_to_positive) / len(data)
    )

    print(
        "Sequences with at least one transition:",
        np.sum(
            np.any(
                x[:, 1:] * x[:, :-1] < 0,
                axis=1
            )
        )
    )

print("\nLobe transition evaluation completed!")