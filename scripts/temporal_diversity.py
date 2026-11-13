import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"

datasets = {
    "Real Lorenz": ROOT / "data" / "lorenz_windows.npy",
    "Experiment 6": OUTPUT_DIR / "synthetic_lorenz_experiment6.npy",
    "Experiment 8": OUTPUT_DIR / "synthetic_lorenz_experiment8.npy",
    "Experiment 9": OUTPUT_DIR / "synthetic_lorenz_experiment9.npy",
}

feature_names = ["X", "Y", "Z"]

results = {}

print("TEMPORAL DIVERSITY ANALYSIS")
print("=" * 60)

for name, path in datasets.items():

    data = np.load(path)

    # Standard deviation across sequences at each time step
    diversity = np.std(data, axis=0)

    results[name] = diversity

    print(f"\n{name}")
    print("Initial diversity:", diversity[0])
    print("Middle diversity:", diversity[49])
    print("Final diversity:", diversity[-1])
    print("Average diversity:", np.mean(diversity, axis=0))


# --------------------------------------------------
# Plot diversity across time
# --------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

for feature in range(3):

    ax = axes[feature]

    for name, diversity in results.items():

        ax.plot(
            np.arange(diversity.shape[0]),
            diversity[:, feature],
            label=name,
            linewidth=1.5
        )

    ax.set_title(f"{feature_names[feature]} Diversity")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Standard Deviation")
    ax.legend()
    ax.grid(alpha=0.3)


fig.suptitle(
    "Temporal Diversity: Real Lorenz vs Synthetic Experiments",
    fontsize=15
)

fig.tight_layout()

output_path = OUTPUT_DIR / "temporal_diversity_comparison.png"

fig.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)

print("\nTemporal diversity analysis completed!")
print("Figure saved to:", output_path)