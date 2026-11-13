import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REAL_PATH = PROJECT_ROOT / "data" / "lorenz.csv"

# Lorenz parameters
SIGMA = 10.0
RHO = 28.0
BETA = 8.0 / 3.0


def lorenz_equations(data):
    x = data[..., 0]
    y = data[..., 1]
    z = data[..., 2]

    dx = SIGMA * (y - x)
    dy = x * (RHO - z) - y
    dz = x * y - BETA * z

    return np.stack([dx, dy, dz], axis=-1)


def evaluate_physics(data, dt):

    midpoint = (
        data[:, 1:, :] + data[:, :-1, :]
    ) / 2.0

    numerical_derivative = (
        data[:, 1:, :] - data[:, :-1, :]
    ) / dt

    expected_derivative = lorenz_equations(midpoint)

    residual = numerical_derivative - expected_derivative

    mae = np.mean(np.abs(residual), axis=(0, 1))

    rmse = np.sqrt(
        np.mean(residual ** 2, axis=(0, 1))
    )

    expected_rms = np.sqrt(
        np.mean(expected_derivative ** 2, axis=(0, 1))
    )

    relative_error = rmse / (expected_rms + 1e-8)

    return mae, rmse, relative_error


# Load real dataset
real_data = np.genfromtxt(
    REAL_PATH,
    delimiter=",",
    skip_header=1,
    usecols=(1, 2, 3)
)

time_values = np.genfromtxt(
    REAL_PATH,
    delimiter=",",
    skip_header=1,
    usecols=0
)

dt = np.mean(np.diff(time_values))

real_data = real_data[np.newaxis, :, :]

# Experiment files
experiments = [
    ("Experiment 4", "synthetic_lorenz_experiment4_previous.npy"),
    ("Experiment 5", "synthetic_lorenz_experiment5.npy"),
    ("Experiment 6", "synthetic_lorenz_experiment6.npy"),
    ("Experiment 7", "synthetic_lorenz_experiment7.npy"),
    ("Experiment 8", "synthetic_lorenz_experiment8.npy"),
    ("Experiment 9", "synthetic_lorenz_experiment9.npy"),
]

print("Lorenz Physics Residual Comparison")
print("Time interval:", dt)

# Evaluate real data
mae, rmse, relative = evaluate_physics(real_data, dt)

print("\n" + "=" * 60)
print("Real Lorenz")
print("=" * 60)
print("MAE:", np.round(mae, 5))
print("RMSE:", np.round(rmse, 5))
print("Relative error:", np.round(relative, 5))

# Evaluate experiments
for name, filename in experiments:

    file_path = PROJECT_ROOT / "outputs" / filename

    if not file_path.exists():
        print(f"\nSkipping {name}: file not found.")
        continue

    synthetic_data = np.load(file_path)

    if synthetic_data.ndim != 3 or synthetic_data.shape[2] != 3:
        print(f"\nSkipping {name}: unexpected shape {synthetic_data.shape}")
        continue

    if not np.isfinite(synthetic_data).all():
        print(f"\nSkipping {name}: contains NaN or Inf.")
        continue

    mae, rmse, relative = evaluate_physics(synthetic_data, dt)

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("Shape:", synthetic_data.shape)
    print("MAE:", np.round(mae, 5))
    print("RMSE:", np.round(rmse, 5))
    print("Relative error:", np.round(relative, 5))
    print("Average relative error:", round(np.mean(relative), 5))

print("\nPhysics residual comparison completed.")
