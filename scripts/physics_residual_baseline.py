import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REAL_PATH = PROJECT_ROOT / "data" / "lorenz.csv"

EXPERIMENT_PATH = (
    PROJECT_ROOT / "outputs" / "synthetic_lorenz_experiment6.npy"
)

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

    # Midpoint states
    midpoint = (
        data[:, 1:, :] + data[:, :-1, :]
    ) / 2.0

    # Numerical derivatives
    numerical_derivative = (
        data[:, 1:, :] - data[:, :-1, :]
    ) / dt

    # Lorenz equation derivatives
    expected_derivative = lorenz_equations(midpoint)

    # Physics residual
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

# Load Experiment 6
synthetic_data = np.load(EXPERIMENT_PATH)

# Evaluate
for name, data in [
    ("Real Lorenz", real_data),
    ("Experiment 6", synthetic_data),
]:

    mae, rmse, relative = evaluate_physics(data, dt)

    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)

    print("MAE:", np.round(mae, 5))
    print("RMSE:", np.round(rmse, 5))
    print("Relative error:", np.round(relative, 5))

print("\nPhysics residual baseline evaluation completed.")