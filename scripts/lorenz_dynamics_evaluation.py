import numpy as np
from pathlib import Path

# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

REAL_PATH = DATA_DIR / "lorenz.csv"

EXPERIMENTS = {
    "Experiment 4": OUTPUT_DIR / "synthetic_lorenz_experiment4_previous.npy",
    "Experiment 5": OUTPUT_DIR / "synthetic_lorenz_experiment5.npy",
    "Experiment 6": OUTPUT_DIR / "synthetic_lorenz_experiment6.npy",
    "Experiment 7": OUTPUT_DIR / "synthetic_lorenz_experiment7.npy",
}

SIGMA = 10.0
RHO = 28.0
BETA = 8.0 / 3.0


# --------------------------------------------------
# 2. Lorenz equations
# --------------------------------------------------

def lorenz_equations(data):

    x = data[..., 0]
    y = data[..., 1]
    z = data[..., 2]

    dx = SIGMA * (y - x)
    dy = x * (RHO - z) - y
    dz = x * y - BETA * z

    return np.stack([dx, dy, dz], axis=-1)


# --------------------------------------------------
# 3. Calculate dynamics residuals
# --------------------------------------------------

def evaluate_dynamics(data, dt):

    numerical_derivative = np.gradient(
        data,
        dt,
        axis=1,
        edge_order=2
    )

    expected_derivative = lorenz_equations(data)

    residual = numerical_derivative - expected_derivative

    mae = np.mean(np.abs(residual), axis=(0, 1))
    rmse = np.sqrt(np.mean(residual ** 2, axis=(0, 1)))

    expected_rms = np.sqrt(
        np.mean(expected_derivative ** 2, axis=(0, 1))
    )

    relative_error = rmse / (expected_rms + 1e-8)

    return mae, rmse, relative_error


# --------------------------------------------------
# 4. Load real data and calculate time interval
# --------------------------------------------------

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

print("Real dataset shape:", real_data.shape)
print("Time interval:", dt)


# Convert real trajectory into one 3D sequence
real_data = real_data[np.newaxis, :, :]

# --------------------------------------------------
# 5. Evaluate real Lorenz dynamics
# --------------------------------------------------

print("\n" + "=" * 65)
print("REAL LORENZ DYNAMICS CONSISTENCY")
print("=" * 65)

real_mae, real_rmse, real_relative = evaluate_dynamics(
    real_data,
    dt
)

print("MAE:", np.round(real_mae, 5))
print("RMSE:", np.round(real_rmse, 5))
print("Relative error:", np.round(real_relative, 5))


# --------------------------------------------------
# 6. Evaluate synthetic experiments
# --------------------------------------------------

for name, path in EXPERIMENTS.items():

    if not path.exists():
        print(f"\nMissing file: {path}")
        continue

    synthetic_data = np.load(path)

    if synthetic_data.ndim != 3 or synthetic_data.shape[2] != 3:
        print(f"\nInvalid shape for {name}: {synthetic_data.shape}")
        continue

    mae, rmse, relative = evaluate_dynamics(
        synthetic_data,
        dt
    )

    print("\n" + "=" * 65)
    print(name.upper())
    print("=" * 65)

    print("Dataset shape:", synthetic_data.shape)
    print("MAE:", np.round(mae, 5))
    print("RMSE:", np.round(rmse, 5))
    print("Relative error:", np.round(relative, 5))


print("\nDynamics evaluation completed!")