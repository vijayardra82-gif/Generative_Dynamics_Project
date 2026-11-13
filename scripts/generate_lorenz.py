import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# Lorenz equations
def lorenz(t, state):
    x, y, z = state

    sigma = 10
    rho = 28
    beta = 8/3

    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z

    return [dx, dy, dz]

# Time range
t_span = (0, 50)

# Time points
t_eval = np.linspace(0, 50, 5000)

# Initial condition
initial_state = [1, 1, 1]

solution = solve_ivp(
    lorenz,
    t_span,
    initial_state,
    t_eval=t_eval
)

df = pd.DataFrame({
    "time": solution.t,
    "x": solution.y[0],
    "y": solution.y[1],
    "z": solution.y[2]
})

df.to_csv("data/lorenz.csv", index=False)

print("Lorenz dataset created")