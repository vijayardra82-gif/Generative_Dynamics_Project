import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# Rossler equations
def rossler(t, state):
    x, y, z = state

    a = 0.2
    b = 0.2
    c = 5.7

    dx = -y - z
    dy = x + a * y
    dz = b + z * (x - c)

    return [dx, dy, dz]

# Time range
t_span = (0, 100)

# Time points
t_eval = np.linspace(0, 100, 5000)

# Initial condition
initial_state = [1, 1, 1]

solution = solve_ivp(
    rossler,
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

df.to_csv("data/rossler.csv", index=False)

print("Rossler dataset created successfully!")