import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

def duffing(t, state):
    x, y = state

    delta = 0.2
    alpha = -1
    beta = 1
    gamma = 0.3
    omega = 1.2

    dx = y
    dy = -delta*y - alpha*x - beta*(x**3) + gamma*np.cos(omega*t)

    return [dx, dy]

t_span = (0, 100)
t_eval = np.linspace(0, 100, 5000)

initial_state = [0.1, 0.1]

solution = solve_ivp(
    duffing,
    t_span,
    initial_state,
    t_eval=t_eval
)

df = pd.DataFrame({
    "time": solution.t,
    "x": solution.y[0],
    "y": solution.y[1]
})

df.to_csv("data/duffing.csv", index=False)

print("Duffing dataset created successfully!")