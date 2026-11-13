import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/lorenz.csv")

plt.figure(figsize=(10,5))
plt.plot(df["time"], df["x"])

plt.title("Lorenz System")
plt.xlabel("Time")
plt.ylabel("X Value")

plt.savefig("lorenz_plot.png")

print("Graph saved successfully")