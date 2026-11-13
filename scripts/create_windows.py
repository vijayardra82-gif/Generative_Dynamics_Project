import pandas as pd
import numpy as np

datasets = [
    ("lorenz.csv", ["x","y","z"], "lorenz_windows.npy"),
    ("rossler.csv", ["x","y","z"], "rossler_windows.npy"),
    ("duffing.csv", ["x","y"], "duffing_windows.npy")
]

window_size = 100

for csv_file, columns, output_file in datasets:

    df = pd.read_csv(f"data/{csv_file}")
    data = df[columns].values

    windows = []

    for i in range(len(data) - window_size):
        windows.append(data[i:i+window_size])

    windows = np.array(windows)

    np.save(f"data/{output_file}", windows)

    print(f"{output_file} saved. Shape = {windows.shape}")