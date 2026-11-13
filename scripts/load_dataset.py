import numpy as np
import torch

data = np.load("data/lorenz_windows.npy")

tensor_data = torch.tensor(data, dtype=torch.float32)

print("Tensor Shape:", tensor_data.shape)