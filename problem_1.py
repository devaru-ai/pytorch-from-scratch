import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_avaialable() else "cpu")
N_TRAIN_SAMPLES = 1000
N_CPU_CORES = 4

X_large = np.random.rand(1_000_000, 2)
Y_small = np.random.randint(0, 2, N_TRAIN_SAMPLES)

class MixedDataDataset(Dataset):
  def __init__(self, X_data_large, Y_data_small, num_samples):
    X_used = X_data_large[:num_samples]
    Y_used = Y_data_small
    self.features = torch.tensor(X_used, dtype=torch.float32)
    self.labels = torch.tensor(Y_used, dtype=torch.float32).unsqueeze(1)
  def __len__(self):
    return len(self.labels)
  def __getitem__(self, idx):
    return self.features[idx], self.labels[idx]

dataset = MixedDataDataset(X_large, Y_small, N_TRAIN_SAMPLES)
BATCH_SIZE = 64

data_loader = DataLoader(
  dataset = dataset,
  batch_size = BATCH_SIZE,
  shuffle = True,
  num_workers = N_CPU_CORES - 1 if N_CPU_CORES > 1 else 0
)

print(f"Dataset Size: {len(dataset)} samples.")
print(f"DataLoader workers set to: {data_loader.num_workers}")
print(f"Sample Batch Shape: {next(iter(data_loader))[0].shape}")
