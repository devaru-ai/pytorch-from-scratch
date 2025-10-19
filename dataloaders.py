import utils
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# Simulate Data
data = {
  'Feature_A' : np.random.rand(100),
  'Feature_B': np.random.rand(100) * 10,
  'Label': np.random.randint(0, 2, 100)
}
simulated_df = pd.DataFrame(data)

# Using Dataset
class CustomCSVDataset(Dataset):
  def __init__(self, dataframe):
    self.features = torch.tensor(dataframe[['Feature_A', 'Feature_B']].values, dtype = torch.float32)
    self.labels = torch.tensor(dataframe['Label'].values, dtype = torch.float32).unsqueeze(1)
    
  def __len__(self):
    return len(self.labels)
    
  def __getitem__(self, idx):
    # Fetch the feature tensor and the label tensor for the given index
    return self.features[idx], self.labels[idx]

my_dataset = CustomCSVDataset(simulated_df)
print(f"Total samples in Dataset: {len(my_dataset)}")


# Using DataLoader
batch_size = 16
data_loader = DataLoader(
  dataset = my_dataset,
  batch_size = batch_size,
  shuffle = True,
  num_workers = 0
)
print(f"DataLoader created with batch size: {batch_size}")

num_epochs = 1
total_batches = 0

print("\nStarting iteration over DataLoader...")
for epoch in range(num_epochs):
  for i, (batch_X, batch_Y) in enumerate(data_loader):
    total_batches += 1
    
    # Training Loop
    # 1. Forward Pass: Y_pred = model(batch_X)
    # 2. Calculate Loss: loss = criterion(Y_pred, batch_Y)
    # 3. Backward Pass: loss.backward()
    # 4. Optimization: optimizer.step()
    # Print information about the first few batches
    
    if i < 3:
      print(f"  Batch {i+1}: Features shape {batch_X.shape}, Labels shape {batch_Y.shape}")

print(f"Finished one epoch. Total batches yielded: {total_batches}")


