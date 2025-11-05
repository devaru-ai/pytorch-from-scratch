import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_moons
import numpy as np

# Setup Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Data pipeline
X_np, Y_np = make_moons(n_samples=1000, noise=0.1, random_state=42)


class MoonDataset(Dataset):
  def __init__(self, X_data, Y_data):
    self.features = torch.tensor(X_data, dtype=torch.float32)
    self.labels = torch.tensor(Y_data, dtype=torch.float32)
  def __len__(self):
    return len(self.labels)
  def __getitem__(self, idx):
    return self.features[idx], self.labels[idx]

dataset = MoonDataset(X_np, Y_np)

BATCH_SIZE = 32
data_loader = DataLoader(
  dataset=dataset,
  batch_size=BATCH_SIZE,
  shuffle=True,
  num_workers=0
)
print(f"Data ready. Total batches: {len(data_loader)}")

# Architecture
class SimpleClassifier(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(SimpleClassifier, self).__init__()
    self.layer_stack = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, output_size)
  )
  def forward(self, x):
    return self.layer_stack(x)

INPUT_SIZE = 2
OUTPUT_SIZE = 1

model = SimpleClassifier(INPUT_SIZE, 64, OUTPUT_SIZE).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
NUM_EPOCHS = 20

print("\n Starting Training Loop..")
for epoch in range(1, NUM_EPOCHS+1):
  model.train()
  epoch_loss = 0.0
  for batch_X, batch_Y in data_loader:
    batch_X = batch_X.to(DEVICE)
    batch_Y = batch_Y.to(DEVICE)
    
    optimizer.zero_grad()
    logits = model(batch_X)
    
    loss = criterion(logits, batch_Y)
    loss.backward()
    
    optimizer.step()
    epoch_loss += loss.item() * batch_X.size(0)
    
  avg_loss = epoch_loss / len(dataset)
  if epoch%5 == 0:
    with torch.no_grad():
      model.eval()
      total_logits = model(dataset.features.to(DEVICE))
      predictions = (torch.sigmoid(total_logits)>0.5).float()
      accuracy = (predictions.cpu()==dataset.labels).float().mean().item()
      print(f"Epoch {epoch:2d}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")

print("\n Training Complete")






























