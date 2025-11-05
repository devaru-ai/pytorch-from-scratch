import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_TRAIN_SAMPLES = 1000
N_CPU_CORES = 4
L2_PENALTY = 0.005
MAX_GRAD_NORM = 1.0
NUM_EPOCHS = 20

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

def custom_activation(z):
  return 0.5 * (torch.sin(z)+1)

dataset = MixedDataDataset(X_large, Y_small, N_TRAIN_SAMPLES)
data_loader = DataLoader(
  dataset=dataset,
  batch_size=64,
  shuffle=True,
  num_workers=N_CPU_CORES - 1 if N_CPU_CORES > 1 else 0
)
print(f"Data ready. Workers: {data_loader.num_workers}, Device: {DEVICE}")

class DeepFreezingClassifier(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super().__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, hidden_size)
    self.fc4 = nn.Linear(hidden_size, output_size)
  def forward(self, x):
    x = custom_activation(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = custom_activation(self.fc3(x))
    return self.fc4(x)

model = DeepFreezingClassifier(2, 64, 1).to(DEVICE)

initial_fc1_weight = model.fc1.weight.data.clone()
for name, param in model.named_parameters():
  if name.startswith('fc1'):
    param.requires_grad = False

criterion = nn.BCEWithLogitsLoss()

# Add L2 Regularization (weight_decay) to Optimizer
optimizer = optim.Adam(
  # Only pass parameters that require gradient tracking (i.e., unfrozen layers)
  filter(lambda p: p.requires_grad, model.parameters()),
  lr=0.001,
  weight_decay=L2_PENALTY
)

print("Start Training...")
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
    nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
    optimizer.step()
    epoch_loss += loss.item()

  if epoch % 5 == 0 or epoch == NUM_EPOCHS:
    model.eval()
    with torch.no_grad():
      total_logits = model(dataset.features.to(DEVICE))
      predictions = (torch.sigmoid(total_logits)>0.5).float()
      accuracy = (predictions.cpu() == dataset.labels).float().mean().item()
      current_fc1_weight = model.fc1.weight.data.clone()
      freeze_status = "Frozen" if torch.equal(initial_fc1_weight, current_fc1_weight) else "FAILED to Freeze"
      print(f"Epoch {epoch:2d}/{NUM_EPOCHS} | Loss: {epoch_loss/len(data_loader):.4f} | Accuracy: {accuracy:.4f} | FC1 Status: {freeze_status}")

print("\nTraining complete!")


















