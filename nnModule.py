import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# 1. Data Generation
X_np , Y_np = make_moons(n_samples=1000, noise=0.3, random_state=42)
# Convert NP arrays to Torch Tensors
X = torch.tensor(X_np, dtype = torch.float32)
Y = torch.tensor(Y_np, dtype = torch.float32).unsqueeze(1) # Needs shape (1000, 1)

print(f"Dataset shape: X={X.shape}, Y={Y.shape}")

# 2. Define the Model
class SimpleClassifier(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(SimpleClassifier, self).__init__()
    self.net = nn.Sequential(
      nn.Linear(input_size, hidden_size),
      nn.ReLU(),
      nn.Linear(hidden_size, hidden_size),
      nn.ReLU(),
      nn.Linear(hidden_size, output_size)
      # NOTE: We do NOT include the sigmoid function here.
      # It's mathematically included in the specialized loss function.
    )
  def forward(self, x):
    return self.net(x)

# Instantiate the model
input_size = X.shape[1] # 2 features (x-coordinate, y-coordinate)
hidden_size = 32
output_size = 1
model = SimpleClassifier(input_size, hidden_size, output_size)
print("Model created using nn.Module.")

# 3. Loss function and Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 500

# 4. Training the Model
print("Starting training...")
for epoch in range(num_epochs):
  # 1. Forward Pass
  logits = model(X)
  loss = criterion(logits, Y)
  
  # 2. Backward Pass and Optimization
  # Zero the gradients
  optimizer.zero_grad()
  loss.backward()

  # Update the parameters
  optimizer.step()
  
  if (epoch + 1) % 100 == 0:
    predictions = (torch.sigmoid(logits)>0.5).float()
    accuracy = (predictions == Y).float().mean().item()
    print(f"Epoch {epoch+1:4d}/{num_epochs} | Loss: {loss.item():.4f} | Accuracy: {accuracy:.4f}")

print("Training finished!")

