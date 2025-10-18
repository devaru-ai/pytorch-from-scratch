# Setup and Data Generation
import torch

TRUE_WEIGHT = 2.0
TRUE_BIAS = 1.0

X = torch.rand(100, 1)
Y = TRUE_WEIGHT * X + TRUE_BIAS + torch.randn(100, 1) * 0.1
print(f"Data generated. X shape: {X.shape}, Y shape: {Y.shape}")

# Initialize Model Parameters
weight = torch.randn(1, 1, requires_grad=True)
bias = torch.randn(1, 1, requires_grad=True)

learning_rate = 0.01
num_epochs = 500

print(f"Initial W: {weight.item():.4f}, Initial B: {bias.item():.4f}")

# The Training Loop
for epoch in range(num_epochs):
  # 1. Forward Pass
  Y_pred = weight * X + bias

  # 2. Calculate loss
  loss = torch.mean((Y_pred - Y)**2)
  
  # 3. Zero gradients at start of each iteration
  if weight.grad is not None:
    weight.grad.zero_()
  if bias.grad is not None:
    bias.grad.zero_()
    
  # 4. Backward Pass
  loss.backward()
  
  # 5. Parameter update
  # We use torch.no_grad() because we don't want this updating step to be
  # part of the *next* iteration's computation graph.
  with torch.no_grad():
    weight -= learning_rate * weight.grad
    bias -= learning_rate * bias.grad
  if (epoch + 1) % 100 == 0:
    print(f"Epoch {epoch+1:4d}/{num_epochs} | Loss: {loss.item():.6f}")

print("\n--- Training Complete ---")
print(f"Learned Weight: {weight.item():.4f} (True: {TRUE_WEIGHT})")
print(f"Learned Bias: {bias.item():.4f} (True: {TRUE_BIAS})")
