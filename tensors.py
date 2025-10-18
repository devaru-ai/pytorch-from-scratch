# pip install torch torchvision torchaudio

# 1. Create Tensors of Different Shapes and Data Types

import torch

# 1D Tensor (3,)
tensor_1d = torch.tensor([1.0, 2.0, 3.0])
print(f"1D Tensor:\n{tensor_1d}\nShape: {tensor_1d.shape}\nData Type: {tensor_1d.dtype}\n")

# 2D Tensor (Matrix) (2, 3)
tensor_2d = torch.zeros(2, 3)
print(f"2D Tensor (zeros):\n{tensor_2d}\nShape: {tensor_2d.shape}\n")

# 3D Tensor (2, 2, 4)
tensor_3d = torch.rand(2, 2, 4)

int_tensor = torch.ones(2, 2, dtype = torch.int16)
print(f"Integer Tensor:\n{int_tensor}\nData Type: {int_tensor.dtype}\n")

float64_tensor = torch.tensor([1.1, 2.2], dtype = torch.float64)
print(f"Float64 Tensor:\n{float64_tensor}\nData Type: {float64_tensor.dtype}\n")

# 2. Perform Element-Wise and Matrix Multiplication

A = torch.tensor([[1, 2],
                 [3, 4]])
B = torch.tensor([[2, 4],
                  [6,5]])
C_elem_wise = A * B
print(f"Tensor A:\n{A}")
print(f"Tensor B:\n{B}")
print(f"Element-Wise (A * B):\n{C_elem_wise}\n")

C_mat_mul = A @ B
print(f"Matrix Multiplication (A @ B) - Shape (2x2):\n{C_mat_mul}")

# 3. Move Tensors Between CPU and GPU

# Check if a GPU is available and set the device
if torch.cuda.is_available():
  device = torch.device("cuda")
  print(f"CUDA is available! Using device: {device}")
else:
  device = torch.device("cpu")
  print("CUDA not available. Using device: cpu")

# Fallback logic
cpu_device = torch.device("cpu") 

cpu_tensor = torch.randn(5,3)
print(f"\nOriginal Tensor Device: {cpu_tensor.device}")

if device.type == "cuda":
  gpu_tensor = cpu_tensor.to(device)
  print(f"Tensor moved to GPU Device: {gpu_tensor.device}")
  back_to_cpu_tensor = gpu_tensor.to(device)
  print(f"Tensor moved back to CPU Device: {back_to_cpu_tensor.device}")

if device.type == "cuda":
  direct_gpu_tensor = torch.ones(2, 2, device=device)
  print(f"Tensor created directly on GPU: {direct_gpu_tensor.device}")

