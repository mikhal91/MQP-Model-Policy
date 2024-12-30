import torch

# Check CUDA availability
print("CUDA Available:", torch.cuda.is_available())

# Perform a simple GPU computation
x = torch.tensor([1.0, 2.0, 3.0]).cuda()
y = torch.tensor([4.0, 5.0, 6.0]).cuda()
print("GPU Computation:", (x + y).cpu())
