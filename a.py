import torch
print(torch.cuda.is_available())  # Should return True
print(torch.version.cuda)        # Should match your CUDA version
