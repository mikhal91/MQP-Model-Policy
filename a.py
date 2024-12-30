import torch

print("PyTorch version:", torch.__version__)
print("Is CUDA available?", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("CUDA device count:", torch.cuda.device_count())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA Device")
