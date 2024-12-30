import torch
import torchvision
import torchaudio

print("PyTorch Version:", torch.__version__)
print("Torchvision Version:", torchvision.__version__)
print("Torchaudio Version:", torchaudio.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
