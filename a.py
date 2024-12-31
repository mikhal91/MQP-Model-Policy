import torch
import triton
from huggingface_hub import HfApi
import os



print("PyTorch version:", torch.__version__)
print("t something version:", triton.__version__)
print("Is CUDA available?", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("CUDA device count:", torch.cuda.device_count())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA Device")




api = HfApi()
token = "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"
print(api.whoami(token=token))



repo_id = "miikhal/test_model"  # Replace with your Hugging Face username and desired repo name
token = "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"

# Create the repository
## api.create_repo(repo_id=repo_id, token=token, private=False, exist_ok=True)

print(os.getenv("HF_TOKEN"))