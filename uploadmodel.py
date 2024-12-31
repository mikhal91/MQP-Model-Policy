from huggingface_hub import HfApi, HfFolder
import os

# Configurations
token = "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"  # Huggingface token
repo_id = "miikhal/Llama-3.1-8B-python-mqp"  # Huggingface model repo
local_model_path = "model"  # Path to the model folder

# Verify token
if not token:
    raise ValueError("Please provide a valid Hugging Face token.")
HfFolder.save_token(token)

# Ensure the folder exists
if not os.path.exists(local_model_path):
    raise FileNotFoundError(f"Local model folder '{local_model_path}' not found.")

# Initialize the API
api = HfApi()

# Upload the model folder to the specified repository
try:
    print(f"Uploading '{local_model_path}' to '{repo_id}'...")
    api.upload_folder(
        folder_path=local_model_path,
        repo_id=repo_id,
        token=token,
        path_in_repo=".",  # Root of the repository
    )
    print(f"Upload successful! View it at: https://huggingface.co/{repo_id}")
except Exception as e:
    print(f"Failed to upload: {e}")
