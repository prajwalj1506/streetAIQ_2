import os
from huggingface_hub import HfApi, login

token = ""

try:
    print("Logging into Hugging Face...")
    login(token=token)
    api = HfApi(token=token)
    
    user_info = api.whoami()
    username = user_info["name"]
    print(f"Logged in as {username}")
    
    space_name = "garbage-ai-backend"
    repo_id = f"{username}/{space_name}"
    
    print(f"Creating Hugging Face Space: {repo_id}...")
    try:
        api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
        print("Space created or already exists!")
    except Exception as e:
        print(f"Error creating space: {e}")
        
    print("Uploading backend files to Hugging Face Space (this may take a few minutes for weights)...")
    
    # We only upload necessary files to avoid uploading __pycache__ or other local large irrelevant files
    local_dir = r"e:\vc code\.vscode\trash tracker\garbage_ai model"
    
    # Using allow_patterns to ignore __pycache__ and big video files if any
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=["__pycache__/*", "*.mp4", "*.avi", "captured_images/*", "road_data_split/*", ".gitignore"]
    )
    print("==================================================")
    print("SUCCESS! Files uploaded.")
    print(f"Your app will be live at: https://{username}-{space_name}.hf.space")
    print("==================================================")
except Exception as e:
    print(f"Deployment failed: {e}")
