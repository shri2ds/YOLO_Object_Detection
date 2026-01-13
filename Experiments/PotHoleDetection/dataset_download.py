import kaggle
import os

dataset_name = "samuelayman/path-holes"
download_path = "data/"
os.makedirs(download_path, exist_ok=True)
print(f"Downloading {dataset_name}...")

# Authenticate automatically using the ~/.kaggle/kaggle.json file
kaggle.api.authenticate()

# Download and Unzip automatically
kaggle.api.dataset_download_files(
    dataset_name,
    path=download_path,
    unzip=True
)

print("✅ Download complete.")