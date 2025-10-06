#!/usr/bin/env python3
"""
Download model files if they don't exist locally.
This script runs on Render startup to fetch models from cloud storage.
"""
import os
import requests
from pathlib import Path

def download_file(url, destination):
    """Download a file from URL to destination."""
    print(f"Downloading {destination}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"✓ Downloaded {destination}")

def main():
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Model URLs - replace with your actual URLs
    models = {
        "models/model.pkl": os.environ.get("MODEL_URL", ""),
        "models/scalers.pkl": os.environ.get("SCALERS_URL", ""),
        "models/qelm_model_improved.pkl": os.environ.get("QELM_URL", "")
    }

    for filepath, url in models.items():
        if not os.path.exists(filepath):
            if url:
                try:
                    download_file(url, filepath)
                except Exception as e:
                    print(f"✗ Failed to download {filepath}: {e}")
                    print("Skipping model download - models must be present locally")
            else:
                print(f"✗ No URL provided for {filepath}")
        else:
            print(f"✓ {filepath} already exists")

if __name__ == "__main__":
    main()
