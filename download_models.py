#!/usr/bin/env python3
"""
Download model files if they don't exist locally.
This script runs on Render startup to fetch models from GitHub releases.
"""
import os
import requests
import tarfile
from pathlib import Path

def download_and_extract_models():
    """Download models archive from GitHub releases and extract."""
    models_dir = Path("models")

    # Check if models already exist
    required_files = [
        models_dir / "model.pkl",
        models_dir / "scalers.pkl",
        models_dir / "qelm_model_improved.pkl"
    ]

    if all(f.exists() for f in required_files):
        print("✓ All model files already exist")
        return True

    # Download from GitHub releases
    release_url = "https://github.com/Gursimranjs/LightYearsAhead/releases/download/v1.0/models.tar.gz"
    archive_path = "models.tar.gz"

    print(f"Downloading models from {release_url}...")
    try:
        response = requests.get(release_url, stream=True, timeout=300)
        response.raise_for_status()

        with open(archive_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"✓ Downloaded models archive ({os.path.getsize(archive_path) / 1024 / 1024:.1f} MB)")

        # Extract archive
        print("Extracting models...")
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall()

        print("✓ Models extracted successfully")

        # Cleanup
        os.remove(archive_path)

        # Verify extraction
        if all(f.exists() for f in required_files):
            print("✓ All model files ready")
            return True
        else:
            print("✗ Some model files missing after extraction")
            return False

    except Exception as e:
        print(f"✗ Failed to download models: {e}")
        print("Models must be trained locally or uploaded manually")
        return False

if __name__ == "__main__":
    download_and_extract_models()
