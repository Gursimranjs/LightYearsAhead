#!/bin/bash
# Render build script - downloads models from GitHub releases

echo "Checking for model files..."

if [ ! -f "models/model.pkl" ] || [ ! -f "models/scalers.pkl" ] || [ ! -f "models/qelm_model_improved.pkl" ]; then
    echo "Models not found. Downloading from GitHub releases..."
    python3 download_models.py

    if [ $? -eq 0 ]; then
        echo "✓ Models downloaded successfully"
    else
        echo "✗ Model download failed. Build cannot continue."
        exit 1
    fi
else
    echo "✓ All models already exist"
fi

echo "Build complete!"
