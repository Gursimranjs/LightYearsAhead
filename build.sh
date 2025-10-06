#!/bin/bash
# Render build script - trains models if they don't exist

echo "Checking for model files..."

if [ ! -f "models/model.pkl" ] || [ ! -f "models/scalers.pkl" ]; then
    echo "Models not found. Training models..."
    python src/models/train.py
else
    echo "✓ Models already exist"
fi

if [ ! -f "models/qelm_model_improved.pkl" ]; then
    echo "QELM model not found. Training QELM..."
    python src/models/train_qelm.py
else
    echo "✓ QELM model already exists"
fi

echo "Build complete!"
