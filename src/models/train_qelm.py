#!/usr/bin/env python3
"""
Train QELM (Quantum Extreme Learning Machine) for atmospheric gas detection.

This script trains a 12-qubit quantum reservoir to predict H2O, CH4, and CO2
abundances from exoplanet transmission spectra.

Performance Target: <6% MAE on gas abundances
"""

import numpy as np
import pandas as pd
import pennylane as qml
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import json
from pathlib import Path

print("=" * 80)
print("QUANTUM EXTREME LEARNING MACHINE (QELM) TRAINING")
print("12-Qubit Reservoir + NASA-Quality Spectral Data")
print("=" * 80)

# Configuration
N_QUBITS = 12
N_LAYERS = 3
TRAIN_DATA_PATH = 'data/spectra/train_spectra_realistic.csv'
TEST_DATA_PATH = 'data/spectra/test_spectra_realistic.csv'
MODEL_SAVE_PATH = 'models/qelm_model_improved.pkl'
RESULTS_SAVE_PATH = 'reports/qelm_results_improved.json'

# Ensure directories exist
Path(MODEL_SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(RESULTS_SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)

print(f"\n📂 Loading spectral data...")
print(f"   Train: {TRAIN_DATA_PATH}")
print(f"   Test: {TEST_DATA_PATH}")

try:
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    print(f"   ✓ Train: {len(train_df)} spectra")
    print(f"   ✓ Test: {len(test_df)} spectra")
except FileNotFoundError as e:
    print(f"\n❌ ERROR: Spectral data not found!")
    print(f"   Please run: python src/data/generate_spectra.py")
    exit(1)

# Quantum reservoir setup
dev = qml.device('default.qubit', wires=N_QUBITS)

@qml.qnode(dev)
def quantum_reservoir(inputs):
    """
    12-qubit quantum reservoir with 3-layer entanglement.

    Args:
        inputs: 12-dimensional vector (spectral patches)

    Returns:
        12 expectation values (Pauli-Z measurements)
    """
    # Data encoding
    for i in range(N_QUBITS):
        qml.RY(inputs[i], wires=i)

    # 3-layer entangling circuit
    for layer in range(N_LAYERS):
        # Linear chain
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[N_QUBITS - 1, 0])  # Ring closure

        # Single-qubit rotations
        for i in range(N_QUBITS):
            qml.RX(0.3 + 0.1 * layer, wires=i)
            qml.RZ(0.2 + 0.1 * layer, wires=i)

        # Skip connections (every 2 qubits)
        for i in range(0, N_QUBITS - 2, 2):
            qml.CNOT(wires=[i, i + 2])

    # Measurements
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


def preprocess_spectrum_advanced(spectrum_df):
    """
    Multi-resolution encoding: 6 global + 3 H2O + 3 CH4/CO2 patches.

    Args:
        spectrum_df: DataFrame with wavelengths and intensities columns

    Returns:
        12-dimensional feature vector per spectrum
    """
    features = []

    for idx, row in spectrum_df.iterrows():
        # Parse wavelengths and intensities
        wavelengths = np.array(eval(row['wavelengths']))
        intensities = np.array(eval(row['intensities']))

        patches = []

        # Global features (6 patches across full 0.6-2.8 μm range)
        for i in range(6):
            start_wl = 0.6 + i * (2.8 - 0.6) / 6
            end_wl = 0.6 + (i + 1) * (2.8 - 0.6) / 6
            mask = (wavelengths >= start_wl) & (wavelengths < end_wl)
            if mask.sum() > 0:
                patches.append(np.mean(intensities[mask]))
            else:
                patches.append(1.0)

        # H2O-focused regions (0.9, 1.4, 1.9 μm)
        h2o_wavelengths = [0.9, 1.4, 1.9]
        for wl in h2o_wavelengths:
            mask = (wavelengths >= wl - 0.1) & (wavelengths < wl + 0.1)
            if mask.sum() > 0:
                patches.append(np.mean(intensities[mask]))
            else:
                patches.append(1.0)

        # CH4/CO2 regions (2.0, 2.3, 2.7 μm)
        co2_ch4_wavelengths = [2.0, 2.3, 2.7]
        for wl in co2_ch4_wavelengths:
            mask = (wavelengths >= wl - 0.1) & (wavelengths < wl + 0.1)
            if mask.sum() > 0:
                patches.append(np.mean(intensities[mask]))
            else:
                patches.append(1.0)

        # Normalize to [0, π] for quantum encoding
        patches = np.array(patches)
        patches_norm = (patches - patches.min()) / (patches.max() - patches.min() + 1e-9) * np.pi
        features.append(patches_norm)

    return np.array(features)


print(f"\n🔧 Preprocessing spectra with {N_QUBITS}-dimensional multi-resolution encoding...")
X_train = preprocess_spectrum_advanced(train_df)
X_test = preprocess_spectrum_advanced(test_df)

y_train = train_df[['H2O', 'CH4', 'CO2']].values
y_test = test_df[['H2O', 'CH4', 'CO2']].values

print(f"   X_train: {X_train.shape} ({X_train.shape[0]} spectra × {X_train.shape[1]} qubits)")
print(f"   y_train: {y_train.shape} (H2O, CH4, CO2)")

# Generate quantum features
print(f"\n⚛️  Generating quantum features ({N_QUBITS}-qubit reservoir)...")
print(f"   Expected time: ~5-8 minutes for {len(X_train) + len(X_test)} spectra...")

def generate_qfeatures(X, label="Train"):
    """Generate quantum features for dataset."""
    qfeatures = []
    for i, sample in enumerate(X):
        qf = quantum_reservoir(sample)
        qfeatures.append(qf)
        if (i + 1) % 100 == 0:
            print(f"   {label} set: {i+1}/{len(X)} samples...")
    return np.array(qfeatures)

qfeatures_train = generate_qfeatures(X_train, "Training")
qfeatures_test = generate_qfeatures(X_test, "Test")

print(f"\n✓ Quantum features: {qfeatures_train.shape}")

# Scale quantum features
scaler = StandardScaler()
qfeatures_train_scaled = scaler.fit_transform(qfeatures_train)
qfeatures_test_scaled = scaler.transform(qfeatures_test)

# Train classical output layer (Ridge regression for each gas)
print(f"\n🎓 Training classical output layer (Ridge regression)...")
models = {}
results = {}

for i, gas in enumerate(['H2O', 'CH4', 'CO2']):
    print(f"   Training {gas} regressor...")
    model = Ridge(alpha=1.0)
    model.fit(qfeatures_train_scaled, y_train[:, i])
    models[gas] = model

    # Evaluate
    y_pred_train = model.predict(qfeatures_train_scaled)
    y_pred_test = model.predict(qfeatures_test_scaled)

    train_mae = mean_absolute_error(y_train[:, i], y_pred_train)
    test_mae = mean_absolute_error(y_test[:, i], y_pred_test)
    test_r2 = r2_score(y_test[:, i], y_pred_test)

    # Detection accuracy (>10% threshold)
    threshold = 0.10
    true_positive = ((y_test[:, i] > threshold) & (y_pred_test > threshold)).sum()
    total_positive = (y_test[:, i] > threshold).sum()
    detection_acc = true_positive / total_positive if total_positive > 0 else 0.0

    results[gas] = {
        'mae': test_mae,
        'r2': test_r2,
        'detection_accuracy': detection_acc
    }

print(f"✓ Training complete")

# Print results
print(f"\n" + "=" * 80)
print("EVALUATION")
print("=" * 80)

print(f"\n🎯 TEST SET PERFORMANCE:")
avg_mae = np.mean([results[gas]['mae'] for gas in ['H2O', 'CH4', 'CO2']])

for gas in ['H2O', 'CH4', 'CO2']:
    mae = results[gas]['mae']
    r2 = results[gas]['r2']
    det_acc = results[gas]['detection_accuracy']
    print(f"   {gas}: MAE={mae:.4f} ({mae*100:.1f}%), R²={r2:.3f}, Detection={det_acc:.1%}")

print(f"\n   Average MAE: {avg_mae:.4f} ({avg_mae*100:.1f}%)")

# Save model
print(f"\n💾 Saving QELM model...")
model_data = {
    'models': models,
    'scaler': scaler,
    'n_qubits': N_QUBITS,
    'n_layers': N_LAYERS
}

with open(MODEL_SAVE_PATH, 'wb') as f:
    pickle.dump(model_data, f)
print(f"   ✓ Saved: {MODEL_SAVE_PATH}")

# Save results
with open(RESULTS_SAVE_PATH, 'w') as f:
    json.dump(results, f, indent=2)
print(f"   ✓ Saved: {RESULTS_SAVE_PATH}")

print(f"\n" + "=" * 80)
print("QELM TRAINING COMPLETE!")
print("=" * 80)
print(f"\n🎉 Average Test MAE: {avg_mae*100:.1f}%")
print(f"   {'✅' if avg_mae < 0.06 else '⚠️ '} Target: <6% MAE")
print(f"   Ready for fusion with transit classifier")
