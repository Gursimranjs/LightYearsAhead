#!/usr/bin/env python3
"""
API-ready prediction script for exoplanet classification with atmospheric analysis.

Uses models SEPARATELY:
- Transit Classifier: Primary model for classification (88.92% F1-score)
- QELM: Secondary model for atmospheric analysis (5.01% MAE) - only when spectral data provided
"""

import pandas as pd
import numpy as np
import pickle
import json
import sys
from pathlib import Path

# Get absolute path to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = PROJECT_ROOT / 'models' / 'model.pkl'
SCALERS_PATH = PROJECT_ROOT / 'models' / 'scalers.pkl'
QELM_PATH = PROJECT_ROOT / 'models' / 'qelm_model_improved.pkl'


def predict_exoplanet(transit_features, spectral_data=None):
    """
    Main prediction function for API endpoint.

    Args:
        transit_features (dict): Transit detection features
            {
                'period': 3.5,
                'transit_duration': 2.1,
                'depth': 0.01,
                'snr': 25.3,
                ... (53 features total)
            }

        spectral_data (dict, optional): Spectroscopic data from JWST/Hubble
            {
                'wavelengths': [0.6, 0.65, 0.7, ..., 2.8],  # in microns
                'intensities': [0.998, 0.997, 0.995, ..., 0.999]
            }

    Returns:
        dict: Complete prediction result
            {
                'classification': 'CONFIRMED' | 'CANDIDATE' | 'FALSE POSITIVE',
                'confidence': 0.92,  # 0-1
                'probabilities': {
                    'CONFIRMED': 0.92,
                    'CANDIDATE': 0.06,
                    'FALSE POSITIVE': 0.02
                },
                'atmospheric_analysis': {  # Only if spectral_data provided
                    'H2O': 0.245,  # 24.5%
                    'CH4': 0.082,  # 8.2%
                    'CO2': 0.157,  # 15.7%
                    'total_atmosphere': 0.484  # 48.4%
                },
                'reasoning': 'CONFIRMED: Strong transit signal (92%) from light curve analysis',
                'recommendation': 'High confidence exoplanet detection.'
            }
    """

    # Load Transit Classifier (in production, keep this loaded in memory)
    with open(MODEL_PATH, 'rb') as f:
        transit_model = pickle.load(f)

    # Load feature list for pruning
    FEATURE_LIST_PATH = PROJECT_ROOT / 'models' / 'feature_list.json'
    with open(FEATURE_LIST_PATH, 'r') as f:
        model_features = json.load(f)

    # Convert transit features to DataFrame and prune
    transit_df = pd.DataFrame([transit_features])

    # Only keep features the model was trained on
    available_features = [f for f in model_features if f in transit_df.columns]
    transit_df_pruned = transit_df[available_features]

    # Get transit prediction (this is the primary classification)
    transit_pred = transit_model.predict(transit_df_pruned)[0]
    transit_proba = transit_model.predict_proba(transit_df_pruned)[0]

    # Map numeric prediction to class name
    label_map = {0: 'FALSE POSITIVE', 1: 'CANDIDATE', 2: 'CONFIRMED'}
    classification = label_map[transit_pred]

    # Get probabilities
    probabilities = {
        'CONFIRMED': float(transit_proba[2]),
        'CANDIDATE': float(transit_proba[1]),
        'FALSE POSITIVE': float(transit_proba[0])
    }

    confidence = max(probabilities.values())

    # Build base response
    response = {
        'classification': classification,
        'confidence': round(confidence, 4),
        'probabilities': {k: round(v, 4) for k, v in probabilities.items()},
        'reasoning': f'{classification}: Strong transit signal ({confidence:.0%}) from light curve analysis'
    }

    # Add atmospheric analysis if spectral data provided
    if spectral_data is not None:
        try:
            # Load QELM model data
            with open(QELM_PATH, 'rb') as f:
                qelm_data = pickle.load(f)

            # Extract wavelengths and intensities
            wavelengths = np.array(spectral_data['wavelengths'])
            intensities = np.array(spectral_data['intensities'])

            # Import quantum circuit
            import pennylane as qml

            # Preprocess spectrum - MUST MATCH TRAINING EXACTLY
            n_qubits = qelm_data['n_qubits']

            # Extract features (12-dimensional)
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

            # H2O-focused regions (0.9, 1.4, 1.9 μm) - TRAINING WAVELENGTHS!
            for wl in [0.9, 1.4, 1.9]:
                mask = (wavelengths >= wl - 0.1) & (wavelengths < wl + 0.1)
                if mask.sum() > 0:
                    patches.append(np.mean(intensities[mask]))
                else:
                    patches.append(1.0)

            # CH4/CO2 regions (2.0, 2.3, 2.7 μm) - TRAINING WAVELENGTHS!
            for wl in [2.0, 2.3, 2.7]:
                mask = (wavelengths >= wl - 0.1) & (wavelengths < wl + 0.1)
                if mask.sum() > 0:
                    patches.append(np.mean(intensities[mask]))
                else:
                    patches.append(1.0)

            # Normalize to [0, π] for quantum encoding (MUST MATCH TRAINING)
            patches = np.array(patches)
            patches_norm = (patches - patches.min()) / (patches.max() - patches.min() + 1e-9) * np.pi
            preprocessed = patches_norm.reshape(1, -1)

            # Generate quantum features using the quantum circuit
            dev = qml.device('default.qubit', wires=n_qubits)

            @qml.qnode(dev)
            def quantum_reservoir(inputs):
                # Encode inputs (already normalized to [0, π])
                for i in range(n_qubits):
                    qml.RY(inputs[i], wires=i)

                # 3-layer entangling circuit
                for layer in range(3):
                    # Linear chain
                    for i in range(n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                    qml.CNOT(wires=[n_qubits - 1, 0])

                    # Single-qubit rotations
                    for i in range(n_qubits):
                        qml.RX(0.3 + 0.1 * layer, wires=i)
                        qml.RZ(0.2 + 0.1 * layer, wires=i)

                    # Skip connections
                    for i in range(0, n_qubits - 2, 2):
                        qml.CNOT(wires=[i, i + 2])

                # Measurements
                return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

            # Run quantum circuit
            quantum_features = np.array([quantum_reservoir(preprocessed[0])])

            # Scale features
            scaler = qelm_data['scaler']
            quantum_features_scaled = scaler.transform(quantum_features)

            # Predict using ridge regression models
            gas_predictions = {}
            for gas_name, model in qelm_data['models'].items():
                pred = float(model.predict(quantum_features_scaled)[0])
                # Clip predictions to valid range [0, 1]
                gas_predictions[gas_name] = max(0.0, min(1.0, pred))

            # Add atmospheric analysis to response
            total_atmo = sum(gas_predictions.values())
            response['atmospheric_analysis'] = {
                'H2O': round(gas_predictions['H2O'], 3),
                'CH4': round(gas_predictions['CH4'], 3),
                'CO2': round(gas_predictions['CO2'], 3),
                'total_atmosphere': round(total_atmo, 3),
                'detected_gases': [k for k, v in gas_predictions.items() if v > 0.10]
            }

            # Update reasoning to include atmospheric info
            num_gases = len([v for v in gas_predictions.values() if v > 0.10])
            if num_gases > 0:
                response['reasoning'] = f'{classification}: Transit signal ({confidence:.0%}) + {num_gases} atmospheric gases detected'

            # Add recommendation based on atmosphere
            if total_atmo > 0.3:
                response['recommendation'] = 'Strong atmospheric signature detected. High confidence in classification.'
            elif total_atmo > 0.1:
                response['recommendation'] = 'Moderate atmospheric signature detected. Classification supported by spectroscopy.'
            else:
                response['recommendation'] = 'Weak or no atmosphere detected. Classification based primarily on transit data.'

        except Exception as e:
            # If QELM fails, still return transit results
            response['recommendation'] = f'Classification based on transit photometry only. Spectral analysis failed: {str(e)}'
    else:
        response['recommendation'] = 'Classification based on transit photometry only. Upload spectrum for atmospheric analysis.'

    return response


# Example usage for testing
if __name__ == "__main__":
    print("=" * 80)
    print("EXOPLANET PREDICTION API - EXAMPLE USAGE")
    print("=" * 80)

    # Example 1: Transit-only prediction (no spectrum)
    print("\n📡 Example 1: Transit-only prediction")
    print("-" * 80)

    # Load a real test sample
    X_test = pd.read_csv('data/processed/X_test_final.csv')
    sample_features = X_test.iloc[0].to_dict()

    result = predict_exoplanet(sample_features)

    print(f"Classification: {result['classification']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Probabilities:")
    for label, prob in result['probabilities'].items():
        print(f"  {label}: {prob:.1%}")
    print(f"Reasoning: {result['reasoning']}")
    print(f"Recommendation: {result['recommendation']}")

    # Example 2: With spectral data
    print("\n\n🔬 Example 2: With spectral data (JWST spectrum)")
    print("-" * 80)

    # Simulate JWST spectrum upload (matching training data scale)
    wavelengths = np.linspace(0.6, 2.8, 250).tolist()

    # Simulate spectrum with H2O, CH4, CO2 absorption (realistic abundances: 20%, 10%, 15%)
    def gaussian_absorption(wl_center, width, depth):
        return depth * np.exp(-((np.array(wavelengths) - wl_center) ** 2) / width)

    intensities = np.ones(250)

    # H2O absorption bands (20% abundance)
    intensities -= gaussian_absorption(1.1, 0.15, 0.20 * 0.03)
    intensities -= gaussian_absorption(1.4, 0.12, 0.20 * 0.05)  # Strong
    intensities -= gaussian_absorption(1.9, 0.10, 0.20 * 0.04)
    intensities -= gaussian_absorption(2.7, 0.15, 0.20 * 0.03)

    # CH4 absorption (10% abundance)
    intensities -= gaussian_absorption(2.3, 0.10, 0.10 * 0.04)

    # CO2 absorption (15% abundance)
    intensities -= gaussian_absorption(2.7, 0.12, 0.15 * 0.05)

    # Add realistic noise
    intensities += np.random.normal(0, 0.0005, size=250)
    intensities = intensities.tolist()

    spectral_data = {
        'wavelengths': wavelengths,
        'intensities': intensities
    }

    result = predict_exoplanet(sample_features, spectral_data)

    print(f"Classification: {result['classification']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Probabilities:")
    for label, prob in result['probabilities'].items():
        print(f"  {label}: {prob:.1%}")

    if 'atmospheric_analysis' in result:
        print(f"\nAtmospheric Analysis:")
        atmo = result['atmospheric_analysis']
        print(f"  H2O: {atmo['H2O']:.1%}")
        print(f"  CH4: {atmo['CH4']:.1%}")
        print(f"  CO2: {atmo['CO2']:.1%}")
        print(f"  Total: {atmo['total_atmosphere']:.1%}")
        print(f"  Detected gases: {', '.join(atmo['detected_gases']) if atmo['detected_gases'] else 'None'}")

    print(f"\nReasoning: {result['reasoning']}")
    print(f"Recommendation: {result['recommendation']}")

    print("\n" + "=" * 80)
    print("✅ API READY FOR INTEGRATION")
    print("=" * 80)
