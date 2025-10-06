#!/usr/bin/env python3
"""
FUSION CLASSIFIER: Transit Detection + Atmospheric Validation
Combines 88.92% F1 transit classifier with QELM gas detection for improved accuracy
"""

import numpy as np
import pandas as pd
import pickle
import ast
import pennylane as qml
from pathlib import Path

class FusionClassifier:
    """
    Fuses transit classification with atmospheric gas detection.

    Strategy:
      1. Use transit classifier as primary detector (88.92% F1)
      2. For borderline cases, use QELM gas detection to correct errors
      3. Apply decision rules to override misclassifications
    """

    def __init__(self, transit_model_path='models/model.pkl',
                 qelm_model_path='models/qelm_model.pkl'):
        """Load both models."""
        print("Loading fusion models...")

        # Load transit classifier
        with open(transit_model_path, 'rb') as f:
            self.transit_model = pickle.load(f)
        print(f"  ✓ Transit classifier loaded: {transit_model_path}")

        # Load QELM
        with open(qelm_model_path, 'rb') as f:
            qelm_data = pickle.load(f)
            self.qelm_models = qelm_data['models']
            self.qelm_scaler = qelm_data['scaler']
            self.n_qubits = qelm_data['n_qubits']
        print(f"  ✓ QELM model loaded: {qelm_model_path}")

        # Setup quantum device
        self.dev = qml.device('default.qubit', wires=self.n_qubits)

        # Create qnode with architecture matching training
        # Support both 8-qubit (old) and 12-qubit (improved) models
        @qml.qnode(self.dev)
        def _qres(inputs):
            """Quantum reservoir circuit (auto-detects 8 or 12 qubit)."""
            n_qubits = self.n_qubits

            # Data encoding
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)

            # Choose architecture based on n_qubits
            if n_qubits == 12:
                # Improved 12-qubit architecture (3 layers)
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
            else:
                # Original 8-qubit architecture (2 layers)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])

                for i in range(n_qubits):
                    qml.RX(0.5, wires=i)
                    qml.RZ(0.3, wires=i)

                for i in range(0, n_qubits - 1, 2):
                    qml.CNOT(wires=[i, i + 1])

            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self._quantum_reservoir = _qres

    def quantum_reservoir(self, inputs):
        """Same reservoir as training."""
        return self._quantum_reservoir(inputs)

    def preprocess_spectrum(self, wavelengths, intensities):
        """
        Convert spectrum to patches (8 or 12 dimensional based on n_qubits).

        8-qubit: Simple binning
        12-qubit: Multi-resolution (6 global + 3 H2O + 3 CH4/CO2)
        """
        if self.n_qubits == 12:
            # Multi-resolution encoding (matches train_qelm_improved.py)
            patches = []

            # Global features (6 bins across full range)
            for i in range(6):
                start_wl = 0.6 + i * (2.8 - 0.6) / 6
                end_wl = 0.6 + (i + 1) * (2.8 - 0.6) / 6
                mask = (wavelengths >= start_wl) & (wavelengths < end_wl)
                if mask.sum() > 0:
                    patches.append(np.mean(intensities[mask]))
                else:
                    patches.append(1.0)

            # H2O-focused region (0.9, 1.4, 1.9 μm)
            h2o_wl = [0.9, 1.4, 1.9]
            for wl in h2o_wl:
                mask = (wavelengths >= wl - 0.1) & (wavelengths < wl + 0.1)
                if mask.sum() > 0:
                    patches.append(np.mean(intensities[mask]))
                else:
                    patches.append(1.0)

            # CH4/CO2 region (2.0, 2.3, 2.7 μm)
            co2_ch4_wl = [2.0, 2.3, 2.7]
            for wl in co2_ch4_wl:
                mask = (wavelengths >= wl - 0.1) & (wavelengths < wl + 0.1)
                if mask.sum() > 0:
                    patches.append(np.mean(intensities[mask]))
                else:
                    patches.append(1.0)

        else:
            # Simple binning for 8-qubit
            n_bins = 8
            bin_size = len(wavelengths) // n_bins
            patches = []
            for i in range(n_bins):
                start = i * bin_size
                end = start + bin_size if i < n_bins - 1 else len(wavelengths)
                patches.append(np.mean(intensities[start:end]))

        # Normalize to [0, π]
        patches = np.array(patches)
        patches = (patches - patches.min()) / (patches.max() - patches.min() + 1e-9) * np.pi
        return patches

    def predict_gases(self, spectral_data):
        """
        Predict gas abundances using QELM.

        Args:
            spectral_data: DataFrame with columns 'wavelengths', 'intensities'
                          (can be strings or actual arrays)

        Returns:
            dict: {'H2O': float, 'CH4': float, 'CO2': float}
        """
        # Parse spectral data (handle both string and array formats)
        wl_data = spectral_data['wavelengths'].iloc[0]
        int_data = spectral_data['intensities'].iloc[0]

        # Convert to arrays (handle string or list/array inputs)
        if isinstance(wl_data, str):
            wavelengths = np.array(ast.literal_eval(wl_data))
            intensities = np.array(ast.literal_eval(int_data))
        else:
            wavelengths = np.array(wl_data)
            intensities = np.array(int_data)

        # Preprocess
        patches = self.preprocess_spectrum(wavelengths, intensities)

        # Generate quantum features
        qfeatures = self.quantum_reservoir(patches)
        qfeatures = np.array([qfeatures])

        # Scale
        qfeatures_scaled = self.qelm_scaler.transform(qfeatures)

        # Predict each gas
        gas_abundances = {}
        for gas in ['H2O', 'CH4', 'CO2']:
            abundance = self.qelm_models[gas].predict(qfeatures_scaled)[0]
            gas_abundances[gas] = max(0, abundance)  # Clip to non-negative

        return gas_abundances

    def fusion_decision(self, transit_proba, gas_abundances):
        """
        Apply fusion rules to correct misclassifications.

        Args:
            transit_proba: [P(CONF), P(CAND), P(FP)] from transit classifier
            gas_abundances: {'H2O': float, 'CH4': float, 'CO2': float}

        Returns:
            final_class: 0=CONFIRMED, 1=CANDIDATE, 2=FALSE POSITIVE
            reasoning: str explaining decision
        """
        p_conf, p_cand, p_fp = transit_proba

        # Gas detection indicators (more stringent thresholds)
        has_atmosphere = any(v > 0.10 for v in gas_abundances.values())  # Raised from 0.05 to 0.10
        very_strong_atmosphere = sum(gas_abundances.values()) > 0.40  # Raised from 0.2 to 0.4
        flat_spectrum = all(v < 0.05 for v in gas_abundances.values())  # Raised from 0.03 to 0.05
        num_gases = sum(1 for v in gas_abundances.values() if v > 0.10)

        # CONSERVATIVE FUSION RULES
        # Goal: Only correct CLEAR errors, don't over-intervene

        # Rule 1: HIGH confidence CONFIRMED + very strong atmosphere = reinforce
        if p_conf > 0.80 and very_strong_atmosphere:
            return 0, f"CONFIRMED: Strong transit ({p_conf:.1%}) + {num_gases} gases → High confidence"

        # Rule 2: HIGH confidence CONFIRMED + flat spectrum = likely FALSE POSITIVE
        # (This catches eclipsing binaries misclassified as planets)
        if p_conf > 0.80 and flat_spectrum:
            return 2, f"FALSE POSITIVE: Transit ({p_conf:.1%}) but NO atmosphere → Eclipsing binary (CORRECTED)"

        # Rule 3: HIGH confidence FALSE POSITIVE + very strong atmosphere = RESCUE
        # (This catches real planets with bad transit fits)
        if p_fp > 0.70 and very_strong_atmosphere:
            return 0, f"CONFIRMED: Clear atmosphere despite poor transit → Real planet (RESCUED)"

        # Rule 4: ONLY upgrade CANDIDATES with VERY strong atmospheres + low FP probability
        if p_cand > 0.60 and p_fp < 0.20 and very_strong_atmosphere:
            return 0, f"CONFIRMED: Strong CANDIDATE + {num_gases} gases → Upgraded"

        # Rule 5: ONLY downgrade CANDIDATES with flat spectra + moderate FP probability
        if p_cand > 0.50 and p_fp > 0.30 and flat_spectrum:
            return 2, f"FALSE POSITIVE: Weak CANDIDATE + no atmosphere → Downgraded"

        # Default: TRUST TRANSIT CLASSIFIER (most cases!)
        predicted_class = np.argmax(transit_proba)
        if predicted_class == 0:
            return 0, f"CONFIRMED: Transit classifier ({p_conf:.1%})"
        elif predicted_class == 1:
            return 1, f"CANDIDATE: Transit classifier ({p_cand:.1%})"
        else:
            return 2, f"FALSE POSITIVE: Transit classifier ({p_fp:.1%})"

    def predict(self, transit_data, spectral_data=None):
        """
        Make final prediction using fusion.

        Args:
            transit_data: DataFrame with 53+ transit features
            spectral_data: Optional DataFrame with 'wavelengths', 'intensities'

        Returns:
            dict with classification, probabilities, and gas abundances
        """
        # Prune low-value features (Phase 4 pruning)
        low_value_features = [
            'observation_quarters', 'fit_quality_score', 'eccentricity',
            'high_eccentricity_flag', 'rocky_planet_score', 'impact',
            'prad_uncertainty_ratio', 'geometry_score'
        ]
        features_to_drop = [f for f in low_value_features if f in transit_data.columns]
        transit_data_pruned = transit_data.drop(columns=features_to_drop)

        # Step 1: Transit classification
        transit_pred = self.transit_model.predict(transit_data_pruned)[0]
        transit_proba = self.transit_model.predict_proba(transit_data_pruned)[0]

        result = {
            'transit_prediction': ['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'][transit_pred],
            'transit_probabilities': {
                'CONFIRMED': float(transit_proba[0]),
                'CANDIDATE': float(transit_proba[1]),
                'FALSE POSITIVE': float(transit_proba[2])
            },
            'gas_abundances': None,
            'final_prediction': None,
            'reasoning': None
        }

        # Step 2: If spectral data available, use QELM
        if spectral_data is not None:
            gas_abundances = self.predict_gases(spectral_data)
            result['gas_abundances'] = gas_abundances

            # Step 3: Fusion decision
            final_class, reasoning = self.fusion_decision(transit_proba, gas_abundances)
            result['final_prediction'] = ['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'][final_class]
            result['reasoning'] = reasoning
        else:
            # No spectral data - trust transit classifier
            result['final_prediction'] = result['transit_prediction']
            result['reasoning'] = f"Transit only (no spectral data)"

        return result


# Command-line demo
if __name__ == "__main__":
    import sys

    print("=" * 80)
    print("FUSION CLASSIFIER DEMO")
    print("=" * 80)

    # Load fusion classifier
    fusion = FusionClassifier()

    # Load test data
    print("\nLoading test transit data...")
    X_test_transit = pd.read_csv('data/processed/X_test_final.csv')

    print("Loading test spectral data...")
    test_spectra = pd.read_csv('data/spectra/test_spectra.csv')

    # Test on first sample
    sample_idx = 0
    transit_sample = X_test_transit.iloc[sample_idx:sample_idx+1]
    spectral_sample = test_spectra.iloc[sample_idx:sample_idx+1]

    print(f"\n🔬 Testing sample {sample_idx}...")
    result = fusion.predict(transit_sample, spectral_sample)

    print("\nRESULTS:")
    print(f"  Transit Prediction: {result['transit_prediction']}")
    print(f"  Transit Probabilities: {result['transit_probabilities']}")
    if result['gas_abundances']:
        print(f"  Gas Abundances: {result['gas_abundances']}")
    print(f"  Final Prediction: {result['final_prediction']}")
    print(f"  Reasoning: {result['reasoning']}")

    print("\n" + "=" * 80)
