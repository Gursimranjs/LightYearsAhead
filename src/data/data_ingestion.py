#!/usr/bin/env python3
"""
Smart Data Ingestion System for Exoplanet Analysis

Handles messy CSV uploads with:
- Flexible column name mapping
- Missing data detection and imputation
- Data quality validation
- Batch processing
- Clear error messages

Author: StarSifter Team
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings


class DataIngestionSystem:
    """
    Flexible data ingestion system that handles various CSV formats,
    missing data, and provides quality validation.
    """

    def __init__(self):
        """Initialize with required features and column aliases."""

        # Load required features for the model
        with open('models/feature_list.json', 'r') as f:
            self.required_features = json.load(f)

        # Column name aliases - maps common variations to standard names
        self.column_aliases = {
            # Period
            'period': ['period', 'orbital_period', 'period_days', 'koi_period'],

            # Depth
            'depth': ['depth', 'transit_depth', 'koi_depth'],

            # Duration
            'duration': ['duration', 'transit_duration', 'koi_duration'],

            # Planet radius
            'prad': ['prad', 'planet_radius', 'radius', 'koi_prad'],

            # Equilibrium temperature
            'teq': ['teq', 'eq_temp', 'equilibrium_temp', 'koi_teq'],

            # Stellar effective temperature
            'steff': ['steff', 'stellar_teff', 'teff', 'koi_steff'],

            # KOI score
            'koi_score': ['koi_score', 'score', 'disposition_score'],

            # False positive flags
            'fpflag_nt': ['fpflag_nt', 'fp_nt', 'not_transit_like'],
            'fpflag_ss': ['fpflag_ss', 'fp_ss', 'stellar_eclipse'],
            'fpflag_co': ['fpflag_co', 'fp_co', 'centroid_offset'],
            'fpflag_ec': ['fpflag_ec', 'fp_ec', 'ephemeris_match'],

            # SNR and other metrics
            'snr': ['snr', 'signal_to_noise', 'signal_noise_ratio'],
            'impact': ['impact', 'impact_parameter', 'b'],
            'ror': ['ror', 'radius_ratio', 'rp_rstar'],
            'insolation': ['insolation', 'insol', 'flux'],

            # Stellar parameters
            'stellar_radius': ['stellar_radius', 'star_radius', 'rstar', 'koi_srad'],
            'stellar_logg': ['stellar_logg', 'logg', 'surface_gravity', 'koi_slogg'],
            'stellar_metallicity': ['stellar_metallicity', 'metallicity', 'feh', 'koi_smet'],
            'stellar_mass': ['stellar_mass', 'star_mass', 'mstar', 'koi_smass'],

            # Orbital parameters
            'semi_major_axis': ['semi_major_axis', 'sma', 'a', 'koi_sma'],
            'inclination': ['inclination', 'inc', 'i', 'koi_incl'],
            'eccentricity': ['eccentricity', 'ecc', 'e', 'koi_eccen'],

            # Observation metrics
            'num_transits': ['num_transits', 'n_transits', 'transit_count'],
            'kepler_magnitude': ['kepler_magnitude', 'kepmag', 'kmag'],
            'observation_quarters': ['observation_quarters', 'quarters', 'num_quarters'],

            # Uncertainties
            'prad_uncertainty': ['prad_uncertainty', 'prad_err', 'koi_prad_err'],
            'depth_uncertainty': ['depth_uncertainty', 'depth_err', 'koi_depth_err'],
            'period_uncertainty': ['period_uncertainty', 'period_err', 'koi_period_err'],
        }

        # Core features that are critical for classification
        self.critical_features = [
            'period', 'depth', 'duration', 'snr', 'prad', 'teq', 'steff',
            'stellar_radius', 'stellar_mass', 'ror', 'insolation'
        ]

        # Load training data statistics for imputation
        self._load_training_statistics()

    def _load_training_statistics(self):
        """Load statistics from training data for intelligent imputation."""
        try:
            X_train = pd.read_csv('data/processed/X_train_final.csv')
            self.training_stats = {
                'mean': X_train.mean().to_dict(),
                'median': X_train.median().to_dict(),
                'std': X_train.std().to_dict(),
                'min': X_train.min().to_dict(),
                'max': X_train.max().to_dict()
            }
        except FileNotFoundError:
            warnings.warn("Training statistics not found. Using default imputation values.")
            self.training_stats = None

    def map_column_names(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Map user's column names to standard feature names.

        Args:
            df: Input DataFrame with potentially non-standard column names

        Returns:
            Tuple of (mapped DataFrame, mapping dictionary)
        """
        mapping = {}

        # First, try exact matches (case-insensitive)
        for std_name in self.required_features:
            for col in df.columns:
                if col.lower() == std_name.lower():
                    mapping[col] = std_name
                    break

        # Then try alias matching
        for std_name, aliases in self.column_aliases.items():
            if std_name in mapping.values():
                continue  # Already mapped

            for col in df.columns:
                col_lower = col.lower().strip()
                if col_lower in [a.lower() for a in aliases]:
                    mapping[col] = std_name
                    break

        # Rename columns
        df_mapped = df.rename(columns=mapping)

        return df_mapped, mapping

    def validate_data_quality(self, df: pd.DataFrame, row_idx: int = None) -> Dict:
        """
        Validate data quality and provide detailed feedback.

        Args:
            df: DataFrame with mapped column names (single row or full DataFrame)
            row_idx: Optional row index for batch processing

        Returns:
            Dictionary with validation results
        """
        result = {
            'is_valid': True,
            'quality_score': 0.0,  # 0-100%
            'available_features': [],
            'missing_features': [],
            'critical_missing': [],
            'warnings': [],
            'errors': [],
            'can_classify': False,
            'can_atmospheric': False
        }

        # Check which required features are present and have valid values
        for feature in self.required_features:
            if feature in df.columns:
                # Check if feature has valid non-null value
                if row_idx is not None:
                    value = df.iloc[row_idx][feature]
                else:
                    value = df[feature].iloc[0] if len(df) > 0 else None

                if pd.notna(value) and not np.isinf(value):
                    result['available_features'].append(feature)
                else:
                    result['missing_features'].append(feature)
            else:
                result['missing_features'].append(feature)

        # Check critical features
        for feature in self.critical_features:
            if feature in result['missing_features']:
                result['critical_missing'].append(feature)

        # Calculate quality score
        total_features = len(self.required_features)
        available_count = len(result['available_features'])
        result['quality_score'] = (available_count / total_features) * 100

        # Determine if we can classify
        critical_available = len([f for f in self.critical_features
                                 if f in result['available_features']])
        min_critical_needed = int(len(self.critical_features) * 0.7)  # Need 70% of critical features

        if critical_available >= min_critical_needed:
            result['can_classify'] = True
        else:
            result['is_valid'] = False
            result['errors'].append(
                f"Insufficient critical features: {critical_available}/{len(self.critical_features)} available. "
                f"Need at least {min_critical_needed}. Missing: {', '.join(result['critical_missing'])}"
            )

        # Add warnings for missing non-critical features
        if result['quality_score'] < 80:
            result['warnings'].append(
                f"Data quality is {result['quality_score']:.1f}%. "
                f"Missing {len(result['missing_features'])} features may reduce prediction accuracy."
            )

        # Check if data is good enough for classification
        if result['quality_score'] < 50:
            result['errors'].append(
                f"Data quality too low ({result['quality_score']:.1f}%). "
                "Please provide more transit features for reliable classification."
            )
            result['is_valid'] = False

        return result

    def validate_spectral_data(self, spectral_data: Dict) -> Dict:
        """
        Validate spectral data format and values.

        Args:
            spectral_data: Dictionary with 'wavelengths' and 'intensities'

        Returns:
            Validation result dictionary
        """
        result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }

        # Check structure
        if not isinstance(spectral_data, dict):
            result['is_valid'] = False
            result['errors'].append("Spectral data must be a dictionary with 'wavelengths' and 'intensities'")
            return result

        if 'wavelengths' not in spectral_data or 'intensities' not in spectral_data:
            result['is_valid'] = False
            result['errors'].append("Spectral data must contain 'wavelengths' and 'intensities' keys")
            return result

        wavelengths = np.array(spectral_data['wavelengths'])
        intensities = np.array(spectral_data['intensities'])

        # Check lengths match
        if len(wavelengths) != len(intensities):
            result['is_valid'] = False
            result['errors'].append(
                f"Wavelengths ({len(wavelengths)}) and intensities ({len(intensities)}) must have same length"
            )
            return result

        # Check wavelength range (QELM trained on 0.6-2.8 μm)
        if wavelengths.min() > 0.7 or wavelengths.max() < 2.6:
            result['warnings'].append(
                f"Wavelength range ({wavelengths.min():.2f}-{wavelengths.max():.2f} μm) is limited. "
                "Best results with 0.6-2.8 μm coverage (JWST/Hubble range)."
            )

        # Check intensity range (should be transmission spectrum, ~0.99-1.00)
        if intensities.min() < 0.95 or intensities.max() > 1.05:
            result['warnings'].append(
                f"Intensity range ({intensities.min():.3f}-{intensities.max():.3f}) unusual. "
                "Expected transmission spectrum values near 0.99-1.00."
            )

        # Check for sufficient data points
        if len(wavelengths) < 50:
            result['warnings'].append(
                f"Only {len(wavelengths)} spectral points. More points (>100) improve accuracy."
            )

        return result

    def impute_missing_features(self, df: pd.DataFrame, validation_result: Dict) -> pd.DataFrame:
        """
        Intelligently impute missing features using training data statistics.

        Args:
            df: DataFrame with mapped columns
            validation_result: Result from validate_data_quality

        Returns:
            DataFrame with imputed values
        """
        df_imputed = df.copy()

        # Add missing columns with imputed values
        for feature in validation_result['missing_features']:
            if self.training_stats:
                # Use median from training data (robust to outliers)
                imputed_value = self.training_stats['median'].get(feature, 0.0)
            else:
                # Fallback to zero (data is already normalized in training)
                imputed_value = 0.0

            df_imputed[feature] = imputed_value

        return df_imputed

    def process_csv(self, csv_path: str, target_name_col: str = None) -> Dict:
        """
        Process a CSV file with flexible format handling.

        Args:
            csv_path: Path to CSV file
            target_name_col: Column name containing target names (optional)

        Returns:
            Dictionary with processed results for each row
        """
        # Read CSV
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to read CSV: {str(e)}"
            }

        if len(df) == 0:
            return {
                'success': False,
                'error': "CSV file is empty"
            }

        # Map column names
        df_mapped, column_mapping = self.map_column_names(df)

        # Process each row
        results = []
        for idx in range(len(df)):
            row_result = self._process_single_row(df_mapped, idx, target_name_col)
            results.append(row_result)

        # Create summary
        valid_count = sum(1 for r in results if r['validation']['is_valid'])
        avg_quality = np.mean([r['validation']['quality_score'] for r in results])

        return {
            'success': True,
            'total_rows': len(df),
            'valid_rows': valid_count,
            'invalid_rows': len(df) - valid_count,
            'average_quality_score': avg_quality,
            'column_mapping': column_mapping,
            'results': results
        }

    def _process_single_row(self, df: pd.DataFrame, idx: int, target_name_col: str = None) -> Dict:
        """Process a single row from the DataFrame."""

        # Get target name if column specified
        target_name = None
        if target_name_col and target_name_col in df.columns:
            target_name = df.iloc[idx][target_name_col]
        else:
            target_name = f"Target_{idx + 1}"

        # Validate data quality
        validation = self.validate_data_quality(df, idx)

        # Extract transit features
        transit_features = {}
        for feature in self.required_features:
            if feature in df.columns:
                value = df.iloc[idx][feature]
                if pd.notna(value) and not np.isinf(value):
                    transit_features[feature] = float(value)

        # Impute missing features if needed
        if validation['can_classify'] and len(validation['missing_features']) > 0:
            # Create single-row DataFrame for imputation
            row_df = pd.DataFrame([transit_features])
            row_df_imputed = self.impute_missing_features(row_df, validation)
            transit_features = row_df_imputed.iloc[0].to_dict()

            validation['imputation_note'] = (
                f"Imputed {len(validation['missing_features'])} missing features using training data statistics. "
                "This may reduce prediction accuracy."
            )

        return {
            'target_name': target_name,
            'row_index': idx,
            'validation': validation,
            'transit_features': transit_features,
            'ready_for_prediction': validation['can_classify']
        }

    def process_single_target(self, data: Dict, target_name: str = "Unknown Target") -> Dict:
        """
        Process a single target from dictionary (for API use).

        Args:
            data: Dictionary with transit features (and optionally spectral_data)
            target_name: Name of the target

        Returns:
            Processed result dictionary
        """
        # Separate spectral data if present
        spectral_data = data.pop('spectral_data', None)
        spectral_validation = None

        if spectral_data:
            spectral_validation = self.validate_spectral_data(spectral_data)

        # Convert to DataFrame for processing
        df = pd.DataFrame([data])

        # Map column names
        df_mapped, column_mapping = self.map_column_names(df)

        # Validate
        validation = self.validate_data_quality(df_mapped, row_idx=0)

        # Get transit features
        transit_features = {}
        for feature in self.required_features:
            if feature in df_mapped.columns:
                value = df_mapped.iloc[0][feature]
                if pd.notna(value) and not np.isinf(value):
                    transit_features[feature] = float(value)

        # Impute missing if needed
        if validation['can_classify'] and len(validation['missing_features']) > 0:
            row_df = pd.DataFrame([transit_features])
            row_df_imputed = self.impute_missing_features(row_df, validation)
            transit_features = row_df_imputed.iloc[0].to_dict()

            validation['imputation_note'] = (
                f"Imputed {len(validation['missing_features'])} missing features. "
                "Prediction accuracy may be reduced."
            )

        return {
            'target_name': target_name,
            'validation': validation,
            'spectral_validation': spectral_validation,
            'transit_features': transit_features,
            'spectral_data': spectral_data,
            'ready_for_prediction': validation['can_classify']
        }


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("DATA INGESTION SYSTEM - TESTING")
    print("=" * 80)

    ingestion = DataIngestionSystem()

    # Test 1: Perfect data
    print("\n✅ Test 1: High-quality data")
    print("-" * 80)
    X_test = pd.read_csv('data/processed/X_test_final.csv')
    sample = X_test.iloc[0].to_dict()

    result = ingestion.process_single_target(sample, "HD 209458b")
    print(f"Target: {result['target_name']}")
    print(f"Quality Score: {result['validation']['quality_score']:.1f}%")
    print(f"Can Classify: {result['validation']['can_classify']}")
    print(f"Available Features: {len(result['validation']['available_features'])}/{len(ingestion.required_features)}")

    # Test 2: Missing some features
    print("\n\n⚠️  Test 2: Missing some features")
    print("-" * 80)
    partial_sample = {k: v for i, (k, v) in enumerate(sample.items()) if i < 30}

    result = ingestion.process_single_target(partial_sample, "Partial Data Target")
    print(f"Target: {result['target_name']}")
    print(f"Quality Score: {result['validation']['quality_score']:.1f}%")
    print(f"Can Classify: {result['validation']['can_classify']}")
    print(f"Missing Features: {len(result['validation']['missing_features'])}")
    if result['validation']['warnings']:
        print(f"Warnings: {result['validation']['warnings'][0]}")

    # Test 3: Very poor data
    print("\n\n❌ Test 3: Insufficient data")
    print("-" * 80)
    poor_sample = {
        'period': 3.5,
        'depth': 0.01,
        'snr': 12.5
    }

    result = ingestion.process_single_target(poor_sample, "Poor Data Target")
    print(f"Target: {result['target_name']}")
    print(f"Quality Score: {result['validation']['quality_score']:.1f}%")
    print(f"Can Classify: {result['validation']['can_classify']}")
    if result['validation']['errors']:
        print(f"Errors:")
        for error in result['validation']['errors']:
            print(f"  - {error}")

    print("\n" + "=" * 80)
    print("✅ DATA INGESTION SYSTEM READY")
    print("=" * 80)
