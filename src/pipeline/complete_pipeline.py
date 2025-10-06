#!/usr/bin/env python3
"""
Complete Analysis Pipeline: CSV Upload → Validation → Prediction → Report

Integrates:
1. Data Ingestion (flexible CSV handling)
2. Prediction API (Transit + QELM)
3. Report Generation (professional PDF)

Author: StarSifter Team
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.data_ingestion import DataIngestionSystem
from src.api.predict import predict_exoplanet
from src.reports.clean_pdf_report import CleanPDFReport


class CompletePipeline:
    """
    End-to-end pipeline for exoplanet analysis.

    Handles everything from messy CSV upload to final PDF report.
    """

    def __init__(self):
        """Initialize pipeline components."""
        self.ingestion = DataIngestionSystem()
        self.report_generator = CleanPDFReport()

    def process_csv_file(
        self,
        csv_path: str,
        target_name_col: str = None,
        notes_col: str = None,
        spectral_wavelength_cols: List[str] = None,
        spectral_intensity_cols: List[str] = None,
        generate_reports: bool = True
    ) -> Dict:
        """
        Process a CSV file end-to-end.

        Args:
            csv_path: Path to CSV file with transit data
            target_name_col: Column containing target names (optional)
            notes_col: Column containing observer notes (optional)
            spectral_wavelength_cols: Columns for wavelengths (optional)
            spectral_intensity_cols: Columns for intensities (optional)
            generate_reports: Whether to generate PDF reports

        Returns:
            Dictionary with processing results
        """

        print("=" * 80)
        print(f"PROCESSING: {Path(csv_path).name}")
        print("=" * 80)

        # Step 1: Ingest and validate data
        print("\n📥 Step 1: Data Ingestion & Validation")
        print("-" * 80)

        ingestion_result = self.ingestion.process_csv(csv_path, target_name_col)

        if not ingestion_result['success']:
            print(f"❌ FAILED: {ingestion_result['error']}")
            return ingestion_result

        print(f"✅ Loaded {ingestion_result['total_rows']} rows")
        print(f"✅ Valid: {ingestion_result['valid_rows']}/{ingestion_result['total_rows']}")
        print(f"✅ Average quality: {ingestion_result['average_quality_score']:.1f}%")

        if ingestion_result['column_mapping']:
            print(f"✅ Mapped {len(ingestion_result['column_mapping'])} column names")

        # Step 2: Run predictions
        print("\n\n🔮 Step 2: Running Predictions")
        print("-" * 80)

        prediction_results = []

        for row_result in ingestion_result['results']:
            if not row_result['ready_for_prediction']:
                print(f"\n❌ {row_result['target_name']}: SKIPPED (data quality too low)")
                print(f"   Quality: {row_result['validation']['quality_score']:.1f}%")
                if row_result['validation']['errors']:
                    print(f"   Error: {row_result['validation']['errors'][0]}")
                prediction_results.append({
                    'target_name': row_result['target_name'],
                    'success': False,
                    'error': 'Data quality too low for prediction',
                    'validation': row_result['validation']
                })
                continue

            # Run prediction
            try:
                prediction = predict_exoplanet(
                    transit_features=row_result['transit_features'],
                    spectral_data=None  # TODO: Add spectral data support
                )

                print(f"\n✅ {row_result['target_name']}: {prediction['classification']}")
                print(f"   Confidence: {prediction['confidence']:.1%}")
                print(f"   Quality: {row_result['validation']['quality_score']:.1f}%")

                if row_result['validation']['warnings']:
                    print(f"   ⚠️  {row_result['validation']['warnings'][0][:60]}...")

                prediction_results.append({
                    'target_name': row_result['target_name'],
                    'success': True,
                    'prediction': prediction,
                    'validation': row_result['validation'],
                    'transit_features': row_result['transit_features']
                })

            except Exception as e:
                print(f"\n❌ {row_result['target_name']}: PREDICTION FAILED")
                print(f"   Error: {str(e)}")
                prediction_results.append({
                    'target_name': row_result['target_name'],
                    'success': False,
                    'error': str(e),
                    'validation': row_result['validation']
                })

        # Step 3: Generate reports
        report_paths = []
        if generate_reports:
            print("\n\n📄 Step 3: Generating Reports")
            print("-" * 80)

            for pred_result in prediction_results:
                if not pred_result['success']:
                    print(f"❌ {pred_result['target_name']}: SKIPPED (no prediction)")
                    continue

                try:
                    # Create observer notes from validation
                    notes_parts = []

                    # Data quality info
                    val = pred_result['validation']
                    notes_parts.append(
                        f"Data Quality: {val['quality_score']:.1f}% "
                        f"({len(val['available_features'])}/{len(self.ingestion.required_features)} features)"
                    )

                    if 'imputation_note' in val:
                        notes_parts.append(val['imputation_note'])

                    if val['warnings']:
                        notes_parts.append("⚠️  " + val['warnings'][0])

                    observer_notes = "\n\n".join(notes_parts)

                    # Generate report
                    report_path = self.report_generator.generate(
                        transit_features=pred_result['transit_features'],
                        spectral_data=None,
                        target_name=pred_result['target_name'],
                        observer_notes=observer_notes
                    )

                    report_paths.append(report_path)
                    print(f"✅ {pred_result['target_name']}: Report generated")
                    print(f"   → {report_path}")

                except Exception as e:
                    print(f"❌ {pred_result['target_name']}: REPORT FAILED")
                    print(f"   Error: {str(e)}")

        # Summary
        print("\n\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        successful = sum(1 for r in prediction_results if r['success'])
        failed = len(prediction_results) - successful

        print(f"Total targets: {len(prediction_results)}")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")

        if generate_reports:
            print(f"📄 Reports generated: {len(report_paths)}")

        # Classification breakdown
        if successful > 0:
            classifications = {}
            for r in prediction_results:
                if r['success']:
                    cls = r['prediction']['classification']
                    classifications[cls] = classifications.get(cls, 0) + 1

            print(f"\nClassifications:")
            for cls, count in classifications.items():
                print(f"  {cls}: {count}")

        return {
            'success': True,
            'total_targets': len(prediction_results),
            'successful_predictions': successful,
            'failed_predictions': failed,
            'average_quality': ingestion_result['average_quality_score'],
            'predictions': prediction_results,
            'report_paths': report_paths
        }

    def process_single_target(
        self,
        data: Dict,
        target_name: str = "Unknown Target",
        observer_notes: str = "",
        generate_report: bool = True
    ) -> Dict:
        """
        Process a single target (for API use).

        Args:
            data: Dictionary with transit features and optional spectral_data
            target_name: Name of the target
            observer_notes: Observation notes
            generate_report: Whether to generate PDF report

        Returns:
            Dictionary with processing results
        """

        print("=" * 80)
        print(f"ANALYZING: {target_name}")
        print("=" * 80)

        # Step 1: Validate data
        print("\n📥 Step 1: Data Validation")
        print("-" * 80)

        ingestion_result = self.ingestion.process_single_target(data, target_name)

        print(f"Quality Score: {ingestion_result['validation']['quality_score']:.1f}%")
        print(f"Available Features: {len(ingestion_result['validation']['available_features'])}/{len(self.ingestion.required_features)}")
        print(f"Can Classify: {'✅ Yes' if ingestion_result['ready_for_prediction'] else '❌ No'}")

        if ingestion_result['validation']['warnings']:
            for warning in ingestion_result['validation']['warnings']:
                print(f"⚠️  {warning}")

        if ingestion_result['validation']['errors']:
            for error in ingestion_result['validation']['errors']:
                print(f"❌ {error}")

        if not ingestion_result['ready_for_prediction']:
            return {
                'success': False,
                'error': 'Data quality insufficient for prediction',
                'validation': ingestion_result['validation']
            }

        # Step 2: Run prediction
        print("\n\n🔮 Step 2: Running Prediction")
        print("-" * 80)

        try:
            prediction = predict_exoplanet(
                transit_features=ingestion_result['transit_features'],
                spectral_data=ingestion_result.get('spectral_data')
            )

            print(f"Classification: {prediction['classification']}")
            print(f"Confidence: {prediction['confidence']:.1%}")
            print(f"Probabilities:")
            for label, prob in prediction['probabilities'].items():
                print(f"  {label}: {prob:.1%}")

            if 'atmospheric_analysis' in prediction:
                print(f"\nAtmospheric Analysis:")
                atmo = prediction['atmospheric_analysis']
                print(f"  H2O: {atmo['H2O']:.1%}")
                print(f"  CH4: {atmo['CH4']:.1%}")
                print(f"  CO2: {atmo['CO2']:.1%}")
                print(f"  Total: {atmo['total_atmosphere']:.1%}")

        except Exception as e:
            print(f"❌ Prediction failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'validation': ingestion_result['validation']
            }

        # Step 3: Generate report
        report_path = None
        if generate_report:
            print("\n\n📄 Step 3: Generating Report")
            print("-" * 80)

            try:
                # Combine observer notes with validation info
                full_notes = observer_notes
                if ingestion_result['validation']['warnings']:
                    full_notes += f"\n\nData Quality: {ingestion_result['validation']['quality_score']:.1f}%"
                    full_notes += "\n" + "\n".join(ingestion_result['validation']['warnings'])

                report_path = self.report_generator.generate(
                    transit_features=ingestion_result['transit_features'],
                    spectral_data=ingestion_result.get('spectral_data'),
                    target_name=target_name,
                    observer_notes=full_notes
                )

                print(f"✅ Report generated: {report_path}")

            except Exception as e:
                print(f"⚠️  Report generation failed: {str(e)}")

        print("\n" + "=" * 80)

        return {
            'success': True,
            'target_name': target_name,
            'validation': ingestion_result['validation'],
            'prediction': prediction,
            'report_path': report_path
        }


# Example usage
if __name__ == "__main__":
    pipeline = CompletePipeline()

    # Test 1: Process messy CSV
    print("\n\n")
    print("🧪 TEST 1: Process messy CSV with missing features")
    result1 = pipeline.process_csv_file(
        'test_data/messy_csv_missing.csv',
        target_name_col='target_name',
        generate_reports=False  # Disable for testing
    )

    # Test 2: Process single target
    print("\n\n")
    print("🧪 TEST 2: Process single high-quality target")
    X_test = pd.read_csv('data/processed/X_test_final.csv')
    sample = X_test.iloc[3414].to_dict()

    result2 = pipeline.process_single_target(
        data=sample,
        target_name="HD 189733b",
        observer_notes="High-quality Kepler data. Clear transit signal.",
        generate_report=False  # Disable for testing
    )

    print("\n" + "=" * 80)
    print("✅ PIPELINE TESTS COMPLETED")
    print("=" * 80)
