#!/usr/bin/env python3
"""
Clean PDF Report Generator - Proper Layout
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for thread safety
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from pathlib import Path

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

class CleanPDFReport:
    """Generate clean, properly formatted PDF reports."""

    def __init__(self, output_dir='reports/generated'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, transit_features, spectral_data=None,
                target_name='Unknown', observer_notes=''):
        """Generate PDF report."""

        print(f"\n{'='*80}")
        print(f"Generating Report: {target_name}")
        print(f"{'='*80}\n")

        # Create directory
        report_id = f"{target_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        report_dir = self.output_dir / report_id
        report_dir.mkdir(parents=True, exist_ok=True)

        # Run analysis
        from src.api.predict import predict_exoplanet
        result = predict_exoplanet(transit_features, spectral_data)

        # Generate PDF
        pdf_path = report_dir / f"{target_name.replace(' ', '_')}_report.pdf"

        with PdfPages(pdf_path) as pdf:
            # Page 1: Summary
            self._create_summary_page(pdf, target_name, observer_notes, result,
                                     transit_features, spectral_data)

            # Page 2: Classification graphs
            self._create_classification_page(pdf, result)

            # Page 3: Spectrum (if available)
            if spectral_data:
                self._create_spectrum_page(pdf, spectral_data, result)

            # Page 4: Details
            self._create_details_page(pdf, result)

        print(f"✅ Report saved: {pdf_path}\n")
        return str(pdf_path)

    def _create_summary_page(self, pdf, target_name, notes, result,
                            transit_features, spectral_data):
        """Summary page with everything on one page."""

        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.95, 'EXOPLANET ANALYSIS REPORT',
                ha='center', fontsize=18, fontweight='bold')
        fig.text(0.5, 0.92, target_name,
                ha='center', fontsize=14, fontweight='bold')
        fig.text(0.5, 0.89, f"Date: {datetime.now().strftime('%B %d, %Y')}",
                ha='center', fontsize=11)

        # Horizontal line
        fig.add_artist(plt.Line2D([0.1, 0.9], [0.87, 0.87],
                                  transform=fig.transFigure, color='black', linewidth=1))

        y = 0.82

        # CLASSIFICATION
        fig.text(0.1, y, '1. CLASSIFICATION RESULTS',
                fontsize=12, fontweight='bold')
        y -= 0.025

        fig.text(0.12, y, f"Classification: {result['classification']}",
                fontsize=11, fontweight='bold')
        y -= 0.02
        fig.text(0.12, y, f"Confidence: {result['confidence']:.1%}",
                fontsize=11)
        y -= 0.025

        fig.text(0.12, y, "Probabilities:", fontsize=10, fontweight='bold')
        y -= 0.02
        for label, prob in result['probabilities'].items():
            fig.text(0.14, y, f"{label}: {prob:.1%}", fontsize=10)
            y -= 0.018

        y -= 0.025

        # ATMOSPHERIC
        if 'atmospheric_analysis' in result:
            fig.text(0.1, y, '2. ATMOSPHERIC COMPOSITION',
                    fontsize=12, fontweight='bold')
            y -= 0.025

            atmo = result['atmospheric_analysis']
            fig.text(0.12, y, f"H₂O: {atmo['H2O']:.1%}  |  CH₄: {atmo['CH4']:.1%}  |  CO₂: {atmo['CO2']:.1%}",
                    fontsize=11, family='monospace')
            y -= 0.02

            detected = atmo.get('detected_gases', [])
            if detected:
                fig.text(0.12, y, f"Detected: {', '.join(detected)}",
                        fontsize=11, fontweight='bold')
            else:
                fig.text(0.12, y, "No gases above 10% threshold",
                        fontsize=11, style='italic')
            y -= 0.025

        # DATA
        fig.text(0.1, y, '3. INPUT DATA', fontsize=12, fontweight='bold')
        y -= 0.025

        fig.text(0.12, y, f"Transit Features: {len(transit_features)} parameters",
                fontsize=10)
        y -= 0.018

        if spectral_data:
            wl = np.array(spectral_data['wavelengths'])
            fig.text(0.12, y, f"Spectrum: {len(wl)} points ({wl.min():.2f}-{wl.max():.2f} μm)",
                    fontsize=10)
        else:
            fig.text(0.12, y, "Spectrum: Not provided", fontsize=10)
        y -= 0.025

        # KEY FINDINGS
        fig.text(0.1, y, '4. KEY FINDINGS', fontsize=12, fontweight='bold')
        y -= 0.025

        findings = [f"Classified as {result['classification']} ({result['confidence']:.1%} confidence)"]

        if 'atmospheric_analysis' in result:
            detected = result['atmospheric_analysis'].get('detected_gases', [])
            if detected:
                findings.append(f"Atmospheric gases: {', '.join(detected)}")

        for finding in findings:
            fig.text(0.12, y, f"• {finding}", fontsize=10)
            y -= 0.02

        y -= 0.025

        # NOTES
        if notes:
            fig.text(0.1, y, 'OBSERVER NOTES', fontsize=12, fontweight='bold')
            y -= 0.025

            # Wrap notes
            words = notes.split()
            lines = []
            current_line = []
            for word in words:
                current_line.append(word)
                if len(' '.join(current_line)) > 80:
                    current_line.pop()
                    lines.append(' '.join(current_line))
                    current_line = [word]
            if current_line:
                lines.append(' '.join(current_line))

            for line in lines[:5]:  # Max 5 lines
                fig.text(0.12, y, line, fontsize=10)
                y -= 0.018

        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    def _create_classification_page(self, pdf, result):
        """Classification page with bar chart."""

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 11),
                                       gridspec_kw={'height_ratios': [1, 1.5]})

        # Title
        fig.suptitle('CLASSIFICATION ANALYSIS', fontsize=16, fontweight='bold', y=0.98)

        # Bar chart
        labels = list(result['probabilities'].keys())
        values = list(result['probabilities'].values())
        colors = ['#2c5f8d', '#d4a017', '#8b0000']

        bars = ax1.bar(labels, values, color=colors, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Probability', fontsize=12, fontweight='bold')
        ax1.set_ylim(0, 1.0)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')

        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, height,
                    f'{val:.1%}', ha='center', va='bottom',
                    fontsize=11, fontweight='bold')

        # Explanation
        ax2.axis('off')

        explanation = f"""INTERPRETATION:

Classification: {result['classification']}
Confidence: {result['confidence']:.1%}

The transit classifier analyzed the light curve and determined this is most
likely a {result['classification']}.

Model Performance:
  • Accuracy: 89.5% on 3,913 test samples
  • F1-Score: 88.9%
  • Training: 15,652 Kepler candidates

Confidence Guide:
  • >90%: High confidence, reliable result
  • 70-90%: Moderate, consider follow-up observations
  • <70%: Low confidence, uncertain classification

{result.get('reasoning', '')}
"""

        ax2.text(0.05, 0.95, explanation, fontsize=10, va='top', family='monospace')

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    def _create_spectrum_page(self, pdf, spectral_data, result):
        """Spectrum analysis page."""

        wavelengths = np.array(spectral_data['wavelengths'])
        intensities = np.array(spectral_data['intensities'])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 11),
                                       gridspec_kw={'height_ratios': [1.5, 1]})

        fig.suptitle('SPECTRAL ANALYSIS', fontsize=16, fontweight='bold', y=0.98)

        # Spectrum plot
        ax1.plot(wavelengths, intensities, 'k-', linewidth=1.5)

        # Mark gas regions
        regions = {
            'H₂O': [(0.9, 1.0), (1.35, 1.45), (1.85, 1.95)],
            'CH₄': [(2.25, 2.35)],
            'CO₂': [(2.65, 2.75)]
        }

        for gas, bands in regions.items():
            for i, (start, end) in enumerate(bands):
                if start >= wavelengths.min() and end <= wavelengths.max():
                    ax1.axvspan(start, end, alpha=0.2,
                               label=f'{gas}' if i == 0 else '')

        ax1.set_xlabel('Wavelength (μm)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Relative Intensity', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(loc='best', fontsize=10)

        # Gas abundances
        if 'atmospheric_analysis' in result:
            atmo = result['atmospheric_analysis']
            gases = ['H₂O', 'CH₄', 'CO₂']
            abundances = [atmo['H2O'], atmo['CH4'], atmo['CO2']]

            bars = ax2.bar(gases, abundances,
                          color=['#2c5f8d', '#5d8aa8', '#8b9dc3'],
                          edgecolor='black', linewidth=1.5)
            ax2.axhline(y=0.10, color='red', linestyle='--', linewidth=2,
                       label='Detection Threshold (10%)')
            ax2.set_ylabel('Volume Abundance', fontsize=12, fontweight='bold')
            ax2.grid(axis='y', alpha=0.3, linestyle='--')
            ax2.legend(loc='upper right')

            for bar, val in zip(bars, abundances):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2, height,
                        f'{val:.1%}', ha='center', va='bottom',
                        fontsize=11, fontweight='bold')

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    def _create_details_page(self, pdf, result):
        """Details and recommendations."""

        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.95, 'DETAILS & RECOMMENDATIONS',
                ha='center', fontsize=16, fontweight='bold')

        y = 0.88

        # Model info
        fig.text(0.1, y, 'MODEL INFORMATION', fontsize=12, fontweight='bold')
        y -= 0.025

        model_text = """Transit Classifier:
  Type: Stacking Ensemble (RF + XGBoost + LightGBM)
  Accuracy: 89.5%
  F1-Score: 88.9%
  Training: 15,652 Kepler exoplanet candidates
"""
        fig.text(0.12, y, model_text, fontsize=10, va='top', family='monospace')
        y -= 0.12

        if 'atmospheric_analysis' in result:
            atmo_text = """Atmospheric Analyzer:
  Type: Quantum ELM (12-qubit reservoir)
  Accuracy: ±3.5% MAE
  Detection Limit: 10% volume abundance
  Training: 1,600 JWST NIRSpec spectra
"""
            fig.text(0.12, y, atmo_text, fontsize=10, va='top', family='monospace')
            y -= 0.12

        # Recommendations
        fig.text(0.1, y, 'RECOMMENDATIONS', fontsize=12, fontweight='bold')
        y -= 0.025

        label = result['classification']
        conf = result['confidence']

        if label == 'CONFIRMED' and conf > 0.90:
            rec = """• High-priority target for detailed study
• JWST spectroscopy recommended for atmosphere
• Radial velocity for mass determination
• Phase curve observations for temperature mapping
"""
        elif label == 'CANDIDATE':
            rec = """• Additional transit observations recommended
• Spectroscopic confirmation needed
• Verify with independent photometric data
• Consider for follow-up target list
"""
        else:
            rec = """• Likely false positive - low priority
• Not recommended for further observations
• Consider removing from candidate list
"""

        fig.text(0.12, y, rec, fontsize=10, va='top')

        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    print("Clean PDF Report Generator loaded")
