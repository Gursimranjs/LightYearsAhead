#!/usr/bin/env python3
"""
Flask Backend API for LightYears Ahead

Provides endpoints for:
- CSV upload and validation
- Batch prediction processing
- Individual PDF report downloads
- Results retrieval

Author: StarSifter Team
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import os
import tempfile
import json
from pathlib import Path
from datetime import datetime

from src.pipeline.complete_pipeline import CompletePipeline

app = Flask(__name__)
CORS(app)  # Enable CORS for Streamlit frontend

# Initialize pipeline
pipeline = CompletePipeline()

# Store results temporarily (in production, use database)
RESULTS_STORE = {}


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'LightYears Ahead Backend',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/upload-csv', methods=['POST'])
def upload_csv():
    """
    Upload CSV file for processing.

    Returns validation info and processing status.
    """

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'File must be CSV format'}), 400

    # Save uploaded file temporarily
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)
    file.save(temp_path)

    try:
        # Process CSV
        result = pipeline.process_csv_file(
            csv_path=temp_path,
            target_name_col=request.form.get('target_name_col'),
            generate_reports=True
        )

        # Generate unique session ID
        session_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')

        # Store results
        RESULTS_STORE[session_id] = result

        # Format response
        response = {
            'session_id': session_id,
            'success': result['success'],
            'total_targets': result['total_targets'],
            'successful': result['successful_predictions'],
            'failed': result['failed_predictions'],
            'average_quality': result['average_quality'],
            'results': []
        }

        # Add individual results
        for pred in result['predictions']:
            result_item = {
                'target_name': pred['target_name'],
                'success': pred['success'],
                'row_index': result['predictions'].index(pred)
            }

            if pred['success']:
                result_item['classification'] = pred['prediction']['classification']
                result_item['confidence'] = pred['prediction']['confidence']
                result_item['probabilities'] = pred['prediction']['probabilities']
                result_item['quality_score'] = pred['validation']['quality_score']

                # Add report path if available
                if result['report_paths']:
                    idx = result['predictions'].index(pred)
                    if idx < len(result['report_paths']):
                        result_item['report_path'] = result['report_paths'][idx]
            else:
                result_item['error'] = pred.get('error', 'Unknown error')
                result_item['quality_score'] = pred['validation']['quality_score']

            response['results'].append(result_item)

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route('/results/<session_id>', methods=['GET'])
def get_results(session_id):
    """Get results for a specific session."""

    if session_id not in RESULTS_STORE:
        return jsonify({'error': 'Session not found'}), 404

    result = RESULTS_STORE[session_id]

    return jsonify({
        'session_id': session_id,
        'total_targets': result['total_targets'],
        'successful': result['successful_predictions'],
        'failed': result['failed_predictions'],
        'average_quality': result['average_quality'],
        'results': result['predictions']
    })


@app.route('/download-report/<session_id>/<int:row_index>', methods=['GET'])
def download_report(session_id, row_index):
    """Download PDF report for a specific target."""

    if session_id not in RESULTS_STORE:
        return jsonify({'error': 'Session not found'}), 404

    result = RESULTS_STORE[session_id]

    if row_index >= len(result['report_paths']):
        return jsonify({'error': 'Report not found'}), 404

    report_path = result['report_paths'][row_index]

    if not os.path.exists(report_path):
        return jsonify({'error': 'Report file not found'}), 404

    return send_file(
        report_path,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f"report_{result['predictions'][row_index]['target_name']}.pdf"
    )


@app.route('/download-all-reports/<session_id>', methods=['GET'])
def download_all_reports(session_id):
    """Download all reports as a ZIP file."""

    if session_id not in RESULTS_STORE:
        return jsonify({'error': 'Session not found'}), 404

    result = RESULTS_STORE[session_id]

    if not result['report_paths']:
        return jsonify({'error': 'No reports available'}), 404

    # Create ZIP file
    import zipfile
    temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')

    with zipfile.ZipFile(temp_zip.name, 'w') as zipf:
        for i, report_path in enumerate(result['report_paths']):
            if os.path.exists(report_path):
                target_name = result['predictions'][i]['target_name']
                zipf.write(report_path, f"{target_name}_report.pdf")

    return send_file(
        temp_zip.name,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f"exoplanet_reports_{session_id}.zip"
    )


@app.route('/validate-csv', methods=['POST'])
def validate_csv():
    """
    Validate CSV without processing (quick check).

    Returns column names and row count.
    """

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'File must be CSV format'}), 400

    try:
        # Read CSV
        df = pd.read_csv(file)

        return jsonify({
            'valid': True,
            'rows': len(df),
            'columns': list(df.columns),
            'has_target_names': any(col.lower() in ['name', 'target_name', 'planet_name', 'object_name']
                                   for col in df.columns)
        })

    except Exception as e:
        return jsonify({
            'valid': False,
            'error': str(e)
        }), 400


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5001))

    print("=" * 80)
    print("LIGHTYEARS AHEAD - BACKEND API")
    print("=" * 80)
    print("\nStarting Flask server...")
    print("API Endpoints:")
    print("  GET  /health")
    print("  POST /upload-csv")
    print("  POST /validate-csv")
    print("  GET  /results/<session_id>")
    print("  GET  /download-report/<session_id>/<row_index>")
    print("  GET  /download-all-reports/<session_id>")
    print(f"\nServer running on port: {port}")
    print("=" * 80)

    app.run(debug=False, host='0.0.0.0', port=port)
