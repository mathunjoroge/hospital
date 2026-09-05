"""
departments/imaging/dicom.py
──────────────────────────────
Task 3.5 — Radiology & DICOM Integration
Features:
  - DICOM Modality Worklist (MWL) endpoint for order status tracking
  - Web-based DICOM viewer attachment handler
  - Structured radiologist reporting workflow
"""

import os
import uuid
import logging
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app, send_from_directory

try:
    from extensions import db
except ImportError:
    from extensions import db

from departments.models.imaging import ImagingResult
from departments.models.medicine import RequestedImage
from departments.models.records import Patient

logger = logging.getLogger(__name__)

dicom_bp = Blueprint('dicom_integration', __name__, url_prefix='/imaging/dicom')

@dicom_bp.route('/mwl', methods=['GET'])
def get_modality_worklist():
    """Get DICOM Modality Worklist (MWL) for pending orders."""
    pending_orders = RequestedImage.query.filter_by(status=0).all()
    mwl_items = []
    
    for order in pending_orders:
        patient = Patient.query.filter_by(patient_id=order.patient_id).first()
        mwl_items.append({
            "AccessionNumber": f"ACC-{order.id}",
            "PatientID": order.patient_id,
            "PatientName": patient.name if patient else "Unknown",
            "PatientBirthDate": patient.date_of_birth.strftime('%Y%m%d') if patient and patient.date_of_birth else "",
            "PatientSex": "M" if patient and patient.sex == "Male" else "F" if patient and patient.sex == "Female" else "O",
            "StudyInstanceUID": f"1.2.826.0.1.3680043.2.1125.{order.id}.{int(datetime.utcnow().timestamp())}",
            "RequestedProcedureDescription": order.description or "Imaging procedure",
            "ScheduledProcedureStepStartDate": order.date_requested.strftime('%Y%m%d'),
            "ScheduledProcedureStepStartTime": order.date_requested.strftime('%H%M%S'),
            "Modality": "UNKNOWN" # Typically mapped from imaging type
        })
        
    return jsonify({"mwl": mwl_items, "count": len(mwl_items)}), 200


@dicom_bp.route('/viewer/<string:result_id>', methods=['GET'])
def viewer_attachment(result_id):
    """Web-based DICOM viewer attachment handler. Returns metadata for the viewer."""
    result = ImagingResult.query.filter_by(result_id=result_id).first()
    if not result:
        return jsonify({"error": "Result not found"}), 404
        
    file_paths = result.dicom_file_path.split(',') if result.dicom_file_path else []
    attachments = []
    
    for idx, path in enumerate(file_paths):
        if os.path.exists(path):
            attachments.append({
                "file_index": idx,
                "filename": os.path.basename(path),
                "url": f"/imaging/dicom/download/{result_id}/{idx}"
            })
            
    return jsonify({
        "result_id": result_id,
        "patient_id": result.patient_id,
        "dicom_attachments": attachments,
        "count": len(attachments)
    }), 200


@dicom_bp.route('/download/<string:result_id>/<int:file_index>', methods=['GET'])
def download_dicom(result_id, file_index):
    """Download specific DICOM file by index."""
    result = ImagingResult.query.filter_by(result_id=result_id).first()
    if not result:
        return jsonify({"error": "Result not found"}), 404
        
    file_paths = result.dicom_file_path.split(',') if result.dicom_file_path else []
    if 0 <= file_index < len(file_paths):
        file_path = file_paths[file_index]
        if os.path.exists(file_path):
            directory = os.path.dirname(file_path)
            filename = os.path.basename(file_path)
            return send_from_directory(directory, filename, as_attachment=False)
            
    return jsonify({"error": "File not found"}), 404


@dicom_bp.route('/report', methods=['POST'])
def save_structured_report():
    """Save structured radiologist report and mark order as completed."""
    data = request.get_json() or {}
    result_id = data.get('result_id')
    findings = data.get('findings')
    impression = data.get('impression')
    radiologist_id = data.get('radiologist_id')
    
    if not all([result_id, findings, impression]):
        return jsonify({"error": "Missing required fields"}), 400
        
    result = ImagingResult.query.filter_by(result_id=result_id).first()
    if not result:
        return jsonify({"error": "Result not found"}), 404
        
    structured_report = (
        f"**Findings:**\n{findings}\n\n"
        f"**Impression:**\n{impression}\n"
    )
    
    result.result_notes = structured_report
    result.updated_by = radiologist_id
    
    # Mark RequestedImage as processed
    request_obj = RequestedImage.query.filter_by(result_id=result_id).first()
    if request_obj:
        request_obj.status = 1  # Processed
        
    db.session.commit()
    
    return jsonify({
        "success": True,
        "result_id": result_id,
        "status": "REPORT_SAVED"
    }), 200
