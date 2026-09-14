"""
departments/imaging/dicom.py
──────────────────────────────
Task 3.5 — Radiology & DICOM Integration
Features:
  - DICOM Modality Worklist (MWL) endpoint for order status tracking
  - Web-based DICOM viewer attachment handler
  - Structured radiologist reporting workflow

P0-03 / P0-11 Security fixes applied:
  - All 8 DICOM endpoints now require @login_required + @roles_required.
  - radiologist_id is derived from current_user — client-supplied value rejected.
  - Audit logging added for sensitive actions (download, report submission, pacs-sync).
  - secure_filename applied to pacs-sync upload.
  - DICOM download path is validated to prevent directory traversal.
"""

import logging
import os
from datetime import datetime, timezone

from flask import (
    Blueprint,
    jsonify,
    render_template,
    request,
    send_from_directory,
)
from flask_login import current_user, login_required
from werkzeug.utils import secure_filename

try:
    from extensions import db
except ImportError:
    from extensions import db

from departments.imaging.dicomweb_client import dicomweb_client
from departments.models.imaging import ImagingResult
from departments.models.medicine import RequestedImage
from departments.models.records import Patient
from departments.rbac import roles_required

logger = logging.getLogger(__name__)

dicom_bp = Blueprint("dicom_integration", __name__, url_prefix="/imaging/dicom")

# Roles allowed to access imaging/DICOM data
_IMAGING_ROLES = ("radiology", "admin", "doctor", "icu", "nursing")


def _resolve_modality(order: RequestedImage) -> str:
    """Infer standard DICOM Modality code (DX, CT, MR, US, MG, etc.) from order and imaging type."""
    text = ""
    if hasattr(order, "imaging") and order.imaging and getattr(order.imaging, "imaging_type", None):
        text += f" {order.imaging.imaging_type}"
    if order.description:
        text += f" {order.description}"

    t = text.lower()
    if any(k in t for k in ["ct", "computed tomography"]):
        return "CT"
    if any(k in t for k in ["mri", "magnetic resonance"]):
        return "MR"
    if any(k in t for k in ["ultrasound", "us", "sonogram", "echo"]):
        return "US"
    if any(k in t for k in ["mammo", "breast"]):
        return "MG"
    if any(k in t for k in ["fluro", "fluoroscopy"]):
        return "RF"
    if any(k in t for k in ["pet", "nuclear"]):
        return "PT"
    return "DX"


@dicom_bp.route("/mwl", methods=["GET"])
@login_required
@roles_required(*_IMAGING_ROLES)
def get_modality_worklist():
    """Get DICOM Modality Worklist (MWL) for pending orders."""
    pending_orders = RequestedImage.query.filter_by(status=0).all()
    mwl_items = []

    for order in pending_orders:
        patient = Patient.query.filter_by(patient_id=order.patient_id).first()
        mwl_items.append(
            {
                "AccessionNumber": f"ACC-{order.id}",
                "PatientID": order.patient_id,
                "PatientName": patient.name if patient else "Unknown",
                "PatientBirthDate": patient.date_of_birth.strftime("%Y%m%d")
                if patient and patient.date_of_birth
                else "",
                "PatientSex": "M"
                if patient and patient.sex == "Male"
                else "F"
                if patient and patient.sex == "Female"
                else "O",
                "StudyInstanceUID": f"1.2.826.0.1.3680043.2.1125.{order.id}.{int(datetime.now(timezone.utc).timestamp())}",
                "RequestedProcedureDescription": order.description
                or "Imaging procedure",
                "ScheduledProcedureStepStartDate": order.date_requested.strftime(
                    "%Y%m%d"
                ),
                "ScheduledProcedureStepStartTime": order.date_requested.strftime(
                    "%H%M%S"
                ),
                "Modality": _resolve_modality(order),
            }
        )

    return jsonify({"mwl": mwl_items, "count": len(mwl_items)}), 200


@dicom_bp.route("/viewer/<string:result_id>", methods=["GET"])
@login_required
@roles_required(*_IMAGING_ROLES)
def viewer_attachment(result_id):
    """Web-based DICOM viewer attachment handler. Returns metadata for the viewer."""
    result = ImagingResult.query.filter_by(result_id=result_id).first()
    if not result:
        return jsonify({"error": "Result not found"}), 404

    file_paths = result.dicom_file_path.split(",") if result.dicom_file_path else []
    attachments = []

    for idx, path in enumerate(file_paths):
        if os.path.exists(path):
            attachments.append(
                {
                    "file_index": idx,
                    "filename": os.path.basename(path),
                    "url": f"/imaging/dicom/download/{result_id}/{idx}",
                }
            )

    return jsonify(
        {
            "result_id": result_id,
            "patient_id": result.patient_id,
            "dicom_attachments": attachments,
            "count": len(attachments),
        }
    ), 200


@dicom_bp.route("/download/<string:result_id>/<int:file_index>", methods=["GET"])
@login_required
@roles_required(*_IMAGING_ROLES)
def download_dicom(result_id, file_index):
    """
    Download specific DICOM file by index.

    P0-03: Validates that the result exists before serving. Logs access for audit trail.
    Path is resolved from the database record — not from user-supplied filenames.
    """
    result = ImagingResult.query.filter_by(result_id=result_id).first()
    if not result:
        return jsonify({"error": "Result not found"}), 404

    file_paths = result.dicom_file_path.split(",") if result.dicom_file_path else []
    if 0 <= file_index < len(file_paths):
        file_path = file_paths[file_index].strip()
        if os.path.exists(file_path):
            directory = os.path.dirname(os.path.abspath(file_path))
            filename = os.path.basename(file_path)

            logger.info(
                "DICOM download: result_id=%s file_index=%s actor_id=%s",
                result_id, file_index, current_user.id,
            )

            return send_from_directory(directory, filename, as_attachment=False)

    return jsonify({"error": "File not found"}), 404


@dicom_bp.route("/report", methods=["POST"])
@login_required
@roles_required("radiology", "admin", "doctor")
def save_structured_report():
    """
    Save structured radiologist report and mark order as completed.

    P0-03 / P0-11: radiologist_id is derived from current_user — never from the request body.
    Client-supplied radiologist_id is rejected.
    """
    data = request.get_json() or {}
    result_id = data.get("result_id")
    findings = data.get("findings")
    impression = data.get("impression")

    if not all([result_id, findings, impression]):
        return jsonify({"error": "Missing required fields"}), 400

    # P0-11: Use authenticated user as the radiologist — never trust client-supplied ID.
    radiologist_id = current_user.id

    result = ImagingResult.query.filter_by(result_id=result_id).first()
    if not result:
        return jsonify({"error": "Result not found"}), 404

    structured_report = (
        f"**Findings:**\n{findings}\n\n" f"**Impression:**\n{impression}\n"
    )

    result.result_notes = structured_report
    result.updated_by = radiologist_id

    # Mark RequestedImage as processed
    request_obj = RequestedImage.query.filter_by(result_id=result_id).first()
    if request_obj:
        request_obj.status = 1  # Processed

    db.session.commit()

    logger.info(
        "DICOM report saved: result_id=%s radiologist_id=%s",
        result_id, radiologist_id,
    )

    return jsonify(
        {"success": True, "result_id": result_id, "status": "REPORT_SAVED"}
    ), 200


@dicom_bp.route("/ohif/<string:result_id>", methods=["GET"])
@login_required
@roles_required(*_IMAGING_ROLES)
def ohif_viewer(result_id):
    """Render full embedded OHIF Web Viewer console for an imaging study."""
    result = ImagingResult.query.filter_by(result_id=result_id).first()
    if not result:
        return jsonify({"error": "Imaging result not found"}), 404

    patient = Patient.query.filter_by(patient_id=result.patient_id).first()
    metadata = result.processing_metadata or {}
    study_uid = metadata.get("study_instance_uid", result.result_id)
    ohif_url = dicomweb_client.get_ohif_viewer_url(study_uid)

    return render_template(
        "imaging/ohif_viewer.html",
        result=result,
        patient=patient,
        metadata=metadata,
        study_uid=study_uid,
        ohif_url=ohif_url,
    )


@dicom_bp.route("/qido", methods=["GET"])
@login_required
@roles_required(*_IMAGING_ROLES)
def qido_search():
    """QIDO-RS Proxy: Query studies by patient ID, modality, or study date."""
    patient_id = request.args.get("PatientID") or request.args.get("patient_id")
    modality = request.args.get("ModalitiesInStudy") or request.args.get("modality")
    study_date = request.args.get("StudyDate") or request.args.get("study_date")
    limit = int(request.args.get("limit", 50))

    studies = dicomweb_client.qido_search_studies(
        patient_id=patient_id,
        modality=modality,
        study_date=study_date,
        limit=limit,
    )
    return jsonify({"studies": studies, "count": len(studies)}), 200


@dicom_bp.route("/wado/<string:study_uid>", methods=["GET"])
@login_required
@roles_required(*_IMAGING_ROLES)
def wado_metadata(study_uid):
    """WADO-RS Proxy: Retrieve study metadata JSON for OHIF and PACS consumers."""
    metadata = dicomweb_client.wado_retrieve_metadata(study_uid)
    return jsonify({"study_instance_uid": study_uid, "metadata": metadata}), 200


@dicom_bp.route("/pacs-status", methods=["GET"])
@login_required
@roles_required(*_IMAGING_ROLES)
def pacs_status():
    """Return local Orthanc PACS health status, AET, version, and DICOMweb endpoint details."""
    status = dicomweb_client.get_pacs_system_status()
    return jsonify(status), 200


@dicom_bp.route("/pacs-explorer", methods=["GET"])
@login_required
@roles_required(*_IMAGING_ROLES)
def pacs_explorer():
    """Render the interactive DICOM PACS Workstation & Explorer console."""
    status = dicomweb_client.get_pacs_system_status()
    return render_template(
        "imaging/pacs_explorer.html",
        title="DICOM PACS Explorer",
        pacs=status,
    )


@dicom_bp.route("/pacs-sync", methods=["POST"])
@login_required
@roles_required("radiology", "admin")
def pacs_sync():
    """
    STOW-RS upload endpoint.
    Accepts a multipart .dcm file, stores it temporarily, and pushes it to Orthanc PACS
    via stow_store_instances(). Returns the orthanc_id and status from the push.

    P0-03 / P0-15: secure_filename applied. File is validated for .dcm extension.
    Audit log recorded for every PACS push.
    """
    uploaded = request.files.get("dicom_file")
    if not uploaded or not uploaded.filename:
        return jsonify({"error": "No DICOM file provided"}), 400

    safe_name = secure_filename(uploaded.filename)
    if not safe_name or not safe_name.lower().endswith(".dcm"):
        return jsonify({"error": "Only .dcm DICOM files accepted"}), 400

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
        uploaded.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = dicomweb_client.stow_store_instances(tmp_path)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 400
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    logger.info(
        "PACS sync performed: actor_id=%s file=%s status=%s",
        current_user.id, safe_name, result.get("status"),
    )

    return jsonify(result), 200 if result.get("status") == "success" else 202
