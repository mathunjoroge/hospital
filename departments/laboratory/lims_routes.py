import json
from datetime import datetime

from flask import jsonify, render_template, request
from flask_login import current_user, login_required
from sqlalchemy.exc import SQLAlchemyError

from departments.models.laboratory import LabQCResult, LabQCSample, Specimen
from departments.rbac import roles_required
from extensions import db

from . import bp
from .lims_service import LIMSService
from departments.shared import queue_service  # Fixed import


@bp.route("/lims-dashboard", methods=["GET"])
@login_required
@roles_required("laboratory", "admin")
def lims_dashboard():
    """Renders the LIMS Specimen Tracking and Westgard QC Dashboard UI."""
    metrics = LIMSService.get_lims_dashboard_metrics()
    recent_specimens = Specimen.query.order_by(Specimen.created_at.desc()).limit(20).all()
    qc_samples = LabQCSample.query.order_by(LabQCSample.control_name).all()
    recent_qc_runs = (
        LabQCResult.query.order_by(LabQCResult.run_timestamp.desc()).limit(15).all()
    )

    return render_template(
        "laboratory/lims_dashboard.html",
        metrics=metrics,
        recent_specimens=recent_specimens,
        qc_samples=qc_samples,
        recent_qc_runs=recent_qc_runs,
    )


@bp.route("/api/lims/specimens/create", methods=["POST"])
@login_required
@roles_required("laboratory", "admin", "medicine")
def api_create_specimen():
    """Creates a new specimen barcode and tracking record."""
    data = request.get_json(silent=True) or request.form
    patient_id = data.get("patient_id")
    requested_lab_id = data.get("requested_lab_id")
    specimen_type = data.get("specimen_type", "WHOLE_BLOOD")
    container_type = data.get("container_type", "EDTA_PURPLE")

    if not patient_id:
        return jsonify({"error": "patient_id is required"}), 400

    try:
        specimen = LIMSService.create_specimen(
            patient_id=patient_id,
            requested_lab_id=int(requested_lab_id) if requested_lab_id else None,
            specimen_type=specimen_type,
            container_type=container_type,
            user_id=current_user.id,
        )
        return (
            jsonify(
                {
                    "message": "Specimen created successfully",
                    "barcode": specimen.barcode,
                    "specimen_id": specimen.id,
                    "status": specimen.status,
                }
            ),
            201,
        )
    except (ValueError, KeyError, SQLAlchemyError) as e:
        return jsonify({"error": str(e)}), 400


@bp.route("/api/lims/specimens/collect", methods=["POST"])
@login_required
@roles_required("laboratory", "admin", "nursing", "medicine")
def api_collect_specimen():
    """Marks a specimen as COLLECTED."""
    data = request.get_json(silent=True) or request.form
    barcode = data.get("barcode") or data.get("specimen_id")
    notes = data.get("notes")

    if not barcode:
        return jsonify({"error": "barcode or specimen_id required"}), 400

    try:
        specimen = LIMSService.update_specimen_status(
            specimen_id_or_barcode=barcode,
            new_status="COLLECTED",
            user_id=current_user.id,
            notes=notes,
        )
        return jsonify(
            {
                "message": f"Specimen {specimen.barcode} marked as COLLECTED",
                "status": specimen.status,
                "collected_at": specimen.collected_at.isoformat()
                if specimen.collected_at
                else None,
            }
        )
    except (ValueError, KeyError, SQLAlchemyError) as e:
        return jsonify({"error": str(e)}), 400


@bp.route("/api/lims/specimens/receive", methods=["POST"])
@login_required
@roles_required("laboratory", "admin")
def api_receive_specimen():
    """Marks a specimen as RECEIVED at the lab."""
    data = request.get_json(silent=True) or request.form
    barcode = data.get("barcode") or data.get("specimen_id")
    notes = data.get("notes")

    if not barcode:
        return jsonify({"error": "barcode or specimen_id required"}), 400

    try:
        specimen = LIMSService.update_specimen_status(
            specimen_id_or_barcode=barcode,
            new_status="RECEIVED",
            user_id=current_user.id,
            notes=notes,
        )
        return jsonify(
            {
                "message": f"Specimen {specimen.barcode} received at lab",
                "status": specimen.status,
                "received_at": specimen.received_at.isoformat()
                if specimen.received_at
                else None,
            }
        )
    except (ValueError, KeyError, SQLAlchemyError) as e:
        return jsonify({"error": str(e)}), 400


@bp.route("/api/lims/specimens/reject", methods=["POST"])
@login_required
@roles_required("laboratory", "admin")
def api_reject_specimen():
    """Marks a specimen as REJECTED due to sample quality issue."""
    data = request.get_json(silent=True) or request.form
    barcode = data.get("barcode") or data.get("specimen_id")
    rejection_reason = data.get("rejection_reason", "INSUFFICIENT_VOLUME")
    notes = data.get("notes")

    if not barcode:
        return jsonify({"error": "barcode or specimen_id required"}), 400

    try:
        specimen = LIMSService.update_specimen_status(
            specimen_id_or_barcode=barcode,
            new_status="REJECTED",
            user_id=current_user.id,
            notes=notes,
            rejection_reason=rejection_reason,
        )
        return jsonify(
            {
                "message": f"Specimen {specimen.barcode} rejected",
                "status": specimen.status,
                "rejection_reason": specimen.rejection_reason,
            }
        )
    except (ValueError, KeyError, SQLAlchemyError) as e:
        return jsonify({"error": str(e)}), 400


@bp.route("/api/lims/specimens/track/<barcode>", methods=["GET"])
@login_required
@roles_required("laboratory", "admin", "medicine")
def api_track_specimen(barcode):
    """Retrieves full specimen details and chain of custody log."""
    specimen = Specimen.query.filter_by(barcode=barcode).first()
    if not specimen:
        return jsonify({"error": f"Specimen not found: {barcode}"}), 404

    coc = json.loads(specimen.chain_of_custody) if specimen.chain_of_custody else []

    return jsonify(
        {
            "id": specimen.id,
            "barcode": specimen.barcode,
            "patient_id": specimen.patient_id,
            "requested_lab_id": specimen.requested_lab_id,
            "specimen_type": specimen.specimen_type,
            "container_type": specimen.container_type,
            "status": specimen.status,
            "collected_at": specimen.collected_at.isoformat()
            if specimen.collected_at
            else None,
            "received_at": specimen.received_at.isoformat()
            if specimen.received_at
            else None,
            "rejection_reason": specimen.rejection_reason,
            "chain_of_custody": coc,
        }
    )


@bp.route("/api/lims/qc/samples", methods=["POST"])
@login_required
@roles_required("laboratory", "admin")
def api_create_qc_sample():
    """Registers a new QC control sample (e.g. Normal, Abnormal Low, Abnormal High)."""
    data = request.get_json(silent=True) or request.form
    control_name = data.get("control_name")
    lot_number = data.get("lot_number")
    analyzer_name = data.get("analyzer_name")
    parameter_name = data.get("parameter_name")
    target_mean = data.get("target_mean")
    target_sd = data.get("target_sd")
    expiration_date_str = data.get("expiration_date")

    if not all([control_name, lot_number, analyzer_name, parameter_name, target_mean, target_sd]):
        return jsonify({"error": "Missing required QC sample parameters"}), 400

    exp_date = None
    if expiration_date_str:
        try:
            exp_date = datetime.strptime(expiration_date_str, "%Y-%m-%d").date()
        except ValueError:
            pass

    qc_sample = LabQCSample(
        control_name=control_name,
        lot_number=lot_number,
        analyzer_name=analyzer_name,
        parameter_name=parameter_name,
        target_mean=float(target_mean),
        target_sd=float(target_sd),
        expiration_date=exp_date,
    )
    db.session.add(qc_sample)
    db.session.commit()

    return (
        jsonify(
            {
                "message": "QC Control Sample registered successfully",
                "qc_sample_id": qc_sample.id,
                "control_name": qc_sample.control_name,
            }
        ),
        201,
    )


@bp.route("/api/lims/qc/log", methods=["POST"])
@login_required
@roles_required("laboratory", "admin")
def api_log_qc_run():
    """Logs a measured QC control value and evaluates Westgard Multi-Rules."""
    data = request.get_json(silent=True) or request.form
    qc_sample_id = data.get("qc_sample_id")
    measured_value = data.get("measured_value")

    if not qc_sample_id or measured_value is None:
        return jsonify({"error": "qc_sample_id and measured_value required"}), 400

    try:
        qc_result = LIMSService.log_qc_result(
            qc_sample_id=int(qc_sample_id),
            measured_value=float(measured_value),
            operator_id=current_user.id,
        )
        violated_rules = (
            json.loads(qc_result.violated_rules) if qc_result.violated_rules else []
        )

        return jsonify(
            {
                "message": "QC run recorded and evaluated",
                "qc_result_id": qc_result.id,
                "z_score": qc_result.z_score,
                "status": qc_result.status,
                "violated_rules": violated_rules,
            }
        )
    except (ValueError, KeyError, SQLAlchemyError) as e:
        return jsonify({"error": str(e)}), 400


@bp.route("/api/lims/dashboard-metrics", methods=["GET"])
@login_required
@roles_required("laboratory", "admin")
def api_lims_dashboard_metrics():
    """Returns real-time LIMS KPIs and QC pass rates."""
    metrics = LIMSService.get_lims_dashboard_metrics()
    return jsonify(metrics)

