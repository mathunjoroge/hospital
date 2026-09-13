"""
departments/nursing/icu_routes.py
──────────────────────────────────
Blueprint routes for ICU / HDU Flowsheet Workstation.
"""

import logging

from flask import Blueprint, jsonify, render_template, request, session
from flask_login import current_user

from departments.models.icu import ICUFlowsheetEntry, ICUFluidBalance
from departments.nursing.icu_engine import (
    calculate_fluid_balance,
    calculate_gcs,
    calculate_map,
    generate_flowsheet_matrix,
)
from extensions import db

logger = logging.getLogger(__name__)

icu_bp = Blueprint("icu", __name__, url_prefix="/nursing/icu")


def _get_current_user_id() -> int:
    """Return logged in user_id, session user_id, or fallback to 1."""
    if current_user and getattr(current_user, "is_authenticated", False):
        return current_user.id
    return session.get("user_id", 1)


@icu_bp.route("/flowsheet/<string:patient_id>", methods=["GET"])
def icu_flowsheet(patient_id: str):
    """
    Render ICU / HDU Flowsheet Workstation UI.
    """
    hours = request.args.get("hours", 24, type=int)
    matrix = generate_flowsheet_matrix(patient_id, hours=hours)
    return render_template("nursing/icu_flowsheet.html", patient_id=patient_id, matrix=matrix)


@icu_bp.route("/api/flowsheet/<string:patient_id>", methods=["GET"])
def api_icu_flowsheet(patient_id: str):
    """
    JSON API endpoint for ICU flowsheet trends and matrix data.
    """
    hours = request.args.get("hours", 24, type=int)
    matrix = generate_flowsheet_matrix(patient_id, hours=hours)
    return jsonify(matrix), 200


@icu_bp.route("/flowsheet/<string:patient_id>/vitals", methods=["POST"])
def log_icu_vitals(patient_id: str):
    """
    Log vitals, ventilator settings, and GCS assessment for ICU patient.
    """
    data = request.get_json(silent=True) or request.form.to_dict()

    nurse_id = _get_current_user_id()

    # Parse inputs safely
    def _int(k):
        v = data.get(k)
        return int(v) if v is not None and str(v).strip() != "" else None

    def _float(k):
        v = data.get(k)
        return float(v) if v is not None and str(v).strip() != "" else None

    sys_bp = _int("bp_systolic")
    dia_bp = _int("bp_diastolic")
    map_val = _float("mean_arterial_pressure") or calculate_map(sys_bp, dia_bp)

    g_eye = _int("gcs_eye")
    g_verbal = _int("gcs_verbal")
    g_motor = _int("gcs_motor")
    gcs_calc = calculate_gcs(g_eye, g_verbal, g_motor)
    gcs_tot = gcs_calc["total"] if gcs_calc["valid"] else None

    entry = ICUFlowsheetEntry(
        patient_id=patient_id,
        nurse_id=nurse_id,
        heart_rate=_int("heart_rate"),
        bp_systolic=sys_bp,
        bp_diastolic=dia_bp,
        mean_arterial_pressure=map_val,
        spo2=_int("spo2"),
        temperature=_float("temperature"),
        central_venous_pressure=_float("central_venous_pressure"),
        ventilator_mode=data.get("ventilator_mode") or None,
        fio2=_float("fio2"),
        peep=_float("peep"),
        tidal_volume=_int("tidal_volume"),
        peak_inspiratory_pressure=_float("peak_inspiratory_pressure"),
        respiratory_rate=_int("respiratory_rate"),
        gcs_eye=g_eye,
        gcs_verbal=g_verbal,
        gcs_motor=g_motor,
        gcs_total=gcs_tot,
        rass_score=_int("rass_score"),
        pain_score=_int("pain_score"),
        notes=data.get("notes"),
    )

    db.session.add(entry)
    db.session.commit()

    return jsonify({
        "success": True,
        "entry_id": entry.id,
        "map": map_val,
        "gcs": gcs_tot,
        "gcs_severity": gcs_calc.get("severity"),
    }), 201


@icu_bp.route("/flowsheet/<string:patient_id>/fluid", methods=["POST"])
def log_icu_fluid(patient_id: str):
    """
    Log Input/Output fluid balance for ICU patient.
    """
    data = request.get_json(silent=True) or request.form.to_dict()
    nurse_id = _get_current_user_id()

    def _float(k, default=0.0):
        v = data.get(k)
        if v is None or str(v).strip() == "":
            return default
        try:
            return float(v)
        except (ValueError, TypeError):
            return default

    iv = _float("iv_fluids")
    blood = _float("blood_products")
    enteral = _float("enteral")
    meds = _float("medications")

    urine = _float("urine")
    drains = _float("drains")
    emesis = _float("ng_emesis")
    stool = _float("stool")
    weight = _float("patient_weight_kg", 70.0)

    eval_io = calculate_fluid_balance(
        iv_fluids=iv,
        blood_products=blood,
        enteral=enteral,
        medications=meds,
        urine=urine,
        drains=drains,
        ng_emesis=emesis,
        stool=stool,
        weight_kg=weight,
    )

    entry = ICUFluidBalance(
        patient_id=patient_id,
        nurse_id=nurse_id,
        iv_fluids_ml=iv,
        blood_products_ml=blood,
        enteral_oral_ml=enteral,
        iv_medications_ml=meds,
        total_input_ml=eval_io["total_input_ml"],
        urine_output_ml=urine,
        drain_output_ml=drains,
        ng_emesis_ml=emesis,
        stool_ml=stool,
        total_output_ml=eval_io["total_output_ml"],
        net_balance_ml=eval_io["net_balance_ml"],
        patient_weight_kg=weight,
        notes=data.get("notes"),
    )

    db.session.add(entry)
    db.session.commit()

    return jsonify({
        "success": True,
        "entry_id": entry.id,
        "total_input_ml": eval_io["total_input_ml"],
        "total_output_ml": eval_io["total_output_ml"],
        "net_balance_ml": eval_io["net_balance_ml"],
        "urine_rate": eval_io["urine_rate_ml_kg_hr"],
        "is_oliguria": eval_io["is_oliguria"],
        "oliguria_warning": eval_io["oliguria_warning"],
    }), 201
