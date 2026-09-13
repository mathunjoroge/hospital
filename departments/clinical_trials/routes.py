"""
departments/clinical_trials/routes.py
────────────────────────────────────────
Routes & API Endpoints for Clinical Trial Registry, e-Consent, Randomization, and SAE Reporting.
"""

from flask import jsonify, render_template, request
from flask_login import current_user, login_required

from . import bp
from .trials_engine import ClinicalTrialsEngine


@bp.route("/console", methods=["GET"])
@login_required
def trials_console_ui():
    """Render Clinical Trial Registry & e-Consent Management Dashboard UI."""
    protocols = ClinicalTrialsEngine.get_trial_registry_summary()
    return render_template("clinical_trials/trials_console.html", protocols=protocols)


@bp.route("/api/protocols", methods=["GET", "POST"])
@login_required
def protocols_api():
    """GET protocols list / POST new Clinical Trial Protocol."""
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        protocol_number = data.get("protocol_number")
        title = data.get("title")
        sponsor = data.get("sponsor")
        pi = data.get("principal_investigator")

        if not all([protocol_number, title, sponsor, pi]):
            return jsonify({"error": "protocol_number, title, sponsor, and principal_investigator are required."}), 400

        try:
            protocol = ClinicalTrialsEngine.create_protocol(
                protocol_number=protocol_number,
                title=title,
                sponsor=sponsor,
                principal_investigator=pi,
                phase=data.get("phase", "Phase III"),
                target_enrollment=int(data.get("target_enrollment", 100)),
                inclusion_criteria=data.get("inclusion_criteria"),
                exclusion_criteria=data.get("exclusion_criteria"),
                treatment_arms=data.get("treatment_arms"),
                irb_approval_number=data.get("irb_approval_number"),
            )
            return jsonify({
                "status": "success",
                "protocol_id": protocol.id,
                "protocol_number": protocol.protocol_number,
            }), 201
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    summary = ClinicalTrialsEngine.get_trial_registry_summary()
    return jsonify({"count": len(summary), "protocols": summary}), 200


@bp.route("/api/screen", methods=["POST"])
@login_required
def screen_eligibility_api():
    """POST /clinical-trials/api/screen — Runs automated patient eligibility screening."""
    data = request.get_json(silent=True) or {}
    protocol_id = data.get("protocol_id")
    patient_id = data.get("patient_id")

    if not protocol_id or not patient_id:
        return jsonify({"error": "protocol_id and patient_id are required."}), 400

    try:
        result = ClinicalTrialsEngine.screen_patient_eligibility(protocol_id, patient_id)
        return jsonify(result), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 404


@bp.route("/api/econsent", methods=["POST"])
@login_required
def econsent_api():
    """POST /clinical-trials/api/econsent — Signs e-Consent document with SHA-256 hash."""
    data = request.get_json(silent=True) or {}
    participant_id = data.get("participant_id")
    witness = data.get("witness_name", current_user.username if hasattr(current_user, "username") else "Witness")

    if not participant_id:
        return jsonify({"error": "participant_id is required."}), 400

    try:
        result = ClinicalTrialsEngine.record_econsent(participant_id, witness_name=witness)
        return jsonify(result), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@bp.route("/api/randomize", methods=["POST"])
@login_required
def randomize_api():
    """POST /clinical-trials/api/randomize — Randomizes participant to treatment arm."""
    data = request.get_json(silent=True) or {}
    participant_id = data.get("participant_id")

    if not participant_id:
        return jsonify({"error": "participant_id is required."}), 400

    try:
        result = ClinicalTrialsEngine.randomize_participant(participant_id)
        return jsonify(result), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@bp.route("/api/adverse-event", methods=["POST"])
@login_required
def log_adverse_event_api():
    """POST /clinical-trials/api/adverse-event — Logs AE / SAE with IRB escalation alert."""
    data = request.get_json(silent=True) or {}
    protocol_id = data.get("protocol_id")
    participant_id = data.get("participant_id")
    event_term = data.get("event_term")

    if not all([protocol_id, participant_id, event_term]):
        return jsonify({"error": "protocol_id, participant_id, and event_term are required."}), 400

    try:
        user_name = current_user.username if hasattr(current_user, "username") else "Investigator"
        result = ClinicalTrialsEngine.log_adverse_event(
            protocol_id=protocol_id,
            participant_id=participant_id,
            event_term=event_term,
            severity_grade=int(data.get("severity_grade", 1)),
            is_serious_ae=str(data.get("is_serious_ae", "false")).lower() in ("true", "1", "on"),
            causality_assessment=data.get("causality_assessment", "POSSIBLE"),
            reported_by=user_name,
        )
        return jsonify(result), 201
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
