"""
departments/theatre/routes.py
──────────────────────────────
Routes and API endpoints for Johns Hopkins–Grade Theatre & Surgical Module.
- Surgical Workbench (Console)
- WHO Surgical Safety Checklist (3 stages: Sign In, Time Out, Sign Out)
- Anaesthetic Record & Intraoperative Vitals
- Post-Operative Operative Note & PACU Aldrete Recovery Score
- Surgical Instrument, Sponge, and Needle Count Reconciliation
"""
import json
from datetime import datetime, timezone

from flask import abort, flash, jsonify, redirect, render_template, request, url_for
from flask_login import current_user, login_required

from departments.models.medicine import TheatreList
from departments.models.records import Patient
from departments.models.theatre import (
    AnaestheticRecord,
    PostOpNote,
    SurgicalInstrumentCount,
    WhoSurgicalChecklist,
)
from departments.theatre.theatre_engine import TheatreOperationsEngine
from extensions import db

from . import bp


def _resolve_entry(entry_id=None):
    """
    Resolve a TheatreList entry by ID, or fall back to the most recent entry.
    P0-10: Does NOT auto-create dummy theatre entries using Patient.query.first().
    Returns None if no entry exists.
    """
    if entry_id is not None:
        entry = db.session.get(TheatreList, entry_id)
        if entry:
            return entry
    # Fall back to most recent theatre booking — do NOT create a dummy one.
    return TheatreList.query.order_by(TheatreList.id.desc()).first()


@bp.route("/workbench", defaults={"entry_id": None}, methods=["GET"])
@bp.route("/workbench/<int:entry_id>", methods=["GET"])
@login_required
def surgical_workbench(entry_id):
    """
    Unified Surgical Workbench console for a theatre list booking entry.
    Displays patient clinical summary, encounter stage, WHO checklist status,
    anaesthetic record, post-op note, and instrument count.

    P0-10: Aborts 404 if no theatre entry or patient found — never substitutes
    the first patient in the database.
    """
    entry = _resolve_entry(entry_id)
    if not entry:
        abort(404)

    entry_id = entry.id
    patient = Patient.query.filter_by(patient_id=entry.patient_id).first()
    if not patient:
        # P0-10: Never substitute an unrelated patient for a clinical workstation.
        abort(404)

    checklist = WhoSurgicalChecklist.query.filter_by(theatre_entry_id=entry_id).first()
    anaesthetic = AnaestheticRecord.query.filter_by(theatre_entry_id=entry_id).first()
    postop = PostOpNote.query.filter_by(theatre_entry_id=entry_id).first()
    instruments = SurgicalInstrumentCount.query.filter_by(theatre_entry_id=entry_id).first()

    return render_template(
        "theatre/workbench.html",
        entry=entry,
        patient=patient,
        checklist=checklist,
        anaesthetic=anaesthetic,
        postop=postop,
        instruments=instruments,
    )


# ---------------------------------------------------------------------------
# WHO Surgical Safety Checklist
# ---------------------------------------------------------------------------
@bp.route("/", methods=["GET"])
@login_required
def theatre_index():
    """Root /theatre/ redirect to OR dashboard."""
    return redirect(url_for("theatre.get_or_dashboard_ui"))


@bp.route("/checklist", defaults={"entry_id": None}, methods=["GET", "POST"])
@bp.route("/checklist/<int:entry_id>", methods=["GET", "POST"])
@login_required
def who_checklist(entry_id):
    """GET/POST WHO 3-stage Surgical Safety Checklist."""
    entry = _resolve_entry(entry_id)
    entry_id = entry.id
    checklist = WhoSurgicalChecklist.query.filter_by(theatre_entry_id=entry_id).first()

    if not checklist:
        checklist = WhoSurgicalChecklist(
            theatre_entry_id=entry.id,
            patient_id=entry.patient_id,
            encounter_id=entry.encounter_id,
        )
        db.session.add(checklist)
        db.session.commit()

    if request.method == "POST":
        data = request.get_json(silent=True) or request.form
        stage = data.get("stage", "sign_in")  # sign_in, time_out, sign_out

        user_id = current_user.id if hasattr(current_user, "id") and isinstance(current_user.id, int) else None

        if stage == "sign_in":
            checklist.patient_identity_confirmed = str(data.get("patient_identity_confirmed", "false")).lower() in ("true", "1", "on")
            checklist.site_marked = str(data.get("site_marked", "false")).lower() in ("true", "1", "on")
            checklist.anaesthesia_safety_check_completed = str(data.get("anaesthesia_safety_check_completed", "false")).lower() in ("true", "1", "on")
            checklist.pulse_oximeter_functioning = str(data.get("pulse_oximeter_functioning", "false")).lower() in ("true", "1", "on")
            checklist.known_allergy = str(data.get("known_allergy", "false")).lower() in ("true", "1", "on")
            checklist.allergy_details = data.get("allergy_details")
            checklist.difficult_airway_risk = str(data.get("difficult_airway_risk", "false")).lower() in ("true", "1", "on")
            checklist.airway_equipment_available = str(data.get("airway_equipment_available", "false")).lower() in ("true", "1", "on")
            checklist.blood_loss_risk_over_500ml = str(data.get("blood_loss_risk_over_500ml", "false")).lower() in ("true", "1", "on")
            checklist.iv_access_and_fluids_planned = str(data.get("iv_access_and_fluids_planned", "false")).lower() in ("true", "1", "on")
            checklist.sign_in_completed = True
            checklist.sign_in_completed_by = user_id
            checklist.sign_in_completed_at = datetime.now(timezone.utc)

        elif stage == "time_out":
            checklist.team_members_introduced = str(data.get("team_members_introduced", "false")).lower() in ("true", "1", "on")
            checklist.verbal_confirm_patient_site_procedure = str(data.get("verbal_confirm_patient_site_procedure", "false")).lower() in ("true", "1", "on")
            checklist.critical_steps_surgeon_reviewed = str(data.get("critical_steps_surgeon_reviewed", "false")).lower() in ("true", "1", "on")
            checklist.critical_steps_anaesthetist_reviewed = str(data.get("critical_steps_anaesthetist_reviewed", "false")).lower() in ("true", "1", "on")
            checklist.critical_steps_nursing_reviewed = str(data.get("critical_steps_nursing_reviewed", "false")).lower() in ("true", "1", "on")
            checklist.antibiotic_prophylaxis_given = str(data.get("antibiotic_prophylaxis_given", "false")).lower() in ("true", "1", "on")
            checklist.antibiotic_given_within_60min = str(data.get("antibiotic_given_within_60min", "false")).lower() in ("true", "1", "on")
            checklist.essential_imaging_displayed = str(data.get("essential_imaging_displayed", "false")).lower() in ("true", "1", "on")
            checklist.time_out_completed = True
            checklist.time_out_completed_by = user_id
            checklist.time_out_completed_at = datetime.now(timezone.utc)

        elif stage == "sign_out":
            checklist.procedure_name_recorded = data.get("procedure_name_recorded", entry.procedure.name if entry.procedure else "")
            checklist.instrument_sponge_needle_counts_correct = str(data.get("instrument_sponge_needle_counts_correct", "false")).lower() in ("true", "1", "on")
            checklist.specimens_labelled = str(data.get("specimens_labelled", "false")).lower() in ("true", "1", "on")
            checklist.equipment_problems_addressed = str(data.get("equipment_problems_addressed", "false")).lower() in ("true", "1", "on")
            checklist.key_recovery_concerns_reviewed = data.get("key_recovery_concerns_reviewed")
            checklist.sign_out_completed = True
            checklist.sign_out_completed_by = user_id
            checklist.sign_out_completed_at = datetime.now(timezone.utc)

        db.session.commit()

        if request.is_json:
            return jsonify({
                "status": "success",
                "message": f"WHO Checklist stage '{stage}' recorded.",
                "checklist_id": checklist.id,
                "is_fully_completed": checklist.is_fully_completed(),
            })

        flash(f"WHO Checklist stage '{stage}' updated successfully!", "success")
        return redirect(url_for("theatre.surgical_workbench", entry_id=entry_id))

    # GET response — render HTML UI
    return render_template(
        "theatre/who_checklist.html",
        entry=entry,
        checklist=checklist,
    )


# ---------------------------------------------------------------------------
# Anaesthetic Record
# ---------------------------------------------------------------------------
@bp.route("/anaesthetic", defaults={"entry_id": None}, methods=["GET", "POST"])
@bp.route("/anaesthetic/<int:entry_id>", methods=["GET", "POST"])
@login_required
def anaesthetic_record(entry_id):
    """GET/POST Anaesthetic Record & Intraoperative Vitals."""
    entry = _resolve_entry(entry_id)
    entry_id = entry.id
    record = AnaestheticRecord.query.filter_by(theatre_entry_id=entry_id).first()

    if not record:
        record = AnaestheticRecord(
            theatre_entry_id=entry.id,
            patient_id=entry.patient_id,
            encounter_id=entry.encounter_id,
            anaesthetist_id=current_user.id if hasattr(current_user, "id") and isinstance(current_user.id, int) else None,
        )
        db.session.add(record)
        db.session.commit()

    if request.method == "POST":
        data = request.get_json(silent=True) or request.form

        record.asa_status = data.get("asa_status", record.asa_status or "ASA I")
        record.is_emergency = str(data.get("is_emergency", "false")).lower() in ("true", "1", "on")
        record.mallampati_class = data.get("mallampati_class")
        record.airway_management = data.get("airway_management")
        record.ett_size = data.get("ett_size")
        record.technique = data.get("technique", record.technique or "General")

        if "agents_administered" in data:
            agents = data.get("agents_administered")
            if isinstance(agents, str):
                try:
                    agents = json.loads(agents)
                except Exception:
                    agents = [agents]
            record.agents_administered = agents

        if "vitals_series" in data:
            vitals = data.get("vitals_series")
            if isinstance(vitals, str):
                try:
                    vitals = json.loads(vitals)
                except Exception:
                    vitals = []
            record.vitals_series = vitals

        record.estimated_blood_loss_ml = int(data.get("estimated_blood_loss_ml", record.estimated_blood_loss_ml or 0))
        record.crystalloids_ml = int(data.get("crystalloids_ml", record.crystalloids_ml or 0))
        record.colloids_ml = int(data.get("colloids_ml", record.colloids_ml or 0))
        record.blood_products_ml = int(data.get("blood_products_ml", record.blood_products_ml or 0))
        record.urine_output_ml = int(data.get("urine_output_ml", record.urine_output_ml or 0))
        record.complications_notes = data.get("complications_notes")

        db.session.commit()

        if request.is_json:
            return jsonify({
                "status": "success",
                "message": "Anaesthetic record updated.",
                "record_id": record.id,
                "asa_status": record.asa_status,
                "technique": record.technique,
            })

        flash("Anaesthetic record saved!", "success")
        return redirect(url_for("theatre.surgical_workbench", entry_id=entry_id))

    return render_template(
        "theatre/anaesthetic_record.html",
        entry=entry,
        record=record,
    )


# ---------------------------------------------------------------------------
# Post-Operative Operative Note & PACU Aldrete Score
# ---------------------------------------------------------------------------
@bp.route("/postop", defaults={"entry_id": None}, methods=["GET", "POST"])
@bp.route("/postop/<int:entry_id>", methods=["GET", "POST"])
@login_required
def postop_note(entry_id):
    """GET/POST Operative Note & PACU Aldrete Recovery Score."""
    entry = _resolve_entry(entry_id)
    entry_id = entry.id
    note = PostOpNote.query.filter_by(theatre_entry_id=entry_id).first()

    if not note:
        note = PostOpNote(
            theatre_entry_id=entry.id,
            patient_id=entry.patient_id,
            encounter_id=entry.encounter_id,
            surgeon_id=current_user.id if hasattr(current_user, "id") and isinstance(current_user.id, int) else None,
            preop_diagnosis=entry.procedure.name if entry.procedure else "Surgical Procedure",
            postop_diagnosis=entry.procedure.name if entry.procedure else "Surgical Procedure",
            procedure_performed=entry.procedure.name if entry.procedure else "Surgical Procedure",
            surgical_findings="Uneventful procedure.",
        )
        db.session.add(note)
        db.session.commit()

    if request.method == "POST":
        data = request.get_json(silent=True) or request.form

        note.preop_diagnosis = data.get("preop_diagnosis", note.preop_diagnosis)
        note.postop_diagnosis = data.get("postop_diagnosis", note.postop_diagnosis)
        note.procedure_performed = data.get("procedure_performed", note.procedure_performed)
        note.surgical_findings = data.get("surgical_findings", note.surgical_findings)
        note.specimens_taken = data.get("specimens_taken")
        note.implants_inserted = data.get("implants_inserted")
        note.drains_tubes_placed = data.get("drains_tubes_placed")
        note.postop_instructions = data.get("postop_instructions")
        note.discharge_to = data.get("discharge_to", note.discharge_to or "PACU")

        # Aldrete scoring components
        note.aldrete_activity = int(data.get("aldrete_activity", note.aldrete_activity or 2))
        note.aldrete_respiration = int(data.get("aldrete_respiration", note.aldrete_respiration or 2))
        note.aldrete_circulation = int(data.get("aldrete_circulation", note.aldrete_circulation or 2))
        note.aldrete_consciousness = int(data.get("aldrete_consciousness", note.aldrete_consciousness or 2))
        note.aldrete_spo2 = int(data.get("aldrete_spo2", note.aldrete_spo2 or 2))

        # Advance status on TheatreList and Encounter if updated
        entry.status = 1  # Completed
        entry.notes_on_post_op = note.postop_instructions or note.surgical_findings

        if entry.encounter_id:
            from departments.medicine.inpatients import transition_surgical_stage
            try:
                transition_surgical_stage(entry.encounter_id, "POST_OP")
            except ValueError:
                pass

        db.session.commit()

        if request.is_json:
            return jsonify({
                "status": "success",
                "message": "Post-op note & Aldrete score recorded.",
                "note_id": note.id,
                "total_aldrete_score": note.total_aldrete_score,
                "fit_for_pacu_discharge": note.is_fit_for_pacu_discharge(),
            })

        flash(f"Post-op note saved! Aldrete Score: {note.total_aldrete_score}/10", "success")
        return redirect(url_for("theatre.surgical_workbench", entry_id=entry_id))

    return render_template(
        "theatre/postop_note.html",
        entry=entry,
        note=note,
    )


# ---------------------------------------------------------------------------
# Surgical Instrument & Sponge Count Reconciliation
# ---------------------------------------------------------------------------
@bp.route("/instruments", defaults={"entry_id": None}, methods=["GET", "POST"])
@bp.route("/instruments/<int:entry_id>", methods=["GET", "POST"])
@login_required
def instrument_count(entry_id):
    """GET/POST Surgical Instrument, Sponge, and Needle Reconciliation."""
    entry = _resolve_entry(entry_id)
    entry_id = entry.id
    counts = SurgicalInstrumentCount.query.filter_by(theatre_entry_id=entry_id).first()

    if not counts:
        counts = SurgicalInstrumentCount(
            theatre_entry_id=entry.id,
            encounter_id=entry.encounter_id,
            scrub_nurse_id=current_user.id if hasattr(current_user, "id") and isinstance(current_user.id, int) else None,
        )
        db.session.add(counts)
        db.session.commit()

    if request.method == "POST":
        data = request.get_json(silent=True) or request.form

        counts.tray_name_or_barcode = data.get("tray_name_or_barcode", counts.tray_name_or_barcode)

        counts.sponges_initial = int(data.get("sponges_initial", counts.sponges_initial or 0))
        counts.sponges_added = int(data.get("sponges_added", counts.sponges_added or 0))
        counts.sponges_closing_cavity = int(data.get("sponges_closing_cavity", counts.sponges_closing_cavity or 0))
        counts.sponges_closing_skin = int(data.get("sponges_closing_skin", counts.sponges_closing_skin or 0))

        counts.needles_initial = int(data.get("needles_initial", counts.needles_initial or 0))
        counts.needles_added = int(data.get("needles_added", counts.needles_added or 0))
        counts.needles_closing_cavity = int(data.get("needles_closing_cavity", counts.needles_closing_cavity or 0))
        counts.needles_closing_skin = int(data.get("needles_closing_skin", counts.needles_closing_skin or 0))

        counts.instruments_initial = int(data.get("instruments_initial", counts.instruments_initial or 0))
        counts.instruments_added = int(data.get("instruments_added", counts.instruments_added or 0))
        counts.instruments_closing_cavity = int(data.get("instruments_closing_cavity", counts.instruments_closing_cavity or 0))
        counts.instruments_closing_skin = int(data.get("instruments_closing_skin", counts.instruments_closing_skin or 0))

        counts.notes = data.get("notes")
        counts.xray_ordered_for_discrepancy = str(data.get("xray_ordered_for_discrepancy", "false")).lower() in ("true", "1", "on")

        reconcile_result = counts.calculate_reconciliation()
        db.session.commit()

        if request.is_json:
            return jsonify({
                "status": "success" if reconcile_result["reconciled"] else "warning_discrepancy",
                "message": "Instrument count reconciled cleanly." if reconcile_result["reconciled"] else "COUNT DISCREPANCY DETECTED! Intraoperative X-Ray required.",
                "count_id": counts.id,
                "reconciliation": reconcile_result,
            })

        if reconcile_result["reconciled"]:
            flash("Surgical instrument count reconciled successfully!", "success")
        else:
            flash("ALERT: Instrument count discrepancy! Verify cavity and order intraoperative X-Ray.", "danger")

        return redirect(url_for("theatre.surgical_workbench", entry_id=entry_id))

    reconcile_result = counts.calculate_reconciliation()
    return render_template(
        "theatre/instrument_count.html",
        entry=entry,
        counts=counts,
        reconcile_result=reconcile_result,
    )


# ---------------------------------------------------------------------------
# Intraoperative Vitals, Fluid Balance, PACU Readiness & OR Flow Console
# ---------------------------------------------------------------------------
@bp.route("/api/vitals/<int:entry_id>", methods=["POST"])
@login_required
def record_intraop_vitals_api(entry_id):
    """POST endpoint to stream intraoperative vital sign snapshot."""
    data = request.get_json() or {}
    try:
        record = TheatreOperationsEngine.record_intraop_vitals(
            entry_id=entry_id,
            hr=data.get("hr"),
            bp_systolic=data.get("bp_sys"),
            bp_diastolic=data.get("bp_dia"),
            spo2=data.get("spo2"),
            etco2=data.get("etco2"),
            agent_concentration=data.get("agent_conc"),
        )
        return jsonify({
            "status": "success",
            "message": "Intraoperative vital snapshot recorded.",
            "record_id": record.id,
            "vitals_count": len(record.vitals_series),
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@bp.route("/api/fluid-balance/<int:entry_id>", methods=["GET"])
@login_required
def get_fluid_balance_api(entry_id):
    """GET endpoint returning intraoperative fluid intake vs. loss calculation."""
    balance = TheatreOperationsEngine.calculate_fluid_balance(entry_id)
    return jsonify(balance), 200


@bp.route("/api/pacu-readiness/<int:entry_id>", methods=["GET"])
@login_required
def get_pacu_readiness_api(entry_id):
    """GET endpoint returning PACU Aldrete Recovery Score & discharge readiness status."""
    readiness = TheatreOperationsEngine.evaluate_pacu_discharge_readiness(entry_id)
    return jsonify(readiness), 200


@bp.route("/dashboard", methods=["GET"])
@login_required
def get_or_dashboard_ui():
    """Render real-time Operating Theatre & OR Flow Console dashboard UI."""
    from departments.theatre.theatre_engine import TheatreOperationsEngine
    metrics = TheatreOperationsEngine.get_or_dashboard_metrics()
    return render_template("theatre/or_dashboard.html", metrics=metrics)


@bp.route("/api/metrics", methods=["GET"])
@login_required
def get_or_metrics_api():
    """JSON API endpoint returning Operating Theatre KPIs and active case list."""
    from departments.theatre.theatre_engine import TheatreOperationsEngine
    metrics = TheatreOperationsEngine.get_or_dashboard_metrics()
    return jsonify(metrics), 200


# ---------------------------------------------------------------------------
# Anesthesia Timeline Matrix & ASA Risk Assessment API
# ---------------------------------------------------------------------------
@bp.route("/api/anesthesia-timeline/<int:entry_id>", methods=["GET", "POST"])
@login_required
def anesthesia_timeline_api(entry_id):
    """GET timeline matrix / POST new timestamped anesthesia timeline event."""
    from departments.theatre.theatre_engine import TheatreOperationsEngine

    if request.method == "POST":
        data = request.get_json() or {}
        event_type = data.get("event_type", "MAINTENANCE")
        notes = data.get("notes")
        user_name = current_user.username if hasattr(current_user, "username") else "Clinician"

        try:
            event_obj = TheatreOperationsEngine.record_timeline_event(
                entry_id=entry_id,
                event_type=event_type,
                notes=notes,
                recorded_by=user_name,
            )
            return jsonify({
                "status": "success",
                "message": f"Anesthesia timeline event '{event_type}' recorded.",
                "event": event_obj,
            }), 201
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    matrix = TheatreOperationsEngine.get_anesthesia_timeline_matrix(entry_id)
    return jsonify(matrix), 200


@bp.route("/api/asa-assessment/<asa_code>", methods=["GET"])
@login_required
def get_asa_assessment_api(asa_code):
    """GET ASA physical status definition & risk grade."""
    from departments.theatre.theatre_engine import TheatreOperationsEngine
    is_emerg = str(request.args.get("emergency", "false")).lower() in ("true", "1")
    eval_result = TheatreOperationsEngine.evaluate_asa_score(asa_code, is_emergency=is_emerg)
    return jsonify(eval_result), 200


@bp.route("/api/or-schedule", methods=["POST"])
@login_required
def update_or_schedule_api():
    """POST endpoint to allocate/schedule OR room with conflict detection."""
    from departments.theatre.theatre_engine import TheatreOperationsEngine
    data = request.get_json() or {}

    entry_id = data.get("entry_id")
    or_room = data.get("or_room", "OR 1")
    start_str = data.get("scheduled_start_time")
    duration_min = int(data.get("estimated_duration_minutes", 120))

    if not entry_id:
        return jsonify({"error": "entry_id is required."}), 400

    entry = TheatreList.query.get_or_404(entry_id)

    start_dt = None
    if start_str:
        try:
            start_dt = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
        except Exception:
            return jsonify({"error": "Invalid scheduled_start_time format ISO 8601 expected."}), 400

    if start_dt:
        has_conflict, conflict_msg = TheatreOperationsEngine.check_room_schedule_conflict(
            or_room=or_room,
            start_time=start_dt,
            duration_minutes=duration_min,
            exclude_entry_id=entry.id,
        )
        if has_conflict:
            return jsonify({
                "status": "conflict_detected",
                "error": conflict_msg,
            }), 409

    entry.or_room = or_room
    if start_dt:
        entry.scheduled_start_time = start_dt
    entry.estimated_duration_minutes = duration_min

    db.session.commit()

    return jsonify({
        "status": "success",
        "message": f"Case #{entry.id} assigned to {or_room}.",
        "entry_id": entry.id,
        "or_room": entry.or_room,
        "scheduled_start_time": entry.scheduled_start_time.isoformat() if entry.scheduled_start_time else None,
        "estimated_duration_minutes": entry.estimated_duration_minutes,
    }), 200


