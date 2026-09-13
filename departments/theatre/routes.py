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

from flask import flash, jsonify, redirect, render_template, request, url_for
from flask_login import current_user, login_required

from departments.models.medicine import TheatreList
from departments.models.records import Patient
from departments.models.theatre import (
    AnaestheticRecord,
    PostOpNote,
    SurgicalInstrumentCount,
    WhoSurgicalChecklist,
)
from extensions import db

from . import bp


@bp.route("/workbench/<int:entry_id>", methods=["GET"])
@login_required
def surgical_workbench(entry_id):
    """
    Unified Surgical Workbench console for a theatre list booking entry.
    Displays patient clinical summary, encounter stage, WHO checklist status,
    anaesthetic record, post-op note, and instrument count.
    """
    entry = TheatreList.query.get_or_404(entry_id)
    patient = Patient.query.filter_by(patient_id=entry.patient_id).first_or_404()

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
@bp.route("/checklist/<int:entry_id>", methods=["GET", "POST"])
@login_required
def who_checklist(entry_id):
    """GET/POST WHO 3-stage Surgical Safety Checklist."""
    entry = TheatreList.query.get_or_404(entry_id)
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

    # GET response
    return jsonify({
        "theatre_entry_id": entry_id,
        "sign_in_completed": checklist.sign_in_completed,
        "time_out_completed": checklist.time_out_completed,
        "sign_out_completed": checklist.sign_out_completed,
        "is_fully_completed": checklist.is_fully_completed(),
    })


# ---------------------------------------------------------------------------
# Anaesthetic Record
# ---------------------------------------------------------------------------
@bp.route("/anaesthetic/<int:entry_id>", methods=["GET", "POST"])
@login_required
def anaesthetic_record(entry_id):
    """GET/POST Anaesthetic Record & Intraoperative Vitals."""
    entry = TheatreList.query.get_or_404(entry_id)
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

    return jsonify({
        "theatre_entry_id": entry_id,
        "asa_status": record.asa_status,
        "technique": record.technique,
        "estimated_blood_loss_ml": record.estimated_blood_loss_ml,
        "agents_administered": record.agents_administered,
        "vitals_series": record.vitals_series,
    })


# ---------------------------------------------------------------------------
# Post-Operative Operative Note & PACU Aldrete Score
# ---------------------------------------------------------------------------
@bp.route("/postop/<int:entry_id>", methods=["GET", "POST"])
@login_required
def postop_note(entry_id):
    """GET/POST Operative Note & PACU Aldrete Recovery Score."""
    entry = TheatreList.query.get_or_404(entry_id)
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

    return jsonify({
        "theatre_entry_id": entry_id,
        "preop_diagnosis": note.preop_diagnosis,
        "postop_diagnosis": note.postop_diagnosis,
        "total_aldrete_score": note.total_aldrete_score,
        "fit_for_pacu_discharge": note.is_fit_for_pacu_discharge(),
    })


# ---------------------------------------------------------------------------
# Surgical Instrument & Sponge Count Reconciliation
# ---------------------------------------------------------------------------
@bp.route("/instruments/<int:entry_id>", methods=["GET", "POST"])
@login_required
def instrument_count(entry_id):
    """GET/POST Surgical Instrument, Sponge, and Needle Reconciliation."""
    entry = TheatreList.query.get_or_404(entry_id)
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
    return jsonify({
        "theatre_entry_id": entry_id,
        "tray_name_or_barcode": counts.tray_name_or_barcode,
        "count_reconciled": counts.count_reconciled,
        "reconciliation": reconcile_result,
    })
