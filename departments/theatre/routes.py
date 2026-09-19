"""
departments/theatre/routes.py
──────────────────────────────
Routes and API endpoints for Johns Hopkins–Grade Theatre & Surgical Module.
- Surgical Workbench (Console)
- WHO Surgical Safety Checklist (3 stages: Sign In, Time Out, Sign Out)
- Anaesthetic Record & Intraoperative Vitals
- Post-Operative Operative Note & PACU Aldrete Recovery Score
- Surgical Instrument, Sponge, and Needle Count Reconciliation

Safety invariants enforced at the route layer:
  - Every route requires a clinical role (RBAC).
  - Clinical documentation routes require an explicit entry_id — a POST can
    never fall back to "the most recent case" (cross-patient documentation).
  - Opening a form (GET) never fabricates clinical documents.
  - Completing a case is gated on the WHO checklist and instrument
    reconciliation; anaesthetic values are validated for plausibility.
"""

import json
import logging
from datetime import datetime, timezone

from flask import (
    abort,
    flash,
    has_request_context,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from flask_login import current_user, login_required

from departments.models.medicine import TheatreList
from departments.models.records import Patient
from departments.models.theatre import (
    AnaestheticRecord,
    PostOpNote,
    SurgicalInstrumentCount,
    WhoSurgicalChecklist,
)
from departments.rbac import roles_required
from departments.theatre.theatre_engine import OR_ROOMS, TheatreOperationsEngine
from extensions import db

from . import bp

logger = logging.getLogger(__name__)

# Roles allowed on theatre documentation. "theatre" is the OR staff role;
# medicine/nursing cover surgeons & perioperative nurses; admin for audit.
THEATRE_ROLES = ("theatre", "medicine", "nursing", "admin")


def _current_user_id():
    """Current user's id if available and int-like, else None."""
    uid = getattr(current_user, "id", None)
    return uid if isinstance(uid, int) else None


def _resolve_entry(entry_id=None, *, strict=False):
    """
    Resolve a TheatreList entry.

    strict=True (all POST paths, and the workbench): entry_id is REQUIRED and
    must exist — there is no "most recent case" fallback, so a request can
    never write to the wrong patient's chart. Returns None when unresolved.

    strict=False (legacy GET convenience): falls back to the most recent
    entry; callers must surface `entry_status_warning` when the resolved
    entry is not the intended one.
    """
    if entry_id is None and has_request_context():
        raw = request.args.get("entry_id") or request.args.get("id")
        if raw:
            try:
                entry_id = int(raw)
            except (TypeError, ValueError):
                return None
    if entry_id is not None:
        entry = db.session.get(TheatreList, entry_id)
        if entry:
            return entry
        if strict:
            return None
    if strict:
        return None
    # GET-only convenience fallback — never reachable from a POST path.
    return TheatreList.query.order_by(TheatreList.id.desc()).first()


def _entry_status_warning(entry):
    """Warning shown when a resolved entry is already completed (status == 1)."""
    if entry is not None and entry.status == 1:
        return (
            f"Case #{entry.id} is already COMPLETED. You are documenting against "
            "a closed case — confirm this is the correct patient/procedure."
        )
    return None


def _form_int(data, key, current, min_value=0):
    """
    Read an integer form/JSON field, falling back to `current` when absent.
    Raises ValueError on present-but-non-numeric input (400 at the route).
    """
    raw = data.get(key)
    if raw is None or raw == "":
        return current or 0
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise ValueError(f"Field '{key}' must be a whole number.") from None
    if value < min_value:
        raise ValueError(f"Field '{key}' cannot be negative.")
    return value


def _validate_aldrete(data, note):
    """Parse the five Aldrete components; each must be an integer 0-2."""
    components = {}
    for field in (
        "aldrete_activity",
        "aldrete_respiration",
        "aldrete_circulation",
        "aldrete_consciousness",
        "aldrete_spo2",
    ):
        current = getattr(note, field, None)
        raw = data.get(field)
        if raw is None or raw == "":
            components[field] = current or 0
            continue
        try:
            value = int(raw)
        except (TypeError, ValueError):
            raise ValueError(
                f"Aldrete component '{field}' must be a whole number 0-2."
            ) from None
        if not 0 <= value <= 2:
            raise ValueError(
                f"Aldrete component '{field}' must be between 0 and 2 (got {value})."
            )
        components[field] = value
    return components


# ---------------------------------------------------------------------------
# Surgical Workbench (Console)
# ---------------------------------------------------------------------------
@bp.route("/workbench", defaults={"entry_id": None}, methods=["GET"])
@bp.route("/workbench/<int:entry_id>", methods=["GET"])
@login_required
@roles_required(*THEATRE_ROLES)
def surgical_workbench(entry_id):
    """
    Unified Surgical Workbench console for a theatre list booking entry.
    Displays patient clinical summary, encounter stage, WHO checklist status,
    anaesthetic record, post-op note, and instrument count.

    P0-10: Aborts 404 if no theatre entry or patient found — never substitutes
    the first patient in the database. Without an explicit entry_id it no
    longer silently opens the most recent case (that fallback cross-patient
    documented); it redirects to the OR dashboard instead.
    """
    if entry_id is None and "entry_id" not in request.args and "id" not in request.args:
        flash(
            "Select a case from the OR dashboard to open its workbench.",
            "info",
        )
        return redirect(url_for("theatre.get_or_dashboard_ui"))

    entry = _resolve_entry(entry_id, strict=True)
    if not entry:
        abort(404)

    patient = Patient.query.filter_by(patient_id=entry.patient_id).first()
    if not patient:
        # P0-10: Never substitute an unrelated patient for a clinical workstation.
        abort(404)

    checklist = WhoSurgicalChecklist.query.filter_by(theatre_entry_id=entry.id).first()
    anaesthetic = AnaestheticRecord.query.filter_by(theatre_entry_id=entry.id).first()
    postop = PostOpNote.query.filter_by(theatre_entry_id=entry.id).first()
    instruments = SurgicalInstrumentCount.query.filter_by(
        theatre_entry_id=entry.id
    ).first()

    return render_template(
        "theatre/workbench.html",
        entry=entry,
        patient=patient,
        checklist=checklist,
        anaesthetic=anaesthetic,
        postop=postop,
        instruments=instruments,
        entry_needs_attention=entry.status == 1,
    )


# ---------------------------------------------------------------------------
# WHO Surgical Safety Checklist
# ---------------------------------------------------------------------------
@bp.route("/", methods=["GET"])
@login_required
@roles_required(*THEATRE_ROLES)
def theatre_index():
    """Root /theatre/ redirect to OR dashboard."""
    return redirect(url_for("theatre.get_or_dashboard_ui"))


@bp.route("/checklist", defaults={"entry_id": None}, methods=["GET", "POST"])
@bp.route("/checklist/<int:entry_id>", methods=["GET", "POST"])
@login_required
@roles_required(*THEATRE_ROLES)
def who_checklist(entry_id):
    """GET/POST WHO 3-stage Surgical Safety Checklist.

    Requires an explicit entry_id on both GET and POST: a checklist is a
    medico-legal patient-safety record and must never be written to "whatever
    case happens to be newest".
    """
    if entry_id is None:
        if request.method == "POST":
            abort(400, description="entry_id is required to record a WHO checklist.")
        flash("Select a case to open its WHO Surgical Safety Checklist.", "info")
        return redirect(url_for("theatre.get_or_dashboard_ui"))

    entry = _resolve_entry(entry_id, strict=True)
    if not entry:
        abort(404)
    entry_id = entry.id

    checklist = WhoSurgicalChecklist.query.filter_by(theatre_entry_id=entry_id).first()

    if request.method == "POST":
        # Create-on-POST only: opening the form must not fabricate a document.
        if not checklist:
            checklist = WhoSurgicalChecklist(
                theatre_entry_id=entry.id,
                patient_id=entry.patient_id,
                encounter_id=entry.encounter_id,
            )
            db.session.add(checklist)

        data = request.get_json(silent=True) or request.form
        stage = data.get("stage", "sign_in")  # sign_in, time_out, sign_out
        if stage not in ("sign_in", "time_out", "sign_out"):
            abort(400, description=f"Unknown WHO checklist stage '{stage}'.")

        user_id = _current_user_id()

        def _bool(key):
            return str(data.get(key, "false")).lower() in ("true", "1", "on")

        if stage == "sign_in":
            checklist.patient_identity_confirmed = _bool("patient_identity_confirmed")
            checklist.site_marked = _bool("site_marked")
            checklist.anaesthesia_safety_check_completed = _bool(
                "anaesthesia_safety_check_completed"
            )
            checklist.pulse_oximeter_functioning = _bool("pulse_oximeter_functioning")
            checklist.known_allergy = _bool("known_allergy")
            checklist.allergy_details = data.get("allergy_details")
            checklist.difficult_airway_risk = _bool("difficult_airway_risk")
            checklist.airway_equipment_available = _bool("airway_equipment_available")
            checklist.blood_loss_risk_over_500ml = _bool("blood_loss_risk_over_500ml")
            checklist.iv_access_and_fluids_planned = _bool("iv_access_and_fluids_planned")
            checklist.sign_in_completed = True
            checklist.sign_in_completed_by = user_id
            checklist.sign_in_completed_at = datetime.now(timezone.utc)

        elif stage == "time_out":
            checklist.team_members_introduced = _bool("team_members_introduced")
            checklist.verbal_confirm_patient_site_procedure = _bool(
                "verbal_confirm_patient_site_procedure"
            )
            checklist.critical_steps_surgeon_reviewed = _bool(
                "critical_steps_surgeon_reviewed"
            )
            checklist.critical_steps_anaesthetist_reviewed = _bool(
                "critical_steps_anaesthetist_reviewed"
            )
            checklist.critical_steps_nursing_reviewed = _bool(
                "critical_steps_nursing_reviewed"
            )
            checklist.antibiotic_prophylaxis_given = _bool("antibiotic_prophylaxis_given")
            checklist.antibiotic_given_within_60min = _bool("antibiotic_given_within_60min")
            checklist.essential_imaging_displayed = _bool("essential_imaging_displayed")
            checklist.time_out_completed = True
            checklist.time_out_completed_by = user_id
            checklist.time_out_completed_at = datetime.now(timezone.utc)

        elif stage == "sign_out":
            checklist.procedure_name_recorded = data.get(
                "procedure_name_recorded",
                entry.procedure.name if entry.procedure else "",
            )
            checklist.instrument_sponge_needle_counts_correct = _bool(
                "instrument_sponge_needle_counts_correct"
            )
            checklist.specimens_labelled = _bool("specimens_labelled")
            checklist.equipment_problems_addressed = _bool("equipment_problems_addressed")
            checklist.key_recovery_concerns_reviewed = data.get(
                "key_recovery_concerns_reviewed"
            )
            checklist.sign_out_completed = True
            checklist.sign_out_completed_by = user_id
            checklist.sign_out_completed_at = datetime.now(timezone.utc)

        db.session.commit()

        if request.is_json:
            return jsonify(
                {
                    "status": "success",
                    "message": f"WHO Checklist stage '{stage}' recorded.",
                    "checklist_id": checklist.id,
                    "is_fully_completed": checklist.is_fully_completed(),
                }
            )

        flash(f"WHO Checklist stage '{stage}' updated successfully!", "success")
        return redirect(url_for("theatre.surgical_workbench", entry_id=entry_id))

    # GET response — render HTML UI (no auto-commit on GET)
    return render_template(
        "theatre/who_checklist.html",
        entry=entry,
        checklist=checklist,
        entry_status_warning=_entry_status_warning(entry),
    )


# ---------------------------------------------------------------------------
# Anaesthetic Record
# ---------------------------------------------------------------------------
@bp.route("/anaesthetic", defaults={"entry_id": None}, methods=["GET", "POST"])
@bp.route("/anaesthetic/<int:entry_id>", methods=["GET", "POST"])
@login_required
@roles_required(*THEATRE_ROLES)
def anaesthetic_record(entry_id):
    """GET/POST Anaesthetic Record & Intraoperative Vitals.

    Requires an explicit entry_id; GET never auto-creates the record.
    """
    if entry_id is None:
        if request.method == "POST":
            abort(
                400, description="entry_id is required to save an anaesthetic record."
            )
        flash("Select a case to open its Anaesthetic Record.", "info")
        return redirect(url_for("theatre.get_or_dashboard_ui"))

    entry = _resolve_entry(entry_id, strict=True)
    if not entry:
        abort(404)
    entry_id = entry.id

    record = AnaestheticRecord.query.filter_by(theatre_entry_id=entry_id).first()

    if request.method == "POST":
        # Create-on-POST only.
        if not record:
            record = AnaestheticRecord(
                theatre_entry_id=entry.id,
                patient_id=entry.patient_id,
                encounter_id=entry.encounter_id,
            )
            db.session.add(record)

        data = request.get_json(silent=True) or request.form
        warnings = []

        asa_status = (data.get("asa_status") or record.asa_status or "").strip()
        if not asa_status:
            abort(400, description="ASA classification is required.")
        try:
            # Fail closed on unknown codes (never silently "ASA I").
            TheatreOperationsEngine.evaluate_asa_score(asa_status)
        except ValueError as ve:
            abort(400, description=str(ve))
        record.asa_status = asa_status

        record.is_emergency = str(data.get("is_emergency", "false")).lower() in (
            "true",
            "1",
            "on",
        )
        record.mallampati_class = data.get("mallampati_class")
        record.airway_management = data.get("airway_management")
        record.ett_size = data.get("ett_size")
        record.technique = data.get("technique", record.technique or "General")

        if "agents_administered" in data:
            agents = data.get("agents_administered")
            if isinstance(agents, str):
                try:
                    agents = json.loads(agents)
                except Exception:  # noqa: BLE001
                    agents = [agents]
            record.agents_administered = agents

        if "vitals_series" in data:
            vitals = data.get("vitals_series")
            if isinstance(vitals, str):
                try:
                    vitals = json.loads(vitals)
                except Exception:  # noqa: BLE001
                    vitals = []
            if not isinstance(vitals, list):
                abort(400, description="vitals_series must be a JSON list.")
            record.vitals_series = vitals

        try:
            record.estimated_blood_loss_ml = _form_int(
                data, "estimated_blood_loss_ml", record.estimated_blood_loss_ml
            )
            record.crystalloids_ml = _form_int(
                data, "crystalloids_ml", record.crystalloids_ml
            )
            record.colloids_ml = _form_int(data, "colloids_ml", record.colloids_ml)
            record.blood_products_ml = _form_int(
                data, "blood_products_ml", record.blood_products_ml
            )
            record.urine_output_ml = _form_int(
                data, "urine_output_ml", record.urine_output_ml
            )
        except ValueError as ve:
            db.session.rollback()
            abort(400, description=str(ve))

        # Clinical-safety sanity: huge EBL deserves an explicit warning, not silence.
        if (record.estimated_blood_loss_ml or 0) > 5000:
            warnings.append(
                f"Estimated blood loss {record.estimated_blood_loss_ml} ml exceeds "
                "5,000 ml — ensure massive transfusion protocol was considered."
            )

        record.complications_notes = data.get("complications_notes")

        # Attribution: the clinician actually saving the record becomes the
        # anaesthetist of record (previously frozen at whoever GET'd first).
        anaesthetist_id = _current_user_id()
        if anaesthetist_id is not None:
            record.anaesthetist_id = anaesthetist_id

        db.session.commit()

        if request.is_json:
            payload = {
                "status": "success",
                "message": "Anaesthetic record updated.",
                "record_id": record.id,
                "asa_status": record.asa_status,
                "technique": record.technique,
                "anaesthetist_id": record.anaesthetist_id,
            }
            if warnings:
                payload["warnings"] = warnings
            return jsonify(payload)

        for w in warnings:
            flash(w, "warning")
        flash("Anaesthetic record saved!", "success")
        return redirect(url_for("theatre.surgical_workbench", entry_id=entry_id))

    return render_template(
        "theatre/anaesthetic_record.html",
        entry=entry,
        record=record,
        entry_status_warning=_entry_status_warning(entry),
    )


# ---------------------------------------------------------------------------
# Post-Operative Operative Note & PACU Aldrete Score
# ---------------------------------------------------------------------------
@bp.route("/postop", defaults={"entry_id": None}, methods=["GET", "POST"])
@bp.route("/postop/<int:entry_id>", methods=["GET", "POST"])
@login_required
@roles_required(*THEATRE_ROLES)
def postop_note(entry_id):
    """GET/POST Operative Note & PACU Aldrete Recovery Score.

    Requires an explicit entry_id; GET never auto-creates the note (no more
    fabricated "Uneventful procedure." attributed to whoever clicked a link).
    Completing the case is gated on the WHO Sign Out + reconciled counts +
    Aldrete >= 9.
    """
    if entry_id is None:
        if request.method == "POST":
            abort(400, description="entry_id is required to save a post-op note.")
        flash("Select a case to open its Post-Operative Note.", "info")
        return redirect(url_for("theatre.get_or_dashboard_ui"))

    entry = _resolve_entry(entry_id, strict=True)
    if not entry:
        abort(404)
    entry_id = entry.id

    note = PostOpNote.query.filter_by(theatre_entry_id=entry_id).first()

    if request.method == "POST":
        data = request.get_json(silent=True) or request.form

        if not note:
            # Create-on-POST only, and every operative finding comes from the
            # clinician — never a fabricated "Uneventful procedure."
            preop = (data.get("preop_diagnosis") or "").strip()
            postop_diag = (data.get("postop_diagnosis") or "").strip()
            procedure = (data.get("procedure_performed") or "").strip()
            findings = (data.get("surgical_findings") or "").strip()
            if not all([preop, postop_diag, procedure, findings]):
                abort(
                    400,
                    description=(
                        "preop_diagnosis, postop_diagnosis, procedure_performed and "
                        "surgical_findings are all required for a new post-op note."
                    ),
                )
            note = PostOpNote(
                theatre_entry_id=entry.id,
                patient_id=entry.patient_id,
                encounter_id=entry.encounter_id,
                preop_diagnosis=preop,
                postop_diagnosis=postop_diag,
                procedure_performed=procedure,
                surgical_findings=findings,
            )
            db.session.add(note)

        note.preop_diagnosis = data.get("preop_diagnosis", note.preop_diagnosis)
        note.postop_diagnosis = data.get("postop_diagnosis", note.postop_diagnosis)
        note.procedure_performed = data.get(
            "procedure_performed", note.procedure_performed
        )
        note.surgical_findings = data.get("surgical_findings", note.surgical_findings)
        note.specimens_taken = data.get("specimens_taken")
        note.implants_inserted = data.get("implants_inserted")
        note.drains_tubes_placed = data.get("drains_tubes_placed")
        note.postop_instructions = data.get("postop_instructions")
        note.discharge_to = data.get("discharge_to", note.discharge_to or "PACU")

        # Aldrete scoring components — validated 0-2 each.
        try:
            aldrete = _validate_aldrete(data, note)
        except ValueError as ve:
            db.session.rollback()
            abort(400, description=str(ve))
        for field, value in aldrete.items():
            setattr(note, field, value)

        # Attribution: the clinician signing the note is the surgeon of record.
        surgeon_id = _current_user_id()
        if surgeon_id is not None:
            note.surgeon_id = surgeon_id

        # Persist the note FIRST — operative documentation must survive even
        # when the safety gates below block case completion.
        db.session.commit()

        # ---- Case-completion safety gate --------------------------------
        # Marking the case completed requires:
        #   1. WHO Sign Out completed (which itself requires sign-in/time-out),
        #   2. instrument/sponge/needle counts reconciled cleanly,
        #   3. Aldrete >= 9 (PACU discharge threshold).
        complete_case = str(data.get("complete_case", "false")).lower() in (
            "true",
            "1",
            "on",
        )
        gate_blockers = []
        if complete_case and entry.status != 1:
            gate_pass, gate_msg = TheatreOperationsEngine.evaluate_who_checklist_gate(
                entry.id, "POST_OP"
            )
            if not gate_pass:
                gate_blockers.append(gate_msg)
            if (note.total_aldrete_score or 0) < 9:
                gate_blockers.append(
                    f"Aldrete score {note.total_aldrete_score or 0}/10 is below the "
                    "PACU discharge threshold of 9 — patient is not stable enough "
                    "for the case to be closed."
                )

        if gate_blockers:
            if request.is_json:
                return jsonify(
                    {
                        "status": "gate_blocked",
                        "message": "Post-op note saved but the case was NOT marked "
                        "completed — safety gates failed.",
                        "note_id": note.id,
                        "total_aldrete_score": note.total_aldrete_score,
                        "gate_blockers": gate_blockers,
                    }
                ), 409
            for blocker in gate_blockers:
                flash(f"CASE NOT COMPLETED: {blocker}", "danger")
            flash(
                "Post-op note saved — resolve the gates above to complete the case.",
                "warning",
            )
            return redirect(url_for("theatre.surgical_workbench", entry_id=entry_id))

        if complete_case and entry.status != 1:
            entry.status = 1  # Completed — all gates passed
            entry.notes_on_post_op = note.postop_instructions or note.surgical_findings

            if entry.encounter_id:
                from departments.medicine.inpatients import transition_surgical_stage

                try:
                    transition_surgical_stage(entry.encounter_id, "POST_OP")
                except ValueError:
                    pass

            db.session.commit()

        if request.is_json:
            return jsonify(
                {
                    "status": "success",
                    "message": "Post-op note & Aldrete score recorded.",
                    "note_id": note.id,
                    "total_aldrete_score": note.total_aldrete_score,
                    "fit_for_pacu_discharge": note.is_fit_for_pacu_discharge(),
                    "surgeon_id": note.surgeon_id,
                }
            )

        flash(
            f"Post-op note saved! Aldrete Score: {note.total_aldrete_score}/10",
            "success",
        )
        return redirect(url_for("theatre.surgical_workbench", entry_id=entry_id))

    return render_template(
        "theatre/postop_note.html",
        entry=entry,
        note=note,
        entry_status_warning=_entry_status_warning(entry),
    )


# ---------------------------------------------------------------------------
# Surgical Instrument & Sponge Count Reconciliation
# ---------------------------------------------------------------------------
@bp.route("/instruments", defaults={"entry_id": None}, methods=["GET", "POST"])
@bp.route("/instruments/<int:entry_id>", methods=["GET", "POST"])
@login_required
@roles_required(*THEATRE_ROLES)
def instrument_count(entry_id):
    """GET/POST Surgical Instrument, Sponge, and Needle Reconciliation.

    Requires an explicit entry_id; GET never auto-creates the count sheet.
    """
    if entry_id is None:
        if request.method == "POST":
            abort(
                400,
                description="entry_id is required to record instrument counts.",
            )
        flash("Select a case to open its Instrument Count sheet.", "info")
        return redirect(url_for("theatre.get_or_dashboard_ui"))

    entry = _resolve_entry(entry_id, strict=True)
    if not entry:
        abort(404)
    entry_id = entry.id

    counts = SurgicalInstrumentCount.query.filter_by(theatre_entry_id=entry_id).first()

    if request.method == "POST":
        # Create-on-POST only.
        if not counts:
            counts = SurgicalInstrumentCount(
                theatre_entry_id=entry.id,
                encounter_id=entry.encounter_id,
            )
            db.session.add(counts)

        data = request.get_json(silent=True) or request.form

        counts.tray_name_or_barcode = data.get(
            "tray_name_or_barcode", counts.tray_name_or_barcode
        )

        try:
            counts.sponges_initial = _form_int(
                data, "sponges_initial", counts.sponges_initial
            )
            counts.sponges_added = _form_int(
                data, "sponges_added", counts.sponges_added
            )
            counts.sponges_closing_cavity = _form_int(
                data, "sponges_closing_cavity", counts.sponges_closing_cavity
            )
            counts.sponges_closing_skin = _form_int(
                data, "sponges_closing_skin", counts.sponges_closing_skin
            )

            counts.needles_initial = _form_int(
                data, "needles_initial", counts.needles_initial
            )
            counts.needles_added = _form_int(
                data, "needles_added", counts.needles_added
            )
            counts.needles_closing_cavity = _form_int(
                data, "needles_closing_cavity", counts.needles_closing_cavity
            )
            counts.needles_closing_skin = _form_int(
                data, "needles_closing_skin", counts.needles_closing_skin
            )

            counts.instruments_initial = _form_int(
                data, "instruments_initial", counts.instruments_initial
            )
            counts.instruments_added = _form_int(
                data, "instruments_added", counts.instruments_added
            )
            counts.instruments_closing_cavity = _form_int(
                data, "instruments_closing_cavity", counts.instruments_closing_cavity
            )
            counts.instruments_closing_skin = _form_int(
                data, "instruments_closing_skin", counts.instruments_closing_skin
            )
        except ValueError as ve:
            db.session.rollback()
            abort(400, description=str(ve))

        counts.notes = data.get("notes")
        counts.xray_ordered_for_discrepancy = str(
            data.get("xray_ordered_for_discrepancy", "false")
        ).lower() in ("true", "1", "on")

        # Attribution: the scrub nurse actually saving the count.
        scrub_id = _current_user_id()
        if scrub_id is not None:
            counts.scrub_nurse_id = scrub_id

        reconcile_result = counts.calculate_reconciliation()
        db.session.commit()

        if request.is_json:
            return jsonify(
                {
                    "status": "success"
                    if reconcile_result["reconciled"]
                    else "warning_discrepancy",
                    "message": "Instrument count reconciled cleanly."
                    if reconcile_result["reconciled"]
                    else "COUNT DISCREPANCY DETECTED! Intraoperative X-Ray required.",
                    "count_id": counts.id,
                    "reconciliation": reconcile_result,
                }
            )

        if reconcile_result["reconciled"]:
            flash("Surgical instrument count reconciled successfully!", "success")
        else:
            flash(
                "ALERT: Instrument count discrepancy! Verify cavity and order intraoperative X-Ray.",
                "danger",
            )

        return redirect(url_for("theatre.surgical_workbench", entry_id=entry_id))

    reconcile_result = counts.calculate_reconciliation() if counts else None
    return render_template(
        "theatre/instrument_count.html",
        entry=entry,
        counts=counts,
        reconcile_result=reconcile_result,
        entry_status_warning=_entry_status_warning(entry),
    )


# ---------------------------------------------------------------------------
# Intraoperative Vitals, Fluid Balance, PACU Readiness & OR Flow Console
# ---------------------------------------------------------------------------
@bp.route("/api/vitals/<int:entry_id>", methods=["POST"])
@login_required
@roles_required(*THEATRE_ROLES)
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
        return jsonify(
            {
                "status": "success",
                "message": "Intraoperative vital snapshot recorded.",
                "record_id": record.id,
                "vitals_count": len(record.vitals_series),
            }
        ), 200
    except ValueError as ve:
        # Implausible/invalid vitals or unknown entry — explicit, actionable 400.
        return jsonify({"error": str(ve)}), 400
    except Exception:  # noqa: BLE001
        logger.exception("Failed to record intraop vitals for entry %s", entry_id)
        return jsonify({"error": "Failed to record intraoperative vitals."}), 500


@bp.route("/api/fluid-balance/<int:entry_id>", methods=["GET"])
@login_required
@roles_required(*THEATRE_ROLES)
def get_fluid_balance_api(entry_id):
    """GET endpoint returning intraoperative fluid intake vs. loss calculation."""
    balance = TheatreOperationsEngine.calculate_fluid_balance(entry_id)
    return jsonify(balance), 200


@bp.route("/api/pacu-readiness/<int:entry_id>", methods=["GET"])
@login_required
@roles_required(*THEATRE_ROLES)
def get_pacu_readiness_api(entry_id):
    """GET endpoint returning PACU Aldrete Recovery Score & discharge readiness status."""
    readiness = TheatreOperationsEngine.evaluate_pacu_discharge_readiness(entry_id)
    return jsonify(readiness), 200


@bp.route("/dashboard", methods=["GET"])
@login_required
@roles_required(*THEATRE_ROLES)
def get_or_dashboard_ui():
    """Render real-time Operating Theatre & OR Flow Console dashboard UI."""
    metrics = TheatreOperationsEngine.get_or_dashboard_metrics()
    return render_template("theatre/or_dashboard.html", metrics=metrics)


@bp.route("/api/metrics", methods=["GET"])
@login_required
@roles_required(*THEATRE_ROLES)
def get_or_metrics_api():
    """JSON API endpoint returning Operating Theatre KPIs and active case list."""
    metrics = TheatreOperationsEngine.get_or_dashboard_metrics()
    return jsonify(metrics), 200


# ---------------------------------------------------------------------------
# Anesthesia Timeline Matrix & ASA Risk Assessment API
# ---------------------------------------------------------------------------
@bp.route("/api/anesthesia-timeline/<int:entry_id>", methods=["GET", "POST"])
@login_required
@roles_required(*THEATRE_ROLES)
def anesthesia_timeline_api(entry_id):
    """GET timeline matrix / POST new timestamped anesthesia timeline event."""
    if request.method == "POST":
        data = request.get_json() or {}
        event_type = data.get("event_type", "MAINTENANCE")
        notes = data.get("notes")
        user_name = (
            current_user.username if hasattr(current_user, "username") else "Clinician"
        )

        try:
            event_obj = TheatreOperationsEngine.record_timeline_event(
                entry_id=entry_id,
                event_type=event_type,
                notes=notes,
                recorded_by=user_name,
            )
            return jsonify(
                {
                    "status": "success",
                    "message": f"Anesthesia timeline event '{event_type}' recorded.",
                    "event": event_obj,
                }
            ), 201
        except ValueError as ve:
            # Invalid event type — explicit 400 instead of silent MAINTENANCE.
            return jsonify({"error": str(ve)}), 400
        except Exception:  # noqa: BLE001
            logger.exception("Failed to record timeline event for entry %s", entry_id)
            return jsonify({"error": "Failed to record timeline event."}), 500

    matrix = TheatreOperationsEngine.get_anesthesia_timeline_matrix(entry_id)
    return jsonify(matrix), 200


@bp.route("/api/asa-assessment/<asa_code>", methods=["GET"])
@login_required
@roles_required(*THEATRE_ROLES)
def get_asa_assessment_api(asa_code):
    """GET ASA physical status definition & risk grade."""
    is_emerg = str(request.args.get("emergency", "false")).lower() in ("true", "1")
    try:
        eval_result = TheatreOperationsEngine.evaluate_asa_score(
            asa_code, is_emergency=is_emerg
        )
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    return jsonify(eval_result), 200


@bp.route("/api/or-schedule", methods=["POST"])
@login_required
@roles_required(*THEATRE_ROLES)
def update_or_schedule_api():
    """POST endpoint to allocate/schedule OR room with conflict detection."""
    data = request.get_json() or {}

    entry_id = data.get("entry_id")
    or_room = (data.get("or_room") or "OR 1").strip()
    start_str = data.get("scheduled_start_time")

    if not entry_id:
        return jsonify({"error": "entry_id is required."}), 400

    if or_room not in OR_ROOMS:
        return jsonify(
            {
                "error": f"Unknown operating room '{or_room}'. "
                f"Valid rooms: {list(OR_ROOMS)}."
            }
        ), 400

    try:
        duration_min = int(data.get("estimated_duration_minutes", 120))
    except (TypeError, ValueError):
        return jsonify(
            {"error": "estimated_duration_minutes must be a whole number of minutes."}
        ), 400
    if duration_min <= 0 or duration_min > 1440:
        return jsonify(
            {
                "error": "estimated_duration_minutes must be between 1 and 1440 "
                "(24 hours)."
            }
        ), 400

    entry = db.session.get(TheatreList, entry_id)
    if not entry:
        return jsonify({"error": f"Theatre case {entry_id} not found."}), 404

    start_dt = None
    if start_str:
        try:
            start_dt = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
        except ValueError:
            return jsonify(
                {"error": "Invalid scheduled_start_time format ISO 8601 expected."}
            ), 400

    if start_dt:
        has_conflict, conflict_msg = TheatreOperationsEngine.check_room_schedule_conflict(
            or_room=or_room,
            start_time=start_dt,
            duration_minutes=duration_min,
            exclude_entry_id=entry.id,
        )
        if has_conflict:
            return jsonify(
                {
                    "status": "conflict_detected",
                    "error": conflict_msg,
                }
            ), 409

    entry.or_room = or_room
    if start_dt:
        entry.scheduled_start_time = start_dt
    entry.estimated_duration_minutes = duration_min

    db.session.commit()

    return jsonify(
        {
            "status": "success",
            "message": f"Case #{entry.id} assigned to {or_room}.",
            "entry_id": entry.id,
            "or_room": entry.or_room,
            "scheduled_start_time": entry.scheduled_start_time.isoformat()
            if entry.scheduled_start_time
            else None,
            "estimated_duration_minutes": entry.estimated_duration_minutes,
        }
    ), 200
