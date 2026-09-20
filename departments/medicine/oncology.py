import json
import os
from datetime import datetime, timezone

from flask import abort, flash, jsonify, redirect, render_template, request, url_for
from flask_login import current_user, login_required
from sqlalchemy.exc import IntegrityError

from departments.api.audit import log_audit_event
from departments.forms import OncologyNoteForm, OncoPatientForm, PatientSearchForm
from departments.medicine.chemotherapy_engine import (
    MAX_CYCLES,
    ChemoInputError,
    calculate_regimen_doses,
    normalize_bsa_formula,
    validate_biometrics,
)
from departments.models.compliance import has_ai_consent
from departments.models.laboratory import LabResultTemplate
from departments.models.medicine import (
    CancerStage,
    CancerType,
    CancerTypeStage,
    Disease,
    DiseaseLab,
    DiseaseManagementPlan,
    OncoDrugCategory,
    OncologyBooking,
    OncologyDrug,
    OncologyNote,
    OncologyRegimen,
    OncoPatient,
    RegimenCategory,
    SpecialWarning,
)
from departments.models.oncology_models import (
    CHEMO_ALLOWED_TRANSITIONS,
    CHEMO_STATUSES,
    ChemotherapyRegimenOrder,
)
from departments.models.records import Patient
from departments.nlp.chatbot import UniversalClinicalSummarizer
from departments.nlp.logging_setup import get_logger
from departments.rbac import has_any_role, roles_required
from extensions import db

from . import bp

logger = get_logger()

# Instantiate the summarizer for use in chatbot_interface

gemini_api_key = os.environ.get("GEMINI_API_KEY")
nvidia_api_key = os.environ.get("NVIDIA_API_KEY")
Summarizer = UniversalClinicalSummarizer(
    gemini_api_key=gemini_api_key, nvidia_api_key=nvidia_api_key
)


ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "pdf", "txt", "csv", "docx"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB max file size

# ---------------------------------------------------------------------------
# Access policy for the oncology module.
#   READ  : anyone who legitimately needs the oncology chart / chemo orders.
#   WRITE : clinicians who diagnose, document, prescribe or cancel.
# Reference-data pages (drugs, regimens, warnings, cancer types) stay open to
# any authenticated user; they contain no patient data.
# ---------------------------------------------------------------------------
CLINICAL_WRITE_ROLES = ("doctor", "medicine", "oncology", "admin")
CLINICAL_READ_ROLES = CLINICAL_WRITE_ROLES + ("nurse", "pharmacist")
SCHEDULING_ROLES = CLINICAL_WRITE_ROLES + ("nurse",)

# Who may move a chemotherapy order to which status.
CHEMO_STATUS_ROLES = {
    "PREPARED": CLINICAL_WRITE_ROLES + ("pharmacist",),
    "ADMINISTERED": CLINICAL_WRITE_ROLES + ("nurse",),
    "CANCELLED": CLINICAL_WRITE_ROLES,
}

MIN_TOXICITY_OVERRIDE_CHARS = 10
MIN_CANCEL_REASON_CHARS = 5
AI_SUMMARY_MAX_CHARS = 12000


def _find_patient(patient_id):
    """
    Exact lookup by patient number. Deliberately NOT fuzzy: a partial match on
    an id or name can silently resolve to a different patient than the one the
    caller was authorised / consented for.
    """
    patient_id = (patient_id or "").strip() if isinstance(patient_id, str) else ""
    if not patient_id:
        return None
    return Patient.query.filter_by(patient_id=patient_id).first()


def _commit_with_audit(action, resource_type, resource_id, details=None):
    """
    Commit pending changes together with an audit row.

    ``log_audit_event`` commits the session, and rolls it back if the audit row
    cannot be written. So a False return means BOTH the audit row and the
    pending change were discarded ("no audit trail, no change").
    """
    return (
        log_audit_event(
            action=action,
            resource_type=resource_type,
            resource_id=str(resource_id),
            details=details,
        )
        is not None
    )


def _json_error(code, message, status, **extra):
    payload = {"error": True, "code": code, "message": message}
    payload.update(extra)
    return jsonify(payload), status


def _utcnow_naive():
    """UTC now without tzinfo (matches the naive DateTime columns used here)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


@bp.route("/diseases/")
@login_required
def list_diseases():
    page = request.args.get("page", default=1, type=int)
    per_page = 10
    diseases = Disease.query.paginate(page=page, per_page=per_page, error_out=False)
    return render_template("medicine/diseases/index.html", diseases=diseases)


@bp.route("/diseases/<int:disease_id>")
@login_required
def view_disease(disease_id):
    disease = Disease.query.get_or_404(disease_id)
    return render_template("medicine/diseases/view_disease.html", disease=disease)


@bp.route("/diseases/add", methods=["GET", "POST"])
@login_required
@roles_required(*CLINICAL_WRITE_ROLES)
def add_disease():
    if request.method == "POST":
        name = request.form["name"].strip()
        cui = request.form["cui"].strip()
        description = request.form["description"].strip()
        management_plan_text = request.form["management_plan"].strip()

        # Step 1: Create and save the disease
        new_disease = Disease(name=name, cui=cui, description=description)
        db.session.add(new_disease)
        db.session.flush()  # Get ID before final commit

        # Step 2: Save the management plan
        plan = DiseaseManagementPlan(
            disease_id=new_disease.id, plan=management_plan_text
        )
        db.session.add(plan)

        # Step 3: Handle lab tests
        lab_test_names = request.form.getlist("lab_test_name")
        lab_test_descriptions = request.form.getlist("lab_test_description")

        for idx, test_name in enumerate(lab_test_names):
            test_name = test_name.strip()
            if not test_name:
                continue  # Skip empty entries

            test_desc = (
                lab_test_descriptions[idx].strip()
                if idx < len(lab_test_descriptions)
                else ""
            )

            lab_test = DiseaseLab(
                disease_id=new_disease.id, lab_test=test_name, description=test_desc
            )
            db.session.add(lab_test)

        # Step 4: Commit all changes
        db.session.commit()

        return redirect(url_for("medicine.list_diseases"))

    return render_template("medicine/diseases/add_disease.html")


@bp.route("/diseases/edit/<int:disease_id>", methods=["GET", "POST"])
@login_required
@roles_required(*CLINICAL_WRITE_ROLES)
def edit_disease(disease_id):
    disease = Disease.query.get_or_404(disease_id)
    plan = disease.management_plan

    # Fetch existing lab tests
    lab_tests = disease.lab_tests

    if request.method == "POST":
        disease.name = request.form["name"].strip()
        disease.cui = request.form["cui"].strip()
        disease.description = request.form["description"].strip()
        plan.plan = request.form["management_plan"].strip()

        # Handle lab tests
        lab_test_ids = request.form.getlist("lab_test_id")
        lab_test_names = request.form.getlist("lab_test_name")
        lab_test_descriptions = request.form.getlist("lab_test_description")

        [t.id for t in lab_tests]

        for idx, test_id in enumerate(lab_test_ids):
            name = lab_test_names[idx]
            desc = lab_test_descriptions[idx]

            if test_id == "new":
                # Add new lab test
                new_test = DiseaseLab(
                    disease_id=disease.id, lab_test=name, description=desc
                )
                db.session.add(new_test)
            else:
                # Update existing lab test
                test = db.session.get(DiseaseLab, int(test_id))
                if test:
                    test.lab_test = name
                    test.description = desc

        # Detect deleted tests
        submitted_ids = {int(i) for i in lab_test_ids if i != "new"}
        for test in lab_tests:
            if test.id not in submitted_ids:
                db.session.delete(test)

        db.session.commit()
        flash("Disease and lab tests updated successfully.", "success")
        return redirect(url_for("medicine.edit_disease", disease_id=disease.id))

    return render_template(
        "medicine/diseases/edit_disease.html",
        disease=disease,
        plan=plan,
        lab_tests=lab_tests,
    )


@bp.route("/diseases/delete/<int:disease_id>", methods=["POST"])
@login_required
@roles_required("admin")
def delete_disease(disease_id):
    """
    P0-09: Disease deletion requires admin role and POST method.
    Clinical reference data should be managed carefully.
    """
    disease = Disease.query.get_or_404(disease_id)
    logger.info(
        "Disease deleted: disease_id=%s disease_name=%s actor_id=%s",
        disease_id,
        disease.name,
        current_user.id,
    )
    db.session.delete(disease)
    db.session.commit()
    flash(f"Disease '{disease.name}' deleted.", "success")
    return redirect(url_for("medicine.list_diseases"))


@bp.route("/oncology", methods=["GET", "POST"])
@login_required
@roles_required(*CLINICAL_READ_ROLES)
def oncology():
    search_form = PatientSearchForm()
    selected_patient = None
    bookings = []

    if search_form.validate_on_submit() and search_form.submit_search.data:
        patient_id = search_form.patient_id.data
        selected_patient = Patient.query.filter_by(patient_id=patient_id).first_or_404()
        bookings = OncologyBooking.query.filter_by(
            patient_id=selected_patient.patient_id
        ).all()

        if not bookings:
            flash("No oncology bookings found for this patient.", "info")

        # Redirect to encounter route
        return redirect(
            url_for(
                "medicine.oncology_encounter", patient_id=selected_patient.patient_id
            )
        )

    return render_template(
        "medicine/oncology/index.html",
        form=search_form,
        selected_patient=selected_patient,
        bookings=bookings,
    )


def _oncology_notes_for(patient_id):
    """All notes for the chart, newest first. Voided notes are kept (and flagged in the UI)."""
    return (
        OncologyNote.query.filter_by(patient_id=patient_id)
        .order_by(OncologyNote.note_date.desc(), OncologyNote.id.desc())
        .all()
    )


def _save_oncology_details(onco_form, selected_patient, onco_patient):
    """
    Persist the oncology details form. Returns (ok, onco_patient).

    The form posts CancerType / CancerStage *ids*; the OncoPatient columns hold
    the human-readable *names* (same as the /oncology/add path), so resolve them.
    """
    cancer_type = db.session.get(CancerType, onco_form.cancer_type.data)
    stage = db.session.get(CancerStage, onco_form.stage.data)
    if not cancer_type or not stage:
        flash("Selected cancer type or stage does not exist.", "danger")
        return False, onco_patient

    # Server-side check that the stage belongs to the cancer type (the
    # client-side filter is only a convenience). Types with no stage links
    # defined yet accept any stage.
    linked_stage_ids = {
        link.cancer_stage_id
        for link in CancerTypeStage.query.filter_by(cancer_type_id=cancer_type.id)
    }
    if linked_stage_ids and stage.id not in linked_stage_ids:
        flash(f"Stage '{stage.label}' is not valid for {cancer_type.name}.", "danger")
        return False, onco_patient

    if onco_patient:
        action = "ONCOLOGY_RECORD_UPDATED"
        onco_patient.diagnosis = onco_form.diagnosis.data
        onco_patient.diagnosis_date = onco_form.diagnosis_date.data
        onco_patient.cancer_type = cancer_type.name
        onco_patient.stage = stage.label
        onco_patient.status = onco_form.status.data
    else:
        action = "ONCOLOGY_ENROLLED"
        onco_patient = OncoPatient(
            patient_id=selected_patient.patient_id,
            diagnosis=onco_form.diagnosis.data,
            diagnosis_date=onco_form.diagnosis_date.data,
            cancer_type=cancer_type.name,
            stage=stage.label,
            status=onco_form.status.data,
            date_enrolled=datetime.now(timezone.utc),
        )
        db.session.add(onco_patient)
    db.session.flush()

    if not _commit_with_audit(
        action,
        "OncoPatient",
        onco_patient.id,
        {
            "patient_id": selected_patient.patient_id,
            "cancer_type": cancer_type.name,
            "stage": stage.label,
            "status": onco_form.status.data,
        },
    ):
        flash("Audit trail unavailable; oncology details were NOT saved.", "danger")
        return False, None
    return True, onco_patient


@bp.route("/oncology/encounter/<patient_id>", methods=["GET", "POST"])
@login_required
@roles_required(*CLINICAL_READ_ROLES)
def oncology_encounter(patient_id):
    # Fetch patient
    selected_patient = Patient.query.filter_by(patient_id=patient_id).first_or_404()

    # Only clinicians may change the oncology chart; nurses/pharmacists read.
    if request.method == "POST" and not has_any_role(*CLINICAL_WRITE_ROLES):
        abort(403)

    # Add age attribute based on date_of_birth
    today = datetime.now(timezone.utc).date()
    dob = selected_patient.date_of_birth
    selected_patient.age = (
        today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    )

    # Initialize forms
    search_form = PatientSearchForm(patient_id=selected_patient.patient_id)
    onco_form = OncoPatientForm()
    note_form = OncologyNoteForm()

    # Fetch existing oncology record. OncoPatient.patient_id is the patient
    # NUMBER (FK to patients.patient_id), not the integer primary key.
    onco_patient = (
        OncoPatient.query.filter_by(patient_id=selected_patient.patient_id)
        .order_by(OncoPatient.id.desc())
        .first()
    )
    bookings = OncologyBooking.query.filter_by(
        patient_id=selected_patient.patient_id
    ).all()

    # Prepopulate the form from the stored record on GET ONLY. Doing it on POST
    # would overwrite the submitted values before they are read, silently
    # discarding every edit while still reporting success.
    if request.method == "GET":
        if onco_patient:
            onco_form.diagnosis.data = onco_patient.diagnosis
            onco_form.diagnosis_date.data = onco_patient.diagnosis_date
            stored_type = CancerType.query.filter_by(
                name=onco_patient.cancer_type
            ).first()
            stored_stage = CancerStage.query.filter_by(label=onco_patient.stage).first()
            if stored_type:
                onco_form.cancer_type.data = stored_type.id
            if stored_stage:
                onco_form.stage.data = stored_stage.id
            onco_form.status.data = onco_patient.status
        log_audit_event(
            action="ONCOLOGY_ENCOUNTER_VIEW",
            resource_type="Patient",
            resource_id=selected_patient.patient_id,
        )

    encounter_url = url_for(
        "medicine.oncology_encounter", patient_id=selected_patient.patient_id
    )

    # Handle oncology form submission
    if onco_form.submit_update.data and onco_form.validate_on_submit():
        ok, onco_patient = _save_oncology_details(
            onco_form, selected_patient, onco_patient
        )
        if ok:
            flash("Oncology patient details saved successfully.", "success")
            return redirect(encounter_url)  # PRG: refresh must not resubmit

    # Handle note form submission
    if note_form.submit_note.data and note_form.validate_on_submit():
        new_note = OncologyNote(
            patient_id=selected_patient.patient_id,
            note_date=note_form.note_date.data,
            note_content=note_form.note_content.data,
        )
        db.session.add(new_note)
        db.session.flush()
        if _commit_with_audit(
            "ONCOLOGY_NOTE_CREATED",
            "OncologyNote",
            new_note.id,
            {"patient_id": selected_patient.patient_id},
        ):
            flash("Oncology note added successfully.", "success")
            return redirect(encounter_url)  # PRG: refresh must not duplicate the note
        flash("Audit trail unavailable; note was NOT saved.", "danger")

    return render_template(
        "medicine/oncology/encounter.html",
        search_form=search_form,
        onco_form=onco_form,
        note_form=note_form,
        selected_patient=selected_patient,
        onco_patient=onco_patient,
        bookings=bookings,
        notes=_oncology_notes_for(selected_patient.patient_id),
    )


@bp.route("/oncology/ai_summary/<patient_id>", methods=["GET", "POST"])
@login_required
@roles_required(*CLINICAL_READ_ROLES)
def oncology_ai_summary(patient_id):
    """Generates AI clinical summary for an oncology patient, gated by DPA 2019 AI consent."""
    try:
        if not has_ai_consent(patient_id):
            log_audit_event(
                action="AI_CONSENT_REFUSED",
                resource_type="Patient",
                resource_id=patient_id,
                details={
                    "feature": "oncology_ai_summary",
                    "reason": "Missing or revoked ai_diagnosis consent",
                },
            )
            return jsonify(
                {
                    "error": "AI-assisted summary unavailable: patient has not consented to AI processing of clinical notes.",
                    "code": "AI_CONSENT_REQUIRED",
                }
            ), 403
    except Exception as consent_err:  # noqa: BLE001
        logger.error(f"Error checking AI consent for oncology summary: {consent_err}")
        return jsonify(
            {
                "error": "Unable to verify patient AI consent status.",
                "code": "AI_CONSENT_ERROR",
            }
        ), 500

    # Consent was verified for EXACTLY this patient number, so the notes sent to
    # the model must belong to exactly that patient. (A fuzzy ILIKE lookup here
    # previously let consent for "P1" release the notes of "P10".)
    selected_patient = _find_patient(patient_id)
    if not selected_patient:
        return jsonify({"error": "Patient not found"}), 404

    # Voided notes are, by definition, entered in error: never feed them to the model.
    notes = (
        OncologyNote.query.filter_by(
            patient_id=selected_patient.patient_id, is_voided=False
        )
        .order_by(OncologyNote.note_date.desc(), OncologyNote.id.desc())
        .all()
    )
    chunks, used = [], 0
    for note in notes:  # newest first, bounded prompt size
        if not note.note_content:
            continue
        if used + len(note.note_content) > AI_SUMMARY_MAX_CHARS and chunks:
            break
        chunks.append(note.note_content[:AI_SUMMARY_MAX_CHARS])
        used += len(note.note_content)
    chunks.reverse()  # chronological for the model
    notes_text = " ".join(chunks) or "Patient enrolled in oncology care."

    # Record the disclosure to the external model BEFORE sending it.
    if not _commit_with_audit(
        "ONCOLOGY_AI_SUMMARY_GENERATED",
        "Patient",
        selected_patient.patient_id,
        {"notes_included": len(chunks)},
    ):
        return jsonify(
            {
                "error": "Audit trail unavailable; AI summary not generated.",
                "code": "AUDIT_UNAVAILABLE",
            }
        ), 503

    summary = Summarizer.answer(notes_text)
    return jsonify({"patient_id": selected_patient.patient_id, "summary": summary})


@bp.route("/oncology/add", methods=["GET", "POST"])
@login_required
@roles_required(*CLINICAL_WRITE_ROLES)
def add_onco_patient():
    if request.method == "POST":
        patient_number = (request.form.get("patient_id") or "").strip()
        diagnosis = (request.form.get("diagnosis") or "").strip()
        cancer_type = (request.form.get("cancer_type") or "").strip()
        stage = (request.form.get("stage") or "").strip()
        diagnosis_date_raw = (request.form.get("diagnosis_date") or "").strip()

        if not all([patient_number, diagnosis, cancer_type, stage, diagnosis_date_raw]):
            flash(
                "Patient, diagnosis, cancer type, stage and date are required.",
                "danger",
            )
            return redirect(url_for("medicine.add_onco_patient"))
        if len(diagnosis) > 200 or len(cancer_type) > 100 or len(stage) > 100:
            flash("Diagnosis, cancer type or stage is too long.", "danger")
            return redirect(url_for("medicine.add_onco_patient"))

        patient = _find_patient(patient_number)
        if not patient:
            flash("Selected patient does not exist.", "danger")
            return redirect(url_for("medicine.add_onco_patient"))

        try:
            diagnosis_date = datetime.strptime(diagnosis_date_raw, "%Y-%m-%d").date()  # noqa: DTZ007
        except ValueError:
            flash("Invalid diagnosis date. Use YYYY-MM-DD.", "danger")
            return redirect(url_for("medicine.add_onco_patient"))
        if diagnosis_date > datetime.now(timezone.utc).date():
            flash("Diagnosis date cannot be in the future.", "danger")
            return redirect(url_for("medicine.add_onco_patient"))

        if OncoPatient.query.filter_by(
            patient_id=patient.patient_id, status="Active"
        ).first():
            flash("Patient is already enrolled in oncology care.", "warning")
            return redirect(
                url_for("medicine.oncology_encounter", patient_id=patient.patient_id)
            )

        onco_patient = OncoPatient(
            patient_id=patient.patient_id,
            diagnosis=diagnosis,
            cancer_type=cancer_type,
            stage=stage,
            diagnosis_date=diagnosis_date,
        )
        db.session.add(onco_patient)
        db.session.flush()
        if not _commit_with_audit(
            "ONCOLOGY_ENROLLED",
            "OncoPatient",
            onco_patient.id,
            {"patient_id": patient.patient_id, "cancer_type": cancer_type},
        ):
            flash("Audit trail unavailable; enrolment was NOT saved.", "danger")
            return redirect(url_for("medicine.add_onco_patient"))
        flash("Patient enrolled in oncology care successfully.", "success")
        return redirect(url_for("medicine.oncology"))

    patients = Patient.query.all()
    return render_template("medicine/oncology/add_onco_patient.html", patients=patients)


@bp.route("/oncology/note/<int:note_id>/edit", methods=["GET", "POST"])
@login_required
@roles_required(*CLINICAL_WRITE_ROLES)
def edit_note(note_id):
    note = db.get_or_404(OncologyNote, note_id)
    patient = Patient.query.filter_by(patient_id=note.patient_id).first_or_404()
    encounter_url = url_for(
        "medicine.oncology_encounter", patient_id=patient.patient_id
    )

    if note.is_voided:
        flash("A voided note cannot be edited.", "warning")
        return redirect(encounter_url)

    form = OncologyNoteForm()
    if form.validate_on_submit() and form.submit_note.data:
        # Amendments must not destroy history (SECURITY.md: immutable clinical
        # history). The append-only audit row carries the text being replaced.
        previous = {
            "patient_id": note.patient_id,
            "previous_note_date": note.note_date.isoformat(),
            "previous_content": note.note_content,
        }
        note.note_date = form.note_date.data
        note.note_content = form.note_content.data
        if not _commit_with_audit(
            "ONCOLOGY_NOTE_AMENDED", "OncologyNote", note.id, previous
        ):
            flash("Audit trail unavailable; note was NOT changed.", "danger")
            return redirect(encounter_url)
        flash("Oncology note updated successfully.", "success")
        return redirect(encounter_url)

    if request.method == "GET":
        form.note_date.data = note.note_date
        form.note_content.data = note.note_content

    return render_template(
        "medicine/oncology/edit_note.html", form=form, note=note, patient=patient
    )


@bp.route("/get_stages/<int:type_id>")
@login_required
def get_stages(type_id):
    links = CancerTypeStage.query.filter_by(cancer_type_id=type_id).all()
    stages = [
        {"id": link.cancer_stage.id, "label": link.cancer_stage.label} for link in links
    ]
    return jsonify(stages)


@bp.route("/oncology/note/<int:note_id>/void", methods=["POST"])
@login_required
@roles_required(*CLINICAL_WRITE_ROLES)
def delete_note(note_id):
    """
    P0-09: Void (soft-delete) an oncology note — never physically delete.
    Original record is preserved for audit trail.
    Requires authentication, role, and a void reason.
    """
    note = db.get_or_404(OncologyNote, note_id)
    patient_id = note.patient_id
    encounter_url = url_for("medicine.oncology_encounter", patient_id=patient_id)

    payload = request.get_json(silent=True) or {}
    void_reason = (
        request.form.get("void_reason") or payload.get("void_reason") or ""
    ).strip()
    if len(void_reason) < 3:
        flash("A void reason is required to void a clinical note.", "danger")
        return redirect(encounter_url)

    if note.is_voided:
        flash("This note has already been voided.", "warning")
        return redirect(encounter_url)

    # Void — never physically delete. Preserve audit trail.
    note.is_voided = True
    note.voided_by = current_user.id
    note.voided_reason = void_reason[:500]
    note.voided_at = _utcnow_naive()
    if not _commit_with_audit(
        "ONCOLOGY_NOTE_VOIDED",
        "OncologyNote",
        note.id,
        {"patient_id": patient_id, "reason": void_reason[:500]},
    ):
        flash("Audit trail unavailable; note was NOT voided.", "danger")
        return redirect(encounter_url)

    logger.info(
        "Oncology note VOIDED: note_id=%s patient_id=%s actor_id=%s",
        note_id,
        patient_id,
        current_user.id,
    )

    flash(
        "Oncology note voided successfully. The original record is preserved.",
        "success",
    )
    return redirect(encounter_url)


@bp.route("/drugs/")
@login_required
def drugs():
    category_id = request.args.get("category_id", type=int)
    severity = request.args.get("severity", type=str)
    therapeutic_class = request.args.get("therapeutic_class", type=str)
    has_black_box = request.args.get("has_black_box", type=str)
    query = OncologyDrug.query
    if category_id:
        query = query.filter_by(category_id=category_id)
    if severity in ["Low", "Moderate", "High"]:
        query = query.join(SpecialWarning).filter(SpecialWarning.severity == severity)
    if therapeutic_class:
        query = query.filter_by(therapeutic_class=therapeutic_class)
    if has_black_box == "yes":
        query = query.filter(OncologyDrug.black_box_warning.isnot(None))
    elif has_black_box == "no":
        query = query.filter(OncologyDrug.black_box_warning.is_(None))
    drugs = query.all()
    categories = OncoDrugCategory.query.all()
    therapeutic_classes = (
        db.session.query(OncologyDrug.therapeutic_class).distinct().all()
    )
    therapeutic_classes = [tc[0] for tc in therapeutic_classes if tc[0]]
    return render_template(
        "medicine/oncology/drugs.html",
        drugs=drugs,
        categories=categories,
        therapeutic_classes=therapeutic_classes,
        selected_category=category_id,
        selected_severity=severity,
        selected_therapeutic_class=therapeutic_class,
        selected_has_black_box=has_black_box,
    )


@bp.route("/regimens/")
@login_required
def regimens():
    category_id = request.args.get("category_id", type=int)
    status = request.args.get("status", type=str)
    query = OncologyRegimen.query
    if category_id:
        query = query.filter_by(category_id=category_id)
    if status in ["Active", "Deprecated", "Under Review"]:
        query = query.filter_by(status=status)
    regimens = query.all()
    categories = RegimenCategory.query.all()
    return render_template(
        "medicine/oncology/regimens.html",
        regimens=regimens,
        categories=categories,
        selected_category=category_id,
        selected_status=status,
    )


@bp.route("/warnings/")
@login_required
def warnings():
    warning_type = request.args.get("warning_type", type=str)
    severity = request.args.get("severity", type=str)
    query = SpecialWarning.query
    if warning_type in ["Warning", "Caution", "Incompatibility"]:
        query = query.filter_by(warning_type=warning_type)
    if severity in ["Low", "Moderate", "High"]:
        query = query.filter_by(severity=severity)
    warnings = query.all()
    return render_template(
        "medicine/oncology/warnings.html",
        warnings=warnings,
        selected_warning_type=warning_type,
        selected_severity=severity,
    )


@bp.route("/bookings/")
@login_required
@roles_required(*CLINICAL_READ_ROLES)
def bookings():
    status = request.args.get("status", type=str)
    purpose = request.args.get("purpose", type=str)
    source = (request.args.get("source") or "").strip().upper()
    selected_source = source if source in ("RECORDS", "ONCOLOGY") else None

    # Build query with join to Patient
    query = OncologyBooking.query.join(
        Patient, OncologyBooking.patient_id == Patient.patient_id
    )

    # Apply filters
    if status in ["Scheduled", "Completed", "Cancelled"]:
        query = query.filter(OncologyBooking.status == status)
    if purpose in ["Consultation", "Chemotherapy", "Follow-up", "Radiation", "Surgery"]:
        query = query.filter(OncologyBooking.purpose == purpose)
    if selected_source:
        query = query.filter(OncologyBooking.source == selected_source)

    bookings = query.all()

    # Stats bar calculations
    booking_count = OncologyBooking.query.count()
    scheduled_booking_count = OncologyBooking.query.filter_by(
        status="Scheduled"
    ).count()
    chemotherapy_booking_count = OncologyBooking.query.filter_by(
        purpose="Chemotherapy"
    ).count()
    now = datetime.now(timezone.utc)
    start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    new_booking_count = OncologyBooking.query.filter(
        OncologyBooking.created_at >= start_of_month
    ).count()

    return render_template(
        "medicine/oncology/bookings.html",
        bookings=bookings,
        selected_status=status,
        selected_purpose=purpose,
        selected_source=selected_source,
        booking_count=booking_count,
        scheduled_booking_count=scheduled_booking_count,
        chemotherapy_booking_count=chemotherapy_booking_count,
        new_booking_count=new_booking_count,
    )


@bp.route("/bookings/new", methods=["GET", "POST"])
@login_required
@roles_required(*SCHEDULING_ROLES)
def new_booking():
    if request.method == "POST":
        patient_id = (request.form.get("patient_id") or "").strip()
        booking_date = request.form.get("booking_date")
        purpose = request.form.get("purpose")
        status = request.form.get("status")
        notes = request.form.get("notes", "").strip() or None

        # Validate required fields
        if not patient_id:
            flash("Patient selection is required.", "danger")
            return redirect(url_for("medicine.new_booking"))
        if not booking_date:
            flash("Booking date is required.", "danger")
            return redirect(url_for("medicine.new_booking"))
        if not purpose:
            flash("Purpose is required.", "danger")
            return redirect(url_for("medicine.new_booking"))
        if not status:
            flash("Status is required.", "danger")
            return redirect(url_for("medicine.new_booking"))

        # Validate patient exists — exact match on the patient number. A fuzzy
        # id/name match could validate one patient and then store another value.
        patient = _find_patient(patient_id)
        if not patient:
            flash("Selected patient does not exist.", "danger")
            return redirect(url_for("medicine.new_booking"))

        # Validate purpose and status
        valid_purposes = [
            "Consultation",
            "Chemotherapy",
            "Follow-up",
            "Radiation",
            "Surgery",
        ]
        valid_statuses = ["Scheduled", "Completed", "Cancelled"]
        if purpose not in valid_purposes:
            flash(
                f'Invalid purpose selected. Choose from: {", ".join(valid_purposes)}',
                "danger",
            )
            return redirect(url_for("medicine.new_booking"))
        if status not in valid_statuses:
            flash(
                f'Invalid status selected. Choose from: {", ".join(valid_statuses)}',
                "danger",
            )
            return redirect(url_for("medicine.new_booking"))

        # Parse booking_date
        try:
            booking_date = datetime.strptime(booking_date, "%Y-%m-%d").date()  # noqa: DTZ007
        except ValueError:
            flash("Invalid date format. Use YYYY-MM-DD.", "danger")
            return redirect(url_for("medicine.new_booking"))

        # Optional slot time (B5) — 24-hour HH:MM anchored to booking_date
        start_time = None
        raw_start = (request.form.get("start_time") or "").strip()
        if raw_start:
            try:
                start_time = datetime.combine(
                    booking_date, datetime.strptime(raw_start, "%H:%M").time()
                )
            except ValueError:
                flash("Invalid start time. Use 24-hour HH:MM.", "danger")
                return redirect(url_for("medicine.new_booking"))

        # Create new booking (canonical patient number, never raw user input)
        new_booking = OncologyBooking(
            patient_id=patient.patient_id,
            booking_date=booking_date,
            start_time=start_time,
            purpose=purpose,
            status=status,
            notes=notes,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        db.session.add(new_booking)
        db.session.flush()
        if not _commit_with_audit(
            "ONCOLOGY_BOOKING_CREATED",
            "OncologyBooking",
            new_booking.id,
            {"patient_id": patient.patient_id, "purpose": purpose, "status": status},
        ):
            flash("Audit trail unavailable; booking was NOT saved.", "danger")
            return redirect(url_for("medicine.new_booking"))
        flash("Booking created successfully!", "success")
        return redirect(url_for("medicine.bookings"))

    patients = Patient.query.all()
    if not patients:
        flash("No patients available. Please add a patient first.", "danger")
        # FIX: 'medicine.patients_list' does not exist — that url_for raised a
        # guaranteed BuildError 500. Route to the oncology home instead.
        return redirect(url_for("medicine.oncology"))
    return render_template("medicine/oncology/new_booking.html", patients=patients)


# Create a new prescription


@bp.route("/cancers")
@login_required
def cancers():
    # Load all cancer types and their details
    cancer_types = CancerType.query.order_by(CancerType.name).all()
    return render_template("medicine/oncology/cancers.html", cancer_types=cancer_types)


def process_lab_result(lab_result, test_name):
    """Helper function to process a single lab result into presentation format."""
    results_dict = {}
    try:
        results_dict = json.loads(lab_result.result) if lab_result.result else {}
    except json.JSONDecodeError:
        flash(f"Invalid result format for result ID {lab_result.result_id}.", "warning")
        results_dict = {}

    # Handle test_date
    test_date = lab_result.test_date
    if isinstance(test_date, str):
        try:
            test_date = datetime.strptime(test_date, "%Y-%m-%d %H:%M:%S")  # noqa: DTZ007
        except (ValueError, TypeError):
            test_date = None

    # Fetch parameters for this lab test
    parameters = LabResultTemplate.query.filter_by(test_id=lab_result.lab_test_id).all()

    test_presentation = []
    for param in parameters:
        result_value = results_dict.get(str(param.id))
        try:
            result_value_float = (
                float(result_value) if result_value is not None else None
            )
        except (ValueError, TypeError):
            result_value_float = None

        status = (
            "Invalid Result"
            if result_value_float is None
            else "Low"
            if result_value_float < param.normal_range_low
            else "High"
            if result_value_float > param.normal_range_high
            else "Normal"
        )

        test_presentation.append(
            {
                "parameter_name": param.parameter_name,
                "normal_range_low": param.normal_range_low,
                "normal_range_high": param.normal_range_high,
                "unit": param.unit,
                "result": result_value if result_value is not None else "N/A",
                "status": status,
            }
        )

    return {
        "test_name": test_name,
        "test_date": test_date.strftime("%Y-%m-%d %H:%M:%S") if test_date else "N/A",
        "result_notes": lab_result.result_notes or "",
        "parameters": test_presentation,
    }


# ---------------------------------------------------------------------------
# Chemotherapy Protocol Builder Routes (Gap #7)
# ---------------------------------------------------------------------------
_CHEMO_ERROR_HTTP_STATUS = {
    "INVALID_INPUT": 400,
    "UNKNOWN_PROTOCOL": 400,
    "CYCLE_OUT_OF_RANGE": 400,
    "CUMULATIVE_HISTORY_UNAVAILABLE": 503,
}


def _serialize_chemo_order(order):
    return {
        "order_id": order.id,
        "patient_id": order.patient_id,
        "protocol_name": order.protocol_name,
        "cycle_number": order.cycle_number,
        "total_cycles": order.total_cycles,
        "status": order.status,
        "bsa_m2": order.bsa_m2,
        "has_toxicity_warning": order.has_toxicity_warning,
        "toxicity_override_reason": order.toxicity_override_reason,
        "status_reason": order.status_reason,
        "physician_id": order.physician_id,
        "created_at": order.created_at.isoformat() if order.created_at else None,
    }


@bp.route("/oncology/chemo-builder/<string:patient_id>", methods=["GET"])
@login_required
@roles_required(*CLINICAL_WRITE_ROLES)
def chemo_builder(patient_id: str):
    """Render Oncology Chemotherapy Protocol Builder Workstation UI."""
    patient = _find_patient(patient_id)
    if not patient:
        abort(404)
    return render_template(
        "medicine/oncology/chemo_builder.html",
        patient_id=patient.patient_id,
        patient_name=patient.name,
    )


@bp.route("/oncology/api/calculate-chemo", methods=["GET"])
@login_required
@roles_required(*CLINICAL_WRITE_ROLES)
def api_calculate_chemo():
    """API endpoint to calculate BSA and chemotherapy protocol doses with toxicity alerts."""
    patient = _find_patient(request.args.get("patient_id", ""))
    if not patient:
        return _json_error("PATIENT_NOT_FOUND", "Patient not found.", 404)

    # Raw strings go straight to the engine, which rejects anything missing,
    # non-numeric, non-finite or implausible. No defaults are invented here.
    calc_res = calculate_regimen_doses(
        patient_id=patient.patient_id,
        protocol_name=request.args.get("protocol", ""),
        height_cm=request.args.get("height"),
        weight_kg=request.args.get("weight"),
        formula=request.args.get("formula", "mosteller"),
        cycle_number=request.args.get("cycle", 1),
    )
    if calc_res.get("error"):
        return jsonify(calc_res), _CHEMO_ERROR_HTTP_STATUS.get(
            calc_res.get("code"), 400
        )

    return jsonify(calc_res), 200


@bp.route("/oncology/api/save-chemo-order", methods=["POST"])
@login_required
@roles_required(*CLINICAL_WRITE_ROLES)
def api_save_chemo_order():
    """API endpoint to save signed Chemotherapy Regimen Order."""
    data = request.get_json(silent=True) or request.form.to_dict()

    patient_id = str(data.get("patient_id") or "").strip()
    protocol = str(data.get("protocol_name") or "").strip()
    if not patient_id or not protocol:
        return _json_error(
            "INVALID_INPUT", "patient_id and protocol_name are required", 400
        )

    patient = _find_patient(patient_id)
    if not patient:
        return _json_error("PATIENT_NOT_FOUND", "Patient not found.", 404)

    try:
        formula = normalize_bsa_formula(data.get("bsa_formula"))
        height, weight = validate_biometrics(
            data.get("height_cm"), data.get("weight_kg")
        )
    except ChemoInputError as exc:
        return _json_error(exc.code, str(exc), 400)

    # The server ALWAYS recomputes doses; nothing dose-related is trusted from the client.
    calc_res = calculate_regimen_doses(
        patient_id=patient.patient_id,
        protocol_name=protocol,
        height_cm=height,
        weight_kg=weight,
        formula=formula,
        cycle_number=data.get("cycle_number", 1),
    )
    if calc_res.get("error"):
        return jsonify(calc_res), _CHEMO_ERROR_HTTP_STATUS.get(
            calc_res.get("code"), 400
        )

    cycle = calc_res["cycle_number"]

    raw_total = data.get("total_cycles")
    if raw_total in (None, ""):
        total_cycles = calc_res.get("default_total_cycles") or 6
    else:
        try:
            as_float = float(raw_total)
            if isinstance(raw_total, bool) or as_float != int(as_float):
                raise ValueError  # noqa: TRY301
            total_cycles = int(as_float)
        except (TypeError, ValueError, OverflowError):
            return _json_error(
                "INVALID_INPUT", "total_cycles must be a whole number.", 400
            )
    if not (1 <= total_cycles <= MAX_CYCLES) or total_cycles < cycle:
        return _json_error(
            "INVALID_INPUT",
            f"total_cycles must be between {cycle} and {MAX_CYCLES}.",
            400,
        )

    # Lifetime-cap warnings are a hard stop unless the prescriber documents a reason.
    override_reason = str(data.get("toxicity_override_reason") or "").strip()
    if (
        calc_res["has_toxicity_warning"]
        and len(override_reason) < MIN_TOXICITY_OVERRIDE_CHARS
    ):
        return _json_error(
            "TOXICITY_OVERRIDE_REQUIRED",
            "Lifetime toxicity cap exceeded. Provide a documented clinical "
            f"justification (at least {MIN_TOXICITY_OVERRIDE_CHARS} characters) to proceed.",
            409,
            toxicity_warnings=calc_res["toxicity_warnings"],
        )
    if not calc_res["has_toxicity_warning"]:
        override_reason = ""

    # One active order per patient/protocol/cycle (a retry or double-click must
    # not double-count cumulative dose).
    duplicate = ChemotherapyRegimenOrder.query.filter(
        ChemotherapyRegimenOrder.patient_id == patient.patient_id,
        ChemotherapyRegimenOrder.protocol_name == calc_res["protocol_name"],
        ChemotherapyRegimenOrder.cycle_number == cycle,
        ChemotherapyRegimenOrder.status != "CANCELLED",
    ).first()
    if duplicate:
        return _json_error(
            "DUPLICATE_CYCLE_ORDER",
            f"An active {calc_res['protocol_name']} order for cycle {cycle} already "
            f"exists (order #{duplicate.id}). Cancel it first if it was entered in error.",
            409,
            order_id=duplicate.id,
        )

    # P0-11: Derive physician_id from authenticated session only.
    # Never fall back to session.get("user_id", 1).
    if not (current_user and getattr(current_user, "is_authenticated", False)):
        abort(401)
    physician_id = current_user.id

    order = ChemotherapyRegimenOrder(
        patient_id=patient.patient_id,
        physician_id=physician_id,
        protocol_name=calc_res["protocol_name"],
        cancer_type=calc_res["cancer_type"],
        weight_kg=weight,
        height_cm=height,
        bsa_m2=calc_res["bsa_m2"],
        bsa_formula=formula,
        cycle_number=cycle,
        total_cycles=total_cycles,
        calculated_doses_json=json.dumps(calc_res["drugs"]),
        has_toxicity_warning=calc_res["has_toxicity_warning"],
        toxicity_warning_details="\n".join(calc_res["toxicity_warnings"])
        if calc_res["toxicity_warnings"]
        else None,
        toxicity_override_reason=override_reason or None,
        status="ORDERED",
    )

    db.session.add(order)
    try:
        db.session.flush()
    except IntegrityError:  # concurrent duplicate hit the partial unique index
        db.session.rollback()
        return _json_error(
            "DUPLICATE_CYCLE_ORDER",
            f"An active {calc_res['protocol_name']} order for cycle {cycle} already exists.",
            409,
        )

    # A chemotherapy order without an audit record is not acceptable: if the
    # audit row cannot be written the order is rolled back as well.
    if not _commit_with_audit(
        "CHEMO_ORDER_CREATED",
        "ChemotherapyRegimenOrder",
        order.id,
        {
            "patient_id": patient.patient_id,
            "protocol": calc_res["protocol_name"],
            "cycle_number": cycle,
            "bsa_m2": calc_res["bsa_m2"],
            "has_toxicity_warning": calc_res["has_toxicity_warning"],
            "toxicity_override_reason": override_reason or None,
        },
    ):
        return _json_error(
            "AUDIT_UNAVAILABLE",
            "Audit trail unavailable; the chemotherapy order was NOT saved.",
            503,
        )

    return jsonify(
        {
            "success": True,
            "order_id": order.id,
            "protocol_name": order.protocol_name,
            "cycle_number": order.cycle_number,
            "status": order.status,
            "bsa_m2": order.bsa_m2,
            "has_toxicity_warning": order.has_toxicity_warning,
        }
    ), 201


@bp.route("/oncology/api/chemo-orders/<string:patient_id>", methods=["GET"])
@login_required
@roles_required(*CLINICAL_READ_ROLES)
def api_list_chemo_orders(patient_id: str):
    """List a patient's chemotherapy orders (newest first), including cancelled ones."""
    patient = _find_patient(patient_id)
    if not patient:
        return _json_error("PATIENT_NOT_FOUND", "Patient not found.", 404)
    orders = (
        ChemotherapyRegimenOrder.query.filter_by(patient_id=patient.patient_id)
        .order_by(
            ChemotherapyRegimenOrder.created_at.desc(),
            ChemotherapyRegimenOrder.id.desc(),
        )
        .all()
    )
    return jsonify(
        {
            "patient_id": patient.patient_id,
            "orders": [_serialize_chemo_order(o) for o in orders],
        }
    )


@bp.route("/oncology/api/chemo-order/<int:order_id>/status", methods=["POST"])
@login_required
@roles_required(*CLINICAL_READ_ROLES)
def api_update_chemo_order_status(order_id: int):
    """
    Move an order along ORDERED -> PREPARED -> ADMINISTERED, or cancel it.

    Orders are never deleted. Cancelling requires a reason and removes the
    order from cumulative lifetime-dose totals; ADMINISTERED is terminal.
    """
    order = db.session.get(ChemotherapyRegimenOrder, order_id)
    if not order:
        return _json_error("ORDER_NOT_FOUND", "Chemotherapy order not found.", 404)

    data = request.get_json(silent=True) or request.form.to_dict()
    new_status = str(data.get("status") or "").strip().upper()
    reason = str(data.get("reason") or "").strip()

    if new_status not in CHEMO_STATUSES:
        return _json_error(
            "INVALID_INPUT", f"status must be one of {list(CHEMO_STATUSES)}.", 400
        )
    if new_status not in CHEMO_ALLOWED_TRANSITIONS.get(order.status, ()):
        return _json_error(
            "INVALID_TRANSITION",
            f"Cannot move an order from {order.status} to {new_status}.",
            409,
        )
    if not has_any_role(*CHEMO_STATUS_ROLES[new_status]):
        return _json_error(
            "FORBIDDEN", f"Your role may not set status {new_status}.", 403
        )
    if new_status == "CANCELLED" and len(reason) < MIN_CANCEL_REASON_CHARS:
        return _json_error(
            "REASON_REQUIRED",
            f"A reason (at least {MIN_CANCEL_REASON_CHARS} characters) is required to cancel an order.",
            400,
        )

    previous_status = order.status
    order.status = new_status
    order.status_reason = reason or None
    order.status_changed_by = current_user.id
    order.status_changed_at = _utcnow_naive()
    if not _commit_with_audit(
        "CHEMO_ORDER_STATUS_CHANGED",
        "ChemotherapyRegimenOrder",
        order.id,
        {
            "patient_id": order.patient_id,
            "from": previous_status,
            "to": new_status,
            "reason": reason or None,
        },
    ):
        return _json_error(
            "AUDIT_UNAVAILABLE",
            "Audit trail unavailable; the status change was NOT saved.",
            503,
        )

    return jsonify({"success": True, **_serialize_chemo_order(order)}), 200
