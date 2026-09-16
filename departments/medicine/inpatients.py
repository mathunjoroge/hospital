import os
from datetime import datetime, timezone

from flask import flash, jsonify, redirect, render_template, request, url_for
from flask_login import current_user, login_required

from departments.forms import AdmitPatientForm
from departments.models.encounter import Encounter
from departments.models.medicine import (
    AdmittedPatient,
    Bed,
    TheatreList,
    TheatreProcedure,
    Ward,
    WardBedHistory,
    WardRoom,
    WardRound,
)
from departments.models.records import Patient
from departments.nlp.chatbot import UniversalClinicalSummarizer
from departments.nlp.logging_setup import get_logger
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


@bp.route("/add-to-theatre", methods=["GET", "POST"])
@login_required
def add_to_theatre():
    """Add a patient to the theatre list."""
    if request.method == "POST":
        try:
            data = (
                request.form
                if request.content_type == "application/x-www-form-urlencoded"
                else request.get_json()
            )

            patient_id = data.get("patient_id")  # Supports text like "P1000"
            procedure_id = data.get("procedure_id")
            created_by = data.get("created_by")
            notes_on_book = data.get("notes_on_book", None)

            if not patient_id or not procedure_id or not created_by:
                flash("All fields are required!", "danger")
                return redirect(url_for("medicine.add_to_theatre"))

            # Check if patient exists
            patient = Patient.query.filter(
                db.or_(
                    Patient.patient_id.ilike(f"%{patient_id}%"),
                    Patient.name.ilike(f"%{patient_id}%"),
                )
            ).first()
            if not patient:
                flash(f"Patient {patient_id} not found.", "danger")
                return redirect(url_for("medicine.add_to_theatre"))

            # Check if procedure exists
            procedure = db.session.get(TheatreProcedure, procedure_id)
            if not procedure:
                flash("Procedure not found.", "danger")
                return redirect(url_for("medicine.add_to_theatre"))

            # Create SURGICAL encounter scoped to this booking (T3.2)
            surgical_enc = create_surgical_encounter(
                patient_id=patient_id,
                provider_id=str(created_by) if created_by else None,
                chief_complaint=f"Surgical procedure: {procedure.name}",
            )

            # Create theatre list entry linked to the surgical encounter
            new_entry = TheatreList(
                patient_id=patient_id,
                procedure_id=procedure_id,
                status=0,
                created_by=created_by,
                notes_on_book=notes_on_book,
                encounter_id=surgical_enc.id,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            db.session.add(new_entry)
            db.session.commit()

            # Explicitly sync the theatre charge to the SURGICAL encounter (T3.8)
            if hasattr(procedure, 'cost') and procedure.cost:
                from departments.billing.sync import sync_charge
                sync_charge(
                    patient_id=patient_id,
                    source_table="theatre_list",
                    source_id=new_entry.id,
                    description=f"Theatre: {procedure.name}",
                    category="theatre",
                    amount=float(procedure.cost),
                    source_encounter_id=surgical_enc.id,
                )
                db.session.commit()

            flash("Patient added to theatre list successfully!", "success")
            return redirect(url_for("medicine.get_theatre_list"))

        except Exception as e:  # noqa: BLE001
            db.session.rollback()
            flash(f"An error occurred: {e!s}", "danger")
            return redirect(url_for("medicine.add_to_theatre"))

    # If GET request, render the form
    patients = Patient.query.all()
    procedures = TheatreProcedure.query.all()
    return render_template(
        "medicine/add_to_theatre.html", patients=patients, procedures=procedures
    )


# ✅ Corrected Route: Display Theatre List
@bp.route("/theatre-list", methods=["GET"])
@login_required
def get_theatre_list():
    """Retrieve all theatre list entries."""
    try:
        status_filter = request.args.get("status", type=int)

        query = (
            db.session.query(
                TheatreList.id,
                TheatreList.patient_id,  # Fetch patient_id as text
                Patient.name.label("patient_name"),
                TheatreProcedure.name.label("procedure_name"),
                TheatreList.status,
                TheatreList.created_at,
                TheatreList.notes_on_book,
                TheatreList.encounter_id,
                Encounter.stage.label("encounter_stage"),
            )
            .select_from(TheatreList)
            .join(TheatreList.patient)
            .join(TheatreList.procedure)
            .outerjoin(TheatreList.encounter)
        )

        if status_filter is not None:
            query = query.filter(TheatreList.status == status_filter)

        theatre_entries = query.order_by(TheatreList.created_at.desc()).all()

        return render_template(
            "medicine/theatre_list.html", theatre_entries=theatre_entries
        )

    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 500


@bp.route("/update-post-op/<int:entry_id>", methods=["GET", "POST"])
@login_required
def update_post_op(entry_id):
    """Update post-operative notes for a theatre list entry."""
    try:
        # Fetch the theatre entry and join with Patient table to get patient name
        entry = (
            db.session.query(TheatreList, Patient.name.label("patient_name"))
            .join(Patient, TheatreList.patient_id == Patient.patient_id)
            .filter(TheatreList.id == entry_id)
            .first()
        )

        if not entry:
            flash("Entry not found.", "danger")
            return redirect(url_for("medicine.get_theatre_list"))

        theatre_entry, patient_name = entry  # Unpack the tuple

        if request.method == "POST":
            notes_on_post_op = request.form.get("notes_on_post_op")

            if not notes_on_post_op:
                flash("Post-op notes are required.", "danger")
                return redirect(url_for("medicine.update_post_op", entry_id=entry_id))

            # Update status and post-op notes
            theatre_entry.status = 1  # Mark as completed
            theatre_entry.notes_on_post_op = notes_on_post_op
            theatre_entry.updated_at = datetime.now(timezone.utc)

            # Advance the SURGICAL encounter through INTRA_OP -> POST_OP -> DISCHARGED (T3.2)
            if getattr(theatre_entry, "encounter_id", None):
                enc = db.session.get(Encounter, theatre_entry.encounter_id)
                if enc and enc.stage not in ("DISCHARGED", "CANCELLED"):
                    enc.set_stage("INTRA_OP")
                    enc.set_stage("POST_OP")
                    enc.close()  # POST_OP -> DISCHARGED

            db.session.commit()

            flash("Post-op notes updated successfully!", "success")
            return redirect(url_for("medicine.get_theatre_list"))

        return render_template(
            "medicine/update_post_op.html",
            entry=theatre_entry,
            patient_name=patient_name,
        )

    except Exception as e:  # noqa: BLE001
        db.session.rollback()
        flash(f"An error occurred: {e!s}", "danger")
        return redirect(url_for("medicine.get_theatre_list"))


@bp.route("/theatre-transition/<int:entry_id>/<string:new_stage>", methods=["POST"])
@login_required
def transition_theatre_stage(entry_id, new_stage):
    """Transitions a SURGICAL encounter to the next valid stage using the state machine."""
    entry = TheatreList.query.get_or_404(entry_id)
    if not entry.encounter_id:
        flash("No surgical encounter found for this booking.", "danger")
        return redirect(url_for("medicine.get_theatre_list"))

    try:
        transition_surgical_stage(entry.encounter_id, new_stage)
        flash(f"Encounter stage successfully updated to {new_stage}.", "success")
    except ValueError as e:
        flash(f"Invalid stage transition: {e!s}", "danger")

    return redirect(url_for("medicine.get_theatre_list"))


@bp.route("/admit-patient", methods=["GET", "POST"])
@login_required
def admit_patient():
    """Admit a patient to a ward, assign a room & bed."""
    form = AdmitPatientForm()

    # ✅ Fetch patients, wards dynamically
    patients = Patient.query.all()
    wards = Ward.query.all()

    # ✅ Set SelectField choices
    form.patient_id.choices = [(p.patient_id, p.name) for p in patients]
    form.ward_id.choices = [(w.id, w.name) for w in wards]
    form.room_id.choices = []  # Will be populated dynamically via JavaScript
    form.bed_id.choices = []  # Will be populated dynamically via JavaScript

    if (
        request.method == "POST"
    ):  # ✅ Use request.method instead of validate_on_submit()
        try:
            patient_id = request.form.get("patient_id")
            ward_id = request.form.get("ward_id")
            room_id = request.form.get("room_id")
            bed_id = request.form.get("bed_id")
            admission_criteria = request.form.get("admission_criteria")
            admitted_by = request.form.get("admitted_by")

            # ✅ Check if patient exists
            patient = Patient.query.filter(
                db.or_(
                    Patient.patient_id.ilike(f"%{patient_id}%"),
                    Patient.name.ilike(f"%{patient_id}%"),
                )
            ).first()
            if not patient:
                flash("Patient not found.", "danger")
                return redirect(url_for("medicine.admit_patient"))

            # ✅ Check if ward exists
            ward = db.session.get(Ward, ward_id)
            if not ward:
                flash("Ward not found.", "danger")
                return redirect(url_for("medicine.admit_patient"))

            # ✅ Check if room exists in the ward
            room = WardRoom.query.filter_by(id=room_id, ward_id=ward_id).first()
            if not room:
                flash("Room not found in the selected ward.", "danger")
                return redirect(url_for("medicine.admit_patient"))

            # ✅ Check if bed exists & is available
            bed = Bed.query.filter_by(
                id=bed_id, room_id=room_id, occupied=False
            ).first()
            if not bed:
                flash("Selected bed is not available.", "danger")
                return redirect(url_for("medicine.admit_patient"))

            # ✅ Admit patient & mark bed as occupied
            admission = AdmittedPatient(
                patient_id=patient_id,
                ward_id=ward_id,
                room_id=room_id,
                bed_id=bed_id,
                admission_criteria=admission_criteria,
                admitted_by=admitted_by,
                admitted_on=datetime.now(timezone.utc),
            )

            bed.occupied = True  # Mark bed as occupied

            db.session.add(admission)

            # Create IPD Encounter for visit tracking
            from departments.models.encounter import Encounter
            ipd_encounter = Encounter(
                patient_id=patient_id,
                encounter_type="IPD",
                status="ACTIVE",
                stage="ADMITTED",
                started_at=datetime.now(timezone.utc)
            )
            db.session.add(ipd_encounter)
            db.session.commit()

            flash(
                f"Patient admitted to Bed {bed.bed_number} in Room {room.room_number}!",
                "success",
            )
            return redirect(url_for("medicine.view_admitted_patients"))

        except Exception as e:  # noqa: BLE001
            db.session.rollback()
            flash(f"Error: {e!s}", "danger")
            return redirect(url_for("medicine.admit_patient"))

    return render_template(
        "medicine/admit_patient.html", form=form, patients=patients, wards=wards
    )


# 2️⃣ Discharge a Patient (Free Up Bed)
@bp.route("/discharge-patient/<int:id>", methods=["POST"])
@login_required
def discharge_patient(id):
    """Discharge a patient and free their assigned bed."""
    try:
        admission = db.session.get(AdmittedPatient, id)
        if not admission:
            flash("Admission record not found.", "danger")
            return redirect(url_for("medicine.view_admitted_patients"))

        ward = db.session.get(Ward, admission.ward_id)
        # FIX 2: Use stored bed_id, fallback to searching the ward
        bed = db.session.get(Bed, admission.bed_id) if admission.bed_id else None
        if not bed:
            bed = (
                Bed.query.join(WardRoom, Bed.room_id == WardRoom.id)
                .filter(WardRoom.ward_id == admission.ward_id, Bed.occupied.is_(True))
                .first()
            )

        if ward and ward.occupied_beds > 0:
            ward.occupied_beds -= 1
        if bed:
            bed.occupied = False

        admission.discharged_on = datetime.now(timezone.utc)

        # Close the IPD Encounter
        from departments.models.encounter import Encounter
        active_enc = Encounter.query.filter_by(
            patient_id=admission.patient_id,
            status="ACTIVE",
            encounter_type="IPD"
        ).order_by(Encounter.started_at.desc()).first()

        if active_enc:
            active_enc.status = "CLOSED"
            active_enc.ended_at = datetime.now(timezone.utc)
            active_enc.close()

        db.session.commit()

        flash("Patient discharged and bed is now available!", "success")
        return redirect(url_for("medicine.view_admitted_patients"))

    except Exception as e:  # noqa: BLE001
        db.session.rollback()
        flash(f"Error: {e!s}", "danger")
        return redirect(url_for("medicine.view_admitted_patients"))


# 3️⃣ patients in ward
@bp.route("/admitted-patients", methods=["GET"])
@login_required
def view_admitted_patients():
    """View all admitted patients."""
    try:
        patients = (
            db.session.query(
                AdmittedPatient.id,
                AdmittedPatient.patient_id,
                Patient.name.label("patient_name"),
                Ward.name.label("ward_name"),
                Ward.sex.label("ward_sex"),
                AdmittedPatient.admitted_on,
                AdmittedPatient.admission_criteria,
                AdmittedPatient.discharged_on,
            )
            .join(Patient, AdmittedPatient.patient_id == Patient.patient_id)
            .join(Ward, AdmittedPatient.ward_id == Ward.id)
            .order_by(AdmittedPatient.admitted_on.desc())
            .all()
        )

        return render_template("medicine/admitted_patients.html", patients=patients)

    except Exception as e:  # noqa: BLE001
        flash(f"Error: {e!s}", "danger")
        return redirect(url_for("medicine.admit_patient"))


@bp.route("/ward-bed-history/<int:ward_id>", methods=["GET"])
@login_required
def ward_bed_history(ward_id):
    """View bed history for a ward."""
    try:
        ward = db.session.get(Ward, ward_id)
        if not ward:
            flash("Ward not found.", "danger")
            return redirect(url_for("medicine.view_admitted_patients"))

        history = (
            WardBedHistory.query.filter_by(ward_id=ward_id)
            .order_by(WardBedHistory.timestamp.desc())
            .all()
        )

        return render_template(
            "medicine/ward_bed_history.html", ward=ward, history=history
        )

    except Exception as e:  # noqa: BLE001
        flash(f"Error: {e!s}", "danger")
        return redirect(url_for("medicine.view_admitted_patients"))


# ✅ Fetch available rooms in a ward
@bp.route("/available-rooms/<int:ward_id>", methods=["GET"])
@login_required
def available_rooms(ward_id):
    """Return available rooms in a ward."""
    try:
        rooms = WardRoom.query.filter_by(ward_id=ward_id, occupied=False).all()
        return jsonify(
            {
                "rooms": [
                    {"id": room.id, "room_number": room.room_number} for room in rooms
                ]
            }
        )
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 500


@bp.route("/available-beds/<int:room_id>", methods=["GET"])
@login_required
def available_beds(room_id):
    """Return available beds in a room."""
    try:
        beds = Bed.query.filter_by(room_id=room_id, occupied=False).all()
        return jsonify(
            {"beds": [{"id": bed.id, "bed_number": bed.bed_number} for bed in beds]}
        )
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 500


# ✅ View Inpatients List
@bp.route("/inpatients", methods=["GET"])
@login_required
def view_inpatients():
    """Show all admitted patients for ward rounds."""
    admitted_patients = (
        AdmittedPatient.query.join(Patient)
        .add_columns(
            AdmittedPatient.id,
            Patient.name.label("patient_name"),
            AdmittedPatient.ward_id,
            AdmittedPatient.admitted_on,
        )
        .order_by(AdmittedPatient.admitted_on.desc())
        .all()
    )

    return render_template(
        "medicine/inpatients.html", admitted_patients=admitted_patients
    )


@bp.route("/ward-rounds", methods=["GET", "POST"])
@login_required
def ward_rounds():
    """View inpatients and allow doctors to update ward rounds."""
    if request.method == "POST":
        try:
            admission_id = request.form.get("admission_id")
            notes = request.form.get("notes")
            status = request.form.get("status")

            if not admission_id or not notes or not status:
                flash("All fields are required!", "danger")
                return redirect(url_for("medicine.ward_rounds"))

            # Check if admission exists
            admission = db.session.get(AdmittedPatient, admission_id)
            if not admission:
                flash("Patient admission not found.", "danger")
                return redirect(url_for("medicine.ward_rounds"))

            # Save ward round entry
            round_entry = WardRound(
                admission_id=admission_id,
                doctor_id=current_user.id,
                notes=notes,
                status=status,
            )

            db.session.add(round_entry)
            db.session.commit()

            flash("Ward round notes updated successfully!", "success")
            return redirect(url_for("medicine.ward_rounds"))

        except Exception as e:  # noqa: BLE001
            db.session.rollback()
            flash(f"Error: {e!s}", "danger")
            return redirect(url_for("medicine.ward_rounds"))

    # Fetch admitted patients and their details
    # Fetch admitted patients and their details, including patient_id
    admitted_patients = (
        db.session.query(
            AdmittedPatient.id,
            AdmittedPatient.patient_id,  # ✅ Ensure this is included
            Patient.name.label("patient_name"),
            Ward.name.label("ward_name"),
            AdmittedPatient.admitted_on,
        )
        .join(Patient, AdmittedPatient.patient_id == Patient.patient_id)
        .join(Ward, AdmittedPatient.ward_id == Ward.id)
        .order_by(AdmittedPatient.admitted_on.desc())
        .all()
    )

    return render_template(
        "medicine/ward_rounds.html", admitted_patients=admitted_patients
    )


@bp.route("/ward-rounds/<int:admission_id>", methods=["GET"])
@login_required
def view_ward_rounds(admission_id):
    """
    Ward round history for a single admission.

    This was referenced as the redirect target from every branch of
    add_ward_round() below (success, missing-fields, and error), and linked
    from inpatients.html, but was never defined anywhere — so submitting a
    ward round note always raised a BuildError after the note had already
    been committed: the note saved, but the request still 500'd.
    """
    admission = db.session.get(AdmittedPatient, admission_id)
    if not admission:
        flash("Patient admission not found.", "danger")
        return redirect(url_for("medicine.ward_rounds"))

    rounds = (
        WardRound.query.filter_by(admission_id=admission_id)
        .order_by(WardRound.created_at.desc())
        .all()
    )
    return render_template(
        "medicine/ward_round_history.html", rounds=rounds, admission=admission
    )


@bp.route("/ward-rounds/add", methods=["POST"])
@login_required
def add_ward_round():
    """Add a ward round note for a patient."""
    try:
        admission_id = request.form.get("admission_id")
        notes = request.form.get("notes")
        status = request.form.get("status", "Under Treatment")

        if not admission_id or not notes:
            flash("Please provide required fields!", "danger")
            return redirect(
                url_for("medicine.view_ward_rounds", admission_id=admission_id)
            )

        new_entry = WardRound(
            admission_id=admission_id,
            doctor_id=current_user.id,
            notes=notes,
            status=status,
        )

        db.session.add(new_entry)
        db.session.commit()

        flash("Ward round note added successfully!", "success")
        return redirect(url_for("medicine.view_ward_rounds", admission_id=admission_id))

    except Exception as e:  # noqa: BLE001
        db.session.rollback()
        flash(f"Error: {e!s}", "danger")
        return redirect(url_for("medicine.view_ward_rounds", admission_id=admission_id))
        # diseases


# --- T3.2: Theatre Booking Stages ---
def create_surgical_encounter(patient_id: str, provider_id: str | None = None, chief_complaint: str = "Surgical Procedure"):
    """Creates a new SURGICAL encounter starting in PRE_OP stage."""
    from departments.models.encounter import Encounter

    enc = Encounter(
        patient_id=patient_id,
        encounter_type="SURGICAL",
        stage="PRE_OP",
        status="ACTIVE",
        provider_id=str(provider_id) if provider_id else None,
        chief_complaint=chief_complaint,
    )
    db.session.add(enc)
    db.session.flush()  # Populate enc.id before any potential linking
    return enc

def transition_surgical_stage(encounter_id: int, new_stage: str):
    """Transitions a SURGICAL encounter to the next valid stage using the state machine."""
    from departments.models.encounter import Encounter

    enc = db.session.get(Encounter, encounter_id)
    if not enc:
        raise ValueError("Encounter not found")
    if enc.encounter_type != "SURGICAL":
        raise ValueError("Encounter is not of type SURGICAL")

    # set_stage enforces legal transitions (PRE_OP -> INTRA_OP -> POST_OP)
    enc.set_stage(new_stage)
    db.session.commit()
    return enc
# ------------------------------------


# --- ADT & Inpatient Bed Management Routes ---

@bp.route("/inpatients/ward-grid", methods=["GET"])
@login_required
def ward_bed_grid_view():
    """Render interactive Inpatient Ward Bed Management & Turnaround Console UI."""
    from departments.medicine.adt_engine import ADTEngine
    matrix = ADTEngine.get_ward_bed_matrix()
    return render_template("medicine/ward_bed_grid.html", matrix=matrix)


@bp.route("/api/beds/status", methods=["GET"])
@login_required
def api_get_bed_status_matrix():
    """Return JSON feeds of live ward bed status matrix."""
    from departments.medicine.adt_engine import ADTEngine
    matrix = ADTEngine.get_ward_bed_matrix()
    return jsonify(matrix), 200


@bp.route("/api/beds/<int:bed_id>/status", methods=["POST"])
@login_required
def api_update_bed_status(bed_id: int):
    """Update bed cleaning / housekeeping turnaround status."""
    from departments.medicine.adt_engine import ADTEngine
    payload = request.get_json() or {}
    new_status = payload.get("status")
    notes = payload.get("notes")

    if not new_status:
        return jsonify({"status": "error", "message": "Missing 'status' parameter."}), 400

    try:
        bed = ADTEngine.update_bed_status(
            bed_id=bed_id,
            new_status=new_status,
            user_id=getattr(current_user, "id", None),
            notes=notes,
        )
        return jsonify({
            "status": "success",
            "message": f"Bed {bed.bed_number} updated to {bed.status}.",
            "bed_id": bed.id,
            "new_status": bed.status,
            "occupied": bed.occupied,
        }), 200
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400


@bp.route("/api/adt/admit", methods=["POST"])
@login_required
def api_adt_admit_a01():
    """Process HL7 ADT^A01 Inpatient Admission API endpoint."""
    from departments.medicine.adt_engine import ADTEngine
    payload = request.get_json() or {}

    patient_id = payload.get("patient_id")
    ward_id = payload.get("ward_id")
    room_id = payload.get("room_id")
    bed_id = payload.get("bed_id")
    criteria = payload.get("admission_criteria", "Inpatient Admission")

    if not (patient_id and ward_id and room_id and bed_id):
        return jsonify({"status": "error", "message": "Missing required fields (patient_id, ward_id, room_id, bed_id)."}), 400

    try:
        adm, adt_log = ADTEngine.admit_patient_a01(
            patient_id=str(patient_id),
            ward_id=int(ward_id),
            room_id=int(room_id),
            bed_id=int(bed_id),
            admission_criteria=str(criteria),
            user_id=getattr(current_user, "id", None),
        )
        return jsonify({
            "status": "success",
            "event_type": "A01",
            "admission_id": adm.id,
            "patient_id": adm.patient_id,
            "ward_id": adm.ward_id,
            "bed_id": adm.bed_id,
            "hl7_message": adt_log.hl7_message,
        }), 201
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400


@bp.route("/api/adt/transfer", methods=["POST"])
@login_required
def api_adt_transfer_a02():
    """Process HL7 ADT^A02 Patient Transfer API endpoint."""
    from departments.medicine.adt_engine import ADTEngine
    payload = request.get_json() or {}

    admission_id = payload.get("admission_id")
    new_ward_id = payload.get("new_ward_id")
    new_room_id = payload.get("new_room_id")
    new_bed_id = payload.get("new_bed_id")

    if not (admission_id and new_ward_id and new_room_id and new_bed_id):
        return jsonify({"status": "error", "message": "Missing required fields (admission_id, new_ward_id, new_room_id, new_bed_id)."}), 400

    try:
        adm, adt_log = ADTEngine.transfer_patient_a02(
            admission_id=int(admission_id),
            new_ward_id=int(new_ward_id),
            new_room_id=int(new_room_id),
            new_bed_id=int(new_bed_id),
            user_id=getattr(current_user, "id", None),
        )
        return jsonify({
            "status": "success",
            "event_type": "A02",
            "admission_id": adm.id,
            "new_ward_id": adm.ward_id,
            "new_bed_id": adm.bed_id,
            "hl7_message": adt_log.hl7_message,
        }), 200
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400


@bp.route("/api/adt/discharge", methods=["POST"])
@login_required
def api_adt_discharge_a03():
    """Process HL7 ADT^A03 Patient Discharge API endpoint."""
    from departments.medicine.adt_engine import ADTEngine
    payload = request.get_json() or {}

    admission_id = payload.get("admission_id")
    summary = payload.get("discharge_summary", "Discharged in stable condition.")

    if not admission_id:
        return jsonify({"status": "error", "message": "Missing required field 'admission_id'."}), 400

    try:
        adm, adt_log = ADTEngine.discharge_patient_a03(
            admission_id=int(admission_id),
            discharge_summary=str(summary),
            user_id=getattr(current_user, "id", None),
        )
        return jsonify({
            "status": "success",
            "event_type": "A03",
            "admission_id": adm.id,
            "discharged_on": adm.discharged_on.isoformat(),
            "hl7_message": adt_log.hl7_message,
        }), 200
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400


@bp.route("/api/adt/events", methods=["GET"])
@login_required
def api_get_adt_events():
    """Fetch historical HL7 ADT event logs."""
    from departments.models.medicine import ADTLog
    logs = ADTLog.query.order_by(ADTLog.id.desc()).limit(100).all()
    data = [
        {
            "id": log_item.id,
            "event_type": log_item.event_type,
            "patient_id": log_item.patient_id,
            "admission_id": log_item.admission_id,
            "from_ward_id": log_item.from_ward_id,
            "to_ward_id": log_item.to_ward_id,
            "to_bed_id": log_item.to_bed_id,
            "created_at": log_item.created_at.isoformat(),
            "hl7_message": log_item.hl7_message,
        }
        for log_item in logs
    ]
    return jsonify({"status": "success", "total": len(data), "events": data}), 200


