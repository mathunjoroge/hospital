import os
import uuid
from typing import Any, Dict, List, Optional

from flask import flash, redirect, render_template, request, url_for
from flask_login import login_required
from psycopg2.extras import RealDictCursor
from sqlalchemy.orm import joinedload

from departments.models.medicine import (
    Imaging,
    LabTest,
    RequestedImage,
    RequestedLab,
    SOAPNote,
    UnmatchedImagingRequest,
)
from departments.models.records import Patient, PatientWaitingList
from departments.nlp.chatbot import UniversalClinicalSummarizer
from departments.nlp.logging_setup import get_logger
from departments.rbac import roles_required
from departments.shared.encounter_utils import active_encounter
from extensions import db, socketio

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


@bp.route("/request_lab_tests/<patient_id>", methods=["GET", "POST"])
@login_required
@roles_required("medicine", "admin")
def request_lab_tests(patient_id):
    """Handles lab test requests."""
    try:
        dept = request.args.get("dept")  # ✅ Capture dept from query string

        # Fetch patient from waiting list
        patient_entry = (
            PatientWaitingList.query.filter_by(patient_id=patient_id)
            .options(joinedload(PatientWaitingList.patient))
            .first()
        )
        if not patient_entry or not patient_entry.patient:
            flash(
                f"Patient with ID {patient_id} not found in the waiting list!", "error"
            )
            return redirect(url_for("medicine.index"))

        patient = patient_entry.patient
        lab_tests = LabTest.query.all()

        if request.method == "POST":
            lab_test_ids = request.form.getlist("lab_tests[]")
            if not lab_test_ids:
                flash("No lab tests selected!", "error")
                return render_template(
                    "medicine/request_lab_tests.html",
                    patient=patient,
                    lab_tests=lab_tests,
                    dept=dept,
                )

            descriptions = {}
            for key, value in request.form.items():
                if key.startswith("descriptions["):
                    lab_id = key.split("[")[1].split("]")[0]
                    descriptions[lab_id] = value.strip() if value else ""

            for lab_test_id in lab_test_ids:
                description = descriptions.get(str(lab_test_id), "").strip()

                if len(description) > 500:
                    flash(
                        f"Description for lab test ID {lab_test_id} exceeds 500 characters.",
                        "error",
                    )
                    return render_template(
                        "medicine/request_lab_tests.html",
                        patient=patient,
                        lab_tests=lab_tests,
                        dept=dept,
                    )

                result_id = str(uuid.uuid4())

                new_lab_request = RequestedLab(
                    patient_id=patient_id,
                    lab_test_id=lab_test_id,
                    result_id=result_id,
                    description=description or None,
                )
                db.session.add(new_lab_request)

            db.session.commit()
            flash("Lab tests requested successfully!", "success")

            # ✅ Redirect accordingly
            if dept == "1":
                return redirect(url_for("medicine.ward_rounds"))
            else:
                return redirect(url_for("medicine.soap_notes", patient_id=patient_id))

        # GET request
        return render_template(
            "medicine/request_lab_tests.html",
            patient=patient,
            lab_tests=lab_tests,
            dept=dept,
        )

    except Exception as e:
        db.session.rollback()
        logger.error(f"Error in medicine.request_lab_tests: {e}", exc_info=True)
        flash("Something went wrong. Please try again.", "error")
        return redirect(url_for("medicine.soap_notes", patient_id=patient_id))


@bp.route("/request_imaging/<patient_id>", methods=["GET", "POST"])
@login_required
@roles_required("medicine", "admin")
def request_imaging(patient_id):
    """Handles imaging requests."""
    try:
        dept = request.args.get("dept")  # ✅ Capture dept from query string

        # Fetch patient from waiting list
        patient_entry = (
            PatientWaitingList.query.filter_by(patient_id=patient_id)
            .options(joinedload(PatientWaitingList.patient))
            .first()
        )
        if not patient_entry or not patient_entry.patient:
            flash(
                f"Patient with ID {patient_id} not found in the waiting list!", "error"
            )
            return redirect(url_for("medicine.index"))

        patient = patient_entry.patient
        soap_notes = (
            SOAPNote.query.filter_by(patient_id=patient_id)
            .order_by(SOAPNote.created_at.desc())
            .first()
        )
        imaging_types = Imaging.query.all()

        if request.method == "POST":
            imaging_ids = request.form.getlist("imaging_types[]")

            if not imaging_ids:
                flash("No imaging types selected!", "error")
                return render_template(
                    "medicine/request_imaging.html",
                    patient=patient,
                    soap_notes=soap_notes,
                    imaging_types=imaging_types,
                    dept=dept,
                )

            descriptions = {}
            for key, value in request.form.items():
                if key.startswith("descriptions["):
                    imaging_id = key.split("[")[1].split("]")[0]
                    descriptions[imaging_id] = value.strip() if value else ""

            for imaging_id in imaging_ids:
                description = descriptions.get(str(imaging_id), "").strip()

                if len(description) > 500:
                    flash(
                        f"Description for imaging ID {imaging_id} exceeds 500 characters.",
                        "error",
                    )
                    return render_template(
                        "medicine/request_imaging.html",
                        patient=patient,
                        soap_notes=soap_notes,
                        imaging_types=imaging_types,
                        dept=dept,
                    )

                result_id = str(uuid.uuid4())
                new_image_request = RequestedImage(
                    patient_id=patient_id,
                    imaging_id=imaging_id,
                    result_id=result_id,
                    description=description or None,
                )
                db.session.add(new_image_request)

            db.session.commit()
            flash("Imaging requested successfully!", "success")

            # ✅ Redirect based on dept
            if dept == "1":
                return redirect(url_for("medicine.ward_rounds"))
            else:
                return redirect(url_for("medicine.soap_notes", patient_id=patient_id))

        # GET: Render form
        return render_template(
            "medicine/request_imaging.html",
            patient=patient,
            soap_notes=soap_notes,
            imaging_types=imaging_types,
            dept=dept,
        )

    except Exception as e:
        db.session.rollback()
        logger.error(f"Error in medicine.request_imaging: {e}", exc_info=True)
        flash("Something went wrong. Please try again.", "error")
        return redirect(url_for("medicine.soap_notes", patient_id=patient_id))


@bp.route("/unmatched_imaging", methods=["GET", "POST"])
@login_required
@roles_required("medicine", "admin")
def unmatched_imaging():
    """Medicine panel to match unmatched imaging requests."""
    # Handle form submission (matching requests)
    if request.method == "POST":
        unmatched_id = request.form.get("unmatched_id")
        imaging_id = request.form.get("imaging_id")

        if unmatched_id and imaging_id:
            unmatched_request = UnmatchedImagingRequest.query.get(unmatched_id)
            if unmatched_request:
                # Move to requested_images table. UnmatchedImagingRequest
                # doesn't carry its own encounter_id, so scope to whatever
                # encounter is active for the patient now (best effort).
                encounter = active_encounter(unmatched_request.patient_id)
                requested_imaging = RequestedImage(
                    patient_id=unmatched_request.patient_id,
                    encounter_id=encounter.id if encounter else None,
                    imaging_id=imaging_id,
                    description=unmatched_request.description,
                )
                db.session.add(requested_imaging)

                # Remove from unmatched list
                db.session.delete(unmatched_request)
                db.session.commit()
                flash("Imaging request successfully matched!", "success")
            else:
                flash("Invalid request!", "error")

    # Get filtering parameters
    patient_name = request.args.get("patient_name", "").strip()
    start_date = request.args.get("start_date", "")
    end_date = request.args.get("end_date", "")

    # Base query for unmatched imaging requests
    unmatched_requests = UnmatchedImagingRequest.query.join(Patient).order_by(
        UnmatchedImagingRequest.date_requested.desc()
    )

    # Apply filters if provided
    if patient_name:
        unmatched_requests = unmatched_requests.filter(
            Patient.name.ilike(f"%{patient_name}%")
        )
    if start_date:
        unmatched_requests = unmatched_requests.filter(
            UnmatchedImagingRequest.date_requested >= start_date
        )
    if end_date:
        unmatched_requests = unmatched_requests.filter(
            UnmatchedImagingRequest.date_requested <= end_date
        )

    unmatched_requests = unmatched_requests.all()
    imaging_options = Imaging.query.all()

    return render_template(
        "unmatched_imaging.html",
        unmatched_requests=unmatched_requests,
        imaging_options=imaging_options,
        patient_name=patient_name,
        start_date=start_date,
        end_date=end_date,
    )


@bp.route("/unmatched_imaging/notify", methods=["GET", "POST"])
@login_required
def notify_admin():
    """Notify medicine users by updating badge count via SocketIO."""
    count = UnmatchedImagingRequest.query.count()
    socketio.emit(
        "update_badge", {"count": count}, namespace="/medicine"
    )  # Updated namespace
    return "", 204  # Return a success response without content


def get_unmatched_count():
    """Get the number of unmatched imaging requests."""
    return UnmatchedImagingRequest.query.count()


@bp.context_processor
def inject_unmatched_count():
    """Inject the unmatched count into the template context."""
    return dict(unmatched_count=get_unmatched_count())


from departments.shared.drugcentral import (  # noqa: E402
    get_drugcentral_connection as get_db_connection,
)


def fetch_drugs_data(search_query: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetch distinct product data with optional search by generic name or brand name."""
    try:
        with get_db_connection() as conn, conn.cursor(
            cursor_factory=RealDictCursor
        ) as cur:
            base_query = """
                SELECT DISTINCT generic_name, product_name, route, form
                FROM product
            """

            params = []
            if search_query:
                search_param = f"%{search_query}%"
                base_query += """
                    WHERE generic_name ILIKE %s OR product_name ILIKE %s
                """
                params = [search_param] * 2  # 2 parameters now

            base_query += " ORDER BY generic_name"
            cur.execute(base_query, params)
            return cur.fetchall()
    except Exception as e:
        print(f"Database error: {str(e)}")
        return []
