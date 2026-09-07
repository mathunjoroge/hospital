import json
import os
from datetime import datetime

import requests
from flask import (
    current_app,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)
from flask_login import login_required
from sqlalchemy.orm import joinedload

from departments.models.laboratory import LabResult
from departments.models.medicine import (
    AdmittedPatient,
    Imaging,
    LabTest,
    Medicine,
    PrescribedMedicine,
    RequestedImage,
    RequestedLab,
    SOAPNote,
    TheatreList,
    UnmatchedImagingRequest,
)
from departments.models.records import Patient, PatientWaitingList
from departments.nlp.chatbot import UniversalClinicalSummarizer
from departments.nlp.logging_setup import get_logger
from departments.rbac import roles_required
from extensions import db

from . import bp

logger = get_logger()


def notify_admin(message):
    """Send notification to admin about unmatched requests."""
    logger.info(f"Admin notification: {message}")


def process_lab_result(lab_result, test_name):
    """Process a lab result into a presentation dictionary."""
    return {
        "test_name": test_name,
        "result_value": getattr(lab_result, "result_value", "N/A"),
        "reference_range": getattr(lab_result, "reference_range", "N/A"),
        "unit": getattr(lab_result, "unit", ""),
        "date": getattr(lab_result, "date_completed", None),
    }


# Instantiate the summarizer for use in chatbot_interface


gemini_api_key = os.environ.get("GEMINI_API_KEY")
nvidia_api_key = os.environ.get("NVIDIA_API_KEY")
Summarizer = UniversalClinicalSummarizer(
    gemini_api_key=gemini_api_key, nvidia_api_key=nvidia_api_key
)


ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "pdf", "txt", "csv", "docx"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB max file size


@bp.route("/submit_soap_notes/<patient_id>", methods=["POST"])
@login_required
@roles_required("medicine", "admin")
def submit_soap_notes(patient_id):
    try:
        # --- Form Data Retrieval ---
        situation = request.form.get("situation")
        hpi = request.form.get("hpi")
        aggravating_factors = request.form.get("aggravating_factors")
        alleviating_factors = request.form.get("alleviating_factors")
        medical_history = request.form.get("medical_history")
        medication_history = request.form.get("medication_history")
        assessment = request.form.get("assessment")
        recommendation = request.form.get("recommendation")
        additional_notes = request.form.get("additional_notes")
        symptoms = request.form.get("symptoms", "")

        # --- File Upload Handling ---
        file = request.files.get("file_upload")
        file_path = None
        if file and file.filename:
            upload_folder = os.path.join(current_app.root_path, "Uploads")
            os.makedirs(upload_folder, exist_ok=True)
            file_path = os.path.join("Uploads", file.filename)
            file.save(os.path.join(upload_folder, file.filename))

        # --- Input Validation ---
        if not all([situation, hpi, assessment, recommendation]):
            flash("All required fields must be filled out!", "error")
            return redirect(url_for("medicine.soap_notes", patient_id=patient_id))

        # --- Symptoms Processing ---
        symptom_list = []
        if symptoms:
            if isinstance(symptoms, str):
                symptom_list = [s.strip() for s in symptoms.split(",") if s.strip()]
            elif isinstance(symptoms, list):
                symptom_list = [
                    s.strip() for s in symptoms if isinstance(s, str) and s.strip()
                ]
            logger.debug(f"Received symptoms for patient {patient_id}: {symptom_list}")

        # --- Create and Save SOAP Note Object ---
        new_soap_note = SOAPNote(
            patient_id=patient_id,
            situation=situation,
            hpi=hpi,
            aggravating_factors=aggravating_factors,
            alleviating_factors=alleviating_factors,
            medical_history=medical_history,
            medication_history=medication_history,
            assessment=assessment,
            recommendation=recommendation,
            additional_notes=additional_notes,
            symptoms=json.dumps(symptom_list) if symptom_list else "",
            file_path=file_path,
            ai_notes=None,
            ai_analysis=None,
        )

        db.session.add(new_soap_note)
        db.session.commit()

        # ⭐ --- START: TRIGGER FASTAPI NLP SERVICE --- ⭐
        try:
            # The note_id is now available on the new_soap_note object
            note_id = new_soap_note.id

            # This URL should ideally be stored in your Flask app's configuration
            nlp_api_url = "http://127.0.0.1:8000/process_note"
            payload = {"note_id": note_id}

            # Send the request to the FastAPI service
            response = requests.post(nlp_api_url, json=payload, timeout=30)

            # This will raise an HTTPError if the HTTP request returned an unsuccessful status code
            response.raise_for_status()

            logger.info(
                f"Successfully triggered AI analysis for SOAP note ID: {note_id}"
            )

        except requests.exceptions.RequestException as e:
            # Catch connection errors, timeouts, and bad responses
            logger.error(f"Failed to trigger AI analysis for note ID {note_id}: {e}")
            flash(
                "Note saved, but the AI analysis service could not be reached. Please ask an admin to process it manually.",
                "warning",
            )
        # ⭐ --- END: TRIGGER FASTAPI NLP SERVICE --- ⭐

        # --- Imaging Request Processing ---
        imaging_keywords = ["ct", "mri", "x-ray", "ultrasound", "pet", "scan"]
        words = recommendation.lower().split()
        matched_imaging = set()
        for i, word in enumerate(words):
            for keyword in imaging_keywords:
                if keyword in word:
                    phrase = " ".join(words[i : i + 2]) if i + 1 < len(words) else word
                    matched_imaging.add(phrase)

        unmatched_requests = []
        for imaging_request in matched_imaging:
            imaging_match = Imaging.query.filter(
                Imaging.imaging_type.ilike(f"%{imaging_request}%")
            ).first()
            if imaging_match:
                requested_imaging = RequestedImage(
                    patient_id=patient_id,
                    imaging_id=imaging_match.id,
                    description=recommendation,
                )
                db.session.add(requested_imaging)
            else:
                unmatched_request = UnmatchedImagingRequest(
                    patient_id=patient_id, description=imaging_request
                )
                db.session.add(unmatched_request)
                unmatched_requests.append(imaging_request)

        # Commit imaging requests
        db.session.commit()

        # --- Notify Admin about Unmatched Requests ---
        if unmatched_requests:
            try:
                message = f"Unmatched imaging requests for patient {patient_id}: {', '.join(unmatched_requests)}"
                notify_admin(message)
            except Exception as e:
                logger.error(
                    f"Failed to notify admin about unmatched imaging: {str(e)}"
                )
            flash(
                f"The following imaging requests need manual review: {', '.join(unmatched_requests)}",
                "warning",
            )

        flash("SOAP note submitted successfully!", "success")
        return redirect(url_for("medicine.notes", patient_id=patient_id))

    except Exception as e:
        flash(f"An unexpected error occurred: {str(e)}", "error")
        db.session.rollback()
        logger.error(
            f"Critical error in submit_soap_notes for patient {patient_id}: {str(e)}",
            exc_info=True,
        )
        return redirect(url_for("medicine.soap_notes", patient_id=patient_id))


@bp.route("/notes/<string:patient_id>", methods=["GET"])
@login_required
@roles_required("medicine", "admin")
def notes(patient_id):
    """Displays SOAP notes for a patient."""
    patient = Patient.query.filter_by(patient_id=patient_id).first_or_404()
    last_soap_note = (
        SOAPNote.query.filter_by(patient_id=patient_id)
        .order_by(SOAPNote.created_at.desc())
        .first()
    )
    return render_template(
        "medicine/notes.html", patient=patient, soap_notes=last_soap_note
    )


@bp.route("/notes/<int:note_id>/reprocess", methods=["POST"])
@login_required
def reprocess_note(note_id):
    return redirect(
        url_for("medicine.notes", patient_id=SOAPNote.query.get(note_id).patient_id)
    )


# Display the medicine waiting list
@bp.route("/")
@login_required
@roles_required("medicine", "admin")
def index():
    """Display the medicine waiting list."""
    try:
        # Fetch all patients in the medicine waiting list who are not yet seen
        waiting_list = (
            PatientWaitingList.query.filter_by(seen=4)
            .options(joinedload(PatientWaitingList.patient))
            .all()
        )
        # Filter out invalid entries (e.g., missing patient relationships)
        valid_waiting_list = [entry for entry in waiting_list if entry.patient]
        if not valid_waiting_list:
            flash(
                "No patients in the medicine waiting list.", "info"
            )  # Inform user if list is empty

        # KPI stats for dashboard
        try:
            total_inpatients = AdmittedPatient.query.filter(
                AdmittedPatient.discharged_on.is_(None)
            ).count()
            pending_labs = RequestedLab.query.filter_by(status=0).count()
            pending_imaging = RequestedImage.query.filter_by(status=0).count()
            theatre_pending = TheatreList.query.filter_by(status=0).count()
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error calculating dashboard KPIs: {e}")
            total_inpatients = pending_labs = pending_imaging = theatre_pending = 0

        return render_template(
            "medicine/index.html",
            waiting_list=valid_waiting_list,
            total_inpatients=total_inpatients,
            pending_labs=pending_labs,
            pending_imaging=pending_imaging,
            theatre_pending=theatre_pending,
        )
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error in medicine.index: {e}")
        flash("Something went wrong loading the clinical dashboard.", "error")
        return render_template(
            "medicine/index.html",
            waiting_list=[],
            total_inpatients=0,
            pending_labs=0,
            pending_imaging=0,
            theatre_pending=0,
        )


# View or submit SOAP notes for a specific patient
@bp.route("/soap_notes/<patient_id>", methods=["GET"])
@login_required
@roles_required("medicine", "admin")
def soap_notes(patient_id):
    """View or submit SOAP notes for a specific patient."""
    try:
        # Fetch the patient from the waiting list
        patient_entry = (
            PatientWaitingList.query.filter_by(patient_id=patient_id)
            .options(joinedload(PatientWaitingList.patient))
            .first()
        )
        if not patient_entry or not patient_entry.patient:
            flash(
                f"Patient with ID {patient_id} not found in the waiting list!", "error"
            )
            return redirect(
                url_for("medicine.index")
            )  # Redirect to index if patient not found
        patient = patient_entry.patient

        # Generate a unique prescription_id for the form
        import uuid

        prescription_id = str(uuid.uuid4())

        # Fetch existing SOAP notes and other related data
        soap_notes = SOAPNote.query.filter_by(patient_id=patient_id).all()
        prescribed_medicines = PrescribedMedicine.query.filter_by(
            patient_id=patient_id
        ).all()
        requested_labs = RequestedLab.query.filter_by(patient_id=patient_id).all()
        requested_images = RequestedImage.query.filter_by(patient_id=patient_id).all()

        # Fetch available lab tests, imaging types, and drugs for dropdowns
        lab_tests = LabTest.query.all()
        imaging_types = Imaging.query.all()
        drugs = Medicine.query.all()

        return render_template(
            "medicine/soap_notes.html",
            patient=patient,
            soap_notes=soap_notes,
            prescribed_medicines=prescribed_medicines,
            requested_labs=requested_labs,
            requested_images=requested_images,
            lab_tests=lab_tests,
            imaging_types=imaging_types,
            drugs=drugs,
            prescription_id=prescription_id,  # Pass the prescription_id to the template
        )
    except Exception as e:
        flash("Something went wrong. Please try again.", "error")
        print(f"Debug: Error in medicine.soap_notes: {e}")  # Debugging
        return redirect(url_for("medicine.index"))  # Redirect to index on error


@bp.route("/lab_patients")
def lab_patients():
    # Query requested labs with status=1 and existing results
    requested_labs = (
        db.session.query(
            RequestedLab,
            LabTest.test_name,
            Patient.name,
            Patient.patient_id,
            LabResult.result_id,
        )
        .join(LabTest, RequestedLab.lab_test_id == LabTest.id)
        .join(Patient, RequestedLab.patient_id == Patient.patient_id)
        .outerjoin(
            LabResult,
            (RequestedLab.patient_id == LabResult.patient_id)
            & (RequestedLab.lab_test_id == LabResult.lab_test_id),
        )
        .filter(RequestedLab.status == 1, LabResult.result_id.isnot(None))
        .order_by(RequestedLab.date_requested.desc())
        .all()
    )

    requested_labs_data = []
    for lab, test_name, patient_name, patient_id, result_id in requested_labs:
        date_requested = lab.date_requested
        if isinstance(date_requested, str):
            try:
                date_requested = datetime.strptime(date_requested, "%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                date_requested = None

        requested_labs_data.append(
            {
                "test_name": test_name,
                "date_requested": date_requested.strftime("%Y-%m-%d %H:%M:%S")
                if date_requested
                else "N/A",
                "status": "Completed",
                "patient_name": patient_name or "N/A",
                "patient_id": patient_id,
                "result_id": result_id,
            }
        )

    return render_template(
        "medicine/lab_patients.html", requested_labs=requested_labs_data
    )


@bp.route("/lab_results/<result_id>")
def lab_results(result_id):
    # Query lab result for the specific result_id
    result = (
        db.session.query(LabResult, LabTest.test_name, Patient.name)
        .join(LabTest, LabResult.lab_test_id == LabTest.id)
        .join(Patient, LabResult.patient_id == Patient.patient_id)
        .filter(LabResult.result_id == result_id)
        .first()
    )

    # If no result found, return an error message
    if not result:
        flash(f"No lab result found for Result ID {result_id}.", "danger")
        return render_template(
            "medicine/lab_results.html", lab_results={}, patient_name="N/A"
        )

    lab_result, test_name, patient_name = result
    patient_name = patient_name or "N/A"

    # Process the lab result
    test_presentations = {}
    test_presentations[lab_result.result_id] = process_lab_result(lab_result, test_name)

    return render_template(
        "medicine/lab_results.html",
        lab_results=test_presentations,
        patient_name=patient_name,
    )


@bp.route("/pending_lab_patients")
def pending_lab_patients():
    # Query requested labs with status=0 and no results
    requested_labs = (
        db.session.query(
            RequestedLab,
            LabTest.test_name,
            Patient.name,
            Patient.patient_id,
            LabResult.result_id,
        )
        .join(LabTest, RequestedLab.lab_test_id == LabTest.id)
        .join(Patient, RequestedLab.patient_id == Patient.patient_id)
        .outerjoin(
            LabResult,
            (RequestedLab.patient_id == LabResult.patient_id)
            & (RequestedLab.lab_test_id == LabResult.lab_test_id),
        )
        .filter(RequestedLab.status == 0, LabResult.result_id.is_(None))
        .order_by(RequestedLab.date_requested.desc())
        .all()
    )

    requested_labs_data = []
    for lab, test_name, patient_name, patient_id, result_id in requested_labs:
        date_requested = lab.date_requested
        if isinstance(date_requested, str):
            try:
                date_requested = datetime.strptime(date_requested, "%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                date_requested = None

        requested_labs_data.append(
            {
                "test_name": test_name,
                "date_requested": date_requested.strftime("%Y-%m-%d %H:%M:%S")
                if date_requested
                else "N/A",
                "status": "not yet done",
                "patient_name": patient_name or "N/A",
                "patient_id": patient_id,
                "result_id": result_id,
            }
        )

    return render_template(
        "medicine/pending_lab_patients.html", requested_labs=requested_labs_data
    )


@bp.route("/patient_lab_results/<patient_id>")
def patient_lab_results(patient_id):
    # Query all lab results for the specific patient_id with status=1 and existing results
    results = (
        db.session.query(LabResult, LabTest.test_name, Patient.name)
        .join(LabTest, LabResult.lab_test_id == LabTest.id)
        .join(Patient, LabResult.patient_id == Patient.patient_id)
        .join(
            RequestedLab,
            (LabResult.patient_id == RequestedLab.patient_id)
            & (LabResult.lab_test_id == RequestedLab.lab_test_id),
        )
        .filter(
            LabResult.patient_id == patient_id,
            RequestedLab.status == 1,
            LabResult.result_id.isnot(None),
        )
        .all()
    )

    # If no results found, return an error message
    if not results:
        flash(f"No lab results found for Patient ID {patient_id}.", "danger")
        return render_template(
            "medicine/patient_lab_results.html", lab_results={}, patient_name="N/A"
        )

    # Correctly extract patient_name from the query results
    patient_name = results[0][2] or "N/A"
    test_presentations = {}

    for lab_result, test_name, _ in results:
        test_presentations[lab_result.result_id] = process_lab_result(
            lab_result, test_name
        )

    return render_template(
        "medicine/patient_lab_results.html",
        lab_results=test_presentations,
        patient_name=patient_name,
    )
