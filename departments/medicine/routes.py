# departments/nlp/soap_notes.py
from flask import render_template, redirect, url_for, request, flash, jsonify,session
from flask_wtf import FlaskForm
from sqlalchemy import func
from wtforms import SelectField
from wtforms.validators import DataRequired
from sqlalchemy import text
import json
from typing import Optional, List, Dict, Any
import psycopg2
from datetime import date
from psycopg2.extras import RealDictCursor
from flask import current_app
from flask_login import login_required, current_user
from flask_wtf.csrf import CSRFProtect
from extensions import db
from flask import session
from flask_socketio import SocketIO
import uuid
from uuid import uuid4
from sqlalchemy.orm import joinedload
import bleach 
from . import bp
from departments.forms import PatientSearchForm, OncoPatientForm, OncologyNoteForm, AdmitPatientForm
import os
from datetime import datetime
from departments.models.laboratory import LabResult,LabResultTemplate
from departments.models.records import PatientWaitingList, Patient
from departments.models.medicine import (
    SOAPNote, LabTest, Imaging, Medicine, PrescribedMedicine, RequestedLab, 
    RequestedImage, UnmatchedImagingRequest, TheatreProcedure, TheatreList, 
    Ward, AdmittedPatient, SpecialWarning,RegimenDrugAssociation, 
    OncologyBooking, OncoDrugCategory, RegimenCategory, 
    WardBedHistory, WardRoom, Bed, WardRound,Disease, 
    DiseaseManagementPlan, DiseaseLab, OncoPatient, 
    OncologyDrug, OncologyRegimen, OncoPrescription, 
    OncoTreatmentRecord,PrescriptionDrugDetail,OncologyNote,
    CancerType, CancerStage, CancerTypeStage, CancerDetail
)
import logging
import json
from departments.nlp.logging_setup import get_logger
logger = get_logger()

socketio = SocketIO()
prescription_id = str(uuid.uuid4())
csrf = CSRFProtect()

@bp.route('/submit_soap_notes/<patient_id>', methods=['POST'])
@login_required
def submit_soap_notes(patient_id):
    if current_user.role not in ['medicine', 'admin']:
        flash('You do not have permission to access this page.', 'error')
        return redirect(url_for('home'))

    try:
        situation = request.form.get('situation')
        hpi = request.form.get('hpi')
        aggravating_factors = request.form.get('aggravating_factors')
        alleviating_factors = request.form.get('alleviating_factors')
        medical_history = request.form.get('medical_history')
        medication_history = request.form.get('medication_history')
        assessment = request.form.get('assessment')
        recommendation = request.form.get('recommendation')
        additional_notes = request.form.get('additional_notes')
        symptoms = request.form.get('symptoms', '')  # New symptoms field

        file = request.files.get('file_upload')
        file_path = None
        if file and file.filename:
            upload_folder = os.path.join(current_app.root_path, 'Uploads')
            os.makedirs(upload_folder, exist_ok=True)
            file_path = os.path.join('Uploads', file.filename)
            file.save(os.path.join(upload_folder, file.filename))

        if not all([situation, hpi, assessment, recommendation]):
            flash('All required fields must be filled out!', 'error')
            return redirect(url_for('medicine.soap_notes', patient_id=patient_id))

        # Process symptoms
        symptom_list = []
        if symptoms:
            if isinstance(symptoms, str):
                symptom_list = [s.strip() for s in symptoms.split(',') if s.strip()]
            elif isinstance(symptoms, list):
                symptom_list = [s.strip() for s in symptoms if isinstance(s, str) and s.strip()]
            logger.debug(f"Received symptoms for patient {patient_id}: {symptom_list}")

        # Create SOAP note
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
            symptoms=json.dumps(symptom_list) if symptom_list else '',  # Store as JSON string
            file_path=file_path,
            ai_notes=None,  # Save as NULL
            ai_analysis=None  # Save as NULL
        )



        db.session.add(new_soap_note)
        db.session.commit()



        imaging_keywords = ["ct", "mri", "x-ray", "ultrasound", "pet", "scan"]
        words = recommendation.lower().split()
        matched_imaging = set()
        for i, word in enumerate(words):
            for keyword in imaging_keywords:
                if keyword in word:
                    phrase = " ".join(words[i:i+2]) if i + 1 < len(words) else word
                    matched_imaging.add(phrase)

        unmatched_requests = []
        for imaging_request in matched_imaging:
            imaging_match = Imaging.query.filter(Imaging.imaging_type.ilike(f"%{imaging_request}%")).first()
            if imaging_match:
                requested_imaging = RequestedImage(
                    patient_id=patient_id,
                    imaging_id=imaging_match.id,
                    description=recommendation
                )
                db.session.add(requested_imaging)
            else:
                unmatched_request = UnmatchedImagingRequest(
                    patient_id=patient_id,
                    description=imaging_request
                )
                db.session.add(unmatched_request)
                unmatched_requests.append(imaging_request)

        db.session.commit()

        if unmatched_requests:
            try:
                message = f"Unmatched imaging requests for patient {patient_id}: {', '.join(unmatched_requests)}"
                notify_admin(message)
            except Exception as e:
                logger.error(f"Failed to notify admin: {str(e)}")
            flash(f"The following imaging requests need manual review: {', '.join(unmatched_requests)}", 'warning')

        return redirect(url_for('medicine.notes', patient_id=patient_id))

    except Exception as e:
        flash(f'Error submitting SOAP notes: {str(e)}', 'error')
        db.session.rollback()
        logger.error(f"Error in submit_soap_notes for patient {patient_id}: {str(e)}")
        return redirect(url_for('medicine.soap_notes', patient_id=patient_id))
@bp.route('/notes/<string:patient_id>', methods=['GET']) 
@login_required
def notes(patient_id):
    """Displays SOAP notes for a patient."""
    if current_user.role not in ['medicine', 'admin']:
        flash('You do not have permission to access this page.', 'error')
        return redirect(url_for('medicine.index'))

    patient = Patient.query.filter_by(patient_id=patient_id).first_or_404()
    last_soap_note = SOAPNote.query.filter_by(patient_id=patient_id).order_by(SOAPNote.created_at.desc()).first()
    return render_template('medicine/notes.html', patient=patient, soap_notes=last_soap_note)      
@bp.route('/notes/<int:note_id>/reprocess', methods=['POST'])
@login_required
def reprocess_note(note_id):
    if current_user.role not in ['medicine', 'admin']:
        flash('You do not have permission to perform this action.', 'error')
        return redirect(url_for('medicine.index'))

    return redirect(url_for('medicine.notes', patient_id=SOAPNote.query.get(note_id).patient_id))


# Display the medicine waiting list
@bp.route('/')
@login_required
def index():
    """Display the medicine waiting list."""
    if current_user.role not in ['medicine', 'admin']: 
        flash('You do not have permission to access this page.', 'error')
        return redirect(url_for('home'))  # Redirect to home if role is invalid
    try:
        # Fetch all patients in the medicine waiting list who are not yet seen
        waiting_list = PatientWaitingList.query.filter_by(seen=4).options(
            joinedload(PatientWaitingList.patient)
        ).all()
        # Filter out invalid entries (e.g., missing patient relationships)
        valid_waiting_list = [
            entry for entry in waiting_list if entry.patient
        ]
        if not valid_waiting_list:
            flash('No patients in the medicine waiting list.', 'info')  # Inform user if list is empty
        return render_template('medicine/index.html', waiting_list=valid_waiting_list)
    except Exception as e:
        flash(f'Error fetching the waiting list: {e}', 'error')
        print(f"Debug: Error in medicine.index: {e}")  # Debugging
        return redirect(url_for('medicine.index'))  # Redirect to home on error


# View or submit SOAP notes for a specific patient
@bp.route('/soap_notes/<patient_id>', methods=['GET'])
@login_required
def soap_notes(patient_id):
    """View or submit SOAP notes for a specific patient."""
    if current_user.role not in ['medicine', 'admin']:
        flash('You do not have permission to access this page.', 'error')
        return redirect(url_for('home'))  # Redirect to home if role is invalid
    try:
        # Fetch the patient from the waiting list
        patient_entry = PatientWaitingList.query.filter_by(patient_id=patient_id).options(
            joinedload(PatientWaitingList.patient)
        ).first()
        if not patient_entry or not patient_entry.patient:
            flash(f'Patient with ID {patient_id} not found in the waiting list!', 'error')
            return redirect(url_for('medicine.index'))  # Redirect to index if patient not found
        patient = patient_entry.patient

        # Generate a unique prescription_id for the form
        

        # Fetch existing SOAP notes and other related data
        soap_notes = SOAPNote.query.filter_by(patient_id=patient_id).all()
        prescribed_medicines = PrescribedMedicine.query.filter_by(patient_id=patient_id).all()
        requested_labs = RequestedLab.query.filter_by(patient_id=patient_id).all()
        requested_images = RequestedImage.query.filter_by(patient_id=patient_id).all()

        # Fetch available lab tests, imaging types, and drugs for dropdowns
        lab_tests = LabTest.query.all()
        imaging_types = Imaging.query.all()
        drugs = Medicine.query.all()

        return render_template(
            'medicine/soap_notes.html',
            patient=patient,
            soap_notes=soap_notes,
            prescribed_medicines=prescribed_medicines,
            requested_labs=requested_labs,
            requested_images=requested_images,
            lab_tests=lab_tests,
            imaging_types=imaging_types,
            drugs=drugs,
            prescription_id=prescription_id  # Pass the prescription_id to the template
        )
    except Exception as e:
        flash(f'Error processing SOAP notes: {e}', 'error')
        print(f"Debug: Error in medicine.soap_notes: {e}")  # Debugging
        return redirect(url_for('medicine.index'))  # Redirect to index on error

@bp.route('/request_lab_tests/<patient_id>', methods=['GET', 'POST'])
@login_required
def request_lab_tests(patient_id):
    """Handles lab test requests."""
    if current_user.role not in ['medicine', 'admin']:
        flash('You do not have permission to access this page.', 'error')
        return redirect(url_for('login'))

    try:
        dept = request.args.get('dept')  # ✅ Capture dept from query string

        # Fetch patient from waiting list
        patient_entry = PatientWaitingList.query.filter_by(patient_id=patient_id).options(
            joinedload(PatientWaitingList.patient)
        ).first()
        if not patient_entry or not patient_entry.patient:
            flash(f'Patient with ID {patient_id} not found in the waiting list!', 'error')
            return redirect(url_for('medicine.index'))

        patient = patient_entry.patient
        lab_tests = LabTest.query.all()

        if request.method == 'POST':
            lab_test_ids = request.form.getlist('lab_tests[]')
            if not lab_test_ids:
                flash('No lab tests selected!', 'error')
                return render_template(
                    'medicine/request_lab_tests.html',
                    patient=patient,
                    lab_tests=lab_tests,
                    dept=dept
                )

            descriptions = {}
            for key, value in request.form.items():
                if key.startswith('descriptions['):
                    lab_id = key.split('[')[1].split(']')[0]
                    descriptions[lab_id] = value.strip() if value else ''

            for lab_test_id in lab_test_ids:
                description = descriptions.get(str(lab_test_id), '').strip()

                if len(description) > 500:
                    flash(f'Description for lab test ID {lab_test_id} exceeds 500 characters.', 'error')
                    return render_template(
                        'medicine/request_lab_tests.html',
                        patient=patient,
                        lab_tests=lab_tests,
                        dept=dept
                    )

                result_id = str(uuid.uuid4())

                new_lab_request = RequestedLab(
                    patient_id=patient_id,
                    lab_test_id=lab_test_id,
                    result_id=result_id,
                    description=description or None
                )
                db.session.add(new_lab_request)

            db.session.commit()
            flash('Lab tests requested successfully!', 'success')

            # ✅ Redirect accordingly
            if dept == '1':
                return redirect(url_for('medicine.ward_rounds'))
            else:
                return redirect(url_for('medicine.soap_notes', patient_id=patient_id))

        # GET request
        return render_template(
            'medicine/request_lab_tests.html',
            patient=patient,
            lab_tests=lab_tests,
            dept=dept
        )

    except Exception as e:
        db.session.rollback()
        logger.error(f"Error in medicine.request_lab_tests: {e}", exc_info=True)
        flash(f'An error occurred: {e}', 'error')
        return redirect(url_for('medicine.soap_notes', patient_id=patient_id))

@bp.route('/request_imaging/<patient_id>', methods=['GET', 'POST'])
@login_required
def request_imaging(patient_id):
    """Handles imaging requests."""
    if current_user.role not in ['medicine', 'admin']:
        flash('You do not have permission to access this page.', 'error')
        return redirect(url_for('login'))

    try:
        dept = request.args.get('dept')  # ✅ Capture dept from query string

        # Fetch patient from waiting list
        patient_entry = PatientWaitingList.query.filter_by(patient_id=patient_id).options(
            joinedload(PatientWaitingList.patient)
        ).first()
        if not patient_entry or not patient_entry.patient:
            flash(f'Patient with ID {patient_id} not found in the waiting list!', 'error')
            return redirect(url_for('medicine.index'))

        patient = patient_entry.patient
        soap_notes = SOAPNote.query.filter_by(patient_id=patient_id).order_by(SOAPNote.created_at.desc()).first()
        imaging_types = Imaging.query.all()

        if request.method == 'POST':
            imaging_ids = request.form.getlist('imaging_types[]')

            if not imaging_ids:
                flash('No imaging types selected!', 'error')
                return render_template(
                    'medicine/request_imaging.html',
                    patient=patient,
                    soap_notes=soap_notes,
                    imaging_types=imaging_types,
                    dept=dept
                )

            descriptions = {}
            for key, value in request.form.items():
                if key.startswith('descriptions['):
                    imaging_id = key.split('[')[1].split(']')[0]
                    descriptions[imaging_id] = value.strip() if value else ''

            for imaging_id in imaging_ids:
                description = descriptions.get(str(imaging_id), '').strip()

                if len(description) > 500:
                    flash(f'Description for imaging ID {imaging_id} exceeds 500 characters.', 'error')
                    return render_template(
                        'medicine/request_imaging.html',
                        patient=patient,
                        soap_notes=soap_notes,
                        imaging_types=imaging_types,
                        dept=dept
                    )

                result_id = str(uuid.uuid4())
                new_image_request = RequestedImage(
                    patient_id=patient_id,
                    imaging_id=imaging_id,
                    result_id=result_id,
                    description=description or None
                )
                db.session.add(new_image_request)

            db.session.commit()
            flash('Imaging requested successfully!', 'success')

            # ✅ Redirect based on dept
            if dept == '1':
                return redirect(url_for('medicine.ward_rounds'))
            else:
                return redirect(url_for('medicine.soap_notes', patient_id=patient_id))

        # GET: Render form
        return render_template(
            'medicine/request_imaging.html',
            patient=patient,
            soap_notes=soap_notes,
            imaging_types=imaging_types,
            dept=dept
        )

    except Exception as e:
        db.session.rollback()
        logger.error(f"Error in medicine.request_imaging: {e}", exc_info=True)
        flash(f'An error occurred: {e}', 'error')
        return redirect(url_for('medicine.soap_notes', patient_id=patient_id))

@bp.route('/prescribe_drugs/<patient_id>', methods=['GET', 'POST'])
@login_required
def prescribe_drugs(patient_id):
    """Handles drug prescription requests for admitted and waiting list patients."""

    if current_user.role not in ['medicine', 'admin']:
        flash('You do not have permission to access this page.', 'error')
        return redirect(url_for('home'))

    try:
        patient_id = str(patient_id).strip()
        logger.debug(f"Checking patient_id={patient_id}")

        # Get department from GET or POST
        dept = request.args.get('dept') or request.form.get('dept')

        # Check if patient is admitted or in waiting list
        admitted_patient = AdmittedPatient.query.filter_by(patient_id=patient_id, discharged_on=None).first()
        waiting_patient = PatientWaitingList.query.filter_by(patient_id=patient_id).first()

        if not admitted_patient and not waiting_patient:
            flash(f'Patient with ID {patient_id} is neither admitted nor in the waiting list.', 'error')
            return redirect(url_for('medicine.index'))

        # Get patient record
        patient = Patient.query.filter_by(patient_id=patient_id).first()
        if not patient:
            flash(f'Patient with ID {patient_id} not found in the system!', 'error')
            return redirect(url_for('medicine.index'))

        # Generate new prescription session if missing
        if 'prescription_id' not in session:
            session['prescription_id'] = str(uuid.uuid4())
        prescription_id = session['prescription_id']

        drugs = Medicine.query.all()

        if request.method == 'POST':
            drugs_selected = request.form.getlist('drugs[]')
            dosage_form = request.form.get('dosage_form')
            strength = request.form.get('strength')
            frequency = request.form.get('frequency')
            custom_frequency = request.form.get('custom_frequency')
            num_days = request.form.get('num_days')

            # Validation
            if not all([drugs_selected, dosage_form, strength, frequency, num_days]):
                flash('All fields are required for prescribing drugs!', 'error')
                return render_template(
                    'medicine/prescribe_drugs.html',
                    patient=patient,
                    drugs=drugs,
                    prescription_id=prescription_id,
                    prescribed_medicines=[],
                    dept=dept
                )

            # Save prescriptions
            for drug_id in drugs_selected:
                prescribed = PrescribedMedicine(
                    patient_id=patient.patient_id,
                    medicine_id=drug_id,
                    dosage=dosage_form,
                    strength=strength.strip(),
                    frequency=custom_frequency.strip() if frequency == "Other" else frequency,
                    prescription_id=prescription_id,
                    num_days=int(num_days)
                )
                db.session.add(prescribed)

            try:
                db.session.commit()
   
            except Exception as e:
                db.session.rollback()
                logger.error(f"Database commit failed: {e}")
                flash(f'Database error: {e}', 'error')
                return redirect(url_for('medicine.prescribe_drugs', patient_id=patient_id, dept=dept))

        # Fetch prescribed medicines
        prescribed_medicines = PrescribedMedicine.query.filter_by(
            prescription_id=prescription_id
        ).all()

        return render_template(
            'medicine/prescribe_drugs.html',
            patient=patient,
            drugs=drugs,
            prescribed_medicines=prescribed_medicines,
            prescription_id=prescription_id,
            dept=dept
        )

    except Exception as e:
        db.session.rollback()
        logger.exception("Unexpected error in prescribe_drugs")
        flash(f'Error prescribing drugs: {e}', 'error')
        return redirect(url_for('medicine.index'))
@bp.route('/edit_prescribed_medicine/<medicine_id>', methods=['GET', 'POST'])
@login_required
def edit_prescribed_medicine(medicine_id):
    """Handles editing a prescribed medicine."""
    if current_user.role not in ['medicine', 'admin']:
        flash('You do not have permission to access this page.', 'error')
        return redirect(url_for('home'))

    try:
        # Get dept from either GET or POST
        dept = request.args.get('dept') or request.form.get('dept')
        prescription_id = request.form.get('prescription_id') or request.args.get('prescription_id')

        prescribed_medicine = PrescribedMedicine.query.get(medicine_id)
        if not prescribed_medicine:
            flash('Prescribed medicine not found.', 'error')
            return redirect(url_for('medicine.index'))

        if request.method == 'POST':
            dosage_form = request.form.get('dosage_form')
            custom_dosage = request.form.get('custom_dosage')
            strength = request.form.get('strength')
            frequency = request.form.get('frequency')
            custom_frequency = request.form.get('custom_frequency')
            num_days = request.form.get('num_days')

            if not all([dosage_form, strength, frequency, num_days]):
                flash('All fields are required!', 'error')
                return redirect(url_for(
                    'medicine.edit_prescribed_medicine',
                    medicine_id=medicine_id,
                    prescription_id=prescription_id,
                    dept=dept
                ))

            prescribed_medicine.dosage = custom_dosage.strip() if dosage_form == "Other" else dosage_form
            prescribed_medicine.strength = strength.strip()
            prescribed_medicine.frequency = custom_frequency.strip() if frequency == "Other" else frequency
            prescribed_medicine.num_days = int(num_days)
            db.session.commit()
            flash('Prescribed medicine updated successfully!', 'success')
            return redirect(url_for('medicine.prescribe_drugs', patient_id=prescribed_medicine.patient_id, dept=dept))

        return render_template(
            'medicine/edit_prescribed_medicine.html',
            prescribed_medicine=prescribed_medicine,
            dept=dept
        )

    except Exception as e:
        logger.exception("Error editing prescribed medicine")
        flash(f'Error editing prescribed medicine: {e}', 'error')
        return redirect(url_for('medicine.index'))

@bp.route('/delete_prescribed_medicine/<medicine_id>', methods=['POST'])
@login_required
def delete_prescribed_medicine(medicine_id):
    """Handles deleting a prescribed medicine."""
    if current_user.role not in ['medicine', 'admin']:
        flash('You do not have permission to access this page.', 'error')
        return redirect(url_for('home'))

    # Get from GET or POST
    prescription_id = request.args.get('prescription_id') or request.form.get('prescription_id')
    dept = request.args.get('dept') or request.form.get('dept')

    try:
        prescribed_medicine = PrescribedMedicine.query.get_or_404(medicine_id)
        patient_id = prescribed_medicine.patient_id
        db.session.delete(prescribed_medicine)
        db.session.commit()
        flash('Prescribed medicine deleted successfully!', 'success')

    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting prescribed medicine: {e}")
        flash(f'Error deleting prescribed medicine: {e}', 'error')
        patient_id = prescribed_medicine.patient_id if prescribed_medicine else None

    return redirect(url_for('medicine.prescribe_drugs', patient_id=patient_id, prescription_id=prescription_id, dept=dept))

@bp.route('/save_prescription/<prescription_id>/<patient_id>', methods=['POST'])
@login_required
def save_prescription(prescription_id, patient_id):
    """Finalize the prescription and redirect appropriately based on dept."""
    if current_user.role not in ['medicine', 'admin']:
        flash('You do not have permission to perform this action.', 'error')
        return redirect(url_for('home'))

    try:
        # Get optional dept from query or form
        dept = request.args.get('dept') or request.form.get('dept')

        # Optional: mark prescription as finalized in DB here

        # ✅ Clear current prescription session
        session.pop('prescription_id', None)

        flash('Prescription finalized and saved successfully.', 'success')

        # Redirect based on dept value
        if dept == "1":
            return redirect(url_for('medicine.ward_rounds'))
        else:
            return redirect(url_for('medicine.index'))

    except Exception as e:
        logger.error(f"Error finalizing prescription: {e}")
        flash(f'Failed to finalize prescription: {e}', 'error')
        return redirect(url_for('medicine.prescribe_drugs', patient_id=patient_id, dept=dept))  
@bp.route('/get_edit_form', methods=['GET'])
@login_required
def get_edit_form():
    """Serves the edit form for a prescribed medicine."""
    if current_user.role not in ['medicine', 'admin']:
        flash('You do not have permission to access this page.', 'error')
        return redirect(url_for('home'))

    try:
        medicine_id = request.args.get('medicine_id')
        prescribed_medicine = PrescribedMedicine.query.get_or_404(medicine_id)
        return render_template(
            'medicine/edit_prescribed_medicine.html',
            prescribed_medicine=prescribed_medicine
        )
    except Exception as e:
        print(f"Debug: Error in medicine.get_edit_form: {e}")
        return "Error loading edit form."                 

@bp.route('/unmatched_imaging', methods=['GET', 'POST'])
@login_required
def unmatched_imaging():
    """Medicine panel to match unmatched imaging requests."""
    if current_user.role not in ['medicine', 'admin']:  # Now only 'medicine' role can access
        flash('Unauthorized access!', 'error')
        return redirect(url_for('home'))

    # Handle form submission (matching requests)
    if request.method == 'POST':
        unmatched_id = request.form.get('unmatched_id')
        imaging_id = request.form.get('imaging_id')

        if unmatched_id and imaging_id:
            unmatched_request = UnmatchedImagingRequest.query.get(unmatched_id)
            if unmatched_request:
                # Move to requested_images table
                requested_imaging = RequestedImage(
                    patient_id=unmatched_request.patient_id,
                    imaging_id=imaging_id,
                    description=unmatched_request.description
                )
                db.session.add(requested_imaging)
                
                # Remove from unmatched list
                db.session.delete(unmatched_request)
                db.session.commit()
                flash('Imaging request successfully matched!', 'success')
            else:
                flash('Invalid request!', 'error')

    # Get filtering parameters
    patient_name = request.args.get('patient_name', '').strip()
    start_date = request.args.get('start_date', '')
    end_date = request.args.get('end_date', '')

    # Base query for unmatched imaging requests
    unmatched_requests = UnmatchedImagingRequest.query.join(Patient).order_by(UnmatchedImagingRequest.date_requested.desc())

    # Apply filters if provided
    if patient_name:
        unmatched_requests = unmatched_requests.filter(Patient.name.ilike(f"%{patient_name}%"))
    if start_date:
        unmatched_requests = unmatched_requests.filter(UnmatchedImagingRequest.date_requested >= start_date)
    if end_date:
        unmatched_requests = unmatched_requests.filter(UnmatchedImagingRequest.date_requested <= end_date)

    unmatched_requests = unmatched_requests.all()
    imaging_options = Imaging.query.all()

    return render_template(
        'unmatched_imaging.html',
        unmatched_requests=unmatched_requests, 
        imaging_options=imaging_options,
        patient_name=patient_name,
        start_date=start_date,
        end_date=end_date
    )

@bp.route('/unmatched_imaging/notify', methods=['GET', 'POST'])
@login_required
def notify_admin():
    """Notify medicine users by updating badge count via SocketIO."""
    count = UnmatchedImagingRequest.query.count()
    socketio.emit('update_badge', {'count': count}, namespace='/medicine')  # Updated namespace
    return '', 204  # Return a success response without content

def get_unmatched_count():
    """Get the number of unmatched imaging requests."""
    return UnmatchedImagingRequest.query.count()

@bp.context_processor
def inject_unmatched_count():
    """Inject the unmatched count into the template context."""
    return dict(unmatched_count=get_unmatched_count()) 

db_params = {
    'dbname': 'drugcentral',
    'user': 'drugman',
    'password': 'dosage',
    'host': 'unmtid-dbs.net',
    'port': '5433'
}



def get_db_connection():
    """Get database connection with context management."""
    return psycopg2.connect(**db_params)


def fetch_drugs_data(search_query: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetch distinct product data with optional search by generic name or brand name."""
    try:
        with get_db_connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
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
@bp.route('/drugs-ref/search', methods=['GET'])
def drugs_ref():
    """Drugs reference route with search functionality."""
    search_query = request.args.get('search', '').strip()
    drugs_data = fetch_drugs_data(search_query)
    
    return render_template('medicine/drugs_ref.html', drugs_data=drugs_data, search_query=search_query)     

@bp.route('/drugs-ref/details/<drug>', methods=['GET'])
def drug_details(drug: str):
    """Fetch and display detailed information about a specific active ingredient."""
    try:
        with get_db_connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Normalize the drug name to uppercase for consistency
            normalized_drug = drug.upper()

            # Table name mapping for user-friendly display
            TABLE_NAME_MAPPING = {
                "faers": "Side Effects",
                "faers_male": "Side Effects (Male)",
                "faers_female": "Side Effects (Female)",
                "faers_ped": "Side Effects (Pediatric)",
                "faers_ger": "Side Effects (Geriatric)",
                "approval": "Regulatory Approvals",
                "ob_patent_view": "Patents",
                "ob_exclusivity_view": "Exclusivity Data",
                "active_ingredient": "Active Ingredients",
                "pharma_class": "Pharmacological Class",
                "act_table_full": "Drug-Target Interactions",
                "pka": "pKa Values",
                "pdb": "Protein Data Bank (PDB) Structures",
                "atc_ddd": "ATC Classification & Defined Daily Dose",
                "struct2obprod": "Marketed Drug Products",
                "struct2atc": "ATC Codes",
                "omop_relationship": "Clinical Data Relationships",
            }

            # Step 1: Fetch struct_id from the active_ingredient table
            cur.execute("""
                SELECT DISTINCT struct_id
                FROM active_ingredient
                WHERE UPPER(substance_name) = %s
            """, [normalized_drug])
            result = cur.fetchone()

            if not result:
                return render_template('medicine/error.html', message=f"No details found for {drug}.")

            struct_id = result["struct_id"]

            # Step 2: Fetch data from all tables using struct_id
            query = """
                SELECT 'active_ingredient' AS table_name, struct_id::TEXT, substance_name::TEXT, quantity::TEXT, unit::TEXT, NULL::TEXT 
                FROM active_ingredient WHERE struct_id = %s
                UNION ALL
                SELECT 'approval' AS table_name, struct_id::TEXT, approval::TEXT, applicant::TEXT, type::TEXT, orphan::TEXT 
                FROM approval WHERE struct_id = %s
                UNION ALL
                SELECT 'faers_male' AS table_name, struct_id::TEXT, meddra_name::TEXT, drug_ae::TEXT, llr_threshold::TEXT, level::TEXT 
                FROM faers_male WHERE struct_id = %s
                UNION ALL
                SELECT 'faers_ped' AS table_name, struct_id::TEXT, meddra_name::TEXT, drug_ae::TEXT, llr_threshold::TEXT, level::TEXT 
                FROM faers_ped WHERE struct_id = %s
                UNION ALL
                SELECT 'omop_relationship' AS table_name, struct_id::TEXT, concept_name::TEXT, cui_semantic_type::TEXT, relationship_name::TEXT, umls_cui::TEXT 
                FROM omop_relationship WHERE struct_id = %s
                UNION ALL
                SELECT 'faers' AS table_name, struct_id::TEXT, meddra_name::TEXT, drug_ae::TEXT, llr_threshold::TEXT, level::TEXT 
                FROM faers WHERE struct_id = %s
                UNION ALL
                SELECT 'atc_ddd' AS table_name, struct_id::TEXT, atc_code::TEXT, route::TEXT, ddd::TEXT, unit_type::TEXT 
                FROM atc_ddd WHERE struct_id = %s
                UNION ALL
                SELECT 'faers_female' AS table_name, struct_id::TEXT, meddra_name::TEXT, drug_ae::TEXT, llr_threshold::TEXT, level::TEXT 
                FROM faers_female WHERE struct_id = %s
                UNION ALL
                SELECT 'faers_ger' AS table_name, struct_id::TEXT, meddra_name::TEXT, drug_ae::TEXT, llr_threshold::TEXT, level::TEXT 
                FROM faers_ger WHERE struct_id = %s
                UNION ALL
                SELECT 'struct2obprod' AS table_name, struct_id::TEXT, prod_id::TEXT, strength::TEXT, NULL::TEXT, NULL::TEXT 
                FROM struct2obprod WHERE struct_id = %s
                UNION ALL
                SELECT 'pdb' AS table_name, struct_id::TEXT, pdb::TEXT, ligand_id::TEXT, accession::TEXT, pubmed_id::TEXT 
                FROM pdb WHERE struct_id = %s
                UNION ALL
                SELECT 'pharma_class' AS table_name, struct_id::TEXT, class_code::TEXT, source::TEXT, name::TEXT, NULL::TEXT 
                FROM pharma_class WHERE struct_id = %s
                UNION ALL
                SELECT 'pka' AS table_name, struct_id::TEXT, value::TEXT, pka_type::TEXT, pka_level::TEXT, NULL::TEXT 
                FROM pka WHERE struct_id = %s
                UNION ALL
                SELECT 'ob_exclusivity_view' AS table_name, struct_id::TEXT, appl_no::TEXT, trade_name::TEXT, exclusivity_date::TEXT, description::TEXT 
                FROM ob_exclusivity_view WHERE struct_id = %s
                UNION ALL
                SELECT 'ob_patent_view' AS table_name, struct_id::TEXT, appl_no::TEXT, trade_name::TEXT, patent_no::TEXT, patent_expire_date::TEXT 
                FROM ob_patent_view WHERE struct_id = %s;
            """
            cur.execute(query, [struct_id] * 15)  # struct_id is used 15 times in the query
            all_data = cur.fetchall()

            # Step 3: Group data by table_name with user-friendly names
            grouped_data = {}
            for row in all_data:
                table_name = row.pop("table_name")
                readable_name = TABLE_NAME_MAPPING.get(table_name, table_name.replace("_", " ").title())
                if readable_name not in grouped_data:
                    grouped_data[readable_name] = []
                grouped_data[readable_name].append(row)

            # Step 4: Fetch additional details (mechanism of action, etc.)
            cur.execute("""
                SELECT DISTINCT
                    UPPER(ai.substance_name) AS active_ingredient,
                    td.name AS target_protein,
                    at.action_type,
                    s.mrdef AS mechanism_of_action
                FROM structures s
                LEFT JOIN active_ingredient ai ON s.id = ai.struct_id
                LEFT JOIN act_table_full act ON s.id = act.struct_id
                LEFT JOIN target_dictionary td ON act.target_id = td.id
                LEFT JOIN action_type at ON act.action_type = at.id::VARCHAR
                WHERE UPPER(ai.substance_name) = %s
            """, [normalized_drug])
            additional_details = cur.fetchall()

            # Ensure uniqueness using Python (removes duplicates missed by DISTINCT)
            additional_details = list({frozenset(item.items()): item for item in additional_details}.values())

            return render_template('medicine/drug_details.html', 
                                 drug=drug, 
                                 additional_details=additional_details,
                                 grouped_data=grouped_data,
                                 struct_id=struct_id)
    except Exception as e:
        print(f"Database error: {str(e)}")
        return render_template('medicine/error.html', message="An error occurred while fetching drug details.")
#add patient to theatre lists
@bp.route('/add-to-theatre', methods=['GET', 'POST'])
def add_to_theatre():
    """Add a patient to the theatre list."""
    if request.method == 'POST':
        try:
            data = request.form if request.content_type == 'application/x-www-form-urlencoded' else request.get_json()

            patient_id = data.get('patient_id')  # Supports text like "P1000"
            procedure_id = data.get('procedure_id')
            created_by = data.get('created_by')
            notes_on_book = data.get('notes_on_book', None)

            if not patient_id or not procedure_id or not created_by:
                flash("All fields are required!", "danger")
                return redirect(url_for('medicine.add_to_theatre'))

            # Check if patient exists
            patient = Patient.query.filter_by(patient_id=patient_id).first()
            if not patient:
                flash(f"Patient {patient_id} not found.", "danger")
                return redirect(url_for('medicine.add_to_theatre'))

            # Check if procedure exists
            procedure = TheatreProcedure.query.get(procedure_id)
            if not procedure:
                flash("Procedure not found.", "danger")
                return redirect(url_for('medicine.add_to_theatre'))

            # Create theatre list entry
            new_entry = TheatreList(
                patient_id=patient_id,
                procedure_id=procedure_id,
                status=0,
                created_by=created_by,
                notes_on_book=notes_on_book,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )

            db.session.add(new_entry)
            db.session.commit()

            flash("Patient added to theatre list successfully!", "success")
            return redirect(url_for('medicine.get_theatre_list'))

        except Exception as e:
            db.session.rollback()
            flash(f"An error occurred: {str(e)}", "danger")
            return redirect(url_for('medicine.add_to_theatre'))

    # If GET request, render the form
    patients = Patient.query.all()
    procedures = TheatreProcedure.query.all()
    return render_template('medicine/add_to_theatre.html', patients=patients, procedures=procedures)

# ✅ Corrected Route: Display Theatre List
@bp.route('/theatre-list', methods=['GET'])
def get_theatre_list():
    """Retrieve all theatre list entries."""
    try:
        status_filter = request.args.get('status', type=int)

        query = TheatreList.query.join(Patient).join(TheatreProcedure).add_columns(
            TheatreList.id,
            TheatreList.patient_id,  # Fetch patient_id as text
            Patient.name.label("patient_name"),
            TheatreProcedure.name.label("procedure_name"),
            TheatreList.status,
            TheatreList.created_at,
            TheatreList.notes_on_book
        )

        if status_filter is not None:
            query = query.filter(TheatreList.status == status_filter)

        theatre_entries = query.order_by(TheatreList.created_at.desc()).all()

        return render_template('medicine/theatre_list.html', theatre_entries=theatre_entries)

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@bp.route('/update-post-op/<int:entry_id>', methods=['GET', 'POST'])
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
            return redirect(url_for('medicine.get_theatre_list'))

        theatre_entry, patient_name = entry  # Unpack the tuple

        if request.method == 'POST':
            notes_on_post_op = request.form.get('notes_on_post_op')

            if not notes_on_post_op:
                flash("Post-op notes are required.", "danger")
                return redirect(url_for('medicine.update_post_op', entry_id=entry_id))

            # Update status and post-op notes
            theatre_entry.status = 1  # Mark as completed
            theatre_entry.notes_on_post_op = notes_on_post_op
            theatre_entry.updated_at = datetime.utcnow()

            db.session.commit()

            flash("Post-op notes updated successfully!", "success")
            return redirect(url_for('medicine.get_theatre_list'))

        return render_template('medicine/update_post_op.html', entry=theatre_entry, patient_name=patient_name)

    except Exception as e:
        db.session.rollback()
        flash(f"An error occurred: {str(e)}", "danger")
        return redirect(url_for('medicine.get_theatre_list'))
    
@bp.route('/admit-patient', methods=['GET', 'POST'])

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
    form.bed_id.choices = []   # Will be populated dynamically via JavaScript

    if request.method == 'POST':  # ✅ Use request.method instead of validate_on_submit()
        try:
            patient_id = request.form.get('patient_id')
            ward_id = request.form.get('ward_id')
            room_id = request.form.get('room_id')
            bed_id = request.form.get('bed_id')
            admission_criteria = request.form.get('admission_criteria')
            admitted_by = request.form.get('admitted_by')

            # ✅ Check if patient exists
            patient = Patient.query.filter_by(patient_id=patient_id).first()
            if not patient:
                flash("Patient not found.", "danger")
                return redirect(url_for('medicine.admit_patient'))

            # ✅ Check if ward exists
            ward = Ward.query.get(ward_id)
            if not ward:
                flash("Ward not found.", "danger")
                return redirect(url_for('medicine.admit_patient'))

            # ✅ Check if room exists in the ward
            room = WardRoom.query.filter_by(id=room_id, ward_id=ward_id).first()
            if not room:
                flash("Room not found in the selected ward.", "danger")
                return redirect(url_for('medicine.admit_patient'))

            # ✅ Check if bed exists & is available
            bed = Bed.query.filter_by(id=bed_id, room_id=room_id, occupied=False).first()
            if not bed:
                flash("Selected bed is not available.", "danger")
                return redirect(url_for('medicine.admit_patient'))

            # ✅ Admit patient & mark bed as occupied
            admission = AdmittedPatient(
                patient_id=patient_id,
                ward_id=ward_id,
                admission_criteria=admission_criteria,
                admitted_by=admitted_by,
                admitted_on=datetime.utcnow()
            )

            bed.occupied = True  # Mark bed as occupied

            db.session.add(admission)
            db.session.commit()

            flash(f"Patient admitted to Bed {bed.bed_number} in Room {room.room_number}!", "success")
            return redirect(url_for('medicine.view_admitted_patients'))

        except Exception as e:
            db.session.rollback()
            flash(f"Error: {str(e)}", "danger")
            return redirect(url_for('medicine.admit_patient'))

    return render_template('medicine/admit_patient.html', form=form, patients=patients, wards=wards)


# 2️⃣ Discharge a Patient (Free Up Bed)
@bp.route('/discharge-patient/<int:id>', methods=['POST'])
def discharge_patient(id):
    """Discharge a patient and free their assigned bed."""
    try:
        admission = AdmittedPatient.query.get(id)
        if not admission:
            flash("Admission record not found.", "danger")
            return redirect(url_for('medicine.view_admitted_patients'))

        ward = Ward.query.get(admission.ward_id)
        bed = Bed.query.filter_by(room_id=admission.ward_id, occupied=True).first()

        if ward and ward.occupied_beds > 0:
            ward.occupied_beds -= 1  # Free up a bed
        if bed:
            bed.occupied = False  # Free up the bed

        admission.discharged_on = datetime.utcnow()
        db.session.commit()

        flash("Patient discharged and bed is now available!", "success")
        return redirect(url_for('medicine.view_admitted_patients'))

    except Exception as e:
        db.session.rollback()
        flash(f"Error: {str(e)}", "danger")
        return redirect(url_for('medicine.view_admitted_patients'))

# 3️⃣ patients in ward
@bp.route('/admitted-patients', methods=['GET'])
def view_admitted_patients():
    """View all admitted patients."""
    try:
        patients = db.session.query(
            AdmittedPatient.id,
            AdmittedPatient.patient_id,
            Patient.name.label("patient_name"),
            Ward.name.label("ward_name"),
            Ward.sex.label("ward_sex"),
            AdmittedPatient.admitted_on,
            AdmittedPatient.admission_criteria,
            AdmittedPatient.discharged_on
        ).join(Patient, AdmittedPatient.patient_id == Patient.patient_id) \
         .join(Ward, AdmittedPatient.ward_id == Ward.id) \
         .order_by(AdmittedPatient.admitted_on.desc()).all()

        return render_template('medicine/admitted_patients.html', patients=patients)

    except Exception as e:
        flash(f"Error: {str(e)}", "danger")
        return redirect(url_for('medicine.admit_patient'))
@bp.route('/ward-bed-history/<int:ward_id>', methods=['GET'])
def ward_bed_history(ward_id):
    """View bed history for a ward."""
    try:
        ward = Ward.query.get(ward_id)
        if not ward:
            flash("Ward not found.", "danger")
            return redirect(url_for('medicine.view_admitted_patients'))

        history = WardBedHistory.query.filter_by(ward_id=ward_id).order_by(WardBedHistory.timestamp.desc()).all()

        return render_template('medicine/ward_bed_history.html', ward=ward, history=history)

    except Exception as e:
        flash(f"Error: {str(e)}", "danger")
        return redirect(url_for('medicine.view_admitted_patients'))
    
# ✅ Fetch available rooms in a ward
@bp.route('/available-rooms/<int:ward_id>', methods=['GET'])
def available_rooms(ward_id):
    """Return available rooms in a ward."""
    try:
        rooms = WardRoom.query.filter_by(ward_id=ward_id, occupied=False).all()
        return jsonify({"rooms": [{"id": room.id, "room_number": room.room_number} for room in rooms]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@bp.route('/available-beds/<int:room_id>', methods=['GET'])
def available_beds(room_id):
    """Return available beds in a room."""
    try:
        beds = Bed.query.filter_by(room_id=room_id, occupied=False).all()
        return jsonify({"beds": [{"id": bed.id, "bed_number": bed.bed_number} for bed in beds]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500    
    
 # ✅ View Inpatients List
@bp.route('/inpatients', methods=['GET'])
@login_required
def view_inpatients():
    """Show all admitted patients for ward rounds."""
    admitted_patients = AdmittedPatient.query.join(Patient).add_columns(
        AdmittedPatient.id, 
        Patient.name.label("patient_name"), 
        AdmittedPatient.ward_id, 
        AdmittedPatient.admitted_on
    ).order_by(AdmittedPatient.admitted_on.desc()).all()

    return render_template('medicine/inpatients.html', admitted_patients=admitted_patients)

@bp.route('/ward-rounds', methods=['GET', 'POST'])
@login_required
def ward_rounds():
    """View inpatients and allow doctors to update ward rounds."""
    if request.method == 'POST':
        try:
            admission_id = request.form.get('admission_id')
            notes = request.form.get('notes')
            status = request.form.get('status')

            if not admission_id or not notes or not status:
                flash("All fields are required!", "danger")
                return redirect(url_for('medicine.ward_rounds'))

            # Check if admission exists
            admission = AdmittedPatient.query.get(admission_id)
            if not admission:
                flash("Patient admission not found.", "danger")
                return redirect(url_for('medicine.ward_rounds'))

            # Save ward round entry
            round_entry = WardRound(
                admission_id=admission_id,
                doctor_id=current_user.id,
                notes=notes,
                status=status
            )

            db.session.add(round_entry)
            db.session.commit()

            flash("Ward round notes updated successfully!", "success")
            return redirect(url_for('medicine.ward_rounds'))

        except Exception as e:
            db.session.rollback()
            flash(f"Error: {str(e)}", "danger")
            return redirect(url_for('medicine.ward_rounds'))

    # Fetch admitted patients and their details
# Fetch admitted patients and their details, including patient_id
    admitted_patients = db.session.query(
        AdmittedPatient.id,
        AdmittedPatient.patient_id,  # ✅ Ensure this is included
        Patient.name.label("patient_name"),
        Ward.name.label("ward_name"),
        AdmittedPatient.admitted_on
    ).join(Patient, AdmittedPatient.patient_id == Patient.patient_id) \
    .join(Ward, AdmittedPatient.ward_id == Ward.id) \
    .order_by(AdmittedPatient.admitted_on.desc()).all()

    return render_template('medicine/ward_rounds.html', admitted_patients=admitted_patients)



@bp.route('/ward-rounds/add', methods=['POST'])
@login_required
def add_ward_round():
    """Add a ward round note for a patient."""
    try:
        admission_id = request.form.get('admission_id')
        notes = request.form.get('notes')
        status = request.form.get('status', "Under Treatment")

        if not admission_id or not notes:
            flash("Please provide required fields!", "danger")
            return redirect(url_for('medicine.view_ward_rounds', admission_id=admission_id))

        new_entry = WardRound(
            admission_id=admission_id,
            doctor_id=current_user.id,
            notes=notes,
            status=status
        )

        db.session.add(new_entry)
        db.session.commit()

        flash("Ward round note added successfully!", "success")
        return redirect(url_for('medicine.view_ward_rounds', admission_id=admission_id))

    except Exception as e:
        db.session.rollback()
        flash(f"Error: {str(e)}", "danger")
        return redirect(url_for('medicine.view_ward_rounds', admission_id=admission_id))     
        #diseases
@bp.route('/diseases/')
@login_required
def list_diseases():
    page = request.args.get('page', default=1, type=int)
    per_page = 10
    diseases = Disease.query.paginate(page=page, per_page=per_page, error_out=False)
    return render_template('medicine/diseases/index.html', diseases=diseases)        
@bp.route('/diseases/<int:disease_id>')
@login_required
def view_disease(disease_id):
    disease = Disease.query.get_or_404(disease_id)
    return render_template('medicine/diseases/view_disease.html', disease=disease)


@bp.route('/diseases/add', methods=['GET', 'POST'])
@login_required
def add_disease():
    if request.method == 'POST':
        name = request.form['name'].strip()
        cui = request.form['cui'].strip()
        description = request.form['description'].strip()
        management_plan_text = request.form['management_plan'].strip()

        # Step 1: Create and save the disease
        new_disease = Disease(name=name, cui=cui, description=description)
        db.session.add(new_disease)
        db.session.flush()  # Get ID before final commit

        # Step 2: Save the management plan
        plan = DiseaseManagementPlan(disease_id=new_disease.id, plan=management_plan_text)
        db.session.add(plan)

        # Step 3: Handle lab tests
        lab_test_names = request.form.getlist('lab_test_name')
        lab_test_descriptions = request.form.getlist('lab_test_description')

        for idx, test_name in enumerate(lab_test_names):
            test_name = test_name.strip()
            if not test_name:
                continue  # Skip empty entries

            test_desc = lab_test_descriptions[idx].strip() if idx < len(lab_test_descriptions) else ''

            lab_test = DiseaseLab(
                disease_id=new_disease.id,
                lab_test=test_name,
                description=test_desc
            )
            db.session.add(lab_test)

        # Step 4: Commit all changes
        db.session.commit()

        return redirect(url_for('medicine.list_diseases'))

    return render_template('medicine/diseases/add_disease.html')

@bp.route('/diseases/edit/<int:disease_id>', methods=['GET', 'POST'])  
@login_required
def edit_disease(disease_id):
    disease = Disease.query.get_or_404(disease_id)
    plan = disease.management_plan

    # Fetch existing lab tests
    lab_tests = disease.lab_tests

    if request.method == 'POST':
        disease.name = request.form['name'].strip()
        disease.cui = request.form['cui'].strip()
        disease.description = request.form['description'].strip()
        plan.plan = request.form['management_plan'].strip()

        # Handle lab tests
        lab_test_ids = request.form.getlist('lab_test_id')
        lab_test_names = request.form.getlist('lab_test_name')
        lab_test_descriptions = request.form.getlist('lab_test_description')

        existing_lab_test_ids = [t.id for t in lab_tests]

        for idx, test_id in enumerate(lab_test_ids):
            name = lab_test_names[idx]
            desc = lab_test_descriptions[idx]

            if test_id == 'new':
                # Add new lab test
                new_test = DiseaseLab(
                    disease_id=disease.id,
                    lab_test=name,
                    description=desc
                )
                db.session.add(new_test)
            else:
                # Update existing lab test
                test = DiseaseLab.query.get(int(test_id))
                if test:
                    test.lab_test = name
                    test.description = desc

        # Detect deleted tests
        submitted_ids = set(int(i) for i in lab_test_ids if i != 'new')
        for test in lab_tests:
            if test.id not in submitted_ids:
                db.session.delete(test)

        db.session.commit()
        flash('Disease and lab tests updated successfully.', 'success')
        return redirect(url_for('medicine.edit_disease', disease_id=disease.id))

    return render_template('medicine/diseases/edit_disease.html', disease=disease, plan=plan, lab_tests=lab_tests)


@bp.route('/diseases/delete/<int:disease_id>')
@login_required
def delete_disease(disease_id):
    disease = Disease.query.get_or_404(disease_id)
    db.session.delete(disease)
    db.session.commit()
    return redirect(url_for('medicine.list_diseases'))

@bp.route('/oncology', methods=['GET', 'POST'])
def oncology():
    search_form = PatientSearchForm()
    selected_patient = None
    bookings = []

    if search_form.validate_on_submit() and search_form.submit_search.data:
        patient_id = search_form.patient_id.data
        selected_patient = Patient.query.filter_by(patient_id=patient_id).first_or_404()
        bookings = OncologyBooking.query.filter_by(patient_id=selected_patient.patient_id).all()
        
        if not bookings:
            flash('No oncology bookings found for this patient.', 'info')
        
        # Redirect to encounter route
        return redirect(url_for('medicine.oncology_encounter', patient_id=selected_patient.patient_id))

    return render_template(
        'medicine/oncology/index.html',
        form=search_form,
        selected_patient=selected_patient,
        bookings=bookings
    )



@bp.route('/oncology/encounter/<patient_id>', methods=['GET', 'POST'])
def oncology_encounter(patient_id):
    # Fetch patient
    selected_patient = Patient.query.filter_by(patient_id=patient_id).first_or_404()

    # Add age attribute based on date_of_birth
    today = date.today()
    dob = selected_patient.date_of_birth
    selected_patient.age = (
        today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    )

    # Initialize forms
    search_form = PatientSearchForm(patient_id=selected_patient.patient_id)
    onco_form = OncoPatientForm()
    note_form = OncologyNoteForm()

    # Fetch existing oncology record
    onco_patient = OncoPatient.query.filter_by(patient_id=selected_patient.id).first()
    bookings = OncologyBooking.query.filter_by(patient_id=selected_patient.patient_id).all()
    notes = OncologyNote.query.filter_by(patient_id=selected_patient.patient_id).order_by(OncologyNote.note_date.desc()).all()

    # Prepopulate the oncology form if data exists
    if onco_patient:
        onco_form.diagnosis.data = onco_patient.diagnosis
        onco_form.diagnosis_date.data = onco_patient.diagnosis_date
        onco_form.cancer_type.data = onco_patient.cancer_type
        onco_form.stage.data = onco_patient.stage
        onco_form.status.data = onco_patient.status

    # Handle oncology form submission
    if onco_form.validate_on_submit() and onco_form.submit_update.data:
        if onco_patient:
            # Update existing record
            onco_patient.diagnosis = onco_form.diagnosis.data
            onco_patient.diagnosis_date = onco_form.diagnosis_date.data
            onco_patient.cancer_type = onco_form.cancer_type.data
            onco_patient.stage = onco_form.stage.data
            onco_patient.status = onco_form.status.data
            flash('Oncology patient details updated successfully.', 'success')
        else:
            # Create new record
            onco_patient = OncoPatient(
                patient_id=selected_patient.id,
                diagnosis=onco_form.diagnosis.data,
                diagnosis_date=onco_form.diagnosis_date.data,
                cancer_type=onco_form.cancer_type.data,
                stage=onco_form.stage.data,
                status=onco_form.status.data,
                date_enrolled=datetime.utcnow()
            )
            db.session.add(onco_patient)
            flash('Oncology patient details created successfully.', 'success')
        db.session.commit()

    # Handle note form submission
    if note_form.validate_on_submit() and note_form.submit_note.data:
        new_note = OncologyNote(
            patient_id=selected_patient.patient_id,
            note_date=note_form.note_date.data,
            note_content=note_form.note_content.data
        )
        db.session.add(new_note)
        db.session.commit()
        flash('Oncology note added successfully.', 'success')

    # Refresh notes after any new submission
    notes = OncologyNote.query.filter_by(patient_id=selected_patient.patient_id).order_by(OncologyNote.note_date.desc()).all()

    return render_template(
        'medicine/oncology/encounter.html',
        search_form=search_form,
        onco_form=onco_form,
        note_form=note_form,
        selected_patient=selected_patient,
        onco_patient=onco_patient,
        bookings=bookings,
        notes=notes
    )

@bp.route('/oncology/add', methods=['GET', 'POST'])
@login_required
def add_onco_patient():
    if request.method == 'POST':
        patient_id = request.form.get('patient_id')
        diagnosis = request.form.get('diagnosis')
        cancer_type = request.form.get('cancer_type')
        stage = request.form.get('stage')
        diagnosis_date = request.form.get('diagnosis_date')

        onco_patient = OncoPatient(
            patient_id=patient_id,
            diagnosis=diagnosis,
            cancer_type=cancer_type,
            stage=stage,
            diagnosis_date=datetime.strptime(diagnosis_date, '%Y-%m-%d')
        )
        db.session.add(onco_patient)
        db.session.commit()
        flash('Patient enrolled in oncology care successfully.', 'success')
        return redirect(url_for('medicine.oncology'))

    patients = Patient.query.all()
    return render_template('medicine/oncology/add_onco_patient.html', patients=patients)
@bp.route('/oncology/note/<int:note_id>/edit', methods=['GET', 'POST'])
def edit_note(note_id):
    note = OncologyNote.query.get_or_404(note_id)
    patient = Patient.query.filter_by(patient_id=note.patient_id).first_or_404()
    
    form = OncologyNoteForm()
    if form.validate_on_submit() and form.submit_note.data:
        note.note_date = form.note_date.data
        note.note_content = form.note_content.data
        db.session.commit()
        flash('Oncology note updated successfully.', 'success')
        return redirect(url_for('medicine.oncology_encounter', patient_id=patient.patient_id))
    
    if request.method == 'GET':
        form.note_date.data = note.note_date
        form.note_content.data = note.note_content
    
    return render_template(
        'medicine/oncology/edit_note.html',
        form=form,
        note=note,
        patient=patient
    )
@bp.route('/get_stages/<int:type_id>')
def get_stages(type_id):
    links = CancerTypeStage.query.filter_by(cancer_type_id=type_id).all()
    stages = [{'id': link.cancer_stage.id, 'label': link.cancer_stage.label} for link in links]
    return jsonify(stages)

@bp.route('/oncology/note/<int:note_id>/delete', methods=['POST'])
def delete_note(note_id):
    note = OncologyNote.query.get_or_404(note_id)
    patient_id = note.patient_id
    db.session.delete(note)
    db.session.commit()
    flash('Oncology note deleted successfully.', 'success')
    return redirect(url_for('medicine.oncology_encounter', patient_id=patient_id))

@bp.route('/drugs/')
@login_required
def drugs():
    category_id = request.args.get('category_id', type=int)
    severity = request.args.get('severity', type=str)
    therapeutic_class = request.args.get('therapeutic_class', type=str)
    has_black_box = request.args.get('has_black_box', type=str)
    query = OncologyDrug.query
    if category_id:
        query = query.filter_by(category_id=category_id)
    if severity in ['Low', 'Moderate', 'High']:
        query = query.join(SpecialWarning).filter(SpecialWarning.severity == severity)
    if therapeutic_class:
        query = query.filter_by(therapeutic_class=therapeutic_class)
    if has_black_box == 'yes':
        query = query.filter(OncologyDrug.black_box_warning.isnot(None))
    elif has_black_box == 'no':
        query = query.filter(OncologyDrug.black_box_warning.is_(None))
    drugs = query.all()
    categories = OncoDrugCategory.query.all()
    therapeutic_classes = db.session.query(OncologyDrug.therapeutic_class).distinct().all()
    therapeutic_classes = [tc[0] for tc in therapeutic_classes if tc[0]]
    return render_template(
        'medicine/oncology/drugs.html',
        drugs=drugs,
        categories=categories,
        therapeutic_classes=therapeutic_classes,
        selected_category=category_id,
        selected_severity=severity,
        selected_therapeutic_class=therapeutic_class,
        selected_has_black_box=has_black_box
    )

@bp.route('/regimens/')
@login_required
def regimens():
    category_id = request.args.get('category_id', type=int)
    status = request.args.get('status', type=str)
    query = OncologyRegimen.query
    if category_id:
        query = query.filter_by(category_id=category_id)
    if status in ['Active', 'Deprecated', 'Under Review']:
        query = query.filter_by(status=status)
    regimens = query.all()
    categories = RegimenCategory.query.all()
    return render_template('medicine/oncology/regimens.html', regimens=regimens, categories=categories, selected_category=category_id, selected_status=status)

@bp.route('/warnings/')
@login_required
def warnings():
    warning_type = request.args.get('warning_type', type=str)
    severity = request.args.get('severity', type=str)
    query = SpecialWarning.query
    if warning_type in ['Warning', 'Caution', 'Incompatibility']:
        query = query.filter_by(warning_type=warning_type)
    if severity in ['Low', 'Moderate', 'High']:
        query = query.filter_by(severity=severity)
    warnings = query.all()
    return render_template('medicine/oncology/warnings.html', warnings=warnings, selected_warning_type=warning_type, selected_severity=severity)
@bp.route('/bookings/')
@login_required
def bookings():
    status = request.args.get('status', type=str)
    purpose = request.args.get('purpose', type=str)
    
    # Build query with join to Patient
    query = OncologyBooking.query.join(Patient, OncologyBooking.patient_id == Patient.patient_id)
    
    # Apply filters
    if status in ['Scheduled', 'Completed', 'Cancelled']:
        query = query.filter(OncologyBooking.status == status)
    if purpose in ['Consultation', 'Chemotherapy', 'Follow-up', 'Radiation', 'Surgery']:
        query = query.filter(OncologyBooking.purpose == purpose)
    
    bookings = query.all()
    
    # Stats bar calculations
    booking_count = OncologyBooking.query.count()
    scheduled_booking_count = OncologyBooking.query.filter_by(status='Scheduled').count()
    chemotherapy_booking_count = OncologyBooking.query.filter_by(purpose='Chemotherapy').count()
    current_month = datetime.now().strftime('%Y-%m')
    new_booking_count = OncologyBooking.query.filter(func.strftime('%Y-%m', OncologyBooking.created_at) == current_month).count()
    
    return render_template(
        'medicine/oncology/bookings.html',
        bookings=bookings,
        selected_status=status,
        selected_purpose=purpose,
        booking_count=booking_count,
        scheduled_booking_count=scheduled_booking_count,
        chemotherapy_booking_count=chemotherapy_booking_count,
        new_booking_count=new_booking_count
    )

@bp.route('/bookings/new', methods=['GET', 'POST'])
@login_required
def new_booking():
    if request.method == 'POST':
        patient_id = request.form.get('patient_id')
        booking_date = request.form.get('booking_date')
        purpose = request.form.get('purpose')
        status = request.form.get('status')
        notes = request.form.get('notes', '').strip() or None
        
        # Log form data for debugging
        print(f"Form data: patient_id={patient_id}, booking_date={booking_date}, purpose={purpose}, status={status}, notes={notes}")
        
        # Validate required fields
        if not patient_id:
            flash('Patient selection is required.', 'danger')
            return redirect(url_for('medicine.new_booking'))
        if not booking_date:
            flash('Booking date is required.', 'danger')
            return redirect(url_for('medicine.new_booking'))
        if not purpose:
            flash('Purpose is required.', 'danger')
            return redirect(url_for('medicine.new_booking'))
        if not status:
            flash('Status is required.', 'danger')
            return redirect(url_for('medicine.new_booking'))
        
        # Validate patient exists
        patient = Patient.query.filter_by(patient_id=patient_id).first()
        if not patient:
            flash('Selected patient does not exist.', 'danger')
            return redirect(url_for('medicine.new_booking'))
        
        # Validate purpose and status
        valid_purposes = ['Consultation', 'Chemotherapy', 'Follow-up', 'Radiation', 'Surgery']
        valid_statuses = ['Scheduled', 'Completed', 'Cancelled']
        if purpose not in valid_purposes:
            flash(f'Invalid purpose selected. Choose from: {", ".join(valid_purposes)}', 'danger')
            return redirect(url_for('medicine.new_booking'))
        if status not in valid_statuses:
            flash(f'Invalid status selected. Choose from: {", ".join(valid_statuses)}', 'danger')
            return redirect(url_for('medicine.new_booking'))
        
        # Parse booking_date
        try:
            booking_date = datetime.strptime(booking_date, '%Y-%m-%d').date()
        except ValueError as e:
            print(f"Date parsing error: {e}")
            flash('Invalid date format. Use YYYY-MM-DD.', 'danger')
            return redirect(url_for('medicine.new_booking'))
        
        # Create new booking
        new_booking = OncologyBooking(
            patient_id=patient_id,
            booking_date=booking_date,
            purpose=purpose,
            status=status,
            notes=notes,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        db.session.add(new_booking)
        db.session.commit()
        flash('Booking created successfully!', 'success')
        return redirect(url_for('medicine.bookings'))
    
    patients = Patient.query.all()
    if not patients:
        flash('No patients available. Please add a patient first.', 'danger')
        return redirect(url_for('medicine.patients_list'))
    return render_template('medicine/oncology/new_booking.html', patients=patients)
# Create a new prescription
@bp.route('/prescriptions/new', methods=['GET', 'POST'])
@login_required
def new_prescription():
    if request.method == 'POST':
        booking_id = request.form.get('booking_id')
        regimen_id = request.form.get('regimen_id')
        start_date = request.form.get('start_date')
        prescribed_by = request.form.get('prescribed_by')
        notes = request.form.get('notes', '').strip() or None
        drug_ids = request.form.getlist('drug_id')  # List of drug IDs

        # Log form data for debugging
        print(f"Form data: booking_id={booking_id}, regimen_id={regimen_id}, start_date={start_date}, prescribed_by={prescribed_by}, notes={notes}, drug_ids={drug_ids}")

        # Validate required fields
        if not booking_id:
            flash('Booking selection is required.', 'danger')
            return redirect(url_for('medicine.new_prescription'))
        if not regimen_id:
            flash('Regimen selection is required.', 'danger')
            return redirect(url_for('medicine.new_prescription'))
        if not start_date:
            flash('Start date is required.', 'danger')
            return redirect(url_for('medicine.new_prescription'))
        if not prescribed_by:
            flash('Prescribed by is required.', 'danger')
            return redirect(url_for('medicine.new_prescription'))

        # Validate booking
        booking = OncologyBooking.query.filter_by(id=booking_id, purpose='Chemotherapy', status='Scheduled').first()
        if not booking:
            flash('Selected booking does not exist or is not a scheduled chemotherapy booking.', 'danger')
            return redirect(url_for('medicine.new_prescription'))

        # Validate patient
        patient = Patient.query.filter_by(patient_id=booking.patient_id).first()
        if not patient:
            flash('Associated patient does not exist.', 'danger')
            return redirect(url_for('medicine.new_prescription'))

        # Validate regimen
        regimen = OncologyRegimen.query.filter_by(id=regimen_id).first()
        if not regimen:
            flash('Selected regimen does not exist.', 'danger')
            return redirect(url_for('medicine.new_prescription'))

        # Validate drug inputs
        regimen_drugs = RegimenDrugAssociation.query.filter_by(regimen_id=regimen_id).all()
        if not regimen_drugs:
            flash('Selected regimen has no associated drugs.', 'danger')
            return redirect(url_for('medicine.new_prescription'))

        # Parse start_date
        try:
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        except ValueError as e:
            print(f"Date parsing error: {e}")
            flash('Invalid date format. Use YYYY-MM-DD.', 'danger')
            return redirect(url_for('medicine.new_prescription'))

        # Validate and collect drug details
        drug_inputs = {}
        for drug_id in drug_ids:
            dosage = request.form.get(f'dosage_{drug_id}', '').strip() or None
            calculated_dose = request.form.get(f'calculated_dose_{drug_id}', '').strip() or None
            infusion_fluid = request.form.get(f'infusion_fluid_{drug_id}', '').strip() or None
            infusion_time = request.form.get(f'infusion_time_{drug_id}', '').strip() or None
            if not dosage or not calculated_dose:
                flash(f'Dosage and calculated dose are required for drug ID {drug_id}.', 'danger')
                return redirect(url_for('medicine.new_prescription'))
            drug_inputs[drug_id] = {
                'dosage': dosage,
                'calculated_dose': calculated_dose,
                'infusion_fluid': infusion_fluid,
                'infusion_time': infusion_time
            }

        # Create prescription
        try:
            new_prescription = OncoPrescription(
                onco_patient_id=booking.patient_id,
                regimen_id=regimen_id,
                start_date=start_date,
                end_date=None,
                prescribed_by=prescribed_by,
                notes=notes,
                created_at=datetime.utcnow()
            )
            db.session.add(new_prescription)
            db.session.flush()  # Get prescription ID

            # Add drug details
            for drug_id, details in drug_inputs.items():
                drug = OncologyDrug.query.get(drug_id)
                if not drug:
                    flash(f'Drug ID {drug_id} does not exist.', 'danger')
                    db.session.rollback()
                    return redirect(url_for('medicine.new_prescription'))
                drug_detail = PrescriptionDrugDetail(
                    prescription_id=new_prescription.id,
                    drug_id=drug_id,
                    dosage=details['dosage'],
                    calculated_dose=details['calculated_dose'],
                    infusion_fluid=details['infusion_fluid'] or drug.reconstitution_fluid,
                    infusion_time=details['infusion_time'] or drug.infusion_time,
                    created_at=datetime.utcnow()
                )
                db.session.add(drug_detail)

            db.session.commit()
            flash('Chemotherapy prescription created successfully!', 'success')
            return redirect(url_for('medicine.list_prescriptions', patient_id=booking.patient_id))
        except Exception as e:
            db.session.rollback()
            print(f"Error creating prescription: {e}")
            flash('An error occurred while creating the prescription.', 'danger')
            return redirect(url_for('medicine.new_prescription'))

    # GET: Render form
    bookings = OncologyBooking.query.filter_by(purpose='Chemotherapy', status='Scheduled').all()
    regimens = OncologyRegimen.query.all()

    # Prepare booking details
    booking_details = [
        {
            'id': booking.id,
            'patient_name': booking.patient.name,
            'patient_id': booking.patient_id,
            'booking_date': booking.booking_date.strftime('%Y-%m-%d')
        } for booking in bookings
    ]

    # Prepare regimen drugs
    regimen_drugs = {}
    for regimen in regimens:
        drugs = RegimenDrugAssociation.query.filter_by(regimen_id=regimen.id).order_by(RegimenDrugAssociation.sequence).all()
        regimen_drugs[regimen.id] = [
            {
                'id': drug.drug.id,
                'name': drug.drug.name,
                'dose': drug.dose or 'N/A',
                'administration_route': drug.administration_route or 'N/A',
                'administration_schedule': drug.administration_schedule or 'N/A',
                'reconstitution_fluid': drug.drug.reconstitution_fluid or 'N/A',
                'infusion_time': drug.drug.infusion_time or 'N/A'
            } for drug in drugs
        ]

    return render_template(
        'medicine/oncology/new_prescription.html',
        bookings=booking_details,
        regimens=regimens,
        regimen_drugs=regimen_drugs,
        no_bookings=len(booking_details) == 0,
        no_regimens=len(regimens) == 0
    )
@bp.route('/prescriptions/<patient_id>', methods=['GET'])
@login_required
def list_prescriptions(patient_id):
    patient = Patient.query.filter_by(patient_id=patient_id).first()
    if not patient:
        flash('Patient does not exist.', 'danger')
        return redirect(url_for('medicine.new_prescription'))

    prescriptions = OncoPrescription.query.filter_by(onco_patient_id=patient_id).all()
    prescription_details = []

    for prescription in prescriptions:
        regimen = OncologyRegimen.query.get(prescription.regimen_id)
        drug_details = PrescriptionDrugDetail.query.filter_by(prescription_id=prescription.id).all()
        drugs = [
            {
                'name': OncologyDrug.query.get(detail.drug_id).name,
                'dosage': detail.dosage,
                'calculated_dose': detail.calculated_dose,
                'infusion_fluid': detail.infusion_fluid,
                'infusion_time': detail.infusion_time
            } for detail in drug_details
        ]
        prescription_details.append({
            'id': prescription.id,
            'regimen_name': regimen.name if regimen else 'Unknown Regimen',
            'start_date': prescription.start_date.strftime('%Y-%m-%d'),
            'prescribed_by': prescription.prescribed_by,
            'notes': prescription.notes,
            'drugs': drugs,
            'created_at': prescription.created_at.strftime('%Y-%m-%d %H:%M:%S')
        })

    return render_template(
        'medicine/oncology/list_prescriptions.html',
        patient=patient,
        prescriptions=prescription_details
    )
# New route to list all prescriptions
@bp.route('/prescriptions', methods=['GET'])
@login_required
def all_prescriptions():
    # Query all prescriptions, joining with patients and regimens for display
    prescriptions = (
        db.session.query(OncoPrescription, Patient, OncologyRegimen)
        .join(Patient, OncoPrescription.onco_patient_id == Patient.patient_id)
        .join(OncologyRegimen, OncoPrescription.regimen_id == OncologyRegimen.id)
        .all()
    )

    prescription_details = []
    for prescription, patient, regimen in prescriptions:
        drug_details = PrescriptionDrugDetail.query.filter_by(prescription_id=prescription.id).all()
        drugs = [
            {
                'name': OncologyDrug.query.get(detail.drug_id).name,
                'dosage': detail.dosage,
                'calculated_dose': detail.calculated_dose,
                'infusion_fluid': detail.infusion_fluid,
                'infusion_time': detail.infusion_time
            } for detail in drug_details
        ]
        prescription_details.append({
            'id': prescription.id,
            'patient_id': patient.patient_id,
            'patient_name': patient.name,
            'regimen_name': regimen.name,
            'start_date': prescription.start_date.strftime('%Y-%m-%d'),
            'prescribed_by': prescription.prescribed_by,
            'notes': prescription.notes,
            'drugs': drugs,
            'created_at': prescription.created_at.strftime('%Y-%m-%d %H:%M:%S')
        })

    return render_template(
        'medicine/oncology/all_prescriptions.html',
        prescriptions=prescription_details
    )
#cancers
@bp.route('/cancers')
@login_required
def cancers():
    # Load all cancer types and their details
    cancer_types = CancerType.query.order_by(CancerType.name).all()
    return render_template('medicine/oncology/cancers.html', cancer_types=cancer_types)

def process_lab_result(lab_result, test_name):
    """Helper function to process a single lab result into presentation format."""
    results_dict = {}
    try:
        results_dict = json.loads(lab_result.result) if lab_result.result else {}
    except json.JSONDecodeError:
        flash(f'Invalid result format for result ID {lab_result.result_id}.', 'warning')
        results_dict = {}

    # Handle test_date
    test_date = lab_result.test_date
    if isinstance(test_date, str):
        try:
            test_date = datetime.strptime(test_date, '%Y-%m-%d %H:%M:%S')
        except (ValueError, TypeError):
            test_date = None

    # Fetch parameters for this lab test
    parameters = LabResultTemplate.query.filter_by(test_id=lab_result.lab_test_id).all()

    test_presentation = []
    for param in parameters:
        result_value = results_dict.get(str(param.id))
        try:
            result_value_float = float(result_value) if result_value is not None else None
        except (ValueError, TypeError):
            result_value_float = None

        status = (
            "Invalid Result" if result_value_float is None else
            "Low" if result_value_float < param.normal_range_low else
            "High" if result_value_float > param.normal_range_high else
            "Normal"
        )

        test_presentation.append({
            'parameter_name': param.parameter_name,
            'normal_range_low': param.normal_range_low,
            'normal_range_high': param.normal_range_high,
            'unit': param.unit,
            'result': result_value if result_value is not None else 'N/A',
            'status': status
        })

    return {
        'test_name': test_name,
        'test_date': test_date.strftime('%Y-%m-%d %H:%M:%S') if test_date else 'N/A',
        'result_notes': lab_result.result_notes or '',
        'parameters': test_presentation
    }

@bp.route('/lab_patients')
def lab_patients():
    # Query requested labs with status=1 and existing results
    requested_labs = (
        db.session.query(RequestedLab, LabTest.test_name, Patient.name, Patient.patient_id, LabResult.result_id)
        .join(LabTest, RequestedLab.lab_test_id == LabTest.id)
        .join(Patient, RequestedLab.patient_id == Patient.patient_id)
        .outerjoin(LabResult, (RequestedLab.patient_id == LabResult.patient_id) & (RequestedLab.lab_test_id == LabResult.lab_test_id))
        .filter(RequestedLab.status == 1, LabResult.result_id.isnot(None))
        .order_by(RequestedLab.date_requested.desc())
        .all()
    )

    requested_labs_data = []
    for lab, test_name, patient_name, patient_id, result_id in requested_labs:
        date_requested = lab.date_requested
        if isinstance(date_requested, str):
            try:
                date_requested = datetime.strptime(date_requested, '%Y-%m-%d %H:%M:%S')
            except (ValueError, TypeError):
                date_requested = None

        requested_labs_data.append({
            'test_name': test_name,
            'date_requested': date_requested.strftime('%Y-%m-%d %H:%M:%S') if date_requested else 'N/A',
            'status': 'Completed',
            'patient_name': patient_name or 'N/A',
            'patient_id': patient_id,
            'result_id': result_id
        })

    return render_template('medicine/lab_patients.html', requested_labs=requested_labs_data)

@bp.route('/lab_results/<result_id>')
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
        flash(f'No lab result found for Result ID {result_id}.', 'danger')
        return render_template('medicine/lab_results.html', lab_results={}, patient_name='N/A')

    lab_result, test_name, patient_name = result
    patient_name = patient_name or 'N/A'

    # Process the lab result
    test_presentations = {}
    test_presentations[lab_result.result_id] = process_lab_result(lab_result, test_name)

    return render_template('medicine/lab_results.html', lab_results=test_presentations, patient_name=patient_name)
@bp.route('/pending_lab_patients')
def pending_lab_patients():
    # Query requested labs with status=0 and no results
    requested_labs = (
        db.session.query(RequestedLab, LabTest.test_name, Patient.name, Patient.patient_id, LabResult.result_id)
        .join(LabTest, RequestedLab.lab_test_id == LabTest.id)
        .join(Patient, RequestedLab.patient_id == Patient.patient_id)
        .outerjoin(LabResult, (RequestedLab.patient_id == LabResult.patient_id) & (RequestedLab.lab_test_id == LabResult.lab_test_id))
        .filter(RequestedLab.status == 0, LabResult.result_id.is_(None))
        .order_by(RequestedLab.date_requested.desc())
        .all()
    )

    requested_labs_data = []
    for lab, test_name, patient_name, patient_id, result_id in requested_labs:
        date_requested = lab.date_requested
        if isinstance(date_requested, str):
            try:
                date_requested = datetime.strptime(date_requested, '%Y-%m-%d %H:%M:%S')
            except (ValueError, TypeError):
                date_requested = None

        requested_labs_data.append({
            'test_name': test_name,
            'date_requested': date_requested.strftime('%Y-%m-%d %H:%M:%S') if date_requested else 'N/A',
            'status': 'not yet done',
            'patient_name': patient_name or 'N/A',
            'patient_id': patient_id,
            'result_id': result_id
        })

    return render_template('medicine/pending_lab_patients.html', requested_labs=requested_labs_data)
@bp.route('/patient_lab_results/<patient_id>')
def patient_lab_results(patient_id):
    # Query all lab results for the specific patient_id with status=1 and existing results
    results = (
        db.session.query(LabResult, LabTest.test_name, Patient.name)
        .join(LabTest, LabResult.lab_test_id == LabTest.id)
        .join(Patient, LabResult.patient_id == Patient.patient_id)
        .join(RequestedLab, (LabResult.patient_id == RequestedLab.patient_id) & (LabResult.lab_test_id == RequestedLab.lab_test_id))
        .filter(LabResult.patient_id == patient_id, RequestedLab.status == 1, LabResult.result_id.isnot(None))
        .all()
    )

    # If no results found, return an error message
    if not results:
        flash(f'No lab results found for Patient ID {patient_id}.', 'danger')
        return render_template('medicine/patient_lab_results.html', lab_results={}, patient_name='N/A')

    # Correctly extract patient_name from the query results
    patient_name = results[0][2] or 'N/A'
    test_presentations = {}

    for lab_result, test_name, _ in results:
        test_presentations[lab_result.result_id] = process_lab_result(lab_result, test_name)

    return render_template('medicine/patient_lab_results.html', lab_results=test_presentations, patient_name=patient_name)
