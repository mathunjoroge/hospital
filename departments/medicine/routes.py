# departments/nlp/soap_notes.py
from flask import render_template, redirect, url_for, request, flash, jsonify
from typing import Optional, List, Dict, Any
import psycopg2

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
import bleach  # For sanitizing descriptions
from . import bp
import os
from datetime import datetime
from departments.models.records import PatientWaitingList, Patient
from departments.models.medicine import (
    SOAPNote, LabTest, Imaging, Medicine, PrescribedMedicine, RequestedLab, 
    RequestedImage, UnmatchedImagingRequest, TheatreProcedure, TheatreList, 
    Ward, AdmittedPatient, WardBedHistory, WardRoom, Bed, WardRound
)
from departments.forms import AdmitPatientForm
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

@bp.route('/notes/<int:note_id>/update_kb', methods=['POST'])
@login_required
def update_knowledge_base(note_id):
    if current_user.role not in ['medicine', 'admin']:
        flash('You do not have permission to perform this action.', 'error')
        return redirect(url_for('medicine.index'))
    from flask import request
    symptom = request.form.get('symptom')
    category = request.form.get('category')
    context = request.form.get('context', '')
    if not symptom or not category:
        flash('Symptom and category are required.', 'error')
        return redirect(url_for('medicine.notes', patient_id=SOAPNote.query.get(note_id).patient_id))


@bp.route('/update_missing_ai_summaries', methods=['POST'])
@login_required
def update_missing_ai_summaries():
    """Manually update SOAP notes with missing AI summaries."""
    if current_user.role not in ['admin']:
        flash('Only admins can update AI summaries.', 'error')
        return redirect(url_for('home'))

    try:
        # Find SOAP notes where ai_notes is null
        soap_notes = SOAPNote.query.filter(SOAPNote.ai_notes.is_(None)).all()
        if not soap_notes:
            flash('No SOAP notes found with missing AI summaries.', 'info')
            return redirect(url_for('home'))

        updated_count = 0
        for note in soap_notes:
            note_text = f"""
Chief Complaint: {note.situation}
HPI: {note.hpi}
Aggravating Factors: {note.aggravating_factors or 'None'}
Alleviating Factors: {note.alleviating_factors or 'None'}
Medical History: {note.medical_history or 'None'}
Medication History: {note.medication_history or 'None'}
Assessment: {note.assessment}
Recommendation: {note.recommendation}
Additional Notes: {note.additional_notes or 'None'}
"""


        db.session.commit()
        flash(f'Updated {updated_count} SOAP notes with AI summaries.', 'success')
        return redirect(url_for('home'))

    except Exception as e:
        flash(f'Error updating AI summaries: {e}', 'error')
        db.session.rollback()
        logger.error(f"Error in update_missing_ai_summaries: {e}")
        return redirect(url_for('home'))

@bp.route('/check_missing_ai_summaries', methods=['GET'])
@login_required
def check_missing_ai_summaries():
    """Display count of SOAP notes with missing AI summaries."""
    if current_user.role not in ['admin']:
        flash('Only admins can view this page.', 'error')
        return redirect(url_for('home'))

    count = SOAPNote.query.filter(SOAPNote.ai_notes.is_(None)).count()
    return render_template('missing_ai_summaries.html', count=count)

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


# Handle prescription submission
@bp.route('/submit_prescription/<patient_id>', methods=['POST'])
@login_required
def submit_prescription(patient_id):
    """Handle prescription submission."""
    if current_user.role not in ['medicine', 'admin']:
        flash('You do not have permission to access this page.', 'error')
        return redirect(url_for('home'))

    try:
        prescription_id = request.form.get('prescription_id')
        drugs = request.form.getlist('drugs[]')
        dosage_form = request.form.get('dosage_form')
        strength = request.form.get('strength')
        frequency = request.form.get('frequency')
        custom_frequency = request.form.get('custom_frequency')
        num_days = request.form.get('num_days')

        # Validate all required fields
        if not all([drugs, dosage_form, strength, frequency, num_days]):
            flash('All fields are required for prescribing drugs!', 'error')
            return redirect(url_for('medicine.prescribe_drugs', patient_id=patient_id))

        # Validate custom frequency if "Other" is selected
        if frequency == "OTHER" and not custom_frequency.strip():
            flash('Please specify a custom frequency!', 'error')
            return redirect(url_for('medicine.prescribe_drugs', patient_id=patient_id))

        # Validate strength
        if not strength.strip():
            flash('Strength is required!', 'error')
            return redirect(url_for('medicine.prescribe_drugs', patient_id=patient_id))

        # Save prescribed medicines to the database
        for drug_id in drugs:
            prescribed_medicine = PrescribedMedicine(
                patient_id=patient_id,
                medicine_id=drug_id,
                dosage=dosage_form,
                strength=strength.strip(),  # Trim whitespace
                frequency=frequency if frequency != "OTHER" else custom_frequency.strip(),
                prescription_id=prescription_id,
                num_days=int(num_days)  # Ensure num_days is an integer
            )
            db.session.add(prescribed_medicine)

        db.session.commit()
        flash('Prescription submitted successfully!', 'success')
        return redirect(url_for('medicine.soap_notes', patient_id=patient_id))

    except Exception as e:
        flash(f'Error submitting prescription: {e}', 'error')
        db.session.rollback()  # Rollback changes in case of error
        print(f"Debug: Error in medicine.submit_prescription: {e}")  # Debugging
        return redirect(url_for('medicine.soap_notes', patient_id=patient_id))
# Handle lab test requests
@bp.route('/request_lab_tests/<patient_id>', methods=['GET', 'POST'])
@login_required
def request_lab_tests(patient_id):
    """Handles lab test requests."""
    if current_user.role not in ['medicine', 'admin']:
        flash('You do not have permission to access this page.', 'error')
        return redirect(url_for('home'))

    try:
        # Fetch the patient from the waiting list
        patient_entry = PatientWaitingList.query.filter_by(patient_id=patient_id).options(
            joinedload(PatientWaitingList.patient)
        ).first()
        if not patient_entry or not patient_entry.patient:
            flash(f'Patient with ID {patient_id} not found in the waiting list!', 'error')
            return redirect(url_for('medicine.index'))  # Redirect to index if patient not found
        patient = patient_entry.patient

        # Fetch available lab tests for dropdowns
        lab_tests = LabTest.query.all()

        if request.method == 'POST':
            # Extract form data
            lab_test_ids = request.form.getlist('lab_tests[]')
            if not lab_test_ids:
                flash('No lab tests selected!', 'error')
                return render_template(
                    'medicine/request_lab_tests.html',
                    patient=patient,
                    lab_tests=lab_tests
                )

            # Save requested lab tests to the database
            for lab_test_id in lab_test_ids:
                new_lab_request = RequestedLab(
                    patient_id=patient_id,
                    lab_test_id=lab_test_id
                )
                db.session.add(new_lab_request)

            db.session.commit()
            flash('Lab tests requested successfully!', 'success')
            return redirect(url_for('medicine.soap_notes', patient_id=patient_id))

        # Render the lab test request form on GET request
        return render_template(
            'medicine/request_lab_tests.html',
            patient=patient,
            lab_tests=lab_tests
        )

    except Exception as e:
        flash(f'Error requesting lab tests: {e}', 'error')
        db.session.rollback()  # Rollback changes in case of error
        print(f"Debug: Error in medicine.request_lab_tests: {e}")  # Debugging
        return redirect(url_for('medicine.soap_notes', patient_id=patient_id))


@bp.route('/request_imaging/<patient_id>', methods=['GET', 'POST'])
@login_required
def request_imaging(patient_id):
    """Handles imaging requests."""
    if current_user.role not in ['medicine', 'admin']:
        flash('You do not have permission to access this page.', 'error')
        return redirect(url_for('home'))

    try:
        # Fetch patient from waiting list
        patient_entry = PatientWaitingList.query.filter_by(patient_id=patient_id).options(
            joinedload(PatientWaitingList.patient)
        ).first()
        if not patient_entry or not patient_entry.patient:
            flash(f'Patient with ID {patient_id} not found in the waiting list!', 'error')
            return redirect(url_for('medicine.index'))
        patient = patient_entry.patient

        # Fetch latest SOAP note
        soap_notes = SOAPNote.query.filter_by(patient_id=patient_id).order_by(SOAPNote.created_at.desc()).first()

        # Fetch available imaging types
        imaging_types = Imaging.query.all()

        if request.method == 'POST':
            # 🔍 DEBUG: Log raw form data
            print("Raw form data:", request.form)
            logger.debug(f"Raw form data for patient {patient_id}: {dict(request.form)}")
            flash("Form submitted. Processing data...", "info")

            # Get selected imaging types
            imaging_ids = request.form.getlist('imaging_types[]')

            if not imaging_ids:
                flash('No imaging types selected!', 'error')
                return render_template(
                    'medicine/request_imaging.html',
                    patient=patient,
                    soap_notes=soap_notes,
                    imaging_types=imaging_types
                )

            # 🔍 DEBUG: Show what imaging IDs were received
            logger.debug(f"Imaging IDs received: {imaging_ids}")
            flash(f"Imaging types selected: {', '.join(imaging_ids)}", "debug")

            # Extract descriptions dynamically (e.g., descriptions[3])
            descriptions = {}
            for key, value in request.form.items():
                if key.startswith('descriptions['):
                    imaging_id = key.split('[')[1].split(']')[0]
                    descriptions[imaging_id] = value.strip() if value else ''

            # 🔍 DEBUG: Show parsed descriptions
            logger.debug(f"Parsed descriptions: {descriptions}")
            flash(f"Parsed descriptions: {descriptions}", "debug")

            # Validate and save each imaging request
            for imaging_id in imaging_ids:
                description = descriptions.get(str(imaging_id), '').strip()

                # 🔍 DEBUG: Show each imaging ID and its description
                logger.debug(f"Processing imaging ID: {imaging_id}, Description: {description[:50]}...")

                # Enforce 500-character limit
                if len(description) > 500:
                    error_msg = f'Description for imaging ID {imaging_id} exceeds 500 characters.'
                    logger.warning(error_msg)
                    flash(error_msg, 'error')
                    return render_template(
                        'medicine/request_imaging.html',
                        patient=patient,
                        soap_notes=soap_notes,
                        imaging_types=imaging_types
                    )
                result_id = str(uuid.uuid4())    
                new_image_request = RequestedImage(
                    patient_id=patient_id,
                    imaging_id=imaging_id,
                    result_id=result_id,  # ✅ Add the UUID here for every request
                    description=description or None  # Store as NULL if empty
                )
                db.session.add(new_image_request)

            db.session.commit()
            success_msg = 'Imaging requested successfully!'
            logger.info(success_msg)
            flash(success_msg, 'success')
            return redirect(url_for('medicine.soap_notes', patient_id=patient_id))

        # GET: Render the form
        return render_template(
            'medicine/request_imaging.html',
            patient=patient,
            soap_notes=soap_notes,
            imaging_types=imaging_types
        )

    except Exception as e:
        db.session.rollback()
        error_msg = f"Error in medicine.request_imaging: {e}"
        logger.error(error_msg, exc_info=True)
        flash(f'An error occurred: {e}', 'error')
        return redirect(url_for('medicine.soap_notes', patient_id=patient_id))
# # Handle drug prescription requests
@bp.route('/prescribe_drugs/<patient_id>', methods=['GET', 'POST'])
@login_required
def prescribe_drugs(patient_id):
    """Handles drug prescription requests for admitted and waiting list patients."""

    if current_user.role not in ['medicine', 'admin']:
        flash('You do not have permission to access this page.', 'error')
        return redirect(url_for('home'))

    try:
        # ✅ Ensure patient_id is treated as a STRING
        patient_id = str(patient_id).strip()
        print(f"Debug: Checking for patient_id={patient_id}")  # Debugging log

        # ✅ Check if the patient is admitted
        admitted_patient = AdmittedPatient.query.filter(
            AdmittedPatient.patient_id == patient_id,
            AdmittedPatient.discharged_on.is_(None)  # Ensure still admitted
        ).first()

        # ✅ Check if the patient is in the waiting list
        waiting_patient = PatientWaitingList.query.filter(
            PatientWaitingList.patient_id == patient_id
        ).first()

        # ✅ Ensure the patient exists in at least one of the tables
        if not admitted_patient and not waiting_patient:
            flash(f'Patient with ID {patient_id} is neither admitted nor in the waiting list!', 'error')
            print(f"Debug: No record found for patient_id={patient_id} in AdmittedPatient or PatientWaitingList.")
            return redirect(url_for('medicine.index'))

        # ✅ Fetch patient details from the `Patient` table
        patient = Patient.query.filter(Patient.patient_id == patient_id).first()
        if not patient:
            flash(f'Patient with ID {patient_id} not found in the system!', 'error')
            print(f"Debug: Patient with ID {patient_id} not found in `patients` table.")
            return redirect(url_for('medicine.index'))

        print(f"Debug: Found patient - {patient.name} (ID: {patient.patient_id})")  # Debugging log

        # ✅ Maintain `prescription_id` in session
        if 'prescription_id' not in session:
            session['prescription_id'] = str(uuid4())  # Generate a unique ID
        
        prescription_id = session.get('prescription_id')

        # ✅ Fetch available drugs
        drugs = Medicine.query.all()

        if request.method == 'POST':
            # ✅ Extract form data
            drugs_selected = request.form.getlist('drugs[]')
            dosage_form = request.form.get('dosage_form')
            strength = request.form.get('strength')
            frequency = request.form.get('frequency')
            custom_frequency = request.form.get('custom_frequency')
            num_days = request.form.get('num_days')

            # ✅ Validate required fields
            if not all([drugs_selected, dosage_form, strength, frequency, num_days]):
                flash('All fields are required for prescribing drugs!', 'error')
                return render_template(
                    'medicine/prescribe_drugs.html',
                    patient=patient,
                    drugs=drugs,
                    prescription_id=prescription_id
                )

            # ✅ Save prescribed medicines to the database
            for drug_id in drugs_selected:
                prescribed_medicine = PrescribedMedicine(
                    patient_id=patient_id,
                    medicine_id=drug_id,
                    dosage=dosage_form,
                    strength=strength.strip(),  # Trim whitespace
                    frequency=custom_frequency.strip() if frequency == "Other" else frequency,
                    prescription_id=prescription_id,
                    num_days=int(num_days)  # Ensure num_days is an integer
                )
                db.session.add(prescribed_medicine)

            try:
                db.session.commit()
                flash('Prescription submitted successfully!', 'success')
            except Exception as e:
                db.session.rollback()
                flash(f'Database error: {e}', 'error')
                print(f"Debug: Database error in medicine.prescribe_drugs: {e}")
                return redirect(url_for('medicine.prescribe_drugs', patient_id=patient_id))

            # ✅ Fetch prescribed medicines for the current `prescription_id`
            prescribed_medicines = PrescribedMedicine.query.filter_by(
                prescription_id=prescription_id
            ).all()

            return render_template(
                'medicine/prescribe_drugs.html',
                patient=patient,
                drugs=drugs,
                prescribed_medicines=prescribed_medicines,
                prescription_id=prescription_id
            )

        # ✅ Fetch prescribed medicines for the current `prescription_id` (if any)
        prescribed_medicines = PrescribedMedicine.query.filter_by(
            prescription_id=prescription_id
        ).all() if prescription_id else []

        return render_template(
            'medicine/prescribe_drugs.html',
            patient=patient,
            drugs=drugs,
            prescribed_medicines=prescribed_medicines,
            prescription_id=prescription_id
        )

    except Exception as e:
        flash(f'Error prescribing drugs: {e}', 'error')
        db.session.rollback()  # Rollback changes in case of error
        print(f"Debug: Error in medicine.prescribe_drugs: {e}")  # Debugging
        return redirect(url_for('medicine.index'))

    
#edit precribed medicine
# 
@bp.route('/edit_prescribed_medicine/<medicine_id>', methods=['GET', 'POST'])
@login_required
def edit_prescribed_medicine(medicine_id):
    """Handles editing a prescribed medicine."""
    if current_user.role not in ['medicine', 'admin']:
        flash('You do not have permission to access this page.', 'error')
        return redirect(url_for('home'))

    try:
        # Fetch the prescribed medicine by ID
        prescribed_medicine = PrescribedMedicine.query.get_or_404(medicine_id)

        if request.method == 'POST':
            # Extract updated form data
            dosage_form = request.form.get('dosage_form')
            strength = request.form.get('strength')
            frequency = request.form.get('frequency')
            custom_frequency = request.form.get('custom_frequency')
            num_days = request.form.get('num_days')

            # Validation
            if not all([dosage_form, strength, frequency, num_days]):
                flash('All fields are required!', 'error')
                return redirect(url_for('medicine.edit_prescribed_medicine', medicine_id=medicine_id))

            # Update the prescribed medicine
            prescribed_medicine.dosage = dosage_form
            prescribed_medicine.strength = strength.strip()
            prescribed_medicine.frequency = frequency if frequency != "OTHER" else custom_frequency.strip()
            prescribed_medicine.num_days = int(num_days)

            db.session.commit()
            flash('Prescribed medicine updated successfully!', 'success')
            return redirect(url_for('medicine.prescribe_drugs', patient_id=prescribed_medicine.patient_id))

        # Pre-fill the form with existing data on GET request
        return render_template(
            'medicine/edit_prescribed_medicine.html',
            prescribed_medicine=prescribed_medicine
        )

    except Exception as e:
        flash(f'Error editing prescribed medicine: {e}', 'error')
        print(f"Debug: Error in medicine.edit_prescribed_medicine: {e}")
        return redirect(url_for('medicine.prescribe_drugs', patient_id=prescribed_medicine.patient_id))  
#delete prescribed medicine 
@bp.route('/delete_prescribed_medicine/<medicine_id>', methods=['POST'])
@login_required
def delete_prescribed_medicine(medicine_id):
    """Handles deleting a prescribed medicine."""
    if current_user.role not in ['medicine', 'admin']:
        flash('You do not have permission to access this page.', 'error')
        return redirect(url_for('home'))

    try:
        # Fetch the prescribed medicine by ID
        prescribed_medicine = PrescribedMedicine.query.get_or_404(medicine_id)

        # Delete the prescribed medicine
        db.session.delete(prescribed_medicine)
        db.session.commit()

        flash('Prescribed medicine deleted successfully!', 'success')
        return redirect(url_for('medicine.prescribe_drugs', patient_id=prescribed_medicine.patient_id))

    except Exception as e:
        flash(f'Error deleting prescribed medicine: {e}', 'error')
        print(f"Debug: Error in medicine.delete_prescribed_medicine: {e}")
        return redirect(url_for('medicine.prescribe_drugs', patient_id=prescribed_medicine.patient_id)) 

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


# ✅ Add a New Ward Round Entry
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
