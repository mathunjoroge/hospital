import requests 
import pickle
import re
from flask import render_template, redirect, url_for, request, flash, jsonify,session
from flask_wtf import FlaskForm
from sqlalchemy import func
from wtforms import SelectField
from wtforms.validators import DataRequired
from sqlalchemy import text
import json
from contextlib import contextmanager
from typing import Optional, List, Dict, Any
import psycopg2
from datetime import date
from psycopg2.extras import RealDictCursor
from flask import current_app
from flask_login import login_required, current_user
from departments.rbac import roles_required, get_effective_role
from flask_wtf.csrf import CSRFProtect,CSRFError
from scipy.spatial.distance import cosine
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
from departments.nlp.chatbot import UniversalClinicalSummarizer
import logging
import json
from flask import Response, stream_with_context, request
import time
from departments.nlp.logging_setup import get_logger
from flask.sessions import SecureCookieSessionInterface
logger = get_logger()
import PyPDF2  # For PDF processing
from docx import Document  # For DOCX processing
import pytesseract  # For OCR on images
from PIL import Image  # For image handling
import csv  #
from werkzeug.utils import secure_filename

# Instantiate the summarizer for use in chatbot_interface


from extensions import csrf

gemini_api_key = os.environ.get("GEMINI_API_KEY")
nvidia_api_key = os.environ.get("NVIDIA_API_KEY")
Summarizer = UniversalClinicalSummarizer(gemini_api_key=gemini_api_key, nvidia_api_key=nvidia_api_key)


from flask import make_response

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf', 'txt', 'csv', 'docx'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB max file size


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
@login_required
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
@login_required
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
@login_required
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

