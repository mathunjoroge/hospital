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


@bp.route('/add-to-theatre', methods=['GET', 'POST'])
@login_required
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
@login_required
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
@login_required
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
@login_required
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
@login_required
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
@login_required
def available_rooms(ward_id):
    """Return available rooms in a ward."""
    try:
        rooms = WardRoom.query.filter_by(ward_id=ward_id, occupied=False).all()
        return jsonify({"rooms": [{"id": room.id, "room_number": room.room_number} for room in rooms]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@bp.route('/available-beds/<int:room_id>', methods=['GET'])
@login_required
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
