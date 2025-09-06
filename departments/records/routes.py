from flask import render_template, redirect, url_for, request, flash,jsonify
from flask_login import login_required, current_user
from . import bp  # Import the blueprint
from departments.models.records import Patient,Clinic,ClinicBooking,PatientWaitingList  # Import the Patient model
from datetime import datetime
from extensions import db
from sqlalchemy.orm import joinedload

@bp.route('/index')  # Ensure the route is correctly defined under the blueprint
@login_required
def index():
    if current_user.role not in ['records', 'admin']:
        return redirect(url_for('login'))
    patients = Patient.query.all()
    clinics = Clinic.query.all()

    # Debug: Print the clinics list
    print(clinics)  # Should display a list of Clinic objects

    return render_template('records/patients_list.html', patients=patients, clinics=clinics)
@bp.route('/search_clinics')
@login_required
def search_clinics():
    query = request.args.get("q", "").strip()

    if not query:
        return jsonify([])

    clinics = Clinic.query.filter(
        Clinic.name.ilike(f"%{query}%")
    ).limit(20).all()

    results = [clinic.to_select2() for clinic in clinics]
    return jsonify(results)
    

@bp.route('/new_patient', methods=['GET', 'POST'])
@login_required
def new_patient():
    if current_user.role not in ['records', 'admin']: 
        return redirect(url_for('login'))

    if request.method == 'POST':
        # Extract form data
        name = request.form['name']
        place_of_residence = request.form['place_of_residence']
        sex = request.form['sex']
        date_of_birth = request.form['date_of_birth']
        marital_status = request.form['marital_status']
        blood_group = request.form['blood_group']
        contact = request.form['contact']
        next_of_kin = request.form['next_of_kin']
        relationship_with_next_of_kin = request.form['relationship_with_next_of_kin']
        next_of_kin_contact = request.form['next_of_kin_contact']
        national_id = request.form['national_id']
        insurance_provider = request.form.get('insurance_provider', None)  # Optional field
        insurance_policy_number = request.form.get('insurance_policy_number', None)  # Optional field
        occupation = request.form.get('occupation', None)  # Optional field
        employer_name = request.form.get('employer_name', None)  # Optional field
        emergency_contact = request.form['emergency_contact']

        # Validate input for required fields
        if not name or not place_of_residence or not sex or not date_of_birth or not marital_status or not blood_group or not contact or not next_of_kin or not relationship_with_next_of_kin or not next_of_kin_contact or not national_id or not emergency_contact:
            flash('All required fields must be filled!', 'error')
            return redirect(url_for('records.new_patient'))

        try:
            # Parse Date of Birth
            date_of_birth = datetime.strptime(date_of_birth, '%Y-%m-%d').date()
        except ValueError:
            flash('Invalid date format for Date of Birth! Use YYYY-MM-DD.', 'error')
            return redirect(url_for('records.new_patient'))

        # Generate unique patient ID
        new_patient_id = Patient.generate_patient_id()

        # Create a new patient
        new_patient = Patient(
            patient_id=new_patient_id,
            name=name,
            place_of_residence=place_of_residence,
            sex=sex,
            date_of_birth=date_of_birth,
            marital_status=marital_status,
            blood_group=blood_group,
            contact=contact,
            next_of_kin=next_of_kin,
            relationship_with_next_of_kin=relationship_with_next_of_kin,
            next_of_kin_contact=next_of_kin_contact,
            national_id=national_id,
            insurance_provider=insurance_provider,
            insurance_policy_number=insurance_policy_number,
            occupation=occupation,
            employer_name=employer_name,
            emergency_contact=emergency_contact
        )

        # Add the patient to the database
        db.session.add(new_patient)
        db.session.commit()

        # Flash success message and redirect to the patient list
        flash(f'Patient {new_patient.name} registered successfully with ID: {new_patient.patient_id}!', 'success')
        return redirect(url_for('records.index'))

    # Render the form for GET requests
    return render_template('records/new_patient.html')
#render clinics
@bp.route('/clinics')
@login_required
def clinics():
    if current_user.role not in ['records', 'admin']:
        return redirect(url_for('login'))
    clinics = Clinic.query.all()  # Fetch all clinics
    return render_template('records/clinics_list.html', clinics=clinics)

#bookings
@bp.route('/bookings')
@login_required
def bookings():
    if current_user.role not in ['records', 'admin']:
        return redirect(url_for('login'))

    # Query clinic bookings with patient and clinic details
    bookings = db.session.query(ClinicBooking).options(
        joinedload(ClinicBooking.patient),  # Load patient details
        joinedload(ClinicBooking.clinic)    # Load clinic details
    ).all()

    return render_template('records/bookings_list.html', bookings=bookings)
#waiting list
@bp.route('/waiting_list')
@login_required
def waiting_list():
    if current_user.role not in ['records', 'admin']:
        return redirect(url_for('login'))

    # Query the waiting list with patient details
    waiting_list = db.session.query(PatientWaitingList, Patient).join(
        Patient, PatientWaitingList.patient_id == Patient.patient_id
    ).all()

    return render_template('records/waiting_list.html', waiting_list=waiting_list)

@bp.route('/book_clinic', methods=['POST'])
@login_required
def book_clinic():
    if current_user.role not in ['records', 'admin']:
        return jsonify({"status": "error", "message": "Unauthorized access!"}), 403

    # Extract form data
    patient_id = request.form.get('patient_id')
    clinic_id = request.form.get('clinic_id')
    clinic_date = request.form.get('clinic_date')

    # Validate input
    if not patient_id or not clinic_id or not clinic_date:
        return jsonify({"status": "error", "message": "All fields are required!"}), 400

    try:
        clinic_date = datetime.strptime(clinic_date, '%Y-%m-%d').date()
    except ValueError:
        return jsonify({"status": "error", "message": "Invalid date format! Use YYYY-MM-DD."}), 400

    # Check if the patient exists
    patient = Patient.query.filter_by(patient_id=patient_id).first()
    if not patient:
        return jsonify({"status": "error", "message": f"Patient with ID {patient_id} does not exist!"}), 400

    # Check if the clinic exists
    clinic = Clinic.query.get(clinic_id)
    if not clinic:
        return jsonify({"status": "error", "message": f"Clinic with ID {clinic_id} does not exist!"}), 400

    # Check if the booking already exists
    existing_booking = ClinicBooking.query.filter_by(
        patient_id=patient_id,
        clinic_id=clinic_id,
        clinic_date=clinic_date
    ).first()
    if existing_booking:
        return jsonify({"status": "error", "message": "This booking already exists!"}), 400

    # Create a new clinic booking
    new_booking = ClinicBooking(
        patient_id=patient_id,
        clinic_id=clinic_id,
        clinic_date=clinic_date
    )
    db.session.add(new_booking)
    db.session.commit()

    return jsonify({
        "status": "success",
        "message": f"Clinic booked successfully for {patient.name} at {clinic.name} on {clinic_date.strftime('%Y-%m-%d')}!"
    }), 200
#live search
@bp.route('/search_patients', methods=['GET'])
@login_required
def search_patients():
    if current_user.role not in ['records', 'admin']:
        return jsonify({"status": "error", "message": "Unauthorized access!"}), 403

    # Get the search term from the request
    search_term = request.args.get('term', '').strip()

    # Query the database for matching patients
    if search_term:
        patients = Patient.query.filter(
            Patient.name.ilike(f"%{search_term}%") | Patient.patient_id.ilike(f"%{search_term}%")
        ).all()
    else:
        patients = []  # Return an empty list if no search term is provided

    # Format the results as a list of dictionaries
    results = [
        {
            "id": patient.patient_id,
            "text": f"{patient.name} ({patient.patient_id})"
        }
        for patient in patients
    ]

    return jsonify({"results": results})