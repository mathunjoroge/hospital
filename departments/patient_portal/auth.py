from datetime import datetime, timedelta
from functools import wraps

from flask import (
    flash,
    g,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from departments.models.patient_user import PatientUser
from departments.models.records import Patient
from extensions import db

from . import patient_portal_bp


def patient_login_required(f):
    """Decorator ensuring request is made by an authenticated PatientUser session."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        patient_user_id = session.get('patient_user_id')
        if not patient_user_id:
            flash('Please log in to access the patient portal.', 'warning')
            return redirect(url_for('patient_portal.login'))

        patient_user = PatientUser.query.get(patient_user_id)
        if not patient_user or not patient_user.is_active:
            session.pop('patient_user_id', None)
            flash('Your portal session is invalid or inactive.', 'danger')
            return redirect(url_for('patient_portal.login'))

        if patient_user.is_locked():
            session.pop('patient_user_id', None)
            flash('Account is locked due to multiple failed login attempts.', 'danger')
            return redirect(url_for('patient_portal.login'))

        g.current_patient_user = patient_user
        g.current_patient = patient_user.patient
        return f(*args, **kwargs)

    return decorated_function


@patient_portal_bp.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('patient_user_id'):
        return redirect(url_for('patient_portal.dashboard'))

    if request.method == 'POST':
        username = (request.form.get('username') or '').strip()
        password = request.form.get('password') or ''

        if not username or not password:
            flash('Username and password are required.', 'danger')
            return render_template('patient_portal/login.html')

        user = PatientUser.query.filter_by(username=username).first()
        if not user:
            flash('Invalid credentials.', 'danger')
            return render_template('patient_portal/login.html')

        if user.is_locked():
            flash('Account is locked due to consecutive failed attempts. Try again later.', 'danger')
            return render_template('patient_portal/login.html')

        if not user.check_password(password):
            user.failed_login_attempts += 1
            if user.failed_login_attempts >= 5:
                user.locked_until = datetime.utcnow() + timedelta(minutes=15)
                flash('Account locked for 15 minutes due to 5 failed login attempts.', 'danger')
            else:
                flash('Invalid credentials.', 'danger')
            db.session.commit()
            return render_template('patient_portal/login.html')

        # Successful login
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login = datetime.utcnow()
        db.session.commit()

        session['patient_user_id'] = user.id
        flash(f'Welcome back, {user.patient.name}!', 'success')
        return redirect(url_for('patient_portal.dashboard'))

    return render_template('patient_portal/login.html')


@patient_portal_bp.route('/register', methods=['GET', 'POST'])
def register():
    if session.get('patient_user_id'):
        return redirect(url_for('patient_portal.dashboard'))

    if request.method == 'POST':
        national_id = (request.form.get('national_id') or '').strip()
        patient_id = (request.form.get('patient_id') or '').strip()
        username = (request.form.get('username') or '').strip()
        password = request.form.get('password') or ''
        confirm_password = request.form.get('confirm_password') or ''

        if not username or not password:
            flash('All required fields must be filled out.', 'danger')
            return render_template('patient_portal/register.html')

        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return render_template('patient_portal/register.html')

        if len(password) < 8:
            flash('Password must be at least 8 characters long.', 'danger')
            return render_template('patient_portal/register.html')

        # Look up existing clinical Patient
        patient = None
        if national_id:
            patients = Patient.query.filter(Patient.national_id.isnot(None)).all()
            patient = next((p for p in patients if p.national_id == national_id), None)
        elif patient_id and patient_id.isdigit():
            patient = Patient.query.get(int(patient_id))

        if not patient:
            flash('No matching patient record found in hospital registry. Please verify National ID or Patient ID.', 'danger')
            return render_template('patient_portal/register.html')

        if patient.portal_user:
            flash('A portal account already exists for this patient. Please log in.', 'warning')
            return redirect(url_for('patient_portal.login'))

        if PatientUser.query.filter_by(username=username).first():
            flash('Username is already taken. Please choose another.', 'danger')
            return render_template('patient_portal/register.html')

        # Create PatientUser
        new_user = PatientUser(
            patient_id=patient.id,
            username=username
        )
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful! Please log in with your credentials.', 'success')
        return redirect(url_for('patient_portal.login'))

    return render_template('patient_portal/register.html')


@patient_portal_bp.route('/logout', methods=['GET', 'POST'])
def logout():
    session.pop('patient_user_id', None)
    flash('You have been logged out of the patient portal.', 'info')
    return redirect(url_for('patient_portal.login'))
