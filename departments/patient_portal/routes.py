from datetime import datetime

from flask import (
    flash,
    g,
    redirect,
    render_template,
    request,
    url_for,
)

from departments.api.audit import log_audit_event
from departments.models.billing import Invoice, Payment
from departments.models.insurance import Claim, PatientInsurance
from departments.models.medicine import RequestedLab
from departments.models.records import Clinic, ClinicBooking
from extensions import db

from . import patient_portal_bp
from .auth import patient_login_required


@patient_portal_bp.route('/dashboard')
@patient_login_required
def dashboard():
    patient = g.current_patient

    # Upcoming bookings (ClinicBooking uses clinic_date, not booking_date)
    upcoming_bookings = ClinicBooking.query.filter_by(
        patient_id=patient.patient_id
    ).filter(ClinicBooking.clinic_date >= datetime.utcnow().date()).all()

    # Released lab results only — integer statuses: 1 = done/released
    # We show status=1 (processed) as the "released" gate
    released_labs = RequestedLab.query.filter_by(
        patient_id=patient.patient_id
    ).filter(RequestedLab.status == 1).order_by(RequestedLab.date_requested.desc()).limit(5).all()

    # Invoices — patient_id FK is the string patient_id column
    recent_invoices = Invoice.query.filter_by(
        patient_id=patient.patient_id
    ).order_by(Invoice.created_at.desc()).limit(5).all()

    return render_template(
        'patient_portal/dashboard.html',
        patient=patient,
        upcoming_bookings=upcoming_bookings,
        released_labs=released_labs,
        recent_invoices=recent_invoices
    )


@patient_portal_bp.route('/appointments', methods=['GET'])
@patient_login_required
def appointments():
    patient = g.current_patient
    all_bookings = ClinicBooking.query.filter_by(
        patient_id=patient.patient_id
    ).order_by(ClinicBooking.clinic_date.desc()).all()

    clinics = Clinic.query.all()

    return render_template(
        'patient_portal/appointments.html',
        patient=patient,
        bookings=all_bookings,
        clinics=clinics
    )


@patient_portal_bp.route('/appointments/book', methods=['POST'])
@patient_login_required
def book_appointment():
    patient = g.current_patient
    clinic_id = request.form.get('clinic_id')
    booking_date_str = request.form.get('booking_date')

    if not clinic_id or not booking_date_str:
        flash('Please select a clinic and booking date.', 'danger')
        return redirect(url_for('patient_portal.appointments'))

    try:
        booking_date = datetime.strptime(booking_date_str, '%Y-%m-%d').date()
    except ValueError:
        flash('Invalid booking date format.', 'danger')
        return redirect(url_for('patient_portal.appointments'))

    # ClinicBooking uses: patient_id (String FK), clinic_id, clinic_date, seen
    booking = ClinicBooking(
        patient_id=patient.patient_id,
        clinic_id=int(clinic_id),
        clinic_date=booking_date,
        seen=0,
    )
    db.session.add(booking)
    db.session.commit()

    log_audit_event(
        action='BOOK_APPOINTMENT',
        resource_type='ClinicBooking',
        resource_id=str(booking.id),
        details=f'Patient {patient.name} booked appointment for {booking_date}',
        user_id=g.current_patient_user.id,
    )

    flash('Appointment request submitted successfully.', 'success')
    return redirect(url_for('patient_portal.appointments'))


@patient_portal_bp.route('/lab-results')
@patient_login_required
def lab_results():
    patient = g.current_patient

    # GATED: Patients only see processed/released lab results (status=1)
    released_labs = RequestedLab.query.filter_by(
        patient_id=patient.patient_id
    ).filter(RequestedLab.status == 1).order_by(RequestedLab.date_requested.desc()).all()

    return render_template(
        'patient_portal/lab_results.html',
        patient=patient,
        labs=released_labs
    )


@patient_portal_bp.route('/billing')
@patient_login_required
def billing():
    patient = g.current_patient

    invoices = Invoice.query.filter_by(
        patient_id=patient.patient_id
    ).order_by(Invoice.created_at.desc()).all()

    payments = Payment.query.filter_by(
        patient_id=patient.patient_id
    ).order_by(Payment.paid_at.desc()).all()

    claims = Claim.query.filter_by(
        patient_id=patient.patient_id
    ).order_by(Claim.created_at.desc()).all()

    return render_template(
        'patient_portal/billing.html',
        patient=patient,
        invoices=invoices,
        payments=payments,
        claims=claims
    )


@patient_portal_bp.route('/insurance')
@patient_login_required
def insurance():
    patient = g.current_patient

    policies = PatientInsurance.query.filter_by(
        patient_id=patient.patient_id
    ).all()

    claims = Claim.query.filter_by(
        patient_id=patient.patient_id
    ).order_by(Claim.created_at.desc()).all()

    return render_template(
        'patient_portal/insurance.html',
        patient=patient,
        policies=policies,
        claims=claims
    )


@patient_portal_bp.route('/profile', methods=['POST'])
@patient_login_required
def update_profile():
    patient = g.current_patient
    phone = (request.form.get('phone') or '').strip()
    address = (request.form.get('address') or '').strip()

    old_contact = patient.contact
    old_residence = patient.place_of_residence

    if phone:
        patient.contact = phone
    if address:
        patient.place_of_residence = address

    db.session.commit()

    log_audit_event(
        action='UPDATE_PROFILE',
        resource_type='Patient',
        resource_id=str(patient.patient_id),
        details=f'Contact updated from {old_contact} to {patient.contact}; Residence from {old_residence} to {patient.place_of_residence}',
        user_id=g.current_patient_user.id,
    )

    flash('Profile contact details updated successfully.', 'success')
    return redirect(url_for('patient_portal.dashboard'))
