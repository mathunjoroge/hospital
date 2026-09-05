from datetime import datetime, timedelta
import logging

from departments.models.notification_log import OutboundNotificationLog
from departments.models.patient_user import PatientUser
from departments.models.records import ClinicBooking, Patient
from departments.notifications.dispatcher import (
    EVENT_APPOINTMENT_REMINDER,
    EVENT_CLAIM_STATUS_CHANGED,
    EVENT_INVOICE_DUE,
    EVENT_LAB_RESULT_READY,
    EVENT_PAYMENT_RECEIVED,
    NotificationDispatcher,
)

logger = logging.getLogger(__name__)


def _get_patient_email(patient_id: str) -> str:
    """Helper to find patient email address from PatientUser or Patient contact."""
    if not patient_id:
        return None

    patient = Patient.query.filter_by(patient_id=patient_id).first()
    if not patient:
        return None

    # Check PatientUser first
    patient_user = PatientUser.query.filter_by(patient_id=patient.id).first()
    if patient_user and patient_user.username and '@' in patient_user.username:
        return patient_user.username

    # Check patient.contact field if it looks like an email
    if patient.contact and '@' in patient.contact:
        return patient.contact

    # Fallback placeholder for notification log tracking in sandbox/demo mode
    return f"patient_{patient_id.lower()}@hospital.org"


def trigger_appointment_reminder(booking: ClinicBooking):
    """Trigger appointment reminder notification for a ClinicBooking."""
    if not booking:
        return None

    email = _get_patient_email(booking.patient_id)
    if not email:
        return None

    subject = "Appointment Reminder — Hospital HMIS"
    body = (
        f"Dear Patient,\n\n"
        f"This is a reminder for your upcoming clinic appointment scheduled on {booking.clinic_date}.\n"
        f"Booking Reference: #{booking.id}\n\n"
        f"Thank you for choosing our Healthcare Facility."
    )

    return NotificationDispatcher.dispatch_event(
        event_type=EVENT_APPOINTMENT_REMINDER,
        recipient=email,
        subject=subject,
        body=body,
        patient_id=booking.patient_id,
        channels=['email'],
    )


def trigger_lab_result_ready(lab_request):
    """Trigger lab result released notification."""
    if not lab_request:
        return None

    email = _get_patient_email(lab_request.patient_id)
    if not email:
        return None

    test_name = (
        lab_request.lab_test.test_name
        if getattr(lab_request, 'lab_test', None)
        else f"Lab Request #{lab_request.id}"
    )

    subject = "Lab Test Results Available — Hospital HMIS"
    body = (
        f"Dear Patient,\n\n"
        f"Your laboratory test results for '{test_name}' are now verified and released.\n"
        f"You may log in to your Patient Portal at /portal/lab-results to view your full results.\n\n"
        f"Hospital Laboratory Department"
    )

    return NotificationDispatcher.dispatch_event(
        event_type=EVENT_LAB_RESULT_READY,
        recipient=email,
        subject=subject,
        body=body,
        patient_id=lab_request.patient_id,
        channels=['email'],
    )


def trigger_invoice_due(invoice):
    """Trigger invoice due notification."""
    if not invoice:
        return None

    email = _get_patient_email(invoice.patient_id)
    if not email:
        return None

    subject = f"Invoice Issued #{invoice.invoice_number} — Hospital HMIS"
    body = (
        f"Dear Patient,\n\n"
        f"A new invoice #{invoice.invoice_number} for KES {invoice.total_amount:,.2f} has been issued.\n"
        f"Status: {invoice.status}\n"
        f"Please log in to your Patient Portal at /portal/billing to review and clear your invoice.\n\n"
        f"Hospital Billing Department"
    )

    return NotificationDispatcher.dispatch_event(
        event_type=EVENT_INVOICE_DUE,
        recipient=email,
        subject=subject,
        body=body,
        patient_id=invoice.patient_id,
        channels=['email'],
    )


def trigger_payment_received(payment):
    """Trigger payment receipt notification."""
    if not payment:
        return None

    invoice = payment.invoice
    patient_id = invoice.patient_id if invoice else None
    email = _get_patient_email(patient_id) if patient_id else None
    if not email:
        return None

    subject = f"Payment Received Receipt #{payment.receipt_number} — Hospital HMIS"
    body = (
        f"Dear Patient,\n\n"
        f"We have received your payment of KES {payment.amount:,.2f} via {payment.payment_method}.\n"
        f"Receipt Number: {payment.receipt_number}\n\n"
        f"Thank you for your prompt payment."
    )

    return NotificationDispatcher.dispatch_event(
        event_type=EVENT_PAYMENT_RECEIVED,
        recipient=email,
        subject=subject,
        body=body,
        patient_id=patient_id,
        channels=['email'],
    )


def trigger_claim_status_changed(claim):
    """Trigger insurance claim status notification."""
    if not claim:
        return None

    patient_id = claim.patient_id or (
        claim.patient_insurance.patient_id if claim.patient_insurance else None
    )
    email = _get_patient_email(patient_id) if patient_id else None
    if not email:
        return None

    subject = f"Insurance Claim #{claim.claim_number} Status Update — Hospital HMIS"
    body = (
        f"Dear Patient,\n\n"
        f"Your SHA/SHIF insurance claim #{claim.claim_number} status has been updated to '{claim.status}'.\n"
        f"Claimed Amount: KES {claim.claimed_amount:,.2f}\n"
        f"Approved Amount: KES {claim.approved_amount:,.2f}\n\n"
        f"Hospital Insurance Department"
    )

    return NotificationDispatcher.dispatch_event(
        event_type=EVENT_CLAIM_STATUS_CHANGED,
        recipient=email,
        subject=subject,
        body=body,
        patient_id=patient_id,
        channels=['email'],
    )


def send_upcoming_appointment_reminders(app=None):
    """Scheduled task sending reminders for appointments happening in the next 24 hours."""

    now = datetime.utcnow().date()
    tomorrow = now + timedelta(days=1)

    # Find bookings within next 24h
    bookings = ClinicBooking.query.filter(
        ClinicBooking.clinic_date >= now, ClinicBooking.clinic_date <= tomorrow
    ).all()

    reminders_sent = 0
    for booking in bookings:
        # Check if already sent in past 24h for this booking
        already_sent = OutboundNotificationLog.query.filter(
            OutboundNotificationLog.patient_id == booking.patient_id,
            OutboundNotificationLog.event_type == EVENT_APPOINTMENT_REMINDER,
            OutboundNotificationLog.created_at
            >= datetime.utcnow() - timedelta(hours=24),
        ).first()

        if not already_sent:
            trigger_appointment_reminder(booking)
            reminders_sent += 1

    logger.info(
        f"Scheduled appointment reminders job complete: sent {reminders_sent} reminders."
    )
    return reminders_sent
