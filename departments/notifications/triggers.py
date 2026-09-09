import logging
from datetime import datetime, timedelta, timezone

from departments.models.notification_log import OutboundNotificationLog
from departments.models.patient_user import PatientUser
from departments.models.records import ClinicBooking, Patient
from departments.notifications.dispatcher import (
    EVENT_APPOINTMENT_REMINDER,
    EVENT_CLAIM_STATUS_CHANGED,
    EVENT_CREDENTIAL_EXPIRED,
    EVENT_CREDENTIAL_EXPIRING,
    EVENT_INVOICE_DUE,
    EVENT_LAB_RESULT_READY,
    EVENT_PASSWORD_RESET,
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
    if patient_user and patient_user.username and "@" in patient_user.username:
        return patient_user.username

    # Check patient.contact field if it looks like an email
    if patient.contact and "@" in patient.contact:
        return patient.contact

    # Fallback placeholder for notification log tracking in sandbox/demo mode
    return f"patient_{patient_id.lower()}@hospital.org"


def trigger_password_reset_email(patient_user: PatientUser, reset_link: str):
    """Trigger password reset email notification for a PatientUser."""
    if not patient_user or not patient_user.patient:
        return None

    email = _get_patient_email(patient_user.patient.patient_id)
    if not email:
        return None

    subject = "Password Reset Request — Hospital HMIS Patient Portal"
    body = (
        f"Dear Patient,\n\n"
        f"We received a request to reset your password. Click the link below to set a new password:\n\n"
        f"{reset_link}\n\n"
        f"This link will expire in 1 hour. If you did not request this, please ignore this email.\n\n"
        f"Hospital IT Support"
    )

    return NotificationDispatcher.dispatch_event(
        event_type=EVENT_PASSWORD_RESET,
        recipient=email,
        subject=subject,
        body=body,
        patient_id=patient_user.patient.patient_id,
        channels=["email"],
    )


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
        channels=["email"],
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
        if getattr(lab_request, "lab_test", None)
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
        channels=["email"],
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
        channels=["email"],
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
        channels=["email"],
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
        channels=["email"],
    )


def send_upcoming_appointment_reminders(app=None):
    """Scheduled task sending reminders for appointments happening in the next 24 hours."""

    now = datetime.now(timezone.utc).date()
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
            >= datetime.now(timezone.utc) - timedelta(hours=24),
        ).first()

        if not already_sent:
            trigger_appointment_reminder(booking)
            reminders_sent += 1

    logger.info(
        f"Scheduled appointment reminders job complete: sent {reminders_sent} reminders."
    )
    return reminders_sent


def trigger_staff_credential_expiry_check(app=None, window_days: int = 30) -> int:
    """
    Scheduled task to query StaffCredential records and notify staff & department admin
    if credential is expiring within window_days or has already expired.
    """
    from datetime import date, timezone

    from departments.models.hr import StaffCredential
    from extensions import db

    today = date.today()
    cutoff_date = today + timedelta(days=window_days)

    # Query credentials expiring on or before cutoff_date (excluding renewed credentials)
    credentials = StaffCredential.query.filter(
        StaffCredential.expiry_date <= cutoff_date, StaffCredential.status != "RENEWED"
    ).all()

    notifications_sent = 0

    for cred in credentials:
        days_until = (cred.expiry_date - today).days

        if days_until < 0 or cred.status == "EXPIRED":
            event_type = EVENT_CREDENTIAL_EXPIRED
            cred.status = "EXPIRED"
            subject = f"EXPIRED: Staff Credential Alert — {cred.credential_type} ({cred.staff_name})"
            body = (
                f"CRITICAL NOTICE:\n\n"
                f"Staff credential '{cred.credential_type}' (License #{cred.credential_number}) "
                f"for {cred.staff_name} EXPIRED on {cred.expiry_date}.\n"
                f"Status updated to EXPIRED. Immediate renewal required before clinical duties resume."
            )
        else:
            event_type = EVENT_CREDENTIAL_EXPIRING
            subject = f"EXPIRING SOON: Staff Credential Warning — {cred.credential_type} ({cred.staff_name})"
            body = (
                f"WARNING:\n\n"
                f"Staff credential '{cred.credential_type}' (License #{cred.credential_number}) "
                f"for {cred.staff_name} is expiring in {days_until} days on {cred.expiry_date}.\n"
                f"Please submit renewal documentation promptly."
            )

        # Staff email
        staff_email = (
            cred.employee.email
            if (cred.employee and cred.employee.email)
            else f"{cred.staff_name.lower().replace(' ', '.')}@hospital.org"
        )

        # Department admin email
        dept_name = (
            cred.employee.department
            if (cred.employee and cred.employee.department)
            else "hr"
        )
        admin_email = f"admin_{dept_name.lower().replace(' ', '_')}@hospital.org"

        for recipient in set([staff_email, admin_email]):
            # Deduplicate if already notified in past 24 hours
            recent = OutboundNotificationLog.query.filter(
                OutboundNotificationLog.recipient == recipient,
                OutboundNotificationLog.event_type == event_type,
                OutboundNotificationLog.body.like(f"%{cred.credential_number}%"),
                OutboundNotificationLog.created_at
                >= datetime.now(timezone.utc) - timedelta(hours=24),
            ).first()

            if not recent:
                NotificationDispatcher.dispatch_event(
                    event_type=event_type,
                    recipient=recipient,
                    subject=subject,
                    body=body,
                    channels=["email"],
                )
                notifications_sent += 1

    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error committing credential status updates: {e}")

    logger.info(
        f"Staff credential expiry check complete: {notifications_sent} notifications sent."
    )
    return notifications_sent


def trigger_batch_expiry_check(app=None, window_days: int = 30) -> int:
    """
    Scheduled task to query Batch records and notify pharmacy & store managers
    if drug batches are expiring within window_days or have already expired.
    """
    from datetime import date, timezone

    from departments.models.pharmacy import Batch

    today = date.today()
    cutoff_date = today + timedelta(days=window_days)

    expiring_batches = Batch.query.filter(
        Batch.expiry_date <= cutoff_date, Batch.quantity_in_stock > 0
    ).all()

    notifications_sent = 0
    recipient = "stores_pharmacy_alerts@hospital.org"

    for batch in expiring_batches:
        days_until = (batch.expiry_date - today).days
        drug_name = batch.drug.generic_name if batch.drug else f"Drug #{batch.drug_id}"

        if days_until < 0:
            subject = f"EXPIRED STOCK ALERT: Batch #{batch.batch_number} ({drug_name})"
            body = (
                f"CRITICAL EXPIRY ALERT:\n\n"
                f"Drug Batch '{batch.batch_number}' for {drug_name} EXPIRED on {batch.expiry_date}.\n"
                f"Stock remaining on shelf: {batch.quantity_in_stock} units.\n"
                f"Immediate quarantine or disposal write-off required."
            )
        else:
            subject = f"SOON EXPIRING STOCK WARNING: Batch #{batch.batch_number} ({drug_name})"
            body = (
                f"EXPIRING STOCK WARNING:\n\n"
                f"Drug Batch '{batch.batch_number}' for {drug_name} expires in {days_until} days on {batch.expiry_date}.\n"
                f"Stock remaining on shelf: {batch.quantity_in_stock} units.\n"
                f"Please prioritize FEFO dispensing or process Return-to-Vendor."
            )

        recent = OutboundNotificationLog.query.filter(
            OutboundNotificationLog.recipient == recipient,
            OutboundNotificationLog.body.like(f"%{batch.batch_number}%"),
            OutboundNotificationLog.created_at
            >= datetime.now(timezone.utc) - timedelta(hours=24),
        ).first()

        if not recent:
            NotificationDispatcher.dispatch_event(
                event_type="BATCH_EXPIRING",
                recipient=recipient,
                subject=subject,
                body=body,
                channels=["email"],
            )
            notifications_sent += 1

    logger.info(
        f"Batch expiry check complete: {notifications_sent} notifications dispatched."
    )
    return notifications_sent
