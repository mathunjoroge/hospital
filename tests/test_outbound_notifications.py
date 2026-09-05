from datetime import datetime, timedelta
from unittest.mock import patch
import pytest

from werkzeug.security import generate_password_hash

from departments.models.billing import Invoice, PaidBill, Payment
from departments.models.insurance import Claim, InsuranceScheme, PatientInsurance
from departments.models.medicine import LabTest, RequestedLab
from departments.models.notification_log import OutboundNotificationLog
from departments.models.records import Clinic, ClinicBooking, Patient
from departments.models.user import User
from departments.notifications.dispatcher import (
    EVENT_APPOINTMENT_REMINDER,
    EVENT_CLAIM_STATUS_CHANGED,
    EVENT_INVOICE_DUE,
    EVENT_LAB_RESULT_READY,
    EVENT_PAYMENT_RECEIVED,
    EmailChannel,
    NotificationDispatcher,
)
from departments.notifications.triggers import (
    send_upcoming_appointment_reminders,
    trigger_appointment_reminder,
    trigger_claim_status_changed,
    trigger_invoice_due,
    trigger_lab_result_ready,
    trigger_payment_received,
)
from extensions import db


@pytest.fixture
def sample_patient(app):
    with app.app_context():
        p = Patient(
            patient_id="P-TEST-001",
            name="John Doe",
            sex="Male",
            date_of_birth=datetime(1990, 1, 1).date(),
            marital_status="Single",
            contact="johndoe@example.com",
            place_of_residence="Nairobi",
            national_id="NAT-12345",
            next_of_kin="Jane Doe",
            relationship_with_next_of_kin="Sister",
            next_of_kin_contact="0700000000",
            emergency_contact="0700000000",
        )
        db.session.add(p)
        db.session.commit()
        return p.patient_id


def test_outbound_log_persistence(app, sample_patient):
    """Verify OutboundNotificationLog model creation and querying."""
    with app.app_context():
        log = OutboundNotificationLog(
            patient_id=sample_patient,
            recipient="johndoe@example.com",
            channel="email",
            event_type=EVENT_APPOINTMENT_REMINDER,
            subject="Test Subject",
            body="Test Body Content",
            status="SENT",
            sent_at=datetime.utcnow(),
        )
        db.session.add(log)
        db.session.commit()

        retrieved = OutboundNotificationLog.query.filter_by(
            patient_id=sample_patient
        ).first()
        assert retrieved is not None
        assert retrieved.recipient == "johndoe@example.com"
        assert retrieved.status == "SENT"
        assert retrieved.event_type == EVENT_APPOINTMENT_REMINDER


def test_notification_dispatcher_email_channel(app, sample_patient):
    """Verify NotificationDispatcher successfully dispatches email and records log."""
    with app.app_context():
        log = NotificationDispatcher.dispatch_event(
            event_type=EVENT_APPOINTMENT_REMINDER,
            recipient="johndoe@example.com",
            subject="Reminder Notice",
            body="Your clinic appointment is tomorrow.",
            patient_id=sample_patient,
            channels=['email'],
        )

        assert log is not None
        assert log.status == "SENT"
        assert log.patient_id == sample_patient
        assert log.channel == "email"


def test_failed_delivery_handling(app, sample_patient):
    """Verify failed channel dispatches log status FAILED and record error message."""
    with app.app_context():
        with patch.object(
            EmailChannel, 'send', side_effect=Exception('SMTP Connection Error')
        ):
            log = NotificationDispatcher.dispatch_event(
                event_type=EVENT_LAB_RESULT_READY,
                recipient="johndoe@example.com",
                subject="Lab Result",
                body="Your results are ready.",
                patient_id=sample_patient,
                channels=['email'],
            )

            assert log is not None
            assert log.status == "FAILED"
            assert "SMTP Connection Error" in log.error_message


def test_all_five_event_triggers(app, sample_patient):
    """Verify all 5 trigger functions create valid OutboundNotificationLog entries."""
    with app.app_context():
        # 1. Appointment Reminder Trigger
        clinic = Clinic(name="General Outpatient", fee=1000.0)
        db.session.add(clinic)
        db.session.commit()

        booking = ClinicBooking(
            patient_id=sample_patient,
            clinic_id=clinic.clinic_id,
            clinic_date=datetime.utcnow().date() + timedelta(days=1),
            seen=0,
        )
        db.session.add(booking)
        db.session.commit()

        log_appt = trigger_appointment_reminder(booking)
        assert log_appt is not None
        assert log_appt.event_type == EVENT_APPOINTMENT_REMINDER
        assert log_appt.status == "SENT"

        # 2. Lab Result Ready Trigger
        lab_test = LabTest(test_name="Full Blood Count", cost=1500.0)
        db.session.add(lab_test)
        db.session.commit()

        lab_req = RequestedLab(
            patient_id=sample_patient, lab_test_id=lab_test.id, status=1
        )
        db.session.add(lab_req)
        db.session.commit()

        log_lab = trigger_lab_result_ready(lab_req)
        assert log_lab is not None
        assert log_lab.event_type == EVENT_LAB_RESULT_READY
        assert log_lab.status == "SENT"

        # 3. Invoice Due Trigger
        inv = Invoice(
            patient_id=sample_patient, total_amount=2500.0, status='UNPAID'
        )
        db.session.add(inv)
        db.session.commit()

        log_inv = trigger_invoice_due(inv)
        assert log_inv is not None
        assert log_inv.event_type == EVENT_INVOICE_DUE
        assert log_inv.status == "SENT"

        # 4. Payment Received Trigger
        paid_record = PaidBill(
            receipt_number="REC-999001",
            patient_id=sample_patient,
            grand_total=2500.0,
            amount_paid=2500.0,
            balance=0.0,
            payment_method="MPESA",
        )
        db.session.add(paid_record)
        db.session.commit()

        pmt = Payment(
            invoice_id=inv.id,
            patient_id=sample_patient,
            amount=2500.0,
            payment_method="MPESA",
            receipt_number="REC-999001",
        )
        db.session.add(pmt)
        db.session.commit()

        log_pmt = trigger_payment_received(pmt)
        assert log_pmt is not None
        assert log_pmt.event_type == EVENT_PAYMENT_RECEIVED
        assert log_pmt.status == "SENT"

        # 5. Claim Status Changed Trigger
        scheme = InsuranceScheme(
            name="SHA Standard", code="SHA-01", scheme_type="public"
        )
        db.session.add(scheme)
        db.session.commit()

        patient_ins = PatientInsurance(
            patient_id=sample_patient,
            scheme_id=scheme.id,
            member_number="CARD-12345",
        )
        db.session.add(patient_ins)
        db.session.commit()

        claim = Claim(
            claim_number=Claim.generate_claim_number(),
            invoice_id=inv.id,
            patient_insurance_id=patient_ins.id,
            scheme_id=scheme.id,
            patient_id=sample_patient,
            claimed_amount=5000.0,
            approved_amount=4500.0,
            status="APPROVED",
        )
        db.session.add(claim)
        db.session.commit()

        log_claim = trigger_claim_status_changed(claim)
        assert log_claim is not None
        assert log_claim.event_type == EVENT_CLAIM_STATUS_CHANGED
        assert log_claim.status == "SENT"


def test_scheduled_appointment_reminders_job(app, sample_patient):
    """Verify scheduler job dispatches reminders for upcoming appointments."""
    with app.app_context():
        clinic = Clinic(name="Dental Clinic", fee=2000.0)
        db.session.add(clinic)
        db.session.commit()

        booking = ClinicBooking(
            patient_id=sample_patient,
            clinic_id=clinic.clinic_id,
            clinic_date=datetime.utcnow().date(),
            seen=0,
        )
        db.session.add(booking)
        db.session.commit()

        sent_count = send_upcoming_appointment_reminders(app)
        assert sent_count == 1

        # Running again should skip sending duplicate reminder
        sent_again = send_upcoming_appointment_reminders(app)
        assert sent_again == 0


def test_admin_outbound_notifications_endpoint(client, app, sample_patient):
    """Verify /admin/outbound-notifications JSON endpoint requires admin login and returns logs."""
    with app.app_context():
        admin_user = User(
            username="admin_notif_test",
            password=generate_password_hash("Password123!", method='pbkdf2:sha256'),
            role="admin",
        )
        db.session.add(admin_user)
        db.session.commit()

        NotificationDispatcher.dispatch_event(
            event_type=EVENT_APPOINTMENT_REMINDER,
            recipient="johndoe@example.com",
            subject="Admin Test Reminder",
            body="Body test",
            patient_id=sample_patient,
        )

    # Login as admin
    client.post(
        '/login',
        data={'username': 'admin_notif_test', 'password': 'Password123!'},
    )

    resp = client.get('/admin/outbound-notifications?format=json')
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'logs' in data
    assert len(data['logs']) >= 1
    assert data['logs'][0]['event_type'] == EVENT_APPOINTMENT_REMINDER
