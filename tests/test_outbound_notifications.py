from datetime import datetime, timedelta, timezone
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
            date_of_birth=datetime(1990, 1, 1).date(),  # noqa: DTZ001
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
            sent_at=datetime.now(timezone.utc),
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
            channels=["email"],
        )

        assert log is not None
        assert log.status == "SENT"
        assert log.patient_id == sample_patient
        assert log.channel == "email"


def test_failed_delivery_handling(app, sample_patient):
    """Verify failed channel dispatches log status FAILED and record error message."""
    with app.app_context(), patch.object(
        EmailChannel, "send", side_effect=Exception("SMTP Connection Error")
    ):
        log = NotificationDispatcher.dispatch_event(
            event_type=EVENT_LAB_RESULT_READY,
            recipient="johndoe@example.com",
            subject="Lab Result",
            body="Your results are ready.",
            patient_id=sample_patient,
            channels=["email"],
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
            clinic_date=datetime.now(timezone.utc).date() + timedelta(days=1),
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
        inv = Invoice(patient_id=sample_patient, total_amount=2500.0, status="UNPAID")
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
            receipt_number="REC-999002",
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
            clinic_date=datetime.now(timezone.utc).date(),
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
            password=generate_password_hash("Password123!", method="pbkdf2:sha256"),
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
        "/login",
        data={"username": "admin_notif_test", "password": "Password123!"},
    )

    resp = client.get("/admin/outbound-notifications?format=json")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "logs" in data
    assert len(data["logs"]) >= 1
    assert data["logs"][0]["event_type"] == EVENT_APPOINTMENT_REMINDER


def test_africas_talking_sms_channel_success(app, sample_patient):
    """Verify AfricasTalkingSMSChannel successfully delivers SMS when API responds with Success."""
    from unittest.mock import MagicMock

    with app.app_context():
        app.config["SMS_CHANNEL"] = "africastalking"
        app.config["AT_API_KEY"] = "mock_at_api_key_123"
        app.config["AT_USERNAME"] = "sandbox"

        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.json.return_value = {
            "SMSMessageData": {
                "Message": "Sent to 1/1 Total Cost: KES 0.8000",
                "Recipients": [
                    {
                        "statusCode": 101,
                        "number": "+254712345678",
                        "status": "Success",
                        "cost": "KES 0.8000",
                        "messageId": "ATXid_test_12345",
                    }
                ],
            }
        }

        with patch("requests.post", return_value=mock_resp) as mock_post:
            log = NotificationDispatcher.dispatch_event(
                event_type=EVENT_APPOINTMENT_REMINDER,
                recipient="+254712345678",
                subject="Appointment Reminder",
                body="Your appointment is tomorrow at 10 AM.",
                patient_id=sample_patient,
                channels=["sms"],
            )

            assert log is not None
            assert log.status == "SENT"
            assert log.channel == "sms"
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args
            assert call_kwargs[1]["headers"]["ApiKey"] == "mock_at_api_key_123"


def test_africas_talking_sms_channel_delivery_failure(app, sample_patient):
    """Verify AfricasTalkingSMSChannel logs FAILED when provider reports recipient failure or error status."""
    from unittest.mock import MagicMock

    with app.app_context():
        app.config["SMS_CHANNEL"] = "africastalking"
        app.config["AT_API_KEY"] = "mock_at_api_key_123"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "SMSMessageData": {
                "Message": "Sent to 0/1",
                "Recipients": [
                    {
                        "statusCode": 403,
                        "number": "+254700000000",
                        "status": "UserInBlackList",
                        "cost": "KES 0.0000",
                        "messageId": "None",
                    }
                ],
            }
        }

        with patch("requests.post", return_value=mock_resp):
            log = NotificationDispatcher.dispatch_event(
                event_type=EVENT_APPOINTMENT_REMINDER,
                recipient="+254700000000",
                subject="Reminder",
                body="Test notification body",
                patient_id=sample_patient,
                channels=["sms"],
            )

            assert log is not None
            assert log.status == "FAILED"
            assert "UserInBlackList" in log.error_message


def test_africas_talking_sms_channel_missing_api_key(app, sample_patient):
    """Verify AfricasTalkingSMSChannel raises error and sets FAILED when AT_API_KEY is omitted."""
    with app.app_context():
        app.config["SMS_CHANNEL"] = "africastalking"
        app.config.pop("AT_API_KEY", None)
        import os

        old_env_key = os.environ.pop("AT_API_KEY", None)
        try:
            log = NotificationDispatcher.dispatch_event(
                event_type=EVENT_APPOINTMENT_REMINDER,
                recipient="+254712345678",
                subject="Reminder",
                body="Test message",
                patient_id=sample_patient,
                channels=["sms"],
            )

            assert log is not None
            assert log.status == "FAILED"
            assert "AT_API_KEY" in log.error_message
        finally:
            if old_env_key:
                os.environ["AT_API_KEY"] = old_env_key
