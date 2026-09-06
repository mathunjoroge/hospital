from datetime import date, timedelta

import pytest

from departments.models.hr import Employee, StaffCredential
from departments.models.notification_log import OutboundNotificationLog
from departments.notifications.dispatcher import (
    EVENT_CREDENTIAL_EXPIRED,
    EVENT_CREDENTIAL_EXPIRING,
)
from departments.notifications.triggers import trigger_staff_credential_expiry_check
from extensions import db


@pytest.fixture
def sample_employee(app):
    with app.app_context():
        emp = Employee(
            employee_id="E-CRED-001",
            name="Dr. Alice Smith",
            role="Doctor",
            department="Medicine",
            job_group="Group A",
            email="alice.smith@hospital.org",
            phone="0711000111",
        )
        db.session.add(emp)
        db.session.commit()
        return emp.id


def test_staff_credential_expiring_soon_notification(app, sample_employee):
    """Verify credential expiring in 5 days triggers 'staff_credential_expiring' notification."""
    with app.app_context():
        today = date.today()
        cred = StaffCredential(
            employee_id=sample_employee,
            staff_name="Dr. Alice Smith",
            credential_type="KMPDC",
            credential_number="KMPDC-LIC-8821",
            issue_date=today - timedelta(days=360),
            expiry_date=today + timedelta(days=5),
            status="ACTIVE",
        )
        db.session.add(cred)
        db.session.commit()

        sent_count = trigger_staff_credential_expiry_check(app, window_days=30)
        assert sent_count > 0

        logs = OutboundNotificationLog.query.filter(
            OutboundNotificationLog.event_type == EVENT_CREDENTIAL_EXPIRING,
            OutboundNotificationLog.body.like("%KMPDC-LIC-8821%"),
        ).all()

        assert len(logs) >= 1
        log = logs[0]
        assert log.status == "SENT"
        assert "EXPIRING SOON" in log.subject
        assert "5 days" in log.body


def test_staff_credential_already_expired_notification(app, sample_employee):
    """Verify already expired credential (-2 days) triggers 'staff_credential_expired' notification and updates status."""
    with app.app_context():
        today = date.today()
        cred = StaffCredential(
            employee_id=sample_employee,
            staff_name="Dr. Alice Smith",
            credential_type="NCK",
            credential_number="NCK-REG-9912",
            issue_date=today - timedelta(days=400),
            expiry_date=today - timedelta(days=2),
            status="ACTIVE",
        )
        db.session.add(cred)
        db.session.commit()

        sent_count = trigger_staff_credential_expiry_check(app, window_days=30)
        assert sent_count > 0

        # Verify model status updated to EXPIRED
        updated_cred = StaffCredential.query.get(cred.id)
        assert updated_cred.status == "EXPIRED"

        logs = OutboundNotificationLog.query.filter(
            OutboundNotificationLog.event_type == EVENT_CREDENTIAL_EXPIRED,
            OutboundNotificationLog.body.like("%NCK-REG-9912%"),
        ).all()

        assert len(logs) >= 1
        log = logs[0]
        assert log.status == "SENT"
        assert "EXPIRED" in log.subject
        assert "CRITICAL NOTICE" in log.body


def test_staff_credential_valid_no_notification(app, sample_employee):
    """Verify valid credential expiring in 60 days (outside 30-day window) generates no notification."""
    with app.app_context():
        today = date.today()
        cred = StaffCredential(
            employee_id=sample_employee,
            staff_name="Dr. Alice Smith",
            credential_type="PPB",
            credential_number="PPB-PHARM-7741",
            issue_date=today - timedelta(days=30),
            expiry_date=today + timedelta(days=60),
            status="ACTIVE",
        )
        db.session.add(cred)
        db.session.commit()

        trigger_staff_credential_expiry_check(app, window_days=30)

        logs = OutboundNotificationLog.query.filter(
            OutboundNotificationLog.body.like("%PPB-PHARM-7741%")
        ).all()
        assert len(logs) == 0
