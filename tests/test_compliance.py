"""
tests/test_compliance.py
────────────────────────
Unit tests for Task 2.6: Data Protection Act 2019 Compliance
(Consent tracking, Subject Access Requests, Anonymization)
"""

from datetime import date

import pytest

try:
    from extensions import db
except ImportError:
    from extensions import db

from departments.models.billing import Invoice
from departments.models.compliance import (
    PatientConsent,
    anonymize_patient_data,
    export_patient_sar_data,
    grant_patient_consent,
)
from departments.models.records import Patient


@pytest.fixture
def sample_patient(app):
    patient = Patient(
        patient_id="PTCOMP01",
        name="Jane Wanjiku",
        sex="Female",
        date_of_birth=date(1990, 4, 12),
        marital_status="Single",
        contact="0722112233",
        national_id="99887766",
        place_of_residence="Nairobi",
        next_of_kin="Peter Wanjiku",
        relationship_with_next_of_kin="Parent",
        next_of_kin_contact="0733445566",
        emergency_contact="0733445566",
    )
    db.session.add(patient)
    db.session.commit()
    return patient


class TestPatientConsent:
    def test_grant_and_revoke_consent(self, app, sample_patient):
        # 1. Grant consent
        consent = grant_patient_consent(
            patient_id=sample_patient.patient_id,
            consent_type="ai_diagnosis",
            ip_address="192.168.1.50",
            notes="Opt-in during registration",
        )
        assert consent.is_granted is True
        assert consent.granted_at is not None
        assert consent.revoked_at is None

        # 2. Revoke consent
        consent.revoke()
        db.session.commit()

        updated = PatientConsent.query.filter_by(
            patient_id=sample_patient.patient_id, consent_type="ai_diagnosis"
        ).first()
        assert updated.is_granted is False
        assert updated.revoked_at is not None


class TestSubjectAccessRequest:
    def test_export_patient_sar_data(self, app, sample_patient):
        # Grant consent & create invoice
        grant_patient_consent(sample_patient.patient_id, "data_sharing")
        inv = Invoice(
            patient_id=sample_patient.patient_id,
            total_amount=5000.0,
            balance_due=5000.0,
        )
        db.session.add(inv)
        db.session.commit()

        export = export_patient_sar_data(sample_patient.patient_id)
        assert "demographics" in export
        assert export["demographics"]["name"] == "Jane Wanjiku"
        assert export["demographics"]["national_id"] == "99887766"
        assert len(export["consents"]) == 1
        assert len(export["invoices"]) == 1


class TestRightToErasureAnonymization:
    def test_anonymize_patient_data(self, app, sample_patient):
        success = anonymize_patient_data(sample_patient.patient_id)
        assert success is True

        # Verify PII scrubbed
        p = Patient.query.filter_by(patient_id=sample_patient.patient_id).first()
        assert p.name.startswith("ANONYMIZED_PATIENT_")
        assert p.contact == "0000000000"
        assert p.national_id is None
        assert p.is_active is False
        assert p.deleted_at is not None
