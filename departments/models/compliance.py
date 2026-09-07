"""
departments/models/compliance.py
─────────────────────────────────
Task 2.6 — Data Protection Act 2019 Compliance Module

Models & Helpers:
  - PatientConsent: Tracks patient consent for AI processing, data sharing, marketing, research
  - export_patient_sar_data: Subject Access Request (SAR) full personal data export
  - anonymize_patient_data: Right to Erasure / Anonymization helper
"""

import json
import logging
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

try:
    from extensions import db
except ImportError:
    from extensions import db

from departments.models.billing import Invoice
from departments.models.insurance import PatientInsurance
from departments.models.records import Patient

logger = logging.getLogger(__name__)


class PatientConsent(db.Model):
    """
    Patient Consent tracking under Kenya Data Protection Act 2019.
    Tracks explicit opt-in/opt-out for processing categories.
    """

    __tablename__ = "patient_consents"

    id = Column(Integer, primary_key=True)
    patient_id = Column(
        String(50), ForeignKey("patients.patient_id"), nullable=False, index=True
    )
    consent_type = Column(
        String(100), nullable=False
    )  # e.g., 'ai_diagnosis', 'third_party_sharing', 'sms_notifications'
    is_granted = Column(Boolean, default=False, nullable=False)
    granted_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    revoked_at = Column(DateTime, nullable=True)
    ip_address = Column(String(50), nullable=True)
    notes = Column(Text, nullable=True)

    patient = relationship("Patient", backref=db.backref("consents", lazy="dynamic"))

    def revoke(self):
        """Revoke consent."""
        self.is_granted = False
        self.revoked_at = datetime.now(timezone.utc)


def grant_patient_consent(
    patient_id: str, consent_type: str, ip_address: str = None, notes: str = None
) -> PatientConsent:
    """Grant or update explicit consent for a patient."""
    consent = PatientConsent.query.filter_by(
        patient_id=patient_id, consent_type=consent_type
    ).first()
    if consent:
        consent.is_granted = True
        consent.granted_at = datetime.now(timezone.utc)
        consent.revoked_at = None
        consent.ip_address = ip_address or consent.ip_address
        consent.notes = notes or consent.notes
    else:
        consent = PatientConsent(
            patient_id=patient_id,
            consent_type=consent_type,
            is_granted=True,
            ip_address=ip_address,
            notes=notes,
        )
        db.session.add(consent)

    db.session.commit()
    return consent


def has_ai_consent(patient_id: str) -> bool:
    """
    Check whether a patient has an active, unrevoked AI-processing consent grant.

    Returns True only when a PatientConsent row exists with:
      - consent_type == 'ai_diagnosis'
      - is_granted == True
      - revoked_at IS NULL

    This is the authoritative consent gate for all external AI calls
    that touch patient-specific clinical data (DPA 2019 s.30 lawful basis).
    """
    if not patient_id:
        return False

    consent = (
        PatientConsent.query.filter_by(
            patient_id=patient_id,
            consent_type="ai_diagnosis",
            is_granted=True,
        )
        .filter(PatientConsent.revoked_at.is_(None))
        .first()
    )

    return consent is not None


def export_patient_sar_data(patient_id: str) -> dict:
    """
    Subject Access Request (SAR) Data Export under DPA 2019 Section 26.
    Generates a structured dict of all personal data held for the patient.
    """
    patient = Patient.query.filter_by(patient_id=patient_id).first()
    if not patient:
        return {"error": "Patient not found"}

    # Demographics
    data = {
        "export_metadata": {
            "requested_at": datetime.now(timezone.utc).isoformat(),
            "compliance": "Kenya Data Protection Act 2019 - Section 26 (Subject Access Request)",
        },
        "demographics": {
            "patient_id": patient.patient_id,
            "name": patient.name,
            "date_of_birth": str(patient.date_of_birth)
            if patient.date_of_birth
            else None,
            "sex": patient.sex,
            "contact": patient.contact,
            "national_id": patient.national_id,
            "place_of_residence": patient.place_of_residence,
            "marital_status": patient.marital_status,
            "next_of_kin": patient.next_of_kin,
            "next_of_kin_contact": patient.next_of_kin_contact,
            "emergency_contact": patient.emergency_contact,
            "insurance_provider": patient.insurance_provider,
            "insurance_policy_number": patient.insurance_policy_number,
        },
        "consents": [],
        "invoices": [],
        "insurance_policies": [],
    }

    # Consents
    consents = PatientConsent.query.filter_by(patient_id=patient_id).all()
    for c in consents:
        data["consents"].append(
            {
                "consent_type": c.consent_type,
                "is_granted": c.is_granted,
                "granted_at": c.granted_at.isoformat() if c.granted_at else None,
                "revoked_at": c.revoked_at.isoformat() if c.revoked_at else None,
            }
        )

    # Invoices
    invoices = Invoice.query.filter_by(patient_id=patient_id).all()
    for inv in invoices:
        data["invoices"].append(
            {
                "invoice_number": inv.invoice_number,
                "total_amount": float(inv.total_amount),
                "balance_due": float(inv.balance_due),
                "status": inv.status,
                "created_at": inv.created_at.isoformat() if inv.created_at else None,
            }
        )

    # Insurance
    policies = PatientInsurance.query.filter_by(patient_id=patient_id).all()
    for pol in policies:
        data["insurance_policies"].append(
            {
                "scheme": pol.scheme.name if pol.scheme else None,
                "member_number": pol.member_number,
                "is_active": pol.is_active,
            }
        )

    return data


def anonymize_patient_data(patient_id: str, operator_id: int = None) -> bool:
    """
    Anonymize patient personal identifiers under Right to Erasure / Anonymization (DPA 2019 Section 40).
    Anonymizes name, phone, national ID, kin info while retaining anonymized clinical structure for statutory medical audit.
    """
    patient = Patient.query.filter_by(patient_id=patient_id).first()
    if not patient:
        return False

    patient.name = f"ANONYMIZED_PATIENT_{patient.id}"
    patient.contact = "0000000000"
    patient.national_id = None
    patient.place_of_residence = "ANONYMIZED"
    patient.next_of_kin = "ANONYMIZED"
    patient.next_of_kin_contact = "0000000000"
    patient.emergency_contact = "0000000000"
    patient.insurance_policy_number = None
    patient.occupation = None
    patient.employer_name = None
    patient.soft_delete()

    db.session.commit()
    logger.info(
        f"Patient {patient_id} anonymized successfully by operator {operator_id}"
    )
    return True


class AuditLog(db.Model):
    """
    Persistent Audit Trail model tracking clinical and administrative actions.
    Append-only record of system activity for statutory compliance and SIEM export.
    """

    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True)
    timestamp = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True
    )
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    username = Column(String(128), nullable=True)

    action = Column(String(64), nullable=False, index=True)
    resource_type = Column(String(64), nullable=True, index=True)
    resource_id = Column(String(64), nullable=True, index=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(255), nullable=True)
    details = Column(Text, nullable=True)

    user = relationship(
        "User", foreign_keys=[user_id], backref=db.backref("audit_logs", lazy="dynamic")
    )

    def to_dict(self) -> dict:
        """Convert audit entry to dict representation."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "user_id": self.user_id,
            "username": self.username,
            "action": self.action,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "details": json.loads(self.details)
            if self.details and self.details.startswith("{")
            else self.details,
        }
