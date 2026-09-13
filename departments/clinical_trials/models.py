"""
departments/clinical_trials/models.py
────────────────────────────────────────
Database models for Clinical Trial Protocols, Participant e-Consent, Randomization & SAE Logging.
"""

import json
import uuid
from datetime import datetime, timezone

from extensions import db


class ClinicalTrialProtocol(db.Model):
    """
    Clinical Trial Protocol Registry Record (Phases I-IV).
    Stores inclusion/exclusion criteria, target enrollment, IRB approval details, and study phase.
    """

    __tablename__ = "clinical_trial_protocols"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    protocol_number = db.Column(db.String(50), unique=True, nullable=False, index=True)
    title = db.Column(db.String(255), nullable=False)
    phase = db.Column(db.String(20), nullable=False, default="Phase III")  # Phase I, Phase II, Phase III, Phase IV
    sponsor = db.Column(db.String(150), nullable=False)
    principal_investigator = db.Column(db.String(150), nullable=False)

    target_enrollment = db.Column(db.Integer, nullable=False, default=100)
    current_enrollment = db.Column(db.Integer, nullable=False, default=0)
    status = db.Column(db.String(30), nullable=False, default="RECRUITING")  # DRAFT, RECRUITING, ACTIVE, SUSPENDED, COMPLETED

    # Criteria stored as JSON string
    inclusion_criteria_json = db.Column(db.Text, nullable=True, default="[]")
    exclusion_criteria_json = db.Column(db.Text, nullable=True, default="[]")
    treatment_arms_json = db.Column(db.Text, nullable=True, default='["Arm A: Investigational", "Arm B: Control"]')

    irb_approval_number = db.Column(db.String(100), nullable=True)
    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    participants = db.relationship("TrialParticipant", backref="protocol", lazy="dynamic")
    adverse_events = db.relationship("TrialAdverseEvent", backref="protocol", lazy="dynamic")

    @property
    def inclusion_criteria(self) -> list:
        try:
            return json.loads(self.inclusion_criteria_json or "[]")
        except Exception:
            return []

    @inclusion_criteria.setter
    def inclusion_criteria(self, value: list):
        self.inclusion_criteria_json = json.dumps(value or [])

    @property
    def exclusion_criteria(self) -> list:
        try:
            return json.loads(self.exclusion_criteria_json or "[]")
        except Exception:
            return []

    @exclusion_criteria.setter
    def exclusion_criteria(self, value: list):
        self.exclusion_criteria_json = json.dumps(value or [])

    @property
    def treatment_arms(self) -> list:
        try:
            return json.loads(self.treatment_arms_json or '["Arm A: Investigational", "Arm B: Control"]')
        except Exception:
            return ["Arm A: Investigational", "Arm B: Control"]

    @treatment_arms.setter
    def treatment_arms(self, value: list):
        self.treatment_arms_json = json.dumps(value or [])


class TrialParticipant(db.Model):
    """
    Trial Participant Enrollment & e-Consent Record.
    Tracks screening status, digital signature hash for e-Consent, and randomization arm assignment.
    """

    __tablename__ = "trial_participants"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    protocol_id = db.Column(db.String(36), db.ForeignKey("clinical_trial_protocols.id"), nullable=False, index=True)
    patient_id = db.Column(db.String(20), db.ForeignKey("patients.patient_id"), nullable=False, index=True)

    # SCREENING, ELIGIBLE, INELIGIBLE, CONSENTED, RANDOMIZED, COMPLETED, WITHDRAWN
    enrollment_status = db.Column(db.String(30), nullable=False, default="SCREENING")
    # PENDING, SIGNED_ECONSENT, DECLINED, REVOKED
    consent_status = db.Column(db.String(30), nullable=False, default="PENDING")

    consent_signed_at = db.Column(db.DateTime(timezone=True), nullable=True)
    digital_signature_hash = db.Column(db.String(64), nullable=True)  # SHA-256 digital signature hash
    randomized_arm = db.Column(db.String(100), nullable=True)

    screening_notes = db.Column(db.Text, nullable=True)
    enrolled_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    adverse_events = db.relationship("TrialAdverseEvent", backref="participant", lazy="dynamic")
    patient = db.relationship("Patient", foreign_keys=[patient_id])


class TrialAdverseEvent(db.Model):
    """
    Safety & Adverse Event (AE) / Serious Adverse Event (SAE) Record.
    Grade 1-5 severity scale with IRB / Regulatory escalation flags.
    """

    __tablename__ = "trial_adverse_events"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    protocol_id = db.Column(db.String(36), db.ForeignKey("clinical_trial_protocols.id"), nullable=False, index=True)
    participant_id = db.Column(db.String(36), db.ForeignKey("trial_participants.id"), nullable=False, index=True)

    event_term = db.Column(db.String(255), nullable=False)
    # Grade 1 (Mild), Grade 2 (Moderate), Grade 3 (Severe), Grade 4 (Life-Threatening), Grade 5 (Death)
    severity_grade = db.Column(db.Integer, nullable=False, default=1)
    is_serious_ae = db.Column(db.Boolean, nullable=False, default=False)  # SAE flag

    # UNRELATED, UNLIKELY, POSSIBLE, PROBABLE, DEFINITE
    causality_assessment = db.Column(db.String(30), nullable=False, default="POSSIBLE")
    sae_reported_to_irb = db.Column(db.Boolean, nullable=False, default=False)
    resolution_status = db.Column(db.String(30), nullable=False, default="ONGOING")  # ONGOING, RESOLVED, RECOVERED_WITH_SEQUELAE, FATAL

    reported_by = db.Column(db.String(100), nullable=True)
    reported_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
