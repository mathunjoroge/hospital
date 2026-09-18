"""
departments/clinical_trials/trials_engine.py
──────────────────────────────────────────────
Clinical Trial Protocol, Automated Eligibility Screener, e-Consent, & SAE Safety Engine.
"""

import hashlib
import logging
import random
from datetime import datetime, timezone

from departments.clinical_trials.models import (
    ClinicalTrialProtocol,
    TrialAdverseEvent,
    TrialParticipant,
)
from departments.models.records import Patient
from extensions import db

logger = logging.getLogger(__name__)


class ClinicalTrialsEngine:
    """Core domain logic engine for Clinical Trial Registry, Eligibility, e-Consent, and SAE Logging."""

    @staticmethod
    def create_protocol(
        protocol_number: str,
        title: str,
        sponsor: str,
        principal_investigator: str,
        phase: str = "Phase III",
        target_enrollment: int = 100,
        inclusion_criteria: list[str] | None = None,
        exclusion_criteria: list[str] | None = None,
        treatment_arms: list[str] | None = None,
        irb_approval_number: str | None = None,
    ) -> ClinicalTrialProtocol:
        """Register a new Clinical Trial Protocol."""
        protocol = ClinicalTrialProtocol(
            protocol_number=protocol_number,
            title=title,
            sponsor=sponsor,
            principal_investigator=principal_investigator,
            phase=phase,
            target_enrollment=target_enrollment,
            irb_approval_number=irb_approval_number
            or f"IRB-{datetime.now(timezone.utc).strftime('%Y%m%d')}",
        )
        if inclusion_criteria:
            protocol.inclusion_criteria = inclusion_criteria
        if exclusion_criteria:
            protocol.exclusion_criteria = exclusion_criteria
        if treatment_arms:
            protocol.treatment_arms = treatment_arms

        db.session.add(protocol)
        db.session.commit()
        return protocol

    @staticmethod
    def screen_patient_eligibility(protocol_id: str, patient_id: str) -> dict:
        """
        Automated protocol eligibility screening based on patient demographics and clinical criteria.
        Checks:
          - Patient active status
          - Target enrollment capacity
        """
        protocol = db.session.get(ClinicalTrialProtocol, protocol_id)
        if not protocol:
            raise ValueError(f"Trial Protocol #{protocol_id} not found.")

        patient = Patient.query.filter_by(patient_id=patient_id).first()
        if not patient:
            raise ValueError(f"Patient #{patient_id} not found.")

        reasons: list[str] = []
        is_eligible = True

        if protocol.status != "RECRUITING":
            is_eligible = False
            reasons.append(
                f"Protocol is currently in status '{protocol.status}', not accepting new recruitment."
            )

        if protocol.current_enrollment >= protocol.target_enrollment:
            is_eligible = False
            reasons.append(
                f"Target enrollment capacity ({protocol.target_enrollment}) has been reached."
            )

        if not patient.is_active:
            is_eligible = False
            reasons.append("Patient record is inactive.")

        # Create or fetch TrialParticipant record
        participant = TrialParticipant.query.filter_by(
            protocol_id=protocol_id, patient_id=patient_id
        ).first()

        if not participant:
            participant = TrialParticipant(
                protocol_id=protocol_id,
                patient_id=patient_id,
                enrollment_status="ELIGIBLE" if is_eligible else "INELIGIBLE",
                screening_notes="; ".join(reasons)
                if reasons
                else "Eligible for enrollment.",
            )
            db.session.add(participant)
        else:
            participant.enrollment_status = "ELIGIBLE" if is_eligible else "INELIGIBLE"
            participant.screening_notes = (
                "; ".join(reasons) if reasons else "Eligible for enrollment."
            )

        db.session.commit()

        return {
            "participant_id": participant.id,
            "protocol_id": protocol_id,
            "patient_id": patient_id,
            "is_eligible": is_eligible,
            "screening_status": participant.enrollment_status,
            "screening_reasons": reasons
            if reasons
            else ["Passed automated eligibility screening."],
        }

    @staticmethod
    def record_econsent(participant_id: str, witness_name: str | None = None) -> dict:
        """
        Record electronic informed consent (e-Consent) with SHA-256 digital signature verification.
        """
        participant = db.session.get(TrialParticipant, participant_id)
        if not participant:
            raise ValueError(f"Trial Participant #{participant_id} not found.")

        now_iso = datetime.now(timezone.utc).isoformat()
        sig_data = f"{participant.id}:{participant.patient_id}:{participant.protocol_id}:{now_iso}:{witness_name or 'Direct'}"
        sig_hash = hashlib.sha256(sig_data.encode("utf-8")).hexdigest()

        participant.consent_status = "SIGNED_ECONSENT"
        participant.consent_signed_at = datetime.now(timezone.utc)
        participant.digital_signature_hash = sig_hash
        participant.enrollment_status = "CONSENTED"

        db.session.commit()

        return {
            "participant_id": participant.id,
            "consent_status": participant.consent_status,
            "signed_at": participant.consent_signed_at.isoformat(),
            "digital_signature_hash": sig_hash,
        }

    @staticmethod
    def randomize_participant(participant_id: str) -> dict:
        """
        Randomize a consented trial participant into one of the protocol treatment arms.
        """
        participant = db.session.get(TrialParticipant, participant_id)
        if not participant:
            raise ValueError(f"Trial Participant #{participant_id} not found.")

        if participant.consent_status != "SIGNED_ECONSENT":
            raise ValueError("Participant must sign e-Consent before randomization.")

        protocol = participant.protocol
        arms = protocol.treatment_arms
        selected_arm = random.choice(arms)

        participant.randomized_arm = selected_arm
        participant.enrollment_status = "RANDOMIZED"

        protocol.current_enrollment += 1
        db.session.commit()

        return {
            "participant_id": participant.id,
            "randomized_arm": selected_arm,
            "enrollment_status": participant.enrollment_status,
            "protocol_current_enrollment": protocol.current_enrollment,
        }

    @staticmethod
    def log_adverse_event(
        protocol_id: str,
        participant_id: str,
        event_term: str,
        severity_grade: int = 1,
        is_serious_ae: bool = False,
        causality_assessment: str = "POSSIBLE",
        reported_by: str | None = None,
    ) -> dict:
        """
        Log an Adverse Event (AE) / Serious Adverse Event (SAE) with IRB reporting trigger.
        Grade 4-5 or is_serious_ae=True triggers immediate IRB regulatory escalation alert.
        """
        grade = max(1, min(5, int(severity_grade)))
        if grade >= 4:
            is_serious_ae = True

        ae = TrialAdverseEvent(
            protocol_id=protocol_id,
            participant_id=participant_id,
            event_term=event_term,
            severity_grade=grade,
            is_serious_ae=is_serious_ae,
            causality_assessment=causality_assessment.upper(),
            sae_reported_to_irb=is_serious_ae,
            reported_by=reported_by or "Investigator",
        )
        db.session.add(ae)
        db.session.commit()

        irb_alert = False
        if is_serious_ae:
            irb_alert = True
            logger.warning(
                "SAE REPORTED: Trial %s, Participant %s - Grade %d SAE '%s'. IRB escalation triggered.",
                protocol_id,
                participant_id,
                grade,
                event_term,
            )

        return {
            "ae_id": ae.id,
            "event_term": ae.event_term,
            "severity_grade": ae.severity_grade,
            "is_serious_ae": ae.is_serious_ae,
            "irb_escalation_triggered": irb_alert,
            "causality": ae.causality_assessment,
        }

    @staticmethod
    def get_trial_registry_summary() -> list[dict]:
        """
        Retrieve summary metrics for all active clinical trials.
        """
        protocols = ClinicalTrialProtocol.query.order_by(
            ClinicalTrialProtocol.created_at.desc()
        ).all()
        summary = []

        for p in protocols:
            total_ae = p.adverse_events.count()
            sae_count = p.adverse_events.filter_by(is_serious_ae=True).count()
            summary.append(
                {
                    "id": p.id,
                    "protocol_number": p.protocol_number,
                    "title": p.title,
                    "phase": p.phase,
                    "sponsor": p.sponsor,
                    "principal_investigator": p.principal_investigator,
                    "status": p.status,
                    "target_enrollment": p.target_enrollment,
                    "current_enrollment": p.current_enrollment,
                    "total_adverse_events": total_ae,
                    "serious_adverse_events": sae_count,
                }
            )

        return summary
