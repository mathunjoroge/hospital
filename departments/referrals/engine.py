"""
Referral and Discharge Continuity Engine.

Manages inter-facility patient transfers and safe discharge workflows.
Ensures continuity of care across Kenya's KEPH tiered healthcare system
by generating standardized clinical summaries during handovers.
"""

import logging
from datetime import datetime, timezone

from extensions import db

from .models import DischargeSummary, Referral

logger = logging.getLogger(__name__)


class ReferralStatus:
    PENDING = "PENDING"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    IN_TRANSIT = "IN_TRANSIT"
    COMPLETED = "COMPLETED"


class ReferralEngine:
    """
    Core engine for managing patient referrals between facilities.
    """

    def initiate(
        self,
        patient_id: int,
        referring_facility: str,
        receiving_facility: str,
        reason: str,
        clinical_summary: str,
    ) -> Referral:
        referral = Referral(
            patient_id=patient_id,
            referring_facility=referring_facility,
            receiving_facility=receiving_facility,
            reason_for_referral=reason,
            clinical_summary=clinical_summary,
            status=ReferralStatus.PENDING,
        )
        db.session.add(referral)
        db.session.commit()

        logger.info(
            "REFERRAL INITIATED: Patient %s from %s to %s (ID: %s)",
            patient_id,
            referring_facility,
            receiving_facility,
            referral.id,
        )
        return referral

    def update_status(self, referral_id: str, new_status: str) -> Referral | None:
        valid_statuses = {
            ReferralStatus.PENDING,
            ReferralStatus.ACCEPTED,
            ReferralStatus.REJECTED,
            ReferralStatus.IN_TRANSIT,
            ReferralStatus.COMPLETED,
        }
        if new_status not in valid_statuses:
            raise ValueError("Invalid referral status")

        referral = Referral.query.get(referral_id)
        if not referral:
            return None

        referral.status = new_status
        db.session.commit()
        logger.info("REFERRAL UPDATED: ID %s -> %s", referral_id, new_status)
        return referral


class DischargeEngine:
    """
    Core engine for generating safe discharge summaries.
    """

    def generate_summary(
        self,
        patient_id: int,
        appointment_id: str | None,
        admission_date: datetime,
        primary_diagnosis: str,
        discharge_medications: str | None = None,
        follow_up_instructions: str | None = None,
        referred_to: str | None = None,
        secondary_diagnoses: str | None = None,
    ) -> DischargeSummary:

        discharge_date = datetime.now(timezone.utc)

        if admission_date > discharge_date:
            raise ValueError(
                "Admission date cannot be in the future relative to discharge."
            )

        summary = DischargeSummary(
            patient_id=patient_id,
            appointment_id=appointment_id,
            admission_date=admission_date,
            discharge_date=discharge_date,
            primary_diagnosis=primary_diagnosis,
            secondary_diagnoses=secondary_diagnoses,
            discharge_medications=discharge_medications,
            follow_up_instructions=follow_up_instructions,
            referred_to=referred_to,
        )
        db.session.add(summary)
        db.session.commit()

        logger.info(
            "DISCHARGE SUMMARY GENERATED: Patient %s (ID: %s)",
            patient_id,
            summary.id,
        )
        return summary
