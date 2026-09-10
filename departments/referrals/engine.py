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
        patient_id: str,
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

        if new_status == ReferralStatus.ACCEPTED:
            self._handle_acceptance(referral)

        db.session.commit()
        logger.info("REFERRAL UPDATED: ID %s -> %s", referral_id, new_status)
        return referral

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _handle_acceptance(self, referral: Referral) -> None:
        """
        When a referral is accepted:
        1. Close the source encounter by transitioning it to REFERRED_OUT
           (uses close() since the stage machine has no REFERRED_OUT arc — we
           set the stage directly after marking status to avoid blocking the
           terminal set()).
        2. Open a new REFERRAL encounter at the receiving facility side so
           clinical orders at the destination are scoped to a distinct visit.
        """
        from departments.models.encounter import Encounter
        from departments.shared.encounter_utils import active_encounter

        patient_id = str(referral.patient_id)

        # 1. Mark the active source encounter as referred-out
        source_enc = active_encounter(patient_id)
        if source_enc:
            # close() sets status=DISCHARGED and stage=DISCHARGED; we override
            # stage to REFERRED_OUT so the clinical record reflects the reason.
            source_enc.close()
            source_enc.stage = "REFERRED_OUT"
            logger.info(
                "REFERRAL ACCEPTED: source encounter %s -> REFERRED_OUT (patient %s)",
                source_enc.id,
                patient_id,
            )

        # 2. Open receiving-facility encounter
        receiving_enc = Encounter(
            patient_id=patient_id,
            encounter_type="REFERRAL",
            stage="IN_CONSULTATION",
            status="ACTIVE",
            chief_complaint=referral.reason_for_referral,
        )
        db.session.add(receiving_enc)
        logger.info(
            "REFERRAL ACCEPTED: receiving encounter created for patient %s (facility: %s)",
            patient_id,
            referral.receiving_facility,
        )


class DischargeEngine:
    """
    Core engine for generating safe discharge summaries.
    """

    def generate_summary(
        self,
        patient_id: str,
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
