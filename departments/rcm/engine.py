"""
Revenue Cycle Management (RCM) Engine.

Handles insurance pre-authorization, automated claim scrubbing,
claim submission, and denial/appeal workflows.
"""

import logging
from datetime import datetime, timezone

from extensions import db

from .models import ClaimDenial, ClaimSubmission, PreAuthorization

logger = logging.getLogger(__name__)


class PreAuthStatus:
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    DENIED = "DENIED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


class ClaimStatus:
    DRAFT = "DRAFT"
    SCRUBBING = "SCRUBBING"
    SUBMITTED = "SUBMITTED"
    UNDER_REVIEW = "UNDER_REVIEW"
    APPROVED = "APPROVED"
    PAID = "PAID"
    DENIED = "DENIED"
    APPEALED = "APPEALED"


class AppealStatus:
    NOT_APPEALED = "NOT_APPEALED"
    APPEAL_IN_PROGRESS = "APPEAL_IN_PROGRESS"
    APPEAL_APPROVED = "APPEAL_APPROVED"
    APPEAL_DENIED = "APPEAL_DENIED"
    WRITTEN_OFF = "WRITTEN_OFF"


class RevenueCycleEngine:
    """
    Core engine for managing hospital financial sustainability,
    SHA/SHIF integration, and private insurance claims.
    """

    def submit_preauth(
        self,
        patient_id: int,
        insurance_scheme_id: int,
        procedure_code: str,
        estimated_amount: float,
        clinical_justification: str | None = None,
    ) -> PreAuthorization:
        """
        Initiates a pre-authorization request for high-cost procedures.
        """
        preauth = PreAuthorization(
            patient_id=patient_id,
            insurance_scheme_id=insurance_scheme_id,
            procedure_code=procedure_code,
            estimated_amount=estimated_amount,
            clinical_justification=clinical_justification,
            status=PreAuthStatus.PENDING,
        )
        db.session.add(preauth)
        db.session.commit()

        logger.info(
            "PRE-AUTH SUBMITTED: Patient %s, Procedure %s (ID: %s)",
            patient_id,
            procedure_code,
            preauth.id,
        )
        return preauth

    def scrub_claim(
        self,
        patient_id: int,
        billed_amount: float,
        service_start_date: datetime,
        service_end_date: datetime,
        primary_diagnosis_icd10: str | None,
    ) -> list[str]:
        """
        Automated claim scrubbing. Validates billing data before submission
        to prevent payer rejections. Returns a list of validation errors.
        """
        errors = []

        if billed_amount <= 0:
            errors.append("Billed amount must be greater than zero.")

        if service_end_date < service_start_date:
            errors.append("Service end date cannot be before service start date.")

        if not primary_diagnosis_icd10:
            errors.append(
                "Primary diagnosis (ICD-10) is required for claim submission."
            )

        return errors

    def submit_claim(
        self,
        patient_id: int,
        billed_amount: float,
        service_start_date: datetime,
        service_end_date: datetime,
        primary_diagnosis_icd10: str | None,
        secondary_diagnosis_icd10: str | None = None,
        insurance_scheme_id: int | None = None,
    ) -> ClaimSubmission | None:
        """
        Submits a claim after passing the scrubbing validation.
        Returns None if scrubbing fails.
        """
        scrub_errors = self.scrub_claim(
            patient_id=patient_id,
            billed_amount=billed_amount,
            service_start_date=service_start_date,
            service_end_date=service_end_date,
            primary_diagnosis_icd10=primary_diagnosis_icd10,
        )

        if scrub_errors:
            logger.warning(
                "CLAIM SCRUBBING FAILED: Patient %s. Errors: %s",
                patient_id,
                "; ".join(scrub_errors),
            )
            return None

        claim = ClaimSubmission(
            patient_id=patient_id,
            insurance_scheme_id=insurance_scheme_id,
            billed_amount=billed_amount,
            service_start_date=service_start_date,
            service_end_date=service_end_date,
            primary_diagnosis_icd10=primary_diagnosis_icd10,
            secondary_diagnosis_icd10=secondary_diagnosis_icd10,
            status=ClaimStatus.SUBMITTED,
            submitted_at=datetime.now(timezone.utc),
        )
        db.session.add(claim)
        db.session.commit()

        logger.info(
            "CLAIM SUBMITTED: Patient %s, Amount %s (ID: %s)",
            patient_id,
            billed_amount,
            claim.id,
        )
        return claim

    def appeal_claim(
        self,
        claim_id: str,
        denial_code: str,
        denial_reason: str,
        appeal_justification: str,
    ) -> ClaimDenial | None:
        """
        Initiates an appeal for a denied claim.
        """
        claim = ClaimSubmission.query.get(claim_id)
        if not claim:
            return None

        if claim.status != ClaimStatus.DENIED:
            raise ValueError("Only denied claims can be appealed.")

        denial = ClaimDenial(
            claim_id=claim_id,
            denial_code=denial_code,
            denial_reason=denial_reason,
            appeal_status=AppealStatus.APPEAL_IN_PROGRESS,
            appeal_justification=appeal_justification,
            appeal_submitted_at=datetime.now(timezone.utc),
        )
        db.session.add(denial)

        claim.status = ClaimStatus.APPEALED
        db.session.commit()

        logger.info(
            "CLAIM APPEALED: Claim %s, Denial Code %s",
            claim_id,
            denial_code,
        )
        return denial
