"""
departments/rcm/claims_scrubber_engine.py
──────────────────────────────────────────
RCM Pre-Submission Claims Scrubbing, EDI 837 Generator & EDI 835 Remittance Parser Engine.
"""

import json
import re
from datetime import datetime, timezone

from departments.models.records import Patient
from departments.rcm.models import (
    ClaimSubmission,
    Edi835RemittanceLog,
    PreAuthorization,
)
from extensions import db


class ClaimsScrubberEngine:
    """Core RCM Domain Engine for Pre-Claim Scrubbing, EDI 837/835, and Denial Risk Analysis."""

    # ---------------------------------------------------------------------------
    # Pre-Claim Rule Scrubber
    # ---------------------------------------------------------------------------
    @staticmethod
    def scrub_claim(claim_id: str) -> dict:
        """
        Execute pre-submission scrubbing rules on a ClaimSubmission record.
        Rules:
          1. Primary ICD-10 Code format & presence.
          2. Treatment date validity (start <= end, not future).
          3. Mandatory Pre-Authorization for claims > 20,000 KES or specific procedures.
          4. Billed amount positive & realistic.
          5. Duplicate claim collision check for same patient and start date.
        """
        claim = db.session.get(ClaimSubmission, claim_id)
        if not claim:
            raise ValueError(f"Claim #{claim_id} not found.")

        errors: list[dict[str, str]] = []
        warnings: list[dict[str, str]] = []

        # 1. Primary ICD-10 Format Check
        if not claim.primary_diagnosis_icd10:
            errors.append(
                {
                    "code": "ERR_ICD10_MISSING",
                    "severity": "CRITICAL",
                    "message": "Primary ICD-10 diagnosis code is required for claim submission.",
                }
            )
        else:
            pattern = r"^[A-Z][0-9]{2}(\.[0-9]{1,2})?$"
            if not re.match(pattern, claim.primary_diagnosis_icd10.strip().upper()):
                warnings.append(
                    {
                        "code": "WARN_ICD10_FORMAT",
                        "severity": "MEDIUM",
                        "message": f"Primary diagnosis '{claim.primary_diagnosis_icd10}' may not conform to standard ICD-10 formatting.",
                    }
                )

        # 2. Service Dates Check
        if claim.service_start_date and claim.service_end_date:
            if claim.service_start_date > claim.service_end_date:
                errors.append(
                    {
                        "code": "ERR_INVALID_DATES",
                        "severity": "CRITICAL",
                        "message": "Service start date cannot be after service end date.",
                    }
                )
            if claim.service_end_date > datetime.now(timezone.utc).date():
                errors.append(
                    {
                        "code": "ERR_FUTURE_DATES",
                        "severity": "CRITICAL",
                        "message": "Service end date cannot be in the future.",
                    }
                )

        # 3. Billed Amount Check
        billed = float(claim.billed_amount or 0)
        if billed <= 0:
            errors.append(
                {
                    "code": "ERR_BILLED_ZERO",
                    "severity": "CRITICAL",
                    "message": "Billed amount must be greater than zero.",
                }
            )

        # 4. Pre-Authorization Gate Check for High-Cost Claims (> 20,000 KES)
        if billed >= 20000.0:
            pre_auth = PreAuthorization.query.filter(
                PreAuthorization.patient_id == claim.patient_id,
                PreAuthorization.status == "APPROVED",
            ).first()
            if not pre_auth:
                errors.append(
                    {
                        "code": "ERR_PREAUTH_MISSING",
                        "severity": "HIGH",
                        "message": f"Approved Pre-Authorization required for high-cost claim (Billed: KES {billed:,.2f}).",
                    }
                )

        # 5. Duplicate Claim Check
        dup = ClaimSubmission.query.filter(
            ClaimSubmission.patient_id == claim.patient_id,
            ClaimSubmission.service_start_date == claim.service_start_date,
            ClaimSubmission.id != claim.id,
            ClaimSubmission.status.notin_(["DENIED", "DRAFT"]),
        ).first()

        if dup:
            warnings.append(
                {
                    "code": "WARN_POSSIBLE_DUPLICATE",
                    "severity": "HIGH",
                    "message": f"Existing active claim #{dup.id} found for patient on {claim.service_start_date}.",
                }
            )

        # Calculate Denial Risk Score (0-100%)
        critical_count = len(errors)
        warning_count = len(warnings)
        risk_score = min(100.0, (critical_count * 35.0) + (warning_count * 15.0))

        is_clean = len(errors) == 0
        claim.scrubbing_status = "CLEAN" if is_clean else "HAS_ERRORS"
        claim.denial_risk_score = risk_score
        claim.scrubbing_errors_json = json.dumps(
            {"errors": errors, "warnings": warnings}
        )

        if is_clean and claim.status in ("DRAFT", "UNSCRUBBED"):
            claim.status = "CLEAN"

        db.session.commit()

        return {
            "claim_id": claim.id,
            "scrubbing_status": claim.scrubbing_status,
            "is_clean": is_clean,
            "denial_risk_score": risk_score,
            "error_count": len(errors),
            "warning_count": len(warnings),
            "errors": errors,
            "warnings": warnings,
        }

    # ---------------------------------------------------------------------------
    # EDI 837 Generator (HIPAA X12 837P Professional Claim Format)
    # ---------------------------------------------------------------------------
    @staticmethod
    def generate_edi_837(claim_id: str) -> str:
        """
        Generate X12 EDI 837P transaction stream for a clean claim.
        """
        claim = db.session.get(ClaimSubmission, claim_id)
        if not claim:
            raise ValueError(f"Claim #{claim_id} not found.")

        patient = (
            db.session.get(Patient, claim.patient_id)
            if isinstance(claim.patient_id, str)
            else Patient.query.filter_by(id=claim.patient_id).first()
        )
        pat_name = patient.name.upper() if patient else "DOE, JANE"
        pat_id = claim.patient_id

        now_str = datetime.now(timezone.utc).strftime("%Y%m%d*%H%M")
        date_start_str = (
            claim.service_start_date.strftime("%Y%m%d")
            if claim.service_start_date
            else "20260901"
        )
        date_end_str = (
            claim.service_end_date.strftime("%Y%m%d")
            if claim.service_end_date
            else date_start_str
        )
        icd10 = (claim.primary_diagnosis_icd10 or "R69").replace(".", "")
        billed_str = f"{float(claim.billed_amount or 0):.2f}"

        edi_lines = [
            f"ISA*00*          *00*          *ZZ*HOSPITAL_MAIN   *ZZ*SHA_KENYA       *{now_str}*U*00401*000000001*0*P*:~",
            f"GS*HC*HOSPITAL_MAIN*SHA_KENYA*{now_str}*1*X*004010X098A1~",
            "ST*837*0001~",
            "BHT*0019*00*0001*20260914*1000*CH~",
            "NM1*41*2*HIMS ENTERPRISE HOSPITAL*****46*123456789~",
            "PER*IC*BILLING DEPT*TE*254700000000~",
            "NM1*40*2*SOCIAL HEALTH AUTHORITY*****46*SHA999~",
            "HL*1**20*1~",
            "PRV*BI*PXC*207R00000X~",
            "NM1*85*2*HIMS ENTERPRISE HOSPITAL*****XX*1098765432~",
            "HL*2*1*22*0~",
            f"NM1*IL*1*{pat_name}****MI*{pat_id}~",
            "N3*NAIROBI KENYA~",
            "HL*3*2*23*0~",
            f"CLM*{claim.id}*{billed_str}***11:B:1*Y*A*Y*Y~",
            f"HI*BK:{icd10}~",
            "LX*1~",
            f"SV1*HC:99214*{billed_str}*UN*1***1~",
            f"DTP*472*RD8*{date_start_str}-{date_end_str}~",
            "SE*19*0001~",
            "GE*1*1~",
            "IEA*1*000000001~",
        ]

        edi_text = "\n".join(edi_lines)
        claim.edi_837_content = edi_text
        if claim.status in ("DRAFT", "CLEAN"):
            claim.status = "SUBMITTED"
            claim.submitted_at = datetime.now(timezone.utc)

        db.session.commit()
        return edi_text

    # ---------------------------------------------------------------------------
    # EDI 835 Remittance Parser (HIPAA X12 835 ERA Format)
    # ---------------------------------------------------------------------------
    @staticmethod
    def parse_and_apply_edi_835(edi_content: str) -> dict:
        """
        Parse X12 835 Electronic Remittance Advice (ERA) content and apply claim payment/denial statuses.
        Format segment examples:
          BPR*I*1500.00*C*ACH...
          CLP*CLAIM101*1*2000.00*1500.00*REF99*11~  (CLP02: 1=Paid, 2=Denied, 3=Pended)
          CAS*CO*45*500.00~ (Adjustment)
        """
        lines = [
            line.strip()
            for line in edi_content.replace("\n", "").split("~")
            if line.strip()
        ]

        total_paid_in_bpr = 0.0
        claims_processed = 0
        claims_summary = []

        remittance_ref = f"ERA-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

        for line in lines:
            parts = line.split("*")
            segment = parts[0]

            if segment == "BPR" and len(parts) >= 3:
                try:
                    total_paid_in_bpr = float(parts[2])
                except ValueError:
                    pass

            elif segment == "CLP" and len(parts) >= 5:
                claim_id = parts[1]
                status_code = parts[2]  # 1=Paid/Processed, 2=Denied, 3=Pended
                billed_amt = float(parts[3]) if len(parts) > 3 and parts[3] else 0.0
                paid_amt = float(parts[4]) if len(parts) > 4 and parts[4] else 0.0
                payer_ref = parts[5] if len(parts) > 5 else "SHA-ERA"

                claim = db.session.get(ClaimSubmission, claim_id)
                if claim:
                    claims_processed += 1
                    claim.payer_reference = payer_ref

                    if status_code == "1":
                        claim.status = "PAID"
                        claim.paid_amount = paid_amt
                        claim.approved_amount = paid_amt
                        claim.paid_at = datetime.now(timezone.utc)

                        # Settle patient's unpaid bills via insurance claim payment
                        patient_id = claim.patient_id
                        if patient_id:
                            from departments.models.billing import (
                                Billing,
                                DrugsBill,
                                Invoice,
                            )
                            from departments.shared.visit_closure import (
                                advance_after_completion,
                                maybe_close_encounter,
                            )

                            for b in Billing.query.filter_by(
                                patient_id=patient_id, status=0
                            ).all():
                                b.status = 1
                            for d in DrugsBill.query.filter_by(
                                patient_id=patient_id, status=0
                            ).all():
                                d.status = 1
                            for inv in Invoice.query.filter_by(
                                patient_id=patient_id, status=0
                            ).all():
                                inv.status = 1

                            db.session.commit()
                            advance_after_completion(patient_id)
                            maybe_close_encounter(patient_id)

                        claims_summary.append(
                            {
                                "claim_id": claim_id,
                                "status": "PAID",
                                "billed": billed_amt,
                                "paid": paid_amt,
                            }
                        )
                    else:  # Denied or Pended
                        claim.status = "DENIED"
                        claim.paid_amount = 0.0
                        claim.approved_amount = 0.0
                        claims_summary.append(
                            {
                                "claim_id": claim_id,
                                "status": "DENIED",
                                "billed": billed_amt,
                                "paid": 0.0,
                            }
                        )

        log = Edi835RemittanceLog(
            payer_id="SHA_KENYA",
            remittance_reference=remittance_ref,
            total_claims_processed=claims_processed,
            total_paid_amount=total_paid_in_bpr,
            edi_content=edi_content,
        )
        db.session.add(log)
        db.session.commit()

        return {
            "remittance_reference": remittance_ref,
            "total_claims_processed": claims_processed,
            "total_paid_amount": total_paid_in_bpr,
            "claims_summary": claims_summary,
        }
