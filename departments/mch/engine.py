"""
Maternal and Child Health (MCH) Workflow Engine.

Aligns with Kenya's MoH 710 (Child Health) and MoH 711 (ANC) registers.
Enforces the Kenya Expanded Programme on Immunization (KEPI) schedule
and validates clinical workflows for Antenatal Care and Immunizations.
"""

import logging
from datetime import datetime, timedelta, timezone

from extensions import db

from .models import AncVisit, ImmunizationRecord

logger = logging.getLogger(__name__)


# Standard KEPI Schedule (Weeks of age)
KEPI_SCHEDULE = {
    "BCG": [0],
    "OPV": [0, 6, 10, 14],
    "Pentavalent": [6, 10, 14],
    "PCV": [6, 10, 14],
    "Rotavirus": [6, 10],
    "IPV": [14],
    "Measles-Rubella": [36, 72],  # ~9 months and 18 months
    "Yellow Fever": [36],  # ~9 months
}


class MchEngine:
    """
    Core engine for MCH workflows, ANC tracking, and immunization scheduling.
    """

    def log_anc_visit(
        self,
        patient_id: str,
        visit_number: int,
        gestation_weeks: int,
        high_risk_factors: str | None = None,
    ) -> AncVisit:
        """
        Records an Antenatal Care visit and calculates the next follow-up date.
        Standard ANC intervals:
          - ANC 1: < 16 weeks
          - ANC 2: 20-24 weeks
          - ANC 3: 28-32 weeks
          - ANC 4+: 36 weeks / weekly until delivery

        Also opens an ANC encounter scoped to this visit so that any orders
        (immunizations, labs) placed during the visit are linked correctly.
        """
        if gestation_weeks < 0 or gestation_weeks > 42:
            raise ValueError("Invalid gestation weeks. Must be between 0 and 42.")

        # Calculate next visit date based on gestation milestones
        if gestation_weeks < 28:
            weeks_until_next = 4
        elif gestation_weeks < 36:
            weeks_until_next = 2
        else:
            weeks_until_next = 1

        next_appointment_date = (
            datetime.now(timezone.utc) + timedelta(weeks=weeks_until_next)
        ).date()

        # Open an ANC encounter for this visit
        from departments.models.encounter import Encounter
        enc = Encounter(
            patient_id=patient_id,
            encounter_type="ANC",
            stage="IN_CONSULTATION",
            status="ACTIVE",
            chief_complaint=f"ANC Visit {visit_number} — {gestation_weeks} weeks gestation",
        )
        db.session.add(enc)
        db.session.flush()  # populate enc.id before linking

        visit = AncVisit(
            patient_id=patient_id,
            visit_number=visit_number,
            gestation_weeks=gestation_weeks,
            high_risk_factors=high_risk_factors,
            next_appointment_date=next_appointment_date,
            encounter_id=enc.id,
        )
        db.session.add(visit)
        db.session.commit()

        logger.info(
            "ANC VISIT LOGGED: Patient %s (Visit %s, %s weeks, encounter %s)",
            patient_id,
            visit_number,
            gestation_weeks,
            enc.id,
        )
        return visit

    def close_anc_visit(self, visit_id: str) -> AncVisit | None:
        """
        Closes an ANC visit by discharging the linked encounter.
        Call this when the patient leaves the MCH clinic after the visit.
        """
        visit = AncVisit.query.get(visit_id)
        if not visit:
            return None

        if visit.encounter_id:
            from departments.models.encounter import Encounter
            enc = Encounter.query.get(visit.encounter_id)
            if enc and enc.stage != "DISCHARGED":
                enc.close()
                db.session.commit()
                logger.info(
                    "ANC VISIT CLOSED: Visit %s encounter %s -> DISCHARGED",
                    visit_id,
                    enc.id,
                )
        return visit

    def record_immunization(
        self,
        child_patient_id: str,
        vaccine_name: str,
        dose_number: int,
        batch_number: str | None = None,
    ) -> ImmunizationRecord:
        """
        Records a vaccine administration, enforcing dose sequencing and preventing duplicates.
        """
        # Check if this exact dose was already given
        existing = ImmunizationRecord.query.filter_by(
            child_patient_id=child_patient_id,
            vaccine_name=vaccine_name,
            dose_number=dose_number,
        ).first()

        if existing:
            raise ValueError(
                "%s Dose %s has already been administered to this child.",
                vaccine_name,
                dose_number,
            )

        # Basic sequence check: ensure previous dose exists (if dose > 1)
        if dose_number > 1:
            prev_dose = ImmunizationRecord.query.filter_by(
                child_patient_id=child_patient_id,
                vaccine_name=vaccine_name,
                dose_number=dose_number - 1,
            ).first()
            if not prev_dose:
                raise ValueError(
                    "Cannot administer %s Dose %s before Dose %s is given.",
                    vaccine_name,
                    dose_number,
                    dose_number - 1,
                )

        # Link to the active ANC encounter for this child (if one is open)
        from departments.shared.encounter_utils import active_encounter
        enc = active_encounter(str(child_patient_id))
        enc_id = enc.id if enc else None

        record = ImmunizationRecord(
            child_patient_id=child_patient_id,
            vaccine_name=vaccine_name,
            dose_number=dose_number,
            batch_number=batch_number,
            encounter_id=enc_id,
        )
        db.session.add(record)
        db.session.commit()

        logger.info(
            "IMMUNIZATION RECORDED: Child %s received %s Dose %s",
            child_patient_id,
            vaccine_name,
            dose_number,
        )
        return record

    def get_due_vaccines(
        self, child_age_weeks: int, child_patient_id: int
    ) -> list[dict]:
        """
        Returns a list of vaccines due for a child based on their current age in weeks.
        """
        due = []

        # Get already administered vaccines for this child
        given = {
            f"{r.vaccine_name}_{r.dose_number}"
            for r in ImmunizationRecord.query.filter_by(
                child_patient_id=child_patient_id
            ).all()
        }

        for vaccine, schedule_weeks in KEPI_SCHEDULE.items():
            for i, week_due in enumerate(schedule_weeks):
                dose_num = i + 1
                key = f"{vaccine}_{dose_num}"

                # If the child has reached the age for this dose and hasn't received it
                if child_age_weeks >= week_due and key not in given:
                    due.append(
                        {
                            "vaccine_name": vaccine,
                            "dose_number": dose_num,
                            "week_due": week_due,
                            "status": "OVERDUE"
                            if child_age_weeks > week_due + 4
                            else "DUE",
                        }
                    )

        return due
