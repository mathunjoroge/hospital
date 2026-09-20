"""
departments/records/clinic_bridge.py
─────────────────────────────────────
Bridge between the Records department's clinic catalog and the specialty
clinical departments (Renal / Dialysis, Oncology).

Two responsibilities:

  1. seed_specialty_clinics() — idempotently guarantee that the specialty
     clinics exist in the `clinics` catalog, so Records staff can book
     patients into the Renal and Oncology departments.

  2. propagate_specialty_booking() — when Records books a patient into one
     of those clinics, create the department-specific record so the booking
     is actually visible where the care happens:
       • Renal / Dialysis → a SCHEDULED DialysisSession (renal console)
       • Oncology         → an OncologyBooking, status "Scheduled" (bookings board)

  Billing safety: dialysis billing only fires when a session reaches
  COMPLETED (departments/billing/event_listeners.py, decision #7), and
  OncologyBooking rows are not billed directly — so propagating a
  scheduled booking from Records never creates a charge.
"""

from __future__ import annotations

import logging
from datetime import date

from sqlalchemy import func

from departments.models.medicine import OncologyBooking
from departments.models.records import Clinic
from departments.models.renal import DialysisSession
from extensions import db

logger = logging.getLogger(__name__)

# Specialty clinics that MUST exist in the Records clinic catalog.
SPECIALTY_CLINICS: list[dict[str, object]] = [
    {"name": "Renal / Dialysis Clinic", "fee": 1000.00},
    {"name": "Oncology Clinic", "fee": 1000.00},
]

_RENAL_KEYWORDS = ("renal", "dialysis", "nephro", "kidney")
_ONCOLOGY_KEYWORDS = ("oncolog", "cancer", "chemo", "tumor", "tumour")


def specialty_for_clinic_name(name: str | None) -> str | None:
    """
    Classify a clinic name as 'renal', 'oncology', or None.

    Keyword-based so staff-created clinics named e.g. "Dialysis Unit" or
    "Cancer Clinic" are picked up as well as the seeded canonical names.
    """
    lowered = (name or "").lower()
    if any(k in lowered for k in _RENAL_KEYWORDS):
        return "renal"
    if any(k in lowered for k in _ONCOLOGY_KEYWORDS):
        return "oncology"
    return None


def seed_specialty_clinics() -> int:
    """
    Idempotently create the specialty clinics in the Records catalog.
    Returns the number of clinics newly created.
    """
    created = 0
    try:
        for spec in SPECIALTY_CLINICS:
            exists = Clinic.query.filter(
                func.lower(Clinic.name) == str(spec["name"]).lower()
            ).first()
            if exists:
                continue
            db.session.add(Clinic(name=str(spec["name"]), fee=spec["fee"]))  # type: ignore[arg-type]
            created += 1
        if created:
            db.session.commit()
            logger.info(
                "Seeded %d specialty clinic(s) into the Records catalog.", created
            )
        return created
    except Exception:  # noqa: BLE001
        db.session.rollback()
        logger.exception("Failed to seed specialty clinics.")
        return 0


def propagate_specialty_booking(
    clinic_name: str | None,
    patient_id: str,
    clinic_date: date,
    actor_id: int | None,
) -> dict[str, str | int] | None:
    """
    Mirror a Records clinic booking into the specialty department's own
    scheduling system.

      • Renal / Dialysis clinic → SCHEDULED DialysisSession (modality "HD"
        placeholder; renal staff confirm HD/CRRT chairside). nurse_id is the
        authenticated Records officer creating the booking — never a
        client-supplied value (renal P0-11 rule).
      • Oncology clinic → OncologyBooking (purpose "Consultation",
        status "Scheduled").

    Deduplicates per patient + date so re-booking after a failed commit
    cannot produce double schedules.

    Rows are added to the session WITHOUT committing — the caller commits
    them atomically together with the ClinicBooking.
    Returns {"department", "model", "id"} or None for non-specialty clinics.
    """
    department = specialty_for_clinic_name(clinic_name)
    if department is None:
        return None

    if department == "renal":
        existing = DialysisSession.query.filter_by(
            patient_id=patient_id, session_date=clinic_date, status="SCHEDULED"
        ).first()
        if existing:
            return {
                "department": "renal",
                "model": "DialysisSession",
                "id": existing.id,
            }
        session = DialysisSession(
            patient_id=patient_id,
            nurse_id=actor_id or 1,
            modality="HD",
            session_date=clinic_date,
            status="SCHEDULED",
            source="RECORDS",
            notes="Booked from Records — modality TBD (HD/CRRT) at chairside.",
        )
        db.session.add(session)
        db.session.flush()
        logger.info(
            "Records booking propagated to Renal Unit: patient=%s date=%s session_id=%s",
            patient_id,
            clinic_date,
            session.id,
        )
        return {
            "department": "renal",
            "model": "DialysisSession",
            "id": session.id,
        }

    # Oncology
    existing = OncologyBooking.query.filter_by(
        patient_id=patient_id, booking_date=clinic_date, status="Scheduled"
    ).first()
    if existing:
        return {
            "department": "oncology",
            "model": "OncologyBooking",
            "id": existing.id,
        }
    booking = OncologyBooking(
        patient_id=patient_id,
        booking_date=clinic_date,
        purpose="Consultation",
        status="Scheduled",
        notes="Booked from Records — Oncology Clinic.",
    )
    db.session.add(booking)
    db.session.flush()
    logger.info(
        "Records booking propagated to Oncology: patient=%s date=%s booking_id=%s",
        patient_id,
        clinic_date,
        booking.id,
    )
    return {
        "department": "oncology",
        "model": "OncologyBooking",
        "id": booking.id,
    }


if __name__ == "__main__":
    from app import app

    with app.app_context():
        count = seed_specialty_clinics()
        print(f"✅ Specialty clinic seeding complete: {count} clinic(s) created.")
