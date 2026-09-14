"""
departments/renal/engine.py
────────────────────────────
Business-logic helpers for the Renal / Dialysis Unit.

Scope (DECISIONS_PENDING.md §23):
  ✅ #1  HD + CRRT modalities            — session CRUD
  ✅ #2  Manual session logging only     — no slot/chair scheduler yet
  ✅ #4  Vascular access surveillance    — access record helpers
  ✅ #7  Flat billing line on COMPLETED  — handled via event_listeners

NOT in scope (pending Nephrology Lead sign-off):
  ❌ #3  Kt/V adequacy alert             — no formula here
  ❌ #5  Prescription / anticoagulation  — no dosing logic
  ❌ #6  CDSS pharmacy hook              — no drug interactions
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

from departments.models.renal import DialysisSession, VascularAccessRecord
from extensions import db

logger = logging.getLogger(__name__)

# ── Allowed values ────────────────────────────────────────────────────────────
VALID_MODALITIES = {"HD", "CRRT"}
VALID_STATUSES = {"SCHEDULED", "IN_PROGRESS", "COMPLETED", "TERMINATED_EARLY"}
VALID_ACCESS_TYPES = {"AVF", "AVG", "Tunnelled Catheter", "Temporary Catheter"}


# ── Session helpers ───────────────────────────────────────────────────────────

def create_session(
    patient_id: str,
    nurse_id: int,
    modality: str,
    session_date: date,
    **kwargs: Any,
) -> DialysisSession:
    """
    Create and persist a new DialysisSession row.

    Raises ValueError for invalid modality.
    Only raw session parameters are accepted; Kt/V fields are NOT accepted.
    """
    modality = modality.upper()
    if modality not in VALID_MODALITIES:
        raise ValueError(f"Invalid modality '{modality}'. Must be one of {VALID_MODALITIES}.")

    session = DialysisSession(
        patient_id=patient_id,
        nurse_id=nurse_id,
        modality=modality,
        session_date=session_date,
        start_time=kwargs.get("start_time"),
        end_time=kwargs.get("end_time"),
        blood_flow_rate=kwargs.get("blood_flow_rate"),
        dialysate_flow_rate=kwargs.get("dialysate_flow_rate"),
        ultrafiltration_volume=kwargs.get("ultrafiltration_volume"),
        pre_weight=kwargs.get("pre_weight"),
        post_weight=kwargs.get("post_weight"),
        status=kwargs.get("status", "SCHEDULED"),
        notes=kwargs.get("notes"),
    )
    db.session.add(session)
    db.session.commit()
    logger.info("DialysisSession created: id=%s patient=%s modality=%s", session.id, patient_id, modality)
    return session


def update_session_status(session_id: int, new_status: str, end_time: datetime | None = None) -> DialysisSession:
    """
    Transition a DialysisSession to a new status.

    Billing is triggered automatically by event_listeners when status → COMPLETED.
    """
    new_status = new_status.upper()
    if new_status not in VALID_STATUSES:
        raise ValueError(f"Invalid status '{new_status}'. Must be one of {VALID_STATUSES}.")

    session = db.session.get(DialysisSession, session_id)
    if session is None:
        raise LookupError(f"DialysisSession #{session_id} not found.")

    session.status = new_status
    if end_time and new_status in ("COMPLETED", "TERMINATED_EARLY"):
        session.end_time = end_time
    db.session.commit()
    logger.info("DialysisSession #%s → status=%s", session_id, new_status)
    return session


def get_patient_sessions(patient_id: str) -> list[DialysisSession]:
    """Return all dialysis sessions for a patient, newest first."""
    return (
        DialysisSession.query
        .filter_by(patient_id=patient_id)
        .order_by(DialysisSession.session_date.desc(), DialysisSession.created_at.desc())
        .all()
    )


# ── Vascular access helpers ───────────────────────────────────────────────────

def log_access_record(
    patient_id: str,
    access_type: str,
    dialysis_session_id: int | None = None,
    **kwargs: Any,
) -> VascularAccessRecord:
    """
    Log or update a vascular access record for complication surveillance.

    Decision #4: track AVF/AVG/tunnelled catheter/temporary catheter.
    """
    if access_type not in VALID_ACCESS_TYPES:
        raise ValueError(f"Invalid access_type '{access_type}'. Must be one of {VALID_ACCESS_TYPES}.")

    record = VascularAccessRecord(
        patient_id=patient_id,
        access_type=access_type,
        insertion_date=kwargs.get("insertion_date"),
        site_description=kwargs.get("site_description"),
        complication_notes=kwargs.get("complication_notes"),
        dialysis_session_id=dialysis_session_id,
    )
    db.session.add(record)
    db.session.commit()
    logger.info(
        "VascularAccessRecord created: id=%s patient=%s type=%s",
        record.id, patient_id, access_type,
    )
    return record


def get_patient_access_records(patient_id: str) -> list[VascularAccessRecord]:
    """Return all vascular access records for a patient, newest first."""
    return (
        VascularAccessRecord.query
        .filter_by(patient_id=patient_id)
        .order_by(VascularAccessRecord.created_at.desc())
        .all()
    )


# ── Summary helper ────────────────────────────────────────────────────────────

def session_summary(session: DialysisSession) -> dict[str, Any]:
    """
    Return a plain-dict summary of a DialysisSession for API responses.
    Does NOT include Kt/V or any calculated adequacy metric (pending §23 #3).
    """
    weight_loss_kg: float | None = None
    if session.pre_weight is not None and session.post_weight is not None:
        weight_loss_kg = round(session.pre_weight - session.post_weight, 2)

    return {
        "id": session.id,
        "patient_id": session.patient_id,
        "modality": session.modality,
        "session_date": session.session_date.isoformat() if session.session_date else None,
        "status": session.status,
        "start_time": session.start_time.isoformat() if session.start_time else None,
        "end_time": session.end_time.isoformat() if session.end_time else None,
        "blood_flow_rate_ml_min": session.blood_flow_rate,
        "dialysate_flow_rate_ml_min": session.dialysate_flow_rate,
        "ultrafiltration_volume_ml": session.ultrafiltration_volume,
        "pre_weight_kg": session.pre_weight,
        "post_weight_kg": session.post_weight,
        "weight_loss_kg": weight_loss_kg,
        "notes": session.notes,
    }
