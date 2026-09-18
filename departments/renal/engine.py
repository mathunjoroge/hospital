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
import math
from datetime import date, datetime
from typing import Any

from departments.models.renal import (
    DialysisPrescription,
    DialysisSession,
    VascularAccessRecord,
)
from extensions import db

logger = logging.getLogger(__name__)

# ── Allowed values ────────────────────────────────────────────────────────────
VALID_MODALITIES = {"HD", "CRRT"}
VALID_STATUSES = {"SCHEDULED", "IN_PROGRESS", "COMPLETED", "TERMINATED_EARLY"}
VALID_ACCESS_TYPES = {"AVF", "AVG", "Tunnelled Catheter", "Temporary Catheter"}


def calculate_spkt_v(
    pre_bun: float | None,
    post_bun: float | None,
    hours: float = 4.0,
    uf_L: float = 0.0,
    post_weight_kg: float = 70.0,
) -> float | None:
    """
    Calculate Single-Pool Kt/V (spKt/V) using Daugirdas II equation (Section 23 #3).

    Formula: spKt/V = -ln(R - 0.008 * t) + (4 - 3.5 * R) * (UF / W)
    where:
      R = post_BUN / pre_BUN
      t = session duration in hours
      UF = ultrafiltration volume in Liters
      W = post-dialysis weight in kg
    """
    if not pre_bun or not post_bun or pre_bun <= 0 or post_bun <= 0 or post_weight_kg <= 0:
        return None

    r = post_bun / pre_bun
    if r >= 1.0 or (r - 0.008 * hours) <= 0:
        return None

    term1 = -math.log(r - 0.008 * hours)
    term2 = (4.0 - 3.5 * r) * (uf_L / post_weight_kg)
    return round(term1 + term2, 2)


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
    """
    modality = modality.upper()
    if modality not in VALID_MODALITIES:
        raise ValueError(
            f"Invalid modality '{modality}'. Must be one of {VALID_MODALITIES}."
        )

    pre_bun = kwargs.get("pre_bun")
    post_bun = kwargs.get("post_bun")
    pre_w = kwargs.get("pre_weight")
    post_w = kwargs.get("post_weight")
    uf_vol = kwargs.get("ultrafiltration_volume") or 0.0
    uf_L = (pre_w - post_w) if (pre_w and post_w) else (uf_vol / 1000.0)

    start_t = kwargs.get("start_time")
    end_t = kwargs.get("end_time")
    hours = 4.0
    if start_t and end_t:
        hours = max(0.5, (end_t - start_t).total_seconds() / 3600.0)

    spkt_v = calculate_spkt_v(
        pre_bun=pre_bun,
        post_bun=post_bun,
        hours=hours,
        uf_L=uf_L,
        post_weight_kg=post_w or 70.0,
    )

    session = DialysisSession(
        patient_id=patient_id,
        nurse_id=nurse_id,
        modality=modality,
        session_date=session_date,
        start_time=start_t,
        end_time=end_t,
        blood_flow_rate=kwargs.get("blood_flow_rate"),
        dialysate_flow_rate=kwargs.get("dialysate_flow_rate"),
        ultrafiltration_volume=kwargs.get("ultrafiltration_volume"),
        pre_weight=pre_w,
        post_weight=post_w,
        pre_bun=pre_bun,
        post_bun=post_bun,
        spkt_v=spkt_v,
        status=kwargs.get("status", "SCHEDULED"),
        notes=kwargs.get("notes"),
    )
    db.session.add(session)
    db.session.commit()
    logger.info(
        "DialysisSession created: id=%s patient=%s modality=%s spKt/V=%s",
        session.id,
        patient_id,
        modality,
        spkt_v,
    )
    return session


def update_session_status(
    session_id: int, new_status: str, end_time: datetime | None = None, **kwargs: Any
) -> DialysisSession:
    """
    Transition a DialysisSession to a new status and update post-session labs/adequacy.
    """
    new_status = new_status.upper()
    if new_status not in VALID_STATUSES:
        raise ValueError(
            f"Invalid status '{new_status}'. Must be one of {VALID_STATUSES}."
        )

    session = db.session.get(DialysisSession, session_id)
    if session is None:
        raise LookupError(f"DialysisSession #{session_id} not found.")

    session.status = new_status
    if end_time and new_status in ("COMPLETED", "TERMINATED_EARLY"):
        session.end_time = end_time

    if "post_bun" in kwargs:
        session.post_bun = kwargs["post_bun"]
    if "post_weight" in kwargs:
        session.post_weight = kwargs["post_weight"]

    # Re-calculate spKt/V adequacy if lab numbers available
    if session.pre_bun and session.post_bun:
        hours = 4.0
        if session.start_time and session.end_time:
            hours = max(0.5, (session.end_time - session.start_time).total_seconds() / 3600.0)
        uf_L = (
            (session.pre_weight - session.post_weight)
            if (session.pre_weight and session.post_weight)
            else ((session.ultrafiltration_volume or 0) / 1000.0)
        )
        session.spkt_v = calculate_spkt_v(
            pre_bun=session.pre_bun,
            post_bun=session.post_bun,
            hours=hours,
            uf_L=uf_L,
            post_weight_kg=session.post_weight or 70.0,
        )

    db.session.commit()
    logger.info("DialysisSession #%s → status=%s (spKt/V=%s)", session_id, new_status, session.spkt_v)
    return session


def get_patient_sessions(patient_id: str) -> list[DialysisSession]:
    """Return all dialysis sessions for a patient, newest first."""
    return (
        DialysisSession.query.filter_by(patient_id=patient_id)
        .order_by(
            DialysisSession.session_date.desc(), DialysisSession.created_at.desc()
        )
        .all()
    )


# ── Prescription helpers (Section 23 #5) ────────────────────────────────────


def create_prescription(
    patient_id: str,
    nephrologist_id: int,
    **kwargs: Any,
) -> DialysisPrescription:
    """
    Create a new Nephrology Dialysis Prescription.
    """
    prescription = DialysisPrescription(
        patient_id=patient_id,
        nephrologist_id=nephrologist_id,
        dialysate_flow_rate=kwargs.get("dialysate_flow_rate", 500.0),
        blood_flow_rate=kwargs.get("blood_flow_rate", 300.0),
        dialysate_composition=kwargs.get("dialysate_composition", "K 2.0, Ca 1.25, Na 138"),
        heparin_bolus_units=kwargs.get("heparin_bolus_units", 1000.0),
        heparin_infusion_rate=kwargs.get("heparin_infusion_rate", 500.0),
        target_uf_liters=kwargs.get("target_uf_liters", 2.5),
        duration_hours=kwargs.get("duration_hours", 4.0),
        notes=kwargs.get("notes"),
    )
    db.session.add(prescription)
    db.session.commit()
    logger.info(
        "DialysisPrescription created: id=%s patient=%s nephrologist_id=%s",
        prescription.id,
        patient_id,
        nephrologist_id,
    )
    return prescription


def get_patient_prescriptions(patient_id: str) -> list[DialysisPrescription]:
    """Return all dialysis prescriptions for a patient, newest first."""
    return (
        DialysisPrescription.query.filter_by(patient_id=patient_id)
        .order_by(DialysisPrescription.created_at.desc())
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
    """
    if access_type not in VALID_ACCESS_TYPES:
        raise ValueError(
            f"Invalid access_type '{access_type}'. Must be one of {VALID_ACCESS_TYPES}."
        )

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
        record.id,
        patient_id,
        access_type,
    )
    return record


def get_patient_access_records(patient_id: str) -> list[VascularAccessRecord]:
    """Return all vascular access records for a patient, newest first."""
    return (
        VascularAccessRecord.query.filter_by(patient_id=patient_id)
        .order_by(VascularAccessRecord.created_at.desc())
        .all()
    )


# ── Summary helper ────────────────────────────────────────────────────────────


def session_summary(session: DialysisSession) -> dict[str, Any]:
    """
    Return a plain-dict summary of a DialysisSession for API responses.
    Includes Kt/V adequacy score when BUN values are provided (Section 23 #3).
    """
    weight_loss_kg: float | None = None
    if session.pre_weight is not None and session.post_weight is not None:
        weight_loss_kg = round(session.pre_weight - session.post_weight, 2)

    return {
        "id": session.id,
        "patient_id": session.patient_id,
        "modality": session.modality,
        "session_date": session.session_date.isoformat()
        if session.session_date
        else None,
        "status": session.status,
        "start_time": session.start_time.isoformat() if session.start_time else None,
        "end_time": session.end_time.isoformat() if session.end_time else None,
        "blood_flow_rate_ml_min": session.blood_flow_rate,
        "dialysate_flow_rate_ml_min": session.dialysate_flow_rate,
        "ultrafiltration_volume_ml": session.ultrafiltration_volume,
        "pre_weight_kg": session.pre_weight,
        "post_weight_kg": session.post_weight,
        "weight_loss_kg": weight_loss_kg,
        "pre_bun_mg_dl": session.pre_bun,
        "post_bun_mg_dl": session.post_bun,
        "spkt_v": session.spkt_v,
        "notes": session.notes,
    }

