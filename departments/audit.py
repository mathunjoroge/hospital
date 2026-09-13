import hashlib
import logging
from datetime import datetime, timezone

from flask import has_request_context
from flask_login import current_user
from sqlalchemy import event

from departments.models.admin import Log

logger = logging.getLogger(__name__)

GENESIS_HASH = "0" * 64


def compute_log_hash(
    ts_str: str,
    level: str,
    message: str,
    user_id: str,
    source: str,
    previous_hash: str,
) -> str:
    """Compute SHA-256 hash for a log entry to guarantee cryptographic immutability."""
    raw = f"{ts_str}|{level}|{message}|{user_id}|{source}|{previous_hash}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def log_audit_event(connection, level: str, message: str, user_id=None, source: str = "audit"):
    """Helper to insert an audit log entry with SHA-256 hash chaining."""
    stmt = Log.__table__.select().order_by(Log.id.desc()).limit(1)
    row = connection.execute(stmt).first()
    previous_hash = row.entry_hash if row and getattr(row, "entry_hash", None) else GENESIS_HASH

    ts = datetime.now(timezone.utc)
    ts_str = ts.isoformat()
    uid_str = str(user_id) if user_id is not None else "SYSTEM"

    entry_hash = compute_log_hash(ts_str, level, message, uid_str, source, previous_hash)

    connection.execute(
        Log.__table__.insert().values(
            timestamp=ts,
            level=level,
            message=message,
            user_id=user_id,
            source=source,
            previous_hash=previous_hash,
            entry_hash=entry_hash,
        )
    )


def verify_audit_log_chain() -> dict:
    """
    Verify full cryptographic SHA-256 hash chain across all Log entries.
    Detects any log tampering, insertion, deletion, or modification (HIPAA § 164.312(b)).
    """
    logs = Log.query.order_by(Log.id.asc()).all()
    if not logs:
        return {
            "valid": True,
            "total_logs": 0,
            "tampered_logs": [],
            "status": "No audit log entries recorded.",
        }

    expected_prev = GENESIS_HASH
    tampered = []

    for log in logs:
        if log.previous_hash and log.previous_hash != expected_prev:
            tampered.append({
                "log_id": log.id,
                "reason": "Previous hash mismatch (broken link in chain)",
                "stored_prev_hash": log.previous_hash,
                "expected_prev_hash": expected_prev,
            })

        if log.entry_hash:
            ts_str = log.timestamp.isoformat() if log.timestamp else ""
            uid_str = str(log.user_id) if log.user_id is not None else "SYSTEM"
            calc_hash = compute_log_hash(
                ts_str, log.level, log.message, uid_str, log.source or "", log.previous_hash or GENESIS_HASH
            )
            if log.entry_hash != calc_hash:
                tampered.append({
                    "log_id": log.id,
                    "reason": "Entry hash mismatch (content modified)",
                    "stored_hash": log.entry_hash,
                    "recalculated_hash": calc_hash,
                })

        expected_prev = log.entry_hash or expected_prev

    return {
        "valid": len(tampered) == 0,
        "total_logs": len(logs),
        "tampered_logs": tampered,
        "status": "Audit trail verified intact and tamper-free."
        if len(tampered) == 0
        else f"TAMPER ALERT: {len(tampered)} compromised log entries detected!",
    }


def register_audit_listeners():
    """Register automatic audit logging event listeners on key models."""
    from departments.models.billing import (
        Billing,
        DrugsBill,
        ImagingBill,
        LabBill,
        PaidBill,
    )
    from departments.models.laboratory import LabResult
    from departments.models.medicine import Imaging, PrescribedMedicine
    from departments.models.pharmacy import DispensedDrug
    from departments.models.records import Patient

    audited_models = [
        Patient,
        Billing,
        DrugsBill,
        PaidBill,
        LabBill,
        ImagingBill,
        DispensedDrug,
        LabResult,
        Imaging,
        PrescribedMedicine,
    ]

    for model in audited_models:

        @event.listens_for(model, "after_insert")
        def receive_after_insert(mapper, connection, target):
            user_id = None
            if (
                has_request_context()
                and hasattr(current_user, "id")
                and getattr(current_user, "is_authenticated", False)
            ):
                user_id = current_user.id
            model_name = target.__class__.__name__
            rec_id = getattr(target, "id", getattr(target, "patient_id", "N/A"))
            msg = f"Audit [INSERT] {model_name} (ID: {rec_id})"
            log_audit_event(connection, "INFO", msg, user_id=user_id, source="audit")

        @event.listens_for(model, "after_update")
        def receive_after_update(mapper, connection, target):
            user_id = None
            if (
                has_request_context()
                and hasattr(current_user, "id")
                and getattr(current_user, "is_authenticated", False)
            ):
                user_id = current_user.id
            model_name = target.__class__.__name__
            rec_id = getattr(target, "id", getattr(target, "patient_id", "N/A"))
            msg = f"Audit [UPDATE] {model_name} (ID: {rec_id})"
            log_audit_event(connection, "INFO", msg, user_id=user_id, source="audit")

        @event.listens_for(model, "after_delete")
        def receive_after_delete(mapper, connection, target):
            user_id = None
            if (
                has_request_context()
                and hasattr(current_user, "id")
                and getattr(current_user, "is_authenticated", False)
            ):
                user_id = current_user.id
            model_name = target.__class__.__name__
            rec_id = getattr(target, "id", getattr(target, "patient_id", "N/A"))
            msg = f"Audit [DELETE] {model_name} (ID: {rec_id})"
            log_audit_event(connection, "INFO", msg, user_id=user_id, source="audit")

