"""
departments/hl7/adt_sender.py
──────────────────────────────
Phase 4 — P4-04: ADT^A01 / ADT^A03 / ADT^A08 Sender Hook

Thin, non-blocking publisher that emits HL7v2 ADT messages to downstream
systems (Mirth Connect or a direct MLLP peer) when patient lifecycle events
fire in the Flask HMIS.

Events handled:
  ADT^A01  — Patient Admission      (new AdmittedPatient row committed)
  ADT^A03  — Patient Discharge      (discharge event / bed cleared)
  ADT^A08  — Patient Info Update    (Patient record update)
  ADT^A28  — Add Person Information (new Patient registration)

GUARDRAILS:
  - This module is imported by SQLAlchemy event listeners, NOT by app.py directly.
  - All network I/O is fire-and-forget (runs in a daemon thread) so it never
    blocks the Flask request / SQLAlchemy flush cycle.
  - If MLLP_DOWNSTREAM_HOST is unset, ADT sending is silently skipped
    (graceful degradation for single-facility deployments without a downstream).
"""

import logging
import os
import socket
import threading
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MLLP_DOWNSTREAM_HOST = os.environ.get("MLLP_DOWNSTREAM_HOST", "")
MLLP_DOWNSTREAM_PORT = int(os.environ.get("MLLP_DOWNSTREAM_PORT", "2575"))
SENDING_APP = os.environ.get("MLLP_SENDING_APP", "HMIS")
SENDING_FAC = os.environ.get("MLLP_SENDING_FAC", "KE_HOSPITAL")
RECEIVING_APP = os.environ.get("MLLP_RECEIVING_APP", "MIRTH")
RECEIVING_FAC = os.environ.get("MLLP_RECEIVING_FAC", "KE")

MLLP_SB = b"\x0b"
MLLP_EB = b"\x1c"
MLLP_CR = b"\x0d"

_CONNECT_TIMEOUT = 5  # seconds


# ── HL7 message builders ──────────────────────────────────────────────────────

def _now_hl7() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def _msh(msg_type: str, ctrl_id: str) -> str:
    now = _now_hl7()
    return (
        f"MSH|^~\\&|{SENDING_APP}|{SENDING_FAC}|"
        f"{RECEIVING_APP}|{RECEIVING_FAC}|{now}||{msg_type}|{ctrl_id}|P|2.5\r"
    )


def _pid(patient) -> str:
    """Build PID segment from a Patient model instance."""
    dob = patient.date_of_birth.strftime("%Y%m%d") if patient.date_of_birth else ""
    sex_map = {"Male": "M", "Female": "F", "Other": "O", "Unknown": "U"}
    sex = sex_map.get(patient.sex, "U")
    name_parts = patient.name.strip().split(" ", 1)
    last = name_parts[0] if name_parts else ""
    first = name_parts[1] if len(name_parts) > 1 else ""
    return (
        f"PID|1||{patient.patient_id}^^^HMIS^MR||"
        f"{last}^{first}^^^||{dob}|{sex}|||"
        f"{patient.place_of_residence or ''}||{patient.contact or ''}|\r"
    )


def _pv1(event_code: str, bed: str = "", ward: str = "") -> str:
    """Build PV1 segment."""
    return f"PV1|1|I|{ward}^{bed}^^^||||||||||||||{event_code}|\r"


def build_adt_a01(patient, ward: str = "", bed: str = "") -> str:
    """ADT^A01 — Admit/Visit Notification."""
    ctrl_id = f"ADT{_now_hl7()}"
    return (
        _msh("ADT^A01^ADT_A01", ctrl_id)
        + _pid(patient)
        + _pv1("A", bed, ward)
    )


def build_adt_a03(patient, ward: str = "", bed: str = "") -> str:
    """ADT^A03 — Discharge/End Visit."""
    ctrl_id = f"ADT{_now_hl7()}"
    return (
        _msh("ADT^A03^ADT_A03", ctrl_id)
        + _pid(patient)
        + _pv1("O", bed, ward)
    )


def build_adt_a08(patient) -> str:
    """ADT^A08 — Update Patient Information."""
    ctrl_id = f"ADT{_now_hl7()}"
    return (
        _msh("ADT^A08^ADT_A08", ctrl_id)
        + _pid(patient)
        + _pv1("U")
    )


def build_adt_a28(patient) -> str:
    """ADT^A28 — Add Person Information (new registration)."""
    ctrl_id = f"ADT{_now_hl7()}"
    return (
        _msh("ADT^A28^ADT_A01", ctrl_id)
        + _pid(patient)
        + _pv1("O")
    )


# ── Fire-and-forget MLLP sender ───────────────────────────────────────────────

def _send_mllp_blocking(raw_msg: str) -> None:
    """Synchronous MLLP send — always called from a daemon thread."""
    if not MLLP_DOWNSTREAM_HOST:
        return  # Graceful skip — no downstream configured

    payload = MLLP_SB + raw_msg.encode("utf-8") + MLLP_EB + MLLP_CR
    try:
        with socket.create_connection(
            (MLLP_DOWNSTREAM_HOST, MLLP_DOWNSTREAM_PORT), timeout=_CONNECT_TIMEOUT
        ) as sock:
            sock.sendall(payload)
            # Read ACK (up to 4 KB)
            ack = sock.recv(4096)
            logger.debug("ADT ACK received (%d bytes)", len(ack))
    except OSError as exc:
        logger.warning(
            "ADT send failed to %s:%s — %s",
            MLLP_DOWNSTREAM_HOST, MLLP_DOWNSTREAM_PORT, exc,
        )


def send_adt_async(raw_msg: str) -> None:
    """
    Fire-and-forget ADT send.
    Spawns a daemon thread so the caller (SQLAlchemy listener / Flask request) is
    never blocked by network I/O.
    """
    t = threading.Thread(target=_send_mllp_blocking, args=(raw_msg,), daemon=True)
    t.start()


# ── Public API used by SQLAlchemy event listeners ────────────────────────────

def on_patient_registered(patient) -> None:
    """Call after a new Patient row is committed."""
    msg = build_adt_a28(patient)
    send_adt_async(msg)
    logger.info("ADT^A28 queued for patient %s", patient.patient_id)


def on_patient_admitted(patient, ward: str = "", bed: str = "") -> None:
    """Call after an AdmittedPatient row is committed."""
    msg = build_adt_a01(patient, ward=ward, bed=bed)
    send_adt_async(msg)
    logger.info("ADT^A01 queued for patient %s ward=%s", patient.patient_id, ward)


def on_patient_discharged(patient, ward: str = "", bed: str = "") -> None:
    """Call after a patient discharge event."""
    msg = build_adt_a03(patient, ward=ward, bed=bed)
    send_adt_async(msg)
    logger.info("ADT^A03 queued for patient %s", patient.patient_id)


def on_patient_updated(patient) -> None:
    """Call after a Patient record update is committed."""
    msg = build_adt_a08(patient)
    send_adt_async(msg)
    logger.info("ADT^A08 queued for patient %s", patient.patient_id)
