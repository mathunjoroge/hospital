"""
departments/api/hl7_receiver.py
────────────────────────────────
Phase 4 — P4-03: HL7v2 ORU^R01 Ingest Endpoint

Accepts either:
  • Raw HL7v2 text (Content-Type: x-application/hl7-v2+er7)
    — sent directly by the asyncio MLLP daemon or Mirth Connect channel
  • FHIR DiagnosticReport JSON (Content-Type: application/fhir+json)
    — used by Mirth's FHIR transformer channel

Security: simple API-key bearer token checked against HL7_INGEST_API_KEY env var.
Rate limit: enforced at nginx/reverse-proxy layer; not duplicated here.

GUARDRAILS:
  - Does NOT touch departments/api/fhir.py (Phase 6 file)
  - Only ADDS source_system / raw_hl7 columns to LabResult (additive, no renames)
  - Calls evaluate_panic_level without modifying its signature
"""

import logging
import os
import uuid
from datetime import datetime, timezone

from flask import Blueprint, jsonify, request

try:
    import hl7 as hl7lib
except ImportError:
    hl7lib = None  # graceful degradation during unit tests that mock the parser

from departments.laboratory.panic_alerts import evaluate_panic_level

try:
    from extensions import db
except ImportError:
    from extensions import db

from departments.models.laboratory import LabResult
from departments.models.records import Patient

logger = logging.getLogger(__name__)

hl7_bp = Blueprint("hl7", __name__, url_prefix="/api/hl7")

# ──────────────────────────────────────────────────────────────────────────────
# Auth helper
# ──────────────────────────────────────────────────────────────────────────────

def _require_api_key():
    """Return (True, None) if authorised, else (False, error_response)."""
    expected = os.environ.get("HL7_INGEST_API_KEY", "changeme")
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer ") and auth[7:] == expected:
        return True, None
    api_key = request.headers.get("X-HL7-API-Key", "")
    if api_key == expected:
        return True, None
    return False, (jsonify({"error": "Unauthorized"}), 401)


# ──────────────────────────────────────────────────────────────────────────────
# ORU^R01 raw HL7 parser helpers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_oru_r01(raw: str) -> dict:
    """
    Parse a raw HL7v2 ORU^R01 message and return a normalised dict:
      {patient_id, parameter_name, result_value, result_notes, source_system}
    Raises ValueError on parse failures.
    """
    if hl7lib is None:
        raise ValueError("hl7 library not installed")

    msg = hl7lib.parse(raw.strip())

    # MSH-3: Sending Application → source_system
    # python-hl7 field indexing: [0]=segment name, [1]=field sep, [2]=encoding chars, [3]=sending app
    try:
        source_system = str(msg["MSH"][0][3][0]) or "UNKNOWN_LIS"
    except Exception:
        source_system = "UNKNOWN_LIS"

    # PID-3: Patient Identifier List (first CX)
    try:
        patient_id = str(msg["PID"][0][3][0][0])
    except Exception:
        raise ValueError("Cannot extract patient ID from PID-3")

    # OBX-3: Observation Identifier (local code)
    # OBX-5: Observation Value
    # OBX-6: Units
    try:
        obx = msg["OBX"][0]
        parameter_name = str(obx[3][0][0])  # OBX-3.1 Identifier
        raw_value = str(obx[5][0])          # OBX-5
        result_value = float(raw_value)
        unit_str = str(obx[6][0]) if obx[6][0] else ""
    except (IndexError, ValueError, TypeError) as exc:
        raise ValueError(f"Cannot parse OBX segment: {exc}") from exc

    # NTE-3: Notes (optional)
    try:
        result_notes = str(msg["NTE"][0][3][0])
    except Exception:
        result_notes = ""

    return {
        "patient_id": patient_id,
        "parameter_name": parameter_name,
        "result_value": result_value,
        "unit": unit_str,
        "result_notes": result_notes,
        "source_system": source_system,
    }


def _parse_fhir_diagnostic_report(data: dict) -> dict:
    """
    Minimal FHIR DiagnosticReport → normalised dict.
    Expects the report to carry exactly one contained Observation with
    valueQuantity.  This mirrors what Mirth's HL7→FHIR transformer emits.
    """
    # subject.reference: e.g. "Patient/P00123"
    subject = data.get("subject", {}).get("reference", "")
    patient_id = subject.replace("Patient/", "").strip()
    if not patient_id:
        raise ValueError("Cannot extract patient ID from DiagnosticReport.subject")

    source_system = (
        data.get("performer", [{}])[0]
        .get("display", "FHIR_SOURCE")
    )

    # Walk contained resources for the first Observation
    parameter_name = "UnknownParameter"
    result_value = 0.0
    unit_str = ""
    result_notes = ""

    for contained in data.get("contained", []):
        if contained.get("resourceType") == "Observation":
            coding = contained.get("code", {}).get("coding", [{}])[0]
            parameter_name = coding.get("display") or coding.get("code", parameter_name)
            vq = contained.get("valueQuantity", {})
            result_value = float(vq.get("value", 0))
            unit_str = vq.get("unit", "")
            result_notes = contained.get("note", [{}])[0].get("text", "")
            break

    return {
        "patient_id": patient_id,
        "parameter_name": parameter_name,
        "result_value": result_value,
        "unit": unit_str,
        "result_notes": result_notes,
        "source_system": source_system,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Core ingest logic (shared by HTTP and daemon-over-HTTP paths)
# ──────────────────────────────────────────────────────────────────────────────

def ingest_lab_result(
    patient_id: str,
    parameter_name: str,
    result_value: float,
    unit: str = "",
    result_notes: str = "",
    source_system: str = "HL7_LIS",
    raw_hl7: str = "",
    lab_test_id: int = 1,
) -> dict:
    """
    Persist a lab result received from an external LIS via HL7/FHIR.
    Returns a dict with result_id, panic_status, panic_message.
    Raises ValueError if the patient does not exist.
    """
    patient = Patient.query.filter_by(patient_id=patient_id).first()
    if not patient:
        raise ValueError(f"Patient {patient_id!r} not found")

    panic_status, panic_message = evaluate_panic_level(parameter_name, result_value)

    res_uuid = f"HL7-{str(uuid.uuid4())[:8].upper()}"
    result_text = f"{parameter_name}: {result_value}"
    if unit:
        result_text += f" {unit}"

    lab_res = LabResult(
        patient_id=patient_id,
        lab_test_id=lab_test_id,
        result_id=res_uuid,
        result=result_text,
        result_notes=result_notes,
        status="PENDING_VERIFICATION",
        panic_status=panic_status,
        panic_message=panic_message,
        # Phase 4 additive columns (added in migration below)
        source_system=source_system,
        raw_hl7=raw_hl7 or None,
    )
    db.session.add(lab_res)
    db.session.commit()

    logger.info(
        "HL7 ingest OK result_id=%s patient=%s param=%s panic=%s source=%s",
        res_uuid, patient_id, parameter_name, panic_status, source_system,
    )

    return {
        "result_id": res_uuid,
        "patient_id": patient_id,
        "parameter_name": parameter_name,
        "result_value": result_value,
        "panic_status": panic_status,
        "panic_message": panic_message,
        "source_system": source_system,
    }


# ──────────────────────────────────────────────────────────────────────────────
# POST /api/hl7/oru  — main ingest route
# ──────────────────────────────────────────────────────────────────────────────

@hl7_bp.route("/oru", methods=["POST"])
def receive_oru():
    """
    Accept ORU^R01 from Mirth Connect or directly from the MLLP daemon.

    Content-Type routing:
      x-application/hl7-v2+er7  → raw HL7 text
      application/fhir+json      → FHIR DiagnosticReport JSON
      application/json           → plain JSON {patient_id, parameter_name, ...}
    """
    ok, err = _require_api_key()
    if not ok:
        return err

    content_type = request.content_type or ""
    raw_hl7_text = ""

    try:
        if "hl7-v2" in content_type or "er7" in content_type:
            raw_hl7_text = request.data.decode("utf-8", errors="replace")
            parsed = _parse_oru_r01(raw_hl7_text)

        elif "fhir+json" in content_type:
            parsed = _parse_fhir_diagnostic_report(request.get_json(force=True) or {})

        else:
            # Plain JSON fallback (used by load tests and unit tests)
            data = request.get_json(force=True) or {}
            parsed = {
                "patient_id": data.get("patient_id", ""),
                "parameter_name": data.get("parameter_name", "Unknown"),
                "result_value": float(data.get("result_value", 0)),
                "unit": data.get("unit", ""),
                "result_notes": data.get("result_notes", ""),
                "source_system": data.get("source_system", "JSON_LIS"),
            }

        if not parsed.get("patient_id"):
            return jsonify({"error": "patient_id is required"}), 400

        # Optional override: allow caller to specify lab_test_id
        lab_test_id = int(request.args.get("lab_test_id", 1))

        result = ingest_lab_result(
            patient_id=parsed["patient_id"],
            parameter_name=parsed["parameter_name"],
            result_value=parsed["result_value"],
            unit=parsed.get("unit", ""),
            result_notes=parsed.get("result_notes", ""),
            source_system=parsed.get("source_system", "HL7_LIS"),
            raw_hl7=raw_hl7_text,
            lab_test_id=lab_test_id,
        )

        return jsonify({"success": True, **result}), 201

    except ValueError as exc:
        logger.warning("HL7 ingest rejected: %s", exc)
        return jsonify({"error": str(exc)}), 422

    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("HL7 ingest unexpected error: %s", exc)
        return jsonify({"error": "Internal server error"}), 500


# ──────────────────────────────────────────────────────────────────────────────
# GET /api/hl7/status  — health probe used by Mirth Connect channel monitor
# ──────────────────────────────────────────────────────────────────────────────

@hl7_bp.route("/status", methods=["GET"])
def hl7_status():
    """Lightweight health endpoint — no auth required (used by Mirth monitor)."""
    return jsonify({"status": "ok", "service": "hl7_receiver", "version": "4.0.0"}), 200
