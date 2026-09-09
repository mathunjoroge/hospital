"""
departments/api/fhir.py
────────────────────────
HL7 FHIR R4 Interoperability Module for HIMS.
Transforms HIMS database records into standardized HL7 FHIR R4 JSON resources and bundles.

Blueprint endpoints (registered under /api/fhir/R4):
  - GET /api/fhir/R4/Patient/<patient_id>
  - GET /api/fhir/R4/Observation?patient=<patient_id>
  - GET /api/fhir/R4/Condition?patient=<patient_id>
  - GET /api/fhir/R4/DiagnosticReport?patient=<patient_id>
  - GET /api/fhir/R4/MedicationRequest?patient=<patient_id>
"""

import logging
from datetime import datetime, timezone

from flask import Blueprint, jsonify, request

from departments.api.auth import jwt_or_session_required
from departments.models.encounter import Encounter
from departments.models.imaging import ImagingResult
from departments.models.laboratory import LabResult
from departments.models.medicine import PrescribedMedicine, SOAPNote
from departments.models.nursing import Vitals
from departments.models.records import Patient
from departments.rbac import roles_required

logger = logging.getLogger(__name__)

fhir_bp = Blueprint("fhir", __name__)

LOINC_CODES = {
    "temperature": {"code": "8310-5", "display": "Body temperature", "unit": "Cel"},
    "heart_rate": {"code": "8867-4", "display": "Heart rate", "unit": "/min"},
    "bp_systolic": {
        "code": "8480-6",
        "display": "Systolic blood pressure",
        "unit": "mmHg",
    },
    "bp_diastolic": {
        "code": "8462-4",
        "display": "Diastolic blood pressure",
        "unit": "mmHg",
    },
    "spo2": {
        "code": "59408-5",
        "display": "Oxygen saturation in Arterial blood by Pulse oximetry",
        "unit": "%",
    },
    "weight": {"code": "29463-7", "display": "Body weight", "unit": "kg"},
    "height": {"code": "8302-2", "display": "Body height", "unit": "cm"},
}


def patient_to_fhir(patient: Patient) -> dict:
    """Map HIMS Patient model to HL7 FHIR R4 Patient Resource."""
    gender_map = {"M": "male", "F": "female", "Male": "male", "Female": "female"}
    gender = gender_map.get(patient.sex, "unknown") if patient.sex else "unknown"

    identifiers = [
        {
            "system": "http://hims.hospital.go.ke/patient-id",
            "value": patient.patient_id,
            "use": "official",
        }
    ]
    if patient.national_id:
        identifiers.append(
            {
                "system": "http://kenya.go.ke/national-id",
                "value": patient.national_id,
                "use": "official",
            }
        )

    telecom = []
    if patient.contact:
        telecom.append({"system": "phone", "value": patient.contact, "use": "mobile"})

    address = []
    if patient.place_of_residence:
        address.append({"text": patient.place_of_residence, "country": "KE"})

    contact_person = []
    if patient.next_of_kin:
        contact_person.append(
            {
                "relationship": [
                    {"text": patient.relationship_with_next_of_kin or "Next of Kin"}
                ],
                "name": {"text": patient.next_of_kin},
                "telecom": [{"system": "phone", "value": patient.next_of_kin_contact}]
                if patient.next_of_kin_contact
                else [],
            }
        )

    resource = {
        "resourceType": "Patient",
        "id": patient.patient_id,
        "identifier": identifiers,
        "active": getattr(patient, "is_active", True),
        "name": [{"use": "official", "text": patient.name}],
        "gender": gender,
        "birthDate": str(patient.date_of_birth) if patient.date_of_birth else None,
        "telecom": telecom,
        "address": address,
        "contact": contact_person,
        "meta": {
            "versionId": "1",
            "lastUpdated": (
                patient.updated_at or patient.date_registered or datetime.now(timezone.utc)
            ).isoformat()
            if hasattr(patient, "updated_at")
            else datetime.now(timezone.utc).isoformat(),
        },
    }
    return resource


def vitals_to_fhir_observations(vitals: Vitals) -> list[dict]:
    """Map HIMS Vitals record into LOINC-coded FHIR Observation resources."""
    observations = []

    metrics = [
        ("temperature", getattr(vitals, "temperature", None)),
        (
            "heart_rate",
            getattr(vitals, "pulse", None) or getattr(vitals, "pulse_rate", None),
        ),
        (
            "bp_systolic",
            getattr(vitals, "blood_pressure_systolic", None)
            or getattr(vitals, "bp_systolic", None),
        ),
        (
            "bp_diastolic",
            getattr(vitals, "blood_pressure_diastolic", None)
            or getattr(vitals, "bp_diastolic", None),
        ),
        (
            "spo2",
            getattr(vitals, "oxygen_saturation", None) or getattr(vitals, "spo2", None),
        ),
        ("weight", getattr(vitals, "weight", None)),
        ("height", getattr(vitals, "height", None)),
    ]

    for key, val in metrics:
        if val is not None:
            loinc = LOINC_CODES[key]
            obs = {
                "resourceType": "Observation",
                "id": f"obs-vitals-{vitals.id}-{key}",
                "status": "final",
                "category": [
                    {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                "code": "vital-signs",
                                "display": "Vital Signs",
                            }
                        ]
                    }
                ],
                "code": {
                    "coding": [
                        {
                            "system": "http://loinc.org",
                            "code": loinc["code"],
                            "display": loinc["display"],
                        }
                    ],
                    "text": loinc["display"],
                },
                "subject": {"reference": f"Patient/{vitals.patient_id}"},
                "effectiveDateTime": vitals.timestamp.isoformat()
                if getattr(vitals, "timestamp", None)
                else datetime.now(timezone.utc).isoformat(),
                "valueQuantity": {
                    "value": float(val),
                    "unit": loinc["unit"],
                    "system": "http://unitsofmeasure.org",
                },
            }
            observations.append(obs)

    return observations


@fhir_bp.route("/Patient/<string:patient_id>", methods=["GET"])
@jwt_or_session_required
@roles_required(
    "admin",
    "records",
    "medicine",
    "nursing",
    "pharmacy",
    "laboratory",
    "imaging",
    "api",
)
def get_fhir_patient(patient_id):
    """Retrieve FHIR R4 Patient resource by patient_id."""
    patient = Patient.query.filter_by(patient_id=patient_id).first_or_404()
    return jsonify(patient_to_fhir(patient))


@fhir_bp.route("/Observation", methods=["GET"])
@jwt_or_session_required
@roles_required(
    "admin",
    "records",
    "medicine",
    "nursing",
    "pharmacy",
    "laboratory",
    "imaging",
    "api",
)
def search_fhir_observations():
    """Search FHIR R4 Observations for a patient (Vitals & Labs)."""
    patient_id = request.args.get("patient")
    if not patient_id:
        return jsonify(
            {
                "resourceType": "OperationOutcome",
                "issue": [
                    {
                        "severity": "error",
                        "code": "required",
                        "diagnostics": "Query parameter 'patient' is required.",
                    }
                ],
            }
        ), 400

    vitals_records = (
        Vitals.query.filter_by(patient_id=patient_id)
        .order_by(Vitals.timestamp.desc())
        .all()
    )

    entries = []
    for v in vitals_records:
        for obs in vitals_to_fhir_observations(v):
            entries.append(
                {
                    "fullUrl": f"{request.host_url}api/fhir/R4/Observation/{obs['id']}",
                    "resource": obs,
                }
            )

    bundle = {
        "resourceType": "Bundle",
        "type": "searchset",
        "total": len(entries),
        "entry": entries,
    }
    return jsonify(bundle)


@fhir_bp.route("/Condition", methods=["GET"])
@jwt_or_session_required
@roles_required("admin", "records", "medicine", "nursing", "api")
def search_fhir_conditions():
    """Search FHIR R4 Conditions (ICD-10 Diagnoses from SOAP notes)."""
    patient_id = request.args.get("patient")
    if not patient_id:
        return jsonify(
            {
                "resourceType": "OperationOutcome",
                "issue": [
                    {
                        "severity": "error",
                        "code": "required",
                        "diagnostics": "Query parameter 'patient' is required.",
                    }
                ],
            }
        ), 400

    notes = SOAPNote.query.filter_by(patient_id=patient_id).all()
    entries = []

    for note in notes:
        if note.assessment:
            condition = {
                "resourceType": "Condition",
                "id": f"cond-soap-{note.id}",
                "clinicalStatus": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                            "code": "active",
                            "display": "Active",
                        }
                    ]
                },
                "code": {
                    "coding": [
                        {
                            "system": "http://hl7.org/fhir/sid/icd-10",
                            "code": "R69",
                            "display": note.assessment,
                        }
                    ],
                    "text": note.assessment,
                },
                "subject": {"reference": f"Patient/{patient_id}"},
                "recordedDate": note.created_at.isoformat()
                if getattr(note, "created_at", None)
                else datetime.now(timezone.utc).isoformat(),
            }
            entries.append(
                {
                    "fullUrl": f"{request.host_url}api/fhir/R4/Condition/{condition['id']}",
                    "resource": condition,
                }
            )

    bundle = {
        "resourceType": "Bundle",
        "type": "searchset",
        "total": len(entries),
        "entry": entries,
    }
    return jsonify(bundle)


@fhir_bp.route("/DiagnosticReport", methods=["GET"])
@jwt_or_session_required
@roles_required(
    "admin", "records", "medicine", "nursing", "laboratory", "imaging", "api"
)
def search_fhir_diagnostic_reports():
    """Search FHIR R4 DiagnosticReport resources (Lab & Imaging)."""
    patient_id = request.args.get("patient")
    if not patient_id:
        return jsonify(
            {
                "resourceType": "OperationOutcome",
                "issue": [
                    {
                        "severity": "error",
                        "code": "required",
                        "diagnostics": "Query parameter 'patient' is required.",
                    }
                ],
            }
        ), 400

    lab_results = LabResult.query.filter_by(patient_id=patient_id).all()
    imaging_results = ImagingResult.query.filter_by(patient_id=patient_id).all()
    entries = []

    for lab in lab_results:
        report = {
            "resourceType": "DiagnosticReport",
            "id": f"report-lab-{lab.id}",
            "status": "final",
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                            "code": "LAB",
                            "display": "Laboratory",
                        }
                    ]
                }
            ],
            "code": {
                "text": lab.test_name
                if hasattr(lab, "test_name")
                else "Laboratory Test"
            },
            "subject": {"reference": f"Patient/{patient_id}"},
            "issued": lab.timestamp.isoformat()
            if getattr(lab, "timestamp", None)
            else datetime.now(timezone.utc).isoformat(),
            "conclusion": str(getattr(lab, "result_value", "")),
        }
        entries.append(
            {
                "fullUrl": f"{request.host_url}api/fhir/R4/DiagnosticReport/{report['id']}",
                "resource": report,
            }
        )

    for img in imaging_results:
        report = {
            "resourceType": "DiagnosticReport",
            "id": f"report-img-{img.id}",
            "status": "final",
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                            "code": "RAD",
                            "display": "Radiology",
                        }
                    ]
                }
            ],
            "code": {
                "text": img.imaging_type
                if hasattr(img, "imaging_type")
                else "Radiology Study"
            },
            "subject": {"reference": f"Patient/{patient_id}"},
            "issued": img.created_at.isoformat()
            if getattr(img, "created_at", None)
            else datetime.now(timezone.utc).isoformat(),
            "conclusion": getattr(img, "ai_impression", "")
            or getattr(img, "result_notes", "")
            or "Imaging Study Completed",
        }
        entries.append(
            {
                "fullUrl": f"{request.host_url}api/fhir/R4/DiagnosticReport/{report['id']}",
                "resource": report,
            }
        )

    bundle = {
        "resourceType": "Bundle",
        "type": "searchset",
        "total": len(entries),
        "entry": entries,
    }
    return jsonify(bundle)


@fhir_bp.route("/MedicationRequest", methods=["GET"])
@jwt_or_session_required
@roles_required("admin", "records", "medicine", "nursing", "pharmacy", "api")
def search_fhir_medication_requests():
    """Search FHIR R4 MedicationRequest resources (Prescriptions)."""
    patient_id = request.args.get("patient")
    if not patient_id:
        return jsonify(
            {
                "resourceType": "OperationOutcome",
                "issue": [
                    {
                        "severity": "error",
                        "code": "required",
                        "diagnostics": "Query parameter 'patient' is required.",
                    }
                ],
            }
        ), 400

    prescriptions = PrescribedMedicine.query.filter_by(patient_id=patient_id).all()
    entries = []

    for rx in prescriptions:
        med_req = {
            "resourceType": "MedicationRequest",
            "id": f"medreq-{rx.id}",
            "status": "active",
            "intent": "order",
            "medicationCodeableConcept": {
                "text": rx.drug_name
                if hasattr(rx, "drug_name")
                else "Prescribed Medication"
            },
            "subject": {"reference": f"Patient/{patient_id}"},
            "authoredOn": rx.date_prescribed.isoformat()
            if getattr(rx, "date_prescribed", None)
            else datetime.now(timezone.utc).isoformat(),
            "dosageInstruction": [
                {
                    "text": f"Dosage: {getattr(rx, 'dosage', 'As directed')}, Duration: {getattr(rx, 'duration', 'N/A')}"
                }
            ],
        }
        entries.append(
            {
                "fullUrl": f"{request.host_url}api/fhir/R4/MedicationRequest/{med_req['id']}",
                "resource": med_req,
            }
        )

    bundle = {
        "resourceType": "Bundle",
        "type": "searchset",
        "total": len(entries),
        "entry": entries,
    }
    return jsonify(bundle)


def encounter_to_fhir(encounter: Encounter) -> dict:
    """Map HIMS Encounter model to HL7 FHIR R4 Encounter Resource."""
    status_map = {
        "ACTIVE": "in-progress",
        "DISCHARGED": "finished",
        "CANCELLED": "cancelled",
        "ABORTED": "entered-in-error",
    }
    fhir_status = status_map.get(encounter.status, "unknown")

    type_code = encounter.encounter_type or "OPD"
    class_map = {
        "OPD": {"code": "AMB", "display": "ambulatory"},
        "IPD": {"code": "IMP", "display": "inpatient encounter"},
        "EMERGENCY": {"code": "EMER", "display": "emergency"},
        "TELEHEALTH": {"code": "VR", "display": "virtual"},
    }
    class_info = class_map.get(type_code, {"code": "AMB", "display": "ambulatory"})

    resource = {
        "resourceType": "Encounter",
        "id": encounter.encounter_id,
        "status": fhir_status,
        "class": {
            "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
            "code": class_info["code"],
            "display": class_info["display"],
        },
        "subject": {"reference": f"Patient/{encounter.patient_id}"},
        "period": {
            "start": encounter.started_at.isoformat()
            if getattr(encounter, "started_at", None)
            else None,
            "end": encounter.ended_at.isoformat()
            if getattr(encounter, "ended_at", None)
            else None,
        },
        "reasonCode": [
            {
                "text": encounter.chief_complaint
            }
        ]
        if encounter.chief_complaint
        else [],
    }
    return resource


@fhir_bp.route("/Encounter", methods=["GET"])
@jwt_or_session_required
@roles_required(
    "admin",
    "records",
    "medicine",
    "nursing",
    "pharmacy",
    "laboratory",
    "imaging",
    "api",
)
def search_fhir_encounters():
    """Search FHIR R4 Encounter resources for a patient."""
    patient_id = request.args.get("patient")
    if not patient_id:
        return jsonify(
            {
                "resourceType": "OperationOutcome",
                "issue": [
                    {
                        "severity": "error",
                        "code": "required",
                        "diagnostics": "Query parameter 'patient' is required.",
                    }
                ],
            }
        ), 400

    encounters = Encounter.query.filter_by(patient_id=patient_id).all()
    entries = []

    for enc in encounters:
        res_data = encounter_to_fhir(enc)
        entries.append(
            {
                "fullUrl": f"{request.host_url}api/fhir/R4/Encounter/{enc.encounter_id}",
                "resource": res_data,
            }
        )

    bundle = {
        "resourceType": "Bundle",
        "type": "searchset",
        "total": len(entries),
        "entry": entries,
    }
    return jsonify(bundle)

