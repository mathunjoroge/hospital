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
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": getattr(lab, "loinc_code", ""),
                        "display": lab.test_name if hasattr(lab, "test_name") else "Laboratory Test"
                    }
                ] if getattr(lab, "loinc_code", None) else [],
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


def imaging_result_to_fhir_study(img: ImagingResult) -> dict:
    """Map HIMS ImagingResult to HL7 FHIR R4 ImagingStudy Resource."""
    metadata = img.processing_metadata or {}
    study_uid = metadata.get("study_instance_uid", img.result_id)
    series_uid = metadata.get("series_instance_uid", "1.2.840.10008.1.1")
    modality = metadata.get("modality", "DX")

    resource = {
        "resourceType": "ImagingStudy",
        "id": f"imgstudy-{img.id}",
        "identifier": [
            {
                "system": "urn:dicom:uid",
                "value": f"urn:oid:{study_uid}",
            }
        ],
        "status": "available",
        "modality": [
            {
                "system": "http://dicom.nema.org/resources/ontology/DCM",
                "code": modality,
                "display": modality,
            }
        ],
        "subject": {"reference": f"Patient/{img.patient_id}"},
        "started": img.test_date.isoformat()
        if getattr(img, "test_date", None)
        else datetime.now(timezone.utc).isoformat(),
        "endpoint": [
            {
                "reference": f"Endpoint/wado-rs-{study_uid}",
                "display": f"PACS WADO-RS Endpoint for Study {study_uid}",
            }
        ],
        "series": [
            {
                "uid": series_uid,
                "modality": {
                    "system": "http://dicom.nema.org/resources/ontology/DCM",
                    "code": modality,
                },
                "numberOfInstances": img.files_processed or 1,
                "instance": [
                    {
                        "uid": img.result_id,
                        "sopClass": {
                            "system": "urn:ietf:rfc:3986",
                            "code": "urn:oid:1.2.840.10008.5.1.4.1.1.1",
                        },
                    }
                ],
            }
        ],
    }
    return resource


@fhir_bp.route("/ImagingStudy/<string:result_id>", methods=["GET"])
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
def get_fhir_imaging_study(result_id):
    """Retrieve FHIR R4 ImagingStudy resource by ImagingResult result_id."""
    img = ImagingResult.query.filter_by(result_id=result_id).first_or_404()
    return jsonify(imaging_result_to_fhir_study(img))


@fhir_bp.route("/ImagingStudy", methods=["GET"])
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
def search_fhir_imaging_studies():
    """Search FHIR R4 ImagingStudy resources for a patient."""
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

    imaging_results = ImagingResult.query.filter_by(patient_id=patient_id).all()
    entries = []

    for img in imaging_results:
        res_data = imaging_result_to_fhir_study(img)
        entries.append(
            {
                "fullUrl": f"{request.host_url}api/fhir/R4/ImagingStudy/{img.result_id}",
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


@fhir_bp.route("/metadata", methods=["GET"])
@fhir_bp.route("/R4/metadata", methods=["GET"])
def get_fhir_metadata():
    """Return FHIR R4 CapabilityStatement."""
    capability = {
        "resourceType": "CapabilityStatement",
        "id": "hims-fhir-r4-capability",
        "status": "active",
        "date": datetime.now(timezone.utc).isoformat(),
        "publisher": "HIMS Enterprise Interoperability",
        "kind": "instance",
        "software": {"name": "HIMS FHIR R4 Engine", "version": "1.0.0"},
        "implementation": {"description": "HIMS HL7 FHIR R4 Enterprise REST API"},
        "fhirVersion": "4.0.1",
        "format": ["json"],
        "rest": [
            {
                "mode": "server",
                "security": {
                    "cors": True,
                    "service": [
                        {
                            "coding": [
                                {
                                    "system": "http://terminology.hl7.org/CodeSystem/restful-security-service",
                                    "code": "SMART-on-FHIR",
                                }
                            ]
                        }
                    ],
                },
                "resource": [
                    {
                        "type": "Patient",
                        "interaction": [{"code": "read"}, {"code": "search-type"}],
                        "searchParam": [
                            {"name": "_id", "type": "token"},
                            {"name": "name", "type": "string"},
                            {"name": "identifier", "type": "token"},
                            {"name": "gender", "type": "token"},
                        ],
                    },
                    {
                        "type": "Observation",
                        "interaction": [{"code": "search-type"}],
                        "searchParam": [{"name": "patient", "type": "reference"}],
                    },
                    {
                        "type": "Condition",
                        "interaction": [{"code": "search-type"}],
                        "searchParam": [{"name": "patient", "type": "reference"}],
                    },
                    {
                        "type": "DiagnosticReport",
                        "interaction": [{"code": "search-type"}],
                        "searchParam": [{"name": "patient", "type": "reference"}],
                    },
                    {
                        "type": "MedicationRequest",
                        "interaction": [{"code": "search-type"}],
                        "searchParam": [{"name": "patient", "type": "reference"}],
                    },
                    {
                        "type": "Encounter",
                        "interaction": [{"code": "search-type"}],
                        "searchParam": [{"name": "patient", "type": "reference"}],
                    },
                    {
                        "type": "ImagingStudy",
                        "interaction": [{"code": "read"}, {"code": "search-type"}],
                        "searchParam": [{"name": "patient", "type": "reference"}],
                    },
                ],
            }
        ],
    }
    return jsonify(capability)


@fhir_bp.route("/.well-known/smart-configuration", methods=["GET"])
def get_smart_configuration():
    """Return SMART on FHIR OAuth2 Configuration."""
    base_url = request.host_url.rstrip("/")
    config = {
        "issuer": f"{base_url}/api/fhir/R4",
        "authorization_endpoint": f"{base_url}/auth/authorize",
        "token_endpoint": f"{base_url}/auth/token",
        "scopes_supported": [
            "openid",
            "profile",
            "launch",
            "patient/*.read",
            "user/*.read",
            "fhirUser",
        ],
        "response_types_supported": ["code", "token"],
        "capabilities": [
            "launch-standalone",
            "client-public",
            "client-confidential-symmetric",
            "context-passthrough-patient",
        ],
        "grant_types_supported": ["authorization_code", "client_credentials"],
    }
    return jsonify(config)


@fhir_bp.route("/Patient", methods=["GET"])
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
def search_fhir_patients():
    """Search FHIR R4 Patient resources by name, identifier, or gender."""
    name_query = request.args.get("name")
    identifier_query = request.args.get("identifier") or request.args.get("patient")
    gender_query = request.args.get("gender")
    patient_id_query = request.args.get("_id") or request.args.get("patient")

    query = Patient.query

    if patient_id_query:
        query = query.filter(Patient.patient_id == patient_id_query)
    if name_query:
        query = query.filter(Patient.name.ilike(f"%{name_query}%"))
    if identifier_query and not patient_id_query:
        query = query.filter(
            (Patient.patient_id == identifier_query)
            | (Patient.national_id == identifier_query)
        )
    if gender_query:
        query = query.filter(Patient.sex.ilike(f"{gender_query}%"))

    patients = query.all()
    entries = []
    for p in patients:
        entries.append(
            {
                "fullUrl": f"{request.host_url}api/fhir/R4/Patient/{p.patient_id}",
                "resource": patient_to_fhir(p),
            }
        )

    bundle = {
        "resourceType": "Bundle",
        "type": "searchset",
        "total": len(entries),
        "entry": entries,
    }
    return jsonify(bundle)


@fhir_bp.route("/", methods=["POST"])
@fhir_bp.route("", methods=["POST"])
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
def process_fhir_batch_bundle():
    """Process FHIR Batch or Transaction Bundle."""
    bundle = request.get_json() or {}
    bundle_type = bundle.get("type")
    if bundle_type not in ("batch", "transaction"):
        return jsonify(
            {
                "resourceType": "OperationOutcome",
                "issue": [
                    {
                        "severity": "error",
                        "code": "invalid",
                        "diagnostics": "Bundle type must be 'batch' or 'transaction'.",
                    }
                ],
            }
        ), 400

    response_entries = []
    for entry in bundle.get("entry", []):
        req = entry.get("request", {})
        url = req.get("url", "")
        method = req.get("method", "GET").upper()

        if method == "GET":
            res_entry = _dispatch_fhir_get(url)
            response_entries.append(res_entry)
        else:
            response_entries.append(
                {
                    "response": {"status": "405 Method Not Allowed"},
                    "resource": {
                        "resourceType": "OperationOutcome",
                        "issue": [
                            {
                                "severity": "warning",
                                "code": "not-supported",
                                "diagnostics": f"Method {method} not supported in batch.",
                            }
                        ],
                    },
                }
            )

    return jsonify(
        {
            "resourceType": "Bundle",
            "type": f"{bundle_type}-response",
            "total": len(response_entries),
            "entry": response_entries,
        }
    )


def _dispatch_fhir_get(url: str) -> dict:
    """Helper to dispatch internal GET requests within batch bundles."""
    clean_url = url.lstrip("/")
    if clean_url.startswith("Patient/"):
        pid = clean_url.split("Patient/")[1]
        patient = Patient.query.filter_by(patient_id=pid).first()
        if patient:
            return {
                "response": {"status": "200 OK"},
                "resource": patient_to_fhir(patient),
            }
        return {
            "response": {"status": "404 Not Found"},
            "resource": {
                "resourceType": "OperationOutcome",
                "issue": [{"severity": "error", "code": "not-found"}],
            },
        }
    elif clean_url.startswith("Patient"):
        parts = clean_url.split("?")
        params = (
            dict(p.split("=") for p in parts[1].split("&") if "=" in p)
            if len(parts) > 1
            else {}
        )
        pid = params.get("patient") or params.get("_id")
        name = params.get("name")
        query = Patient.query
        if pid:
            query = query.filter(Patient.patient_id == pid)
        if name:
            query = query.filter(Patient.name.ilike(f"%{name}%"))
        patients = query.all()
        return {
            "response": {"status": "200 OK"},
            "resource": {
                "resourceType": "Bundle",
                "type": "searchset",
                "total": len(patients),
                "entry": [{"resource": patient_to_fhir(p)} for p in patients],
            },
        }
    elif clean_url.startswith("Observation"):
        parts = clean_url.split("?")
        params = (
            dict(p.split("=") for p in parts[1].split("&") if "=" in p)
            if len(parts) > 1
            else {}
        )
        pid = params.get("patient")
        vitals_records = Vitals.query.filter_by(patient_id=pid).all() if pid else []
        obs_list = []
        for v in vitals_records:
            obs_list.extend(vitals_to_fhir_observations(v))
        return {
            "response": {"status": "200 OK"},
            "resource": {
                "resourceType": "Bundle",
                "type": "searchset",
                "total": len(obs_list),
                "entry": [{"resource": o} for o in obs_list],
            },
        }
    elif clean_url.startswith("Condition"):
        parts = clean_url.split("?")
        params = (
            dict(p.split("=") for p in parts[1].split("&") if "=" in p)
            if len(parts) > 1
            else {}
        )
        pid = params.get("patient")
        notes = SOAPNote.query.filter_by(patient_id=pid).all() if pid else []
        conds = [
            {
                "resourceType": "Condition",
                "id": f"cond-soap-{n.id}",
                "clinicalStatus": {"coding": [{"code": "active"}]},
                "code": {"text": n.assessment},
                "subject": {"reference": f"Patient/{pid}"},
            }
            for n in notes
            if n.assessment
        ]
        return {
            "response": {"status": "200 OK"},
            "resource": {
                "resourceType": "Bundle",
                "type": "searchset",
                "total": len(conds),
                "entry": [{"resource": c} for c in conds],
            },
        }
    elif clean_url.startswith("DiagnosticReport"):
        parts = clean_url.split("?")
        params = (
            dict(p.split("=") for p in parts[1].split("&") if "=" in p)
            if len(parts) > 1
            else {}
        )
        pid = params.get("patient")
        lab_results = LabResult.query.filter_by(patient_id=pid).all() if pid else []
        img_results = ImagingResult.query.filter_by(patient_id=pid).all() if pid else []
        reports = []
        for lab in lab_results:
            reports.append(
                {
                    "resourceType": "DiagnosticReport",
                    "id": f"report-lab-{lab.id}",
                    "subject": {"reference": f"Patient/{pid}"},
                    "conclusion": str(getattr(lab, "result_value", "")),
                }
            )
        for i in img_results:
            reports.append(
                {
                    "resourceType": "DiagnosticReport",
                    "id": f"report-img-{i.id}",
                    "subject": {"reference": f"Patient/{pid}"},
                    "conclusion": getattr(i, "ai_impression", "")
                    or getattr(i, "result_notes", ""),
                }
            )
        return {
            "response": {"status": "200 OK"},
            "resource": {
                "resourceType": "Bundle",
                "type": "searchset",
                "total": len(reports),
                "entry": [{"resource": r} for r in reports],
            },
        }
    elif clean_url.startswith("MedicationRequest"):
        parts = clean_url.split("?")
        params = (
            dict(p.split("=") for p in parts[1].split("&") if "=" in p)
            if len(parts) > 1
            else {}
        )
        pid = params.get("patient")
        prescriptions = (
            PrescribedMedicine.query.filter_by(patient_id=pid).all() if pid else []
        )
        meds = [
            {
                "resourceType": "MedicationRequest",
                "id": f"medreq-{rx.id}",
                "status": "active",
                "subject": {"reference": f"Patient/{pid}"},
                "medicationCodeableConcept": {
                    "text": rx.drug_name
                    if hasattr(rx, "drug_name")
                    else "Prescribed Medication"
                },
            }
            for rx in prescriptions
        ]
        return {
            "response": {"status": "200 OK"},
            "resource": {
                "resourceType": "Bundle",
                "type": "searchset",
                "total": len(meds),
                "entry": [{"resource": m} for m in meds],
            },
        }
    elif clean_url.startswith("Encounter"):
        parts = clean_url.split("?")
        params = (
            dict(p.split("=") for p in parts[1].split("&") if "=" in p)
            if len(parts) > 1
            else {}
        )
        pid = params.get("patient")
        encs = Encounter.query.filter_by(patient_id=pid).all() if pid else []
        enc_resources = [encounter_to_fhir(e) for e in encs]
        return {
            "response": {"status": "200 OK"},
            "resource": {
                "resourceType": "Bundle",
                "type": "searchset",
                "total": len(enc_resources),
                "entry": [{"resource": e} for e in enc_resources],
            },
        }
    elif clean_url.startswith("ImagingStudy"):
        parts = clean_url.split("?")
        params = (
            dict(p.split("=") for p in parts[1].split("&") if "=" in p)
            if len(parts) > 1
            else {}
        )
        pid = params.get("patient")
        imgs = ImagingResult.query.filter_by(patient_id=pid).all() if pid else []
        img_resources = [imaging_result_to_fhir_study(i) for i in imgs]
        return {
            "response": {"status": "200 OK"},
            "resource": {
                "resourceType": "Bundle",
                "type": "searchset",
                "total": len(img_resources),
                "entry": [{"resource": e} for e in img_resources],
            },
        }

    return {
        "response": {"status": "400 Bad Request"},
        "resource": {
            "resourceType": "OperationOutcome",
            "issue": [
                {
                    "severity": "error",
                    "code": "not-found",
                    "diagnostics": f"Unknown URL {url}",
                }
            ],
        },
    }



