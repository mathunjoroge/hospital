"""
departments/api/fhir_hie.py
───────────────────────────
HL7 FHIR R4 $everything operation for Multi-Facility HIE Interoperability.
Assembles complete medical record bundle for a patient across all clinical modules.
NOTE: Serializer helpers are inlined here to avoid circular imports with fhir.py.
"""

from datetime import datetime, timezone
from typing import Any

from departments.models.encounter import Encounter
from departments.models.imaging import ImagingResult
from departments.models.laboratory import LabResult
from departments.models.medicine import PrescribedMedicine, SOAPNote
from departments.models.nursing import Vitals
from departments.models.records import Patient

# LOINC codes for vital observations (duplicated from fhir.py to avoid circular imports)
_LOINC_CODES = {
    "temperature": {"code": "8310-5", "display": "Body temperature", "unit": "Cel"},
    "pulse": {"code": "8867-4", "display": "Heart rate", "unit": "/min"},
    "blood_pressure_systolic": {
        "code": "8480-6",
        "display": "Systolic blood pressure",
        "unit": "mmHg",
    },
    "blood_pressure_diastolic": {
        "code": "8462-4",
        "display": "Diastolic blood pressure",
        "unit": "mmHg",
    },
    "oxygen_saturation": {
        "code": "59408-5",
        "display": "Oxygen saturation",
        "unit": "%",
    },
    "weight": {"code": "29463-7", "display": "Body weight", "unit": "kg"},
    "height": {"code": "8302-2", "display": "Body height", "unit": "cm"},
}


def _vitals_to_observations(vitals: Vitals) -> list[dict[str, Any]]:
    """Convert a Vitals row to a list of FHIR R4 Observation resources."""
    observations = []
    for field, loinc in _LOINC_CODES.items():
        val = getattr(vitals, field, None)
        if val is None:
            continue
        obs = {
            "resourceType": "Observation",
            "id": f"obs-{field}-{vitals.id}",
            "status": "final",
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": loinc["code"],
                        "display": loinc["display"],
                    }
                ]
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


def _encounter_to_fhir(encounter: Encounter) -> dict[str, Any]:
    """Convert an Encounter row to a FHIR R4 Encounter resource."""
    return {
        "resourceType": "Encounter",
        "id": f"enc-{encounter.id}",
        "status": getattr(encounter, "status", "finished"),
        "class": {
            "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
            "code": "IMP",
            "display": getattr(encounter, "encounter_type", "Inpatient"),
        },
        "subject": {"reference": f"Patient/{encounter.patient_id}"},
    }


def _imaging_to_fhir_study(img: ImagingResult) -> dict[str, Any]:
    """Convert an ImagingResult row to a FHIR R4 ImagingStudy resource."""
    return {
        "resourceType": "ImagingStudy",
        "id": f"img-{img.id}",
        "status": "available",
        "subject": {"reference": f"Patient/{img.patient_id}"},
        "description": getattr(img, "description", ""),
    }


def export_patient_everything_bundle(patient_id: str) -> dict[str, Any]:
    """
    Execute FHIR R4 Patient/$everything operation.
    Collects Patient, Encounters, Conditions, Observations, DiagnosticReports,
    ImagingStudies, and MedicationRequests into a single FHIR Collection Bundle.
    """
    patient = Patient.query.filter_by(patient_id=patient_id).first()
    if not patient:
        return {
            "resourceType": "OperationOutcome",
            "status": 404,
            "issue": [
                {
                    "severity": "error",
                    "code": "not-found",
                    "diagnostics": f"Patient with ID '{patient_id}' was not found.",
                }
            ],
        }

    entries: list[dict[str, Any]] = []

    # 1. Patient resource
    gender_map = {"M": "male", "F": "female", "Male": "male", "Female": "female"}
    pat_res = {
        "resourceType": "Patient",
        "id": patient_id,
        "identifier": [
            {"system": "http://hims.hospital.go.ke/patient-id", "value": patient_id}
        ],
        "name": [{"text": patient.name}],
        "gender": gender_map.get(patient.sex, "unknown") if patient.sex else "unknown",
    }
    entries.append({"resource": pat_res})

    # 2. Encounters
    encounters = Encounter.query.filter_by(patient_id=patient_id).all()
    for enc in encounters:
        entries.append({"resource": _encounter_to_fhir(enc)})

    # 3. Conditions (SOAP Assessment)
    soap_notes = SOAPNote.query.filter_by(patient_id=patient_id).all()
    for note in soap_notes:
        if note.assessment:
            cond_res = {
                "resourceType": "Condition",
                "id": f"cond-soap-{note.id}",
                "clinicalStatus": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                            "code": "active",
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
            }
            entries.append({"resource": cond_res})

    # 4. Observations (Vitals)
    vitals_records = Vitals.query.filter_by(patient_id=patient_id).all()
    for v in vitals_records:
        for obs in _vitals_to_observations(v):
            entries.append({"resource": obs})

    # 5. DiagnosticReports (Lab Results)
    labs = LabResult.query.filter_by(patient_id=patient_id).all()
    for lab in labs:
        rep_res = {
            "resourceType": "DiagnosticReport",
            "id": f"lab-{lab.id}",
            "status": "final"
            if getattr(lab, "status", "") == "Completed"
            else "registered",
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                            "code": "LAB",
                        }
                    ]
                }
            ],
            "code": {"text": getattr(lab, "test_name", "Laboratory Test")},
            "subject": {"reference": f"Patient/{patient_id}"},
            "conclusion": getattr(lab, "result_notes", "") or "",
        }
        entries.append({"resource": rep_res})

    # 6. ImagingStudies
    images = ImagingResult.query.filter_by(patient_id=patient_id).all()
    for img in images:
        entries.append({"resource": _imaging_to_fhir_study(img)})

    # 7. MedicationRequests
    prescriptions = PrescribedMedicine.query.filter_by(patient_id=patient_id).all()
    for rx in prescriptions:
        med_res = {
            "resourceType": "MedicationRequest",
            "id": f"medrx-{rx.id}",
            "status": "active",
            "intent": "order",
            "subject": {"reference": f"Patient/{patient_id}"},
            "medicationCodeableConcept": {
                "text": getattr(rx, "drug_name", "Prescribed Medication")
            },
            "dosageInstruction": [
                {
                    "text": f"{getattr(rx, 'dosage', '')} {getattr(rx, 'frequency', '')}".strip()
                }
            ],
        }
        entries.append({"resource": med_res})

    return {
        "resourceType": "Bundle",
        "id": f"bundle-everything-{patient_id}",
        "type": "collection",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total": len(entries),
        "entry": entries,
    }
