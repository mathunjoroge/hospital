"""
departments/api/ccda.py
───────────────────────
P8-08 — C-CDA (CCD) Continuity of Care Document generator from FHIR resources.

Generates XML Clinical Document Architecture R2 / C-CDA 2.1 CCD documents
for inter-facility continuity of care exchange. Includes sections for:
  - Header (Patient, Author, Custodian, Document ID)
  - Allergies section
  - Medication History section
  - Problem List / Conditions section
  - Vital Signs section
  - Laboratory / Diagnostic Results section
"""

import xml.etree.ElementTree as ET
from datetime import datetime, timezone

from flask import Blueprint, Response

from departments.api.auth import jwt_or_session_required
from departments.models.laboratory import LabResult
from departments.models.medicine import PrescribedMedicine, SOAPNote
from departments.models.nursing import Vitals
from departments.models.records import Patient
from departments.rbac import roles_required

ccda_bp = Blueprint("ccda", __name__)


def generate_c_cda_xml(patient: Patient) -> str:
    """Generate HL7 C-CDA 2.1 Continuity of Care Document XML for a patient."""
    now_utc = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%z")

    root = ET.Element("ClinicalDocument", {
        "xmlns": "urn:hl7-org:v3",
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:schemaLocation": "urn:hl7-org:v3 CDA.xsd"
    })

    # Realm & Type ID
    ET.SubElement(root, "realmCode", {"code": "US"})
    ET.SubElement(root, "typeId", {"root": "2.16.840.1.113883.1.3", "extension": "POCD_HD000040"})

    # CCD Template IDs
    ET.SubElement(root, "templateId", {"root": "2.16.840.1.113883.10.20.22.1.1", "extension": "2015-08-01"}) # General Header
    ET.SubElement(root, "templateId", {"root": "2.16.840.1.113883.10.20.22.1.2", "extension": "2015-08-01"}) # CCD

    # Document ID & Code
    ET.SubElement(root, "id", {"root": "2.16.840.1.113883.19.5", "extension": f"CCD-{patient.patient_id}"})
    ET.SubElement(root, "code", {
        "code": "34133-9",
        "codeSystem": "2.16.840.1.113883.6.1",
        "codeSystemName": "LOINC",
        "displayName": "Summarization of Episode Note"
    })
    ET.SubElement(root, "title").text = f"Continuity of Care Document - {patient.name}"
    ET.SubElement(root, "effectiveTime", {"value": now_utc})
    ET.SubElement(root, "confidentialityCode", {"code": "N", "codeSystem": "2.16.840.1.113883.5.25"})
    ET.SubElement(root, "languageCode", {"code": "en-US"})

    # Patient Role / Record Target
    record_target = ET.SubElement(root, "recordTarget")
    patient_role = ET.SubElement(record_target, "patientRole")
    ET.SubElement(patient_role, "id", {"root": "2.16.840.1.113883.19.5", "extension": str(patient.patient_id)})
    if patient.national_id:
        ET.SubElement(patient_role, "id", {"root": "2.16.840.1.113883.4.1", "extension": str(patient.national_id)})

    patient_elem = ET.SubElement(patient_role, "patient")
    name_elem = ET.SubElement(patient_elem, "name")
    parts = (patient.name or "Unknown Patient").split(" ", 1)
    ET.SubElement(name_elem, "given").text = parts[0]
    if len(parts) > 1:
        ET.SubElement(name_elem, "family").text = parts[1]

    gender_code = "M" if patient.sex in ["M", "Male"] else ("F" if patient.sex in ["F", "Female"] else "UN")
    ET.SubElement(patient_elem, "administrativeGenderCode", {"code": gender_code, "codeSystem": "2.16.840.1.113883.5.1"})

    if patient.date_of_birth:
        dob_str = patient.date_of_birth.strftime("%Y%m%d") if hasattr(patient.date_of_birth, "strftime") else str(patient.date_of_birth).replace("-", "")
        ET.SubElement(patient_elem, "birthTime", {"value": dob_str})

    # Author
    author = ET.SubElement(root, "author")
    ET.SubElement(author, "time", {"value": now_utc})
    assigned_author = ET.SubElement(author, "assignedAuthor")
    ET.SubElement(assigned_author, "id", {"root": "2.16.840.1.113883.19.5", "extension": "HIMS-SYSTEM"})
    assigned_person = ET.SubElement(assigned_author, "assignedPerson")
    person_name = ET.SubElement(assigned_person, "name")
    ET.SubElement(person_name, "given").text = "HIMS EHR System"

    # Custodian
    custodian = ET.SubElement(root, "custodian")
    assigned_custodian = ET.SubElement(custodian, "assignedCustodian")
    represented_org = ET.SubElement(assigned_custodian, "representedCustodianOrganization")
    ET.SubElement(represented_org, "id", {"root": "2.16.840.1.113883.19.5"})
    ET.SubElement(represented_org, "name").text = "Kenya National Teaching & Referral Hospital"

    # Structured Body
    component = ET.SubElement(root, "component")
    structured_body = ET.SubElement(component, "structuredBody")

    # 1. Allergies Section
    allergies_sec = ET.SubElement(structured_body, "component")
    section_all = ET.SubElement(allergies_sec, "section")
    ET.SubElement(section_all, "templateId", {"root": "2.16.840.1.113883.10.20.22.2.6.1", "extension": "2015-08-01"})
    ET.SubElement(section_all, "code", {"code": "48765-2", "codeSystem": "2.16.840.1.113883.6.1", "displayName": "Allergies and Adverse Reactions"})
    ET.SubElement(section_all, "title").text = "Allergies & Adverse Reactions"
    text_all = ET.SubElement(section_all, "text")
    if hasattr(patient, "allergies") and patient.allergies:
        try:
            items = [f"{a.allergen} ({getattr(a, 'reaction', 'Adverse reaction') or 'Adverse reaction'})" for a in patient.allergies]
            allergies_list = ", ".join(items) if items else "No known allergies."
        except Exception:
            allergies_list = str(patient.allergies)
    else:
        allergies_list = "No known allergies."
    text_all.text = allergies_list

    # 2. Medications Section
    meds_sec = ET.SubElement(structured_body, "component")
    section_med = ET.SubElement(meds_sec, "section")
    ET.SubElement(section_med, "templateId", {"root": "2.16.840.1.113883.10.20.22.2.1.1", "extension": "2015-08-01"})
    ET.SubElement(section_med, "code", {"code": "10160-0", "codeSystem": "2.16.840.1.113883.6.1", "displayName": "History of Medication Use"})
    ET.SubElement(section_med, "title").text = "Medication History"
    text_med = ET.SubElement(section_med, "text")

    rx_list = PrescribedMedicine.query.filter_by(patient_id=patient.patient_id).all()
    if rx_list:
        table_med = ET.SubElement(text_med, "table")
        thead = ET.SubElement(table_med, "thead")
        tr_h = ET.SubElement(thead, "tr")
        ET.SubElement(tr_h, "th").text = "Drug Name"
        ET.SubElement(tr_h, "th").text = "Dosage"
        ET.SubElement(tr_h, "th").text = "Frequency"
        tbody = ET.SubElement(table_med, "tbody")
        for rx in rx_list:
            tr = ET.SubElement(tbody, "tr")
            ET.SubElement(tr, "td").text = getattr(rx, "drug_name", "N/A")
            ET.SubElement(tr, "td").text = getattr(rx, "dosage", "N/A")
            ET.SubElement(tr, "td").text = getattr(rx, "frequency", "N/A")
    else:
        text_med.text = "No prescribed medications recorded."

    # 3. Problem List / Conditions Section
    prob_sec = ET.SubElement(structured_body, "component")
    section_prob = ET.SubElement(prob_sec, "section")
    ET.SubElement(section_prob, "templateId", {"root": "2.16.840.1.113883.10.20.22.2.5.1", "extension": "2015-08-01"})
    ET.SubElement(section_prob, "code", {"code": "11450-4", "codeSystem": "2.16.840.1.113883.6.1", "displayName": "Problem List"})
    ET.SubElement(section_prob, "title").text = "Problems & Diagnoses"
    text_prob = ET.SubElement(section_prob, "text")

    soap_notes = SOAPNote.query.filter_by(patient_id=patient.patient_id).all()
    assessments = [note.assessment for note in soap_notes if note.assessment]
    if assessments:
        table_p = ET.SubElement(text_prob, "table")
        tbody_p = ET.SubElement(table_p, "tbody")
        for ass in assessments:
            tr = ET.SubElement(tbody_p, "tr")
            ET.SubElement(tr, "td").text = ass
    else:
        text_prob.text = "No active problems recorded."

    # 4. Vital Signs Section
    vitals_sec = ET.SubElement(structured_body, "component")
    section_vit = ET.SubElement(vitals_sec, "section")
    ET.SubElement(section_vit, "templateId", {"root": "2.16.840.1.113883.10.20.22.2.4.1", "extension": "2015-08-01"})
    ET.SubElement(section_vit, "code", {"code": "8716-3", "codeSystem": "2.16.840.1.113883.6.1", "displayName": "Vital Signs"})
    ET.SubElement(section_vit, "title").text = "Vital Signs"
    text_vit = ET.SubElement(section_vit, "text")

    vitals_records = Vitals.query.filter_by(patient_id=patient.patient_id).order_by(Vitals.timestamp.desc()).limit(5).all()
    if vitals_records:
        table_v = ET.SubElement(text_vit, "table")
        thead_v = ET.SubElement(table_v, "thead")
        tr_vh = ET.SubElement(thead_v, "tr")
        ET.SubElement(tr_vh, "th").text = "Date/Time"
        ET.SubElement(tr_vh, "th").text = "Temp (°C)"
        ET.SubElement(tr_vh, "th").text = "Pulse (/min)"
        ET.SubElement(tr_vh, "th").text = "BP (mmHg)"
        ET.SubElement(tr_vh, "th").text = "SpO2 (%)"
        tbody_v = ET.SubElement(table_v, "tbody")
        for v in vitals_records:
            tr = ET.SubElement(tbody_v, "tr")
            ET.SubElement(tr, "td").text = v.timestamp.strftime("%Y-%m-%d %H:%M") if getattr(v, "timestamp", None) else "N/A"
            ET.SubElement(tr, "td").text = str(getattr(v, "temperature", "-"))
            ET.SubElement(tr, "td").text = str(getattr(v, "pulse", "-") or getattr(v, "pulse_rate", "-"))
            sys = getattr(v, "blood_pressure_systolic", None) or getattr(v, "bp_systolic", "-")
            dia = getattr(v, "blood_pressure_diastolic", None) or getattr(v, "bp_diastolic", "-")
            ET.SubElement(tr, "td").text = f"{sys}/{dia}"
            ET.SubElement(tr, "td").text = str(getattr(v, "oxygen_saturation", "-") or getattr(v, "spo2", "-"))
    else:
        text_vit.text = "No vital signs recorded."

    # 5. Diagnostic Results Section
    results_sec = ET.SubElement(structured_body, "component")
    section_res = ET.SubElement(results_sec, "section")
    ET.SubElement(section_res, "templateId", {"root": "2.16.840.1.113883.10.20.22.2.3.1", "extension": "2015-08-01"})
    ET.SubElement(section_res, "code", {"code": "30954-2", "codeSystem": "2.16.840.1.113883.6.1", "displayName": "Relevant Diagnostic Tests and/or Laboratory Data"})
    ET.SubElement(section_res, "title").text = "Diagnostic & Laboratory Results"
    text_res = ET.SubElement(section_res, "text")

    labs = LabResult.query.filter_by(patient_id=patient.patient_id).all()
    if labs:
        table_l = ET.SubElement(text_res, "table")
        thead_l = ET.SubElement(table_l, "thead")
        tr_lh = ET.SubElement(thead_l, "tr")
        ET.SubElement(tr_lh, "th").text = "Test Name"
        ET.SubElement(tr_lh, "th").text = "Result"
        tbody_l = ET.SubElement(table_l, "tbody")
        for lab in labs:
            tr = ET.SubElement(tbody_l, "tr")
            ET.SubElement(tr, "td").text = getattr(lab, "test_name", "Lab Test")
            ET.SubElement(tr, "td").text = str(getattr(lab, "result_value", "Pending"))
    else:
        text_res.text = "No diagnostic results recorded."

    # Convert XML tree to string
    ET.indent(root, space="  ")
    xml_str = '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="utf-8").decode("utf-8")
    return xml_str


@ccda_bp.route("/api/ccda/<string:patient_id>", methods=["GET"])
@ccda_bp.route("/api/fhir/R4/Patient/<string:patient_id>/$ccda", methods=["GET"])
@jwt_or_session_required
@roles_required("admin", "records", "medicine", "nursing", "api")
def export_ccda(patient_id):
    """Export C-CDA XML Continuity of Care document for patient."""
    patient = Patient.query.filter_by(patient_id=patient_id).first_or_404()
    xml_content = generate_c_cda_xml(patient)
    return Response(xml_content, mimetype="application/xml")

