"""
departments/nursing/triage.py
──────────────────────────────
Task 3.1 — Triage Workflow & Emergency Severity Index (ESI 1-5) Scoring

Features:
  - Pediatric & Adult age-adjusted vitals validator
  - Emergency Severity Index (ESI Level 1 to 5) decision algorithm
  - Real-time priority queue generation
  - Emergency escalation alert trigger for Level 1 & Level 2 patients
"""

import logging
from datetime import date, datetime

from flask import Blueprint, jsonify, request

try:
    from extensions import db
except ImportError:
    from extensions import db

from departments.models.nursing import Notifications, TriageAssessment, Vitals
from departments.models.records import Patient

logger = logging.getLogger(__name__)

triage_bp = Blueprint('triage', __name__, url_prefix='/nursing/triage')


def validate_vitals(age_years: float, hr: int = None, rr: int = None, sbp: int = None, temp: float = None, spo2: int = None) -> dict:
    """
    Validate vitals against age-adjusted pediatric and adult physiological reference ranges.
    Returns dict: {"is_abnormal": bool, "warnings": list[str], "risk_level": str}
    """
    warnings = []
    risk_level = "NORMAL"

    if spo2 is not None:
        if spo2 < 88:
            warnings.append(f"CRITICAL HYPOXIA: SpO2 {spo2}% (<88%)")
            risk_level = "CRITICAL"
        elif spo2 < 94:
            warnings.append(f"Moderate Hypoxia: SpO2 {spo2}% (<94%)")
            if risk_level != "CRITICAL":
                risk_level = "WARNING"

    if temp is not None:
        if temp >= 39.5:
            warnings.append(f"High Fever: Temp {temp}°C (>=39.5°C)")
            if risk_level != "CRITICAL":
                risk_level = "WARNING"
        elif temp < 35.0:
            warnings.append(f"Hypothermia: Temp {temp}°C (<35.0°C)")
            if risk_level != "CRITICAL":
                risk_level = "WARNING"

    # Age-specific HR, RR, SBP thresholds
    if age_years < 1:  # Infant
        hr_min, hr_max = 100, 160
        rr_min, rr_max = 30, 60
        sbp_min, sbp_max = 70, 100
    elif age_years <= 3:  # Toddler
        hr_min, hr_max = 90, 150
        rr_min, rr_max = 24, 40
        sbp_min, sbp_max = 80, 105
    elif age_years <= 6:  # Preschool
        hr_min, hr_max = 80, 140
        rr_min, rr_max = 20, 30
        sbp_min, sbp_max = 80, 110
    elif age_years <= 12:  # School age
        hr_min, hr_max = 70, 120
        rr_min, rr_max = 15, 25
        sbp_min, sbp_max = 90, 120
    else:  # Adolescent / Adult
        hr_min, hr_max = 60, 100
        rr_min, rr_max = 12, 20
        sbp_min, sbp_max = 90, 139

    if hr is not None:
        if hr < (hr_min - 15) or hr > (hr_max + 30):
            warnings.append(f"Extreme Heart Rate: {hr} bpm (Normal: {hr_min}-{hr_max})")
            risk_level = "CRITICAL"
        elif hr < hr_min or hr > hr_max:
            warnings.append(f"Abnormal Heart Rate: {hr} bpm (Normal: {hr_min}-{hr_max})")
            if risk_level != "CRITICAL":
                risk_level = "WARNING"

    if rr is not None:
        if rr < (rr_min - 5) or rr > (rr_max + 15):
            warnings.append(f"Extreme Respiratory Rate: {rr} rpm (Normal: {rr_min}-{rr_max})")
            risk_level = "CRITICAL"
        elif rr < rr_min or rr > rr_max:
            warnings.append(f"Abnormal Respiratory Rate: {rr} rpm (Normal: {rr_min}-{rr_max})")
            if risk_level != "CRITICAL":
                risk_level = "WARNING"

    if sbp is not None:
        if sbp < sbp_min or sbp >= 180:
            warnings.append(f"Critical Blood Pressure SBP: {sbp} mmHg (Normal: {sbp_min}-{sbp_max})")
            risk_level = "CRITICAL"
        elif sbp > sbp_max:
            warnings.append(f"High Blood Pressure SBP: {sbp} mmHg (Normal: {sbp_min}-{sbp_max})")
            if risk_level != "CRITICAL":
                risk_level = "WARNING"

    return {
        "is_abnormal": len(warnings) > 0,
        "warnings": warnings,
        "risk_level": risk_level
    }


def calculate_esi_level(vitals_dict: dict, chief_complaint: str = "", resources_needed: int = 1, age_years: float = 30.0) -> tuple[int, str]:
    """
    Calculate Emergency Severity Index (ESI Level 1-5).
    Returns (esi_level, reasoning_string).
    """
    validation = validate_vitals(
        age_years=age_years,
        hr=vitals_dict.get('pulse'),
        rr=vitals_dict.get('respiratory_rate'),
        sbp=vitals_dict.get('blood_pressure_systolic'),
        temp=vitals_dict.get('temperature'),
        spo2=vitals_dict.get('oxygen_saturation')
    )

    complaint_lower = (chief_complaint or "").lower()

    # ESI 1: Immediate life-saving intervention required
    life_threatening_keywords = ['unresponsive', 'cardiac arrest', 'apneic', 'severe anaphylaxis', 'gasping']
    if any(k in complaint_lower for k in life_threatening_keywords) or validation['risk_level'] == 'CRITICAL':
        return 1, "ESI Level 1: Immediate life-saving intervention required (Critical Vitals / Unresponsive)."

    # ESI 2: High risk / confused / severe pain or distress / abnormal vitals
    high_risk_keywords = ['chest pain', 'stroke', 'severe pain', 'suicidal', 'confused', 'shortness of breath']
    if any(k in complaint_lower for k in high_risk_keywords) or validation['risk_level'] == 'WARNING':
        return 2, "ESI Level 2: High risk situation, severe distress, or abnormal vital signs."

    # ESI 3, 4, 5 based on expected resource consumption
    if resources_needed >= 2:
        return 3, "ESI Level 3: Urgent — Patient requires 2 or more resources with stable vitals."
    elif resources_needed == 1:
        return 4, "ESI Level 4: Less Urgent — Patient requires 1 resource (e.g. Lab or X-Ray only)."
    else:
        return 5, "ESI Level 5: Non-Urgent — Patient requires no additional diagnostic resources."


@triage_bp.route('/assess', methods=['POST'])
def assess_patient_triage():
    """Submit a triage assessment for an emergency/outpatient visit."""
    data = request.get_json() or {}
    patient_id = data.get('patient_id')
    nurse_id = data.get('nurse_id', 1)
    chief_complaint = data.get('chief_complaint', 'Routine Checkup')
    resources_needed = int(data.get('resources_needed', 1))
    vitals_data = data.get('vitals', {})

    patient = Patient.query.filter_by(patient_id=patient_id).first()
    if not patient:
        return jsonify({'error': 'Patient not found'}), 404

    # Calculate age in years
    age_years = 30.0
    if patient.date_of_birth:
        dob = patient.date_of_birth
        if isinstance(dob, str):
            try:
                dob = datetime.strptime(dob, '%Y-%m-%d').date()
            except ValueError:
                dob = date(1995, 1, 1)
        today = date.today()
        age_years = (today - dob).days / 365.25

    # Compute ESI level
    esi_level, reasoning = calculate_esi_level(
        vitals_dict=vitals_data,
        chief_complaint=chief_complaint,
        resources_needed=resources_needed,
        age_years=age_years
    )

    # Save Vitals record
    v = Vitals(
        patient_id=patient_id,
        nurse_id=nurse_id,
        temperature=vitals_data.get('temperature'),
        pulse=vitals_data.get('pulse'),
        blood_pressure_systolic=vitals_data.get('blood_pressure_systolic'),
        blood_pressure_diastolic=vitals_data.get('blood_pressure_diastolic'),
        respiratory_rate=vitals_data.get('respiratory_rate'),
        oxygen_saturation=vitals_data.get('oxygen_saturation'),
        blood_glucose=vitals_data.get('blood_glucose'),
        weight=vitals_data.get('weight'),
        height=vitals_data.get('height')
    )
    db.session.add(v)
    db.session.flush()

    vitals_val = validate_vitals(
        age_years=age_years,
        hr=v.pulse,
        rr=v.respiratory_rate,
        sbp=v.blood_pressure_systolic,
        temp=v.temperature,
        spo2=v.oxygen_saturation
    )

    # Save TriageAssessment record
    assessment = TriageAssessment(
        patient_id=patient_id,
        nurse_id=nurse_id,
        esi_level=esi_level,
        chief_complaint=chief_complaint,
        vitals_id=v.id,
        is_pediatric=(age_years < 12),
        vitals_warning=" | ".join(vitals_val['warnings']) if vitals_val['warnings'] else None,
        priority_status="ESCALATED" if esi_level in (1, 2) else "WAITING"
    )
    db.session.add(assessment)

    # Trigger alert notification if critical ESI 1 or 2
    if esi_level in (1, 2):
        alert_msg = f"EMERGENCY ESCALATION [ESI Level {esi_level}]: Patient {patient.name} ({patient_id}) — {chief_complaint}"
        notification = Notifications(receiver_id=nurse_id, message=alert_msg)
        db.session.add(notification)

    db.session.commit()

    return jsonify({
        "success": True,
        "assessment_id": assessment.id,
        "esi_level": esi_level,
        "reasoning": reasoning,
        "is_pediatric": age_years < 12,
        "vitals_warnings": vitals_val['warnings'],
        "priority_status": assessment.priority_status
    }), 201


@triage_bp.route('/queue', methods=['GET'])
def get_triage_queue():
    """Retrieve active emergency triage queue sorted by ESI level (1 highest)."""
    assessments = TriageAssessment.query.filter_by(priority_status="WAITING").order_by(
        TriageAssessment.esi_level.asc(),
        TriageAssessment.created_at.asc()
    ).all()

    queue = []
    for a in assessments:
        queue.append({
            "assessment_id": a.id,
            "patient_id": a.patient_id,
            "patient_name": a.patient.name if a.patient else "Unknown",
            "esi_level": a.esi_level,
            "chief_complaint": a.chief_complaint,
            "vitals_warning": a.vitals_warning,
            "created_at": a.created_at.isoformat()
        })

    return jsonify({"queue": queue, "total_waiting": len(queue)}), 200
