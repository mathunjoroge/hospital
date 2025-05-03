# departments/nlp/ai_summary.py
from typing import List, Dict, Tuple
from departments.models.medicine import SOAPNote
from departments.models.records import Patient
from departments.nlp.clinical_analyzer import ClinicalAnalyzer
from departments.nlp.symptom_tracker import SymptomTracker
import re
import logging

logger = logging.getLogger(__name__)

def generate_rationale(features: Dict, differentials: List[Tuple], plan: Dict) -> str:
    rationale = []
    for diff in differentials:
        if not isinstance(diff, tuple) or len(diff) != 3:
            continue
        dx, score, reason = diff
        rationale.append(f"{dx} (Confidence: {score:.2f}): {reason}")
    return '; '.join(rationale) or 'Based on clinical features and history.'

def generate_ai_analysis(note: SOAPNote, patient: Patient = None) -> str:
    logger.debug(f"Generating analysis for note {getattr(note, 'id', 'unknown')}, situation: {note.situation}")
    if not isinstance(note, SOAPNote):
        logger.error(f"Invalid note type: {type(note)}")
        return """
[=== AI CLINICAL ANALYSIS ===]
[PATIENT PROFILE]
Not provided
[CHIEF CONCERN]
Unknown
[SYMPTOMS]
None identified
[NEGATED SYMPTOMS]
None reported
[DIAGNOSIS]
Not specified
[DIFFERENTIAL DIAGNOSIS]
• Undetermined: Insufficient data
[CLINICAL RATIONALE]
No clinical features provided
[RECOMMENDED WORKUP]
• Urgent: None
• Routine: None
[TREATMENT OPTIONS]
• Symptomatic: None
• Definitive: Pending diagnosis
• Lifestyle: None
[FOLLOW-UP]
As needed
[REFERENCES]
None available
DISCLAIMER: This AI-generated analysis requires clinical correlation.
"""
    try:
        analyzer = ClinicalAnalyzer()
        tracker = SymptomTracker()
        situation = getattr(note, 'situation', '').lower().strip()
        demographic_pattern = r"^(male|female|man|woman|boy|girl),\s*(.*)"
        match = re.match(demographic_pattern, situation)
        chief_complaint = match.group(2).strip() if match else situation
        symptoms = tracker.process_note(note, chief_complaint)
        features = {
            'chief_complaint': chief_complaint,
            'symptoms': symptoms,
            'hpi': note.hpi or '',
            'assessment': note.assessment or '',
            'additional_notes': note.additional_notes or ''
        }
        differentials = analyzer.generate_differential_dx(features, patient)
        plan = analyzer.generate_management_plan(features, differentials)
        assessment = note.assessment or 'Not specified'
        if differentials and differentials[0][0] == "Undetermined":
            assessment = "Not specified (insufficient data)"
        patient_profile = []
        patient_sex = getattr(patient, 'sex', None)
        if match and not patient_sex:
            patient_sex = match.group(1).capitalize()
        if patient_sex:
            patient_profile.append(f"Sex: {patient_sex}")
        if patient and hasattr(patient, 'age') and patient.age:
            patient_profile.append(f"Age: {patient.age}")
        patient_profile_text = '; '.join(patient_profile) or 'Not provided'
        symptom_text = []
        for symptom in symptoms:
            desc = symptom.get('description', '')
            severity = symptom.get('severity', 'Unknown')
            duration = symptom.get('duration', 'Unknown')
            location = symptom.get('location', 'Unknown')
            symptom_text.append(f"{desc}, Severity: {severity}, Duration: {duration}, Location: {location}")
        symptoms_output = '\n• '.join(symptom_text) or 'None identified'
        negated_terms = set()
        text = f"{note.situation or ''} {note.hpi or ''} {note.assessment or ''} {note.additional_notes or ''}"
        negation_pattern = r"\b(no|denies|without|not)\b\s+([\w\s]+?)(?=\.|,|;|\band\b|\bor\b|$)"
        for match in re.finditer(negation_pattern, text.lower(), re.IGNORECASE):
            term = match.group(2).strip()
            negated_terms.add(term)
        negated_output = '; '.join(sorted(negated_terms)) or 'None reported'
        differential_text = []
        for i, diff in enumerate(differentials):
            if not isinstance(diff, tuple) or len(diff) != 3:
                continue
            dx, score, reason = diff
            likelihood = 'Most likely' if i == 0 else 'Less likely'
            differential_text.append(f"{dx} (Confidence: {score:.2f}): {reason} ({likelihood})")
        differential_output = '\n• '.join(differential_text) or 'Undetermined: Insufficient data'
        high_risk_conditions = {
            'temporal arteritis', 'atrial fibrillation', 'subarachnoid hemorrhage',
            'myocardial infarction', 'pulmonary embolism', 'aortic dissection'
        }
        high_risk = any(dx.lower() in high_risk_conditions for dx, _, _ in differentials if isinstance(dx, str))
        disclaimer = "High-risk conditions detected; urgent review recommended." if high_risk else ""
        references = set()
        for category, pathways in analyzer.clinical_pathways.items():
            for key, path in pathways.items():
                if any(k.lower() in chief_complaint.lower() for k in key.split('|')):
                    metadata = path.get('metadata', {})
                    sources = metadata.get('source', [])
                    if isinstance(sources, list):
                        references.update(sources)
                    for update in metadata.get('updates', []):
                        title = update.get('title', 'Guideline')
                        url = update.get('url', '')
                        references.add(f"{title} ({url})")
        references_output = '\n• '.join(sorted(references)) or 'None available'
        pathways_warning = ""
        if not any(any(k.lower() in chief_complaint.lower() for k in key.split('|'))
                   for pathways in analyzer.clinical_pathways.values()
                   for key in pathways):
            pathways_warning = "Warning: No clinical pathway found for chief complaint; recommendations may be limited."
        analysis_output = f"""
[=== AI CLINICAL ANALYSIS ===]
[PATIENT PROFILE]
{patient_profile_text}
[CHIEF CONCERN]
{chief_complaint or 'Unknown'}
[SYMPTOMS]
• {symptoms_output}
[NEGATED SYMPTOMS]
{negated_output}
[DIAGNOSIS]
{assessment}
[DIFFERENTIAL DIAGNOSIS]
• {differential_output}
[CLINICAL RATIONALE]
{generate_rationale(features, differentials, plan)}
[RECOMMENDED WORKUP]
• Urgent: {'; '.join(sorted(plan['workup']['urgent'])) or 'None'}
• Routine: {'; '.join(sorted(plan['workup']['routine'])) or 'None'}
[TREATMENT OPTIONS]
• Symptomatic: {'; '.join(sorted(plan['treatment']['symptomatic'])) or 'None'}
• Definitive: {'; '.join(sorted(plan['treatment']['definitive'])) or 'Pending diagnosis'}
• Lifestyle: {'; '.join(sorted(plan['treatment'].get('lifestyle', []))) or 'None'}
[FOLLOW-UP]
{'; '.join(plan['follow_up']) or 'As needed'}
[REFERENCES]
• {references_output}
DISCLAIMER: This AI-generated analysis requires clinical correlation. {disclaimer} {pathways_warning}
"""
        logger.debug(f"Analysis output for note {getattr(note, 'id', 'unknown')}")
        return analysis_output.strip()
    except Exception as e:
        logger.error(f"Analysis failed for note {getattr(note, 'id', 'unknown')}: {str(e)}", exc_info=True)
        default_workup = {
            'palpitations': 'ECG;Holter monitor',
            'vaginal discharge': 'Vaginal swab;Pelvic exam',
            'headache': 'Neurological exam;CT head if severe',
            'chest pain': 'ECG;Cardiac enzymes;Chest X-ray',
            'fever': 'CBC;Blood cultures',
            'fatigue': 'CBC;TSH;Vitamin D',
            'abdominal pain': 'Abdominal ultrasound;CBC',
            'urinary symptoms': 'Urinalysis;Urine culture',
            'unknown': 'CBC;Basic metabolic panel'
        }
        chief_complaint = getattr(note, 'situation', 'Unknown').lower()
        workup_key = next((k for k in default_workup if k in chief_complaint), 'unknown')
        return f"""
[=== AI CLINICAL ANALYSIS ===]
[PATIENT PROFILE]
Not provided
[CHIEF CONCERN]
{chief_complaint or 'Unknown'}
[SYMPTOMS]
None identified
[NEGATED SYMPTOMS]
None reported
[DIAGNOSIS]
Not specified
[DIFFERENTIAL DIAGNOSIS]
• Undetermined: Insufficient data
[CLINICAL RATIONALE]
Analysis failed due to processing error
[RECOMMENDED WORKUP]
• Urgent: {default_workup[workup_key]}
• Routine: None
[TREATMENT OPTIONS]
• Symptomatic: None
• Definitive: Pending diagnosis
• Lifestyle: None
[FOLLOW-UP]
As needed
[REFERENCES]
None available
DISCLAIMER: This AI-generated analysis requires clinical correlation.
"""