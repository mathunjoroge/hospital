from typing import List, Dict, Tuple
import re
from departments.models.medicine import SOAPNote
from departments.models.records import Patient
from departments.nlp.clinical_analyzer import ClinicalAnalyzer
from departments.nlp.note_processing import build_note_text, generate_ai_summary
from departments.nlp.logging_setup import logger
from departments.nlp.config import CONFIDENCE_THRESHOLD

def generate_rationale(features: Dict, differentials: List[Tuple], plan: Dict) -> str:
    """Generate clinical rationale for the analysis.
    
    Args:
        features: Dictionary of extracted clinical features
        differentials: List of tuples (diagnosis: str, score: float, reasoning: str)
        plan: Dictionary with workup, treatment, and follow-up plans
    
    Returns:
        Formatted rationale string
    """
    rationale = []
    for diff in differentials:
        if not isinstance(diff, tuple) or len(diff) != 3:
            continue
        dx, score, reason = diff
        rationale.append(f"{dx} (Confidence: {score:.2f}): {reason}")
    return '; '.join(rationale) or 'Based on clinical features and history.'

def generate_ai_analysis(note: SOAPNote, patient: Patient = None) -> str:
    """Generate clinical analysis with ranked differentials and reasoning."""
    logger.debug(f"Generating analysis for note {note.id}, situation: {note.situation}")
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
[FOLLOW-UP]
As needed
[REFERENCES]
None available
DISCLAIMER: This AI-generated analysis requires clinical correlation.
"""
    try:
        analyzer = ClinicalAnalyzer()
        features = analyzer.extract_clinical_features(note)
        differentials = analyzer.generate_differential_dx(features, patient)
        plan = analyzer.generate_management_plan(features, differentials)
        
        # Validate assessment
        assessment = features['assessment'] or 'Not specified'
        if differentials[0][0] == "Undetermined":
            assessment = "Not specified (insufficient data)"
        
        # Format patient profile
        patient_profile = []
        if patient and hasattr(patient, 'age') and patient.age:
            patient_profile.append(f"Age: {patient.age}")
        if patient and hasattr(patient, 'sex') and patient.sex:
            patient_profile.append(f"Sex: {patient.sex}")
        patient_profile_text = '; '.join(patient_profile) or 'Not provided'
        
        # Format symptoms
        symptom_text = []
        for symptom in features.get('symptoms', []):
            desc = symptom.get('description', '')
            severity = symptom.get('severity', '')
            duration = symptom.get('duration', '')
            location = symptom.get('location', '')
            details = [desc]
            if severity: details.append(f"Severity: {severity}")
            if duration: details.append(f"Duration: {duration}")
            if location: details.append(f"Location: {location}")
            symptom_text.append(', '.join(details))
        symptoms_output = '\n• '.join(symptom_text) or 'None identified'
        
        # Format negated symptoms
        negated_terms = set()
        text = f"{features.get('chief_complaint', '')} {features.get('hpi', '')} {features.get('additional_notes', '')}"
        for match in re.finditer(r'\b(?:no|denies|without)\s+([\w\s]+?)(?:\s|$)', text.lower()):
            term = match.group(1).strip()
            if term in analyzer.medical_terms or term in analyzer.common_symptoms.get_all_symptoms():
                negated_terms.add(term)
        negated_output = '; '.join(sorted(negated_terms)) or 'None reported'
        
        # Format differentials
        differential_text = []
        for i, diff in enumerate(differentials):
            if not isinstance(diff, tuple) or len(diff) != 3:
                continue
            dx, score, reason = diff
            likelihood = 'Most likely' if i == 0 else 'Less likely'
            differential_text.append(f"{dx} (Confidence: {score:.2f}): {reason} ({likelihood})")
        differential_output = '\n• '.join(differential_text) or 'Undetermined: Insufficient data'
        
        # Check high-risk conditions
        high_risk_conditions = {
            'temporal arteritis', 'atrial fibrillation', 'subarachnoid hemorrhage',
            'myocardial infarction', 'pulmonary embolism', 'aortic dissection'
        }
        high_risk = any(dx.lower() in high_risk_conditions for dx, _, _ in differentials if isinstance(dx, str))
        disclaimer = "High-risk conditions detected; urgent review recommended." if high_risk else ""
        
        # Extract references from pathways
        references = []
        for category, pathways in analyzer.clinical_pathways.items():
            for key, path in pathways.items():
                if key.lower() in features.get('chief_complaint', '').lower():
                    metadata = path.get('metadata', {})
                    sources = metadata.get('source', [])
                    if isinstance(sources, list):
                        references.extend(sources)
                    for update in metadata.get('updates', []):
                        title = update.get('title', 'Guideline')
                        url = update.get('url', '')
                        references.append(f"{title} ({url})")
        references_output = '\n• '.join(set(references)) or 'None available'
        
        # Check for incomplete pathways
        pathways_warning = ""
        chief_complaint = features.get('chief_complaint', '').lower()
        if not any(key.lower() == chief_complaint for pathways in analyzer.clinical_pathways.values() for key in pathways):
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
        logger.debug(f"Analysis output for note {note.id}: {analysis_output}")
        return analysis_output.strip()
    except Exception as e:
        logger.error(f"Analysis failed for note {note.id}: {str(e)}", exc_info=True)
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