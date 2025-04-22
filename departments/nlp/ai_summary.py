# departments/nlp/ai_summary.py

from typing import List, Dict, Tuple
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
        dx, _, reason = diff
        rationale.append(f"{dx}: {reason}")
    return ';'.join(rationale) or 'Based on clinical features and history.'

def generate_ai_analysis(note: SOAPNote, patient: Patient = None) -> str:
    """Generate clinical analysis with ranked differentials and reasoning."""
    logger.debug(f"Generating analysis for note {note.id}, situation: {note.situation}")
    if not isinstance(note, SOAPNote):
        logger.error(f"Invalid note type: {type(note)}")
        return """
[=== AI CLINICAL ANALYSIS ===]
[CHIEF CONCERN]
Unknown
[DIAGNOSIS]
Not specified
[DIFFERENTIAL DIAGNOSIS]
Undetermined: Insufficient data
[CLINICAL RATIONALE]
No clinical features provided
[RECOMMENDED WORKUP]
■ Urgent: None
■ Routine: None
[TREATMENT OPTIONS]
▲ Symptomatic: None
● Definitive: Pending diagnosis
[FOLLOW-UP]
As needed
DISCLAIMER: This AI-generated analysis requires clinical correlation.
"""
    try:
        analyzer = ClinicalAnalyzer()
        features = analyzer.extract_clinical_features(note)
        differentials = analyzer.generate_differential_dx(features, patient)
        plan = analyzer.generate_management_plan(features, differentials)
        
        # Format differentials with qualitative likelihood
        differential_text = []
        for i, diff in enumerate(differentials):
            if not isinstance(diff, tuple) or len(diff) != 3:
                continue
            dx, score, reason = diff
            likelihood = 'Most likely' if i == 0 else 'Less likely'
            differential_text.append(f"{dx}: {reason} ({likelihood})")
        differential_output = ';'.join(differential_text) or 'Undetermined: Insufficient data'
        
        # Check for high-risk conditions
        high_risk_conditions = {'temporal arteritis', 'atrial fibrillation', 'subarachnoid hemorrhage'}
        high_risk = any(dx.lower() in high_risk_conditions for dx, _, _ in differentials if isinstance(dx, str))
        disclaimer = "High-risk conditions detected; urgent review recommended." if high_risk else ""

        analysis_output = f"""
[=== AI CLINICAL ANALYSIS ===]
[CHIEF CONCERN]
{features['chief_complaint'].lower() or 'unknown'}
[DIAGNOSIS]
{features['assessment'] or 'Not specified'}
[DIFFERENTIAL DIAGNOSIS]
{differential_output}
[CLINICAL RATIONALE]
{generate_rationale(features, differentials, plan)}
[RECOMMENDED WORKUP]
■ Urgent: {';'.join(sorted(plan['workup']['urgent'])) or 'None'}
■ Routine: {';'.join(sorted(plan['workup']['routine'])) or 'None'}
[TREATMENT OPTIONS]
▲ Symptomatic: {';'.join(sorted(plan['treatment']['symptomatic'])) or 'None'}
● Definitive: {';'.join(sorted(plan['treatment']['definitive'])) or 'Pending diagnosis'}
[FOLLOW-UP]
{';'.join(plan['follow_up']) or 'As needed'}
DISCLAIMER: This AI-generated analysis requires clinical correlation. {disclaimer}
"""
        logger.debug(f"Analysis output for note {note.id}: {analysis_output}")
        return analysis_output.strip()
    except Exception as e:
        logger.error(f"Analysis failed for note {note.id}: {str(e)}", exc_info=True)
        default_workup = {
            'palpitations': 'ECG',
            'vaginal discharge': 'Vaginal swab',
            'headache': 'Neurological exam',
            'unknown': 'None'
        }
        chief_complaint = getattr(note, 'situation', 'Unknown').lower()
        workup_key = next((k for k in default_workup if k in chief_complaint), 'unknown')
        return f"""
[=== AI CLINICAL ANALYSIS ===]
[CHIEF CONCERN]
{chief_complaint or 'Unknown'}
[DIAGNOSIS]
Not specified
[DIFFERENTIAL DIAGNOSIS]
Undetermined: Insufficient data
[CLINICAL RATIONALE]
Analysis failed due to processing error
[RECOMMENDED WORKUP]
■ Urgent: {default_workup[workup_key]}
■ Routine: CBC;Basic metabolic panel
[TREATMENT OPTIONS]
▲ Symptomatic: None
● Definitive: Pending diagnosis
[FOLLOW-UP]
As needed
DISCLAIMER: This AI-generated analysis requires clinical correlation.
"""