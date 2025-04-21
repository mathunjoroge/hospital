from departments.models.medicine import SOAPNote
from departments.models.records import Patient
from departments.nlp.clinical_analyzer import ClinicalAnalyzer
from departments.nlp.note_processing import build_note_text, generate_ai_summary
from departments.nlp.logging_setup import logger
from departments.nlp.config import CONFIDENCE_THRESHOLD

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
Undetermined
[RECOMMENDED WORKUP]
■ Urgent: None
■ Routine: CBC;Basic metabolic panel
[TREATMENT OPTIONS]
▲ Symptomatic: None
● Definitive: Pending diagnosis
[FOLLOW-UP]
As needed
DISCLAIMER: This AI-generated analysis requires clinical correlation.
"""
    try:
        logger.debug(f"Initializing ClinicalAnalyzer for note {note.id}")
        analyzer = ClinicalAnalyzer()
        logger.debug(f"Extracting clinical features for note {note.id}")
        features = analyzer.extract_clinical_features(note)
        logger.debug(f"Features extracted: {features}")
        if not features['symptoms']:
            logger.warning(f"No symptoms extracted for note {note.id}")
        logger.debug(f"Generating differential diagnosis for note {note.id}")
        differentials = analyzer.generate_differential_dx(features)
        logger.debug(f"Differentials: {differentials}")
        logger.debug(f"Generating management plan for note {note.id}")
        plan = analyzer.generate_management_plan(features, differentials)
        logger.debug(f"Management plan: {plan}")
        high_risk = any(len(diff) >= 2 and diff[1] >= CONFIDENCE_THRESHOLD for diff in differentials if isinstance(diff, tuple))
        analysis_output = f"""
[=== AI CLINICAL ANALYSIS ===]
[CHIEF CONCERN]
{features['chief_complaint'].lower() or 'unknown'}
[DIAGNOSIS]
{features['assessment'] or 'Not specified'}
[DIFFERENTIAL DIAGNOSIS]
{';'.join(f'{dx} ({score:.2%}): {reason}' for dx, score, reason in differentials if isinstance(dx, str) and isinstance(score, (int, float))) or 'Undetermined'}
[RECOMMENDED WORKUP]
■ Urgent: {';'.join(sorted(plan['workup']['urgent'])) or 'None'}
■ Routine: {';'.join(sorted(plan['workup']['routine'])) or 'None'}
[TREATMENT OPTIONS]
▲ Symptomatic: {';'.join(sorted(plan['treatment']['symptomatic'])) or 'None'}
● Definitive: {';'.join(sorted(plan['treatment']['definitive'])) or 'Pending diagnosis'}
[FOLLOW-UP]
{';'.join(plan['follow_up']) or 'As needed'}
DISCLAIMER: This AI-generated analysis requires clinical correlation. {'High-risk conditions detected; urgent review recommended.' if high_risk else ''}
"""
        logger.debug(f"Analysis output for note {note.id}: {analysis_output}")
        return analysis_output.strip()
    except Exception as e:
        logger.error(f"Analysis failed for note {note.id}: {str(e)}", exc_info=True)
        return """
[=== AI CLINICAL ANALYSIS ===]
[CHIEF CONCERN]
Unknown
[DIAGNOSIS]
Not specified
[DIFFERENTIAL DIAGNOSIS]
Undetermined
[RECOMMENDED WORKUP]
■ Urgent: None
■ Routine: CBC;Basic metabolic panel
[TREATMENT OPTIONS]
▲ Symptomatic: None
● Definitive: Pending diagnosis
[FOLLOW-UP]
As needed
DISCLAIMER: This AI-generated analysis requires clinical correlation.
"""