from departments.models.medicine import SOAPNote
from departments.nlp.logging_setup import get_logger
from departments.nlp.clinical_analyzer import ClinicalAnalyzer
import re

logger = get_logger()

def generate_ai_summary(note, analyzer: ClinicalAnalyzer = None) -> str:
    """Generate an AI summary for a SOAP note, including symptoms with UMLS metadata."""
    logger.debug(f"Starting generate_ai_summary for note ID {getattr(note, 'id', 'unknown')}")
    if not isinstance(note, SOAPNote):
        logger.error(f"Invalid input: note must be a SOAPNote object, got {type(note)}")
        return "Summary unavailable"

    try:
        # Initialize ClinicalAnalyzer if not provided
        if analyzer is None:
            analyzer = ClinicalAnalyzer()
            logger.debug("Initialized new ClinicalAnalyzer instance")

        summary_parts = []
        situation = getattr(note, 'situation', None) or ""

        # Extract chief complaint
        demographic_pattern = r"^(male|female|man|woman|boy|girl),\s*(.*)"
        match = re.match(demographic_pattern, situation.lower().strip())
        chief_complaint = ""
        if match:
            _, chief_complaint = match.groups()
            summary_parts.append(f"Chief Complaint: {chief_complaint.strip()}.")
        else:
            chief_complaint = situation.lower().strip()
            summary_parts.append(f"Chief Complaint: {chief_complaint}.")

        # Extract clinical features using ClinicalAnalyzer
        try:
            features = analyzer.extract_clinical_features(note, expected_symptoms=[chief_complaint] if chief_complaint else [])
            valid_symptoms = features.get('symptoms', [])
            logger.debug(f"Extracted symptoms: {[s['description'] for s in valid_symptoms]}")
        except Exception as e:
            logger.error(f"Error extracting clinical features: {str(e)}")
            valid_symptoms = []

        # Add HPI
        hpi = getattr(note, 'hpi', None)
        if hpi:
            summary_parts.append(f"HPI: {hpi.strip()}")

        # Add medication history
        medication_history = getattr(note, 'medication_history', None)
        if medication_history:
            summary_parts.append(f"Medications: {medication_history.strip()}")

        # Add assessment
        assessment = getattr(note, 'assessment', None)
        if assessment:
            primary_dx = re.search(r"Primary Assessment: (.*?)(?:\.|$)", assessment, re.DOTALL)
            if primary_dx:
                summary_parts.append(f"Primary Diagnosis: {primary_dx.group(1).strip()}")
            else:
                summary_parts.append(f"Assessment: {assessment.strip()}")

        # Add symptoms with UMLS metadata
        if valid_symptoms:
            symptom_text = "\nExtracted Symptoms:\n" + "\n".join(
                f"- {s['description']} (Category: {s['category']}"
                + (f", CUI: {s['umls_cui']}, Semantic Type: {s['semantic_type']})"
                   if s['umls_cui'] else ")")
                for s in valid_symptoms
            )
            summary_parts.append(symptom_text)

        if not summary_parts:
            logger.warning("No valid fields found for summary")
            return "Summary unavailable"

        summary = "\n".join(summary_parts)
        logger.debug(f"Generated summary: {summary}")
        return summary
    except Exception as e:
        logger.error(f"Error in generate_ai_summary for note ID {getattr(note, 'id', 'unknown')}: {str(e)}")
        return "Summary unavailable"