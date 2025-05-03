# departments/nlp/note_processing.py
from departments.models.medicine import SOAPNote
import logging
import re

logger = logging.getLogger(__name__)

def generate_ai_summary(note):
    logger.debug(f"Starting generate_ai_summary for note ID {getattr(note, 'id', 'unknown')}")
    if not isinstance(note, SOAPNote):
        logger.error(f"Invalid input: note must be a SOAPNote object, got {type(note)}")
        return "Summary unavailable"
    try:
        summary_parts = []
        situation = getattr(note, 'situation', None) or ""
        demographic_pattern = r"^(male|female|man|woman|boy|girl),\s*(.*)"
        match = re.match(demographic_pattern, situation.lower().strip())
        if match:
            _, symptoms = match.groups()
            summary_parts.append(f"Chief Complaint: {symptoms.strip()}.")
        else:
            summary_parts.append(f"Chief Complaint: {situation.lower().strip()}.")
        hpi = getattr(note, 'hpi', None)
        if hpi:
            summary_parts.append(f"HPI: {hpi.strip()}")
        medication_history = getattr(note, 'medication_history', None)
        if medication_history:
            summary_parts.append(f"Medications: {medication_history.strip()}")
        assessment = getattr(note, 'assessment', None)
        if assessment:
            primary_dx = re.search(r"Primary Assessment: (.*?)(?:\.|$)", assessment, re.DOTALL)
            if primary_dx:
                summary_parts.append(f"Primary Diagnosis: {primary_dx.group(1).strip()}")
            else:
                summary_parts.append(f"Assessment: {assessment.strip()}")
        if not summary_parts:
            logger.warning("No valid fields found for summary")
            return "Summary unavailable"
        summary = "\n".join(summary_parts)
        logger.debug(f"Generated summary: {summary}")
        return summary
    except Exception as e:
        logger.error(f"Error in generate_ai_summary for note ID {getattr(note, 'id', 'unknown')}: {str(e)}")
        return "Summary unavailable"