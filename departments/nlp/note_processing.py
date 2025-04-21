from departments.models.medicine import SOAPNote
from departments.nlp.logging_setup import logger

def build_note_text(note: SOAPNote) -> str:
    """Construct full note text."""
    if not isinstance(note, SOAPNote):
        logger.error(f"Invalid note type: {type(note)}")
        return ""
    sections = [
        ("Chief Complaint", note.situation),
        ("HPI", note.hpi),
        ("Medical History", note.medical_history),
        ("Medications", note.medication_history),
        ("Assessment", note.assessment),
        ("Plan", note.recommendation),
        ("Additional Notes", note.additional_notes)
    ]
    return "\n".join(f"{k}: {v}" for k, v in sections if v)

def generate_ai_summary(note: SOAPNote) -> str:
    """Generate concise clinical summary."""
    if not isinstance(note, SOAPNote):
        logger.error(f"Invalid note type: {type(note)}")
        return "Summary unavailable"
    try:
        sections = [
            ("Chief Complaint", note.situation),
            ("HPI", note.hpi),
            ("Medications", note.medication_history),
            ("Assessment", note.assessment)
        ]
        selected = []
        for key, value in sections:
            if value:
                prefix = key + ": "
                if key == "Chief Complaint" and not value.startswith("Patient"):
                    value = "Patient experiencing " + value
                selected.append(f"{prefix}{value}")
        return "\n".join(selected)
    except Exception as e:
        logger.error(f"Summary failed for note {note.id}: {str(e)}")
        return "Summary unavailable"