from typing import Optional
import re
from datetime import datetime
from departments.nlp.logging_setup import get_logger
logger = get_logger()

def calculate_age(dob: datetime.date) -> Optional[int]:
    """Calculate age from date of birth."""
    if not isinstance(dob, datetime.date):
        logger.warning(f"Invalid dob type: {type(dob)}")
        return None
    today = datetime.now().date()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

def extract_duration(text: str) -> str:
    """Extract duration from clinical text."""
    if not isinstance(text, str):
        logger.error(f"Invalid text type for duration: {type(text)}")
        return "Unknown"
    match = re.search(r'(\d+\s*(day|week|month|year)s?)', text.lower())
    return match.group(0) if match else "Unknown"

def classify_severity(text: str) -> str:
    """Classify symptom severity."""
    if not isinstance(text, str):
        logger.error(f"Invalid text type for severity: {type(text)}")
        return "Mild"
    text = text.lower()
    if 'severe' in text or '8/10' in text:
        return "Severe"
    elif 'moderate' in text:
        return "Moderate"
    return "Mild"

def extract_location(text: str) -> str:
    """Extract symptom location."""
    if not isinstance(text, str):
        logger.error(f"Invalid text type for location: {type(text)}")
        return "Unspecified"
    symptom_specific = {
        'headache': 'Head',
        'photophobia': 'Head',
        'chest pain': 'Chest',
        'shortness of breath': 'Chest',
        'epigastric pain': 'Abdomen',
        'nausea': 'Abdomen',
        'knee pain': 'Knee',
        'swelling': 'Knee',
        'wheezing': 'Chest',
        'cough': 'Chest',
        'rash': 'Skin',
        'back pain': 'Back',
        'diarrhea': 'Abdomen',
        'cramping': 'Abdomen',
        'constipation': 'Abdomen',
        'abdominal discomfort': 'Abdomen',
        'fatigue': 'Generalized',
        'weakness': 'Generalized',
        'fatigue': 'Systemic',
        'headache': 'Head',
        'chest pain': 'Chest',
        'abdominal pain': 'Abdomen',
        'fever': 'Systemic',
    }
    text = text.lower()
    for term, loc in symptom_specific.items():
        if term in text:
            return loc
    locations = [
        'head', 'chest', 'abdomen', 'back', 'extremity', 'joint', 'neck',
        'hand', 'arm', 'leg', 'knee', 'ankle', 'foot', 'face', 'eyes',
        'cheeks', 'flank', 'epigastric', 'bilateral', 'skin'
    ]
    found = [loc.capitalize() for loc in locations if loc in text]
    return ", ".join(found) or "Unspecified"

def extract_aggravating_alleviating(text: str, factor: str) -> str:
    """Extract aggravating or alleviating factors."""
    if not isinstance(text, str):
        logger.error(f"Invalid text type for {factor}: {type(text)}")
        return "Unknown"
    text = text.lower()
    if factor == "aggravating":
        match = re.search(r'(?:(aggravat|worse)\s+(?:by|with|on|after)\s+[\w\s,]+)', text)
    else:
        match = re.search(r'(?:(alleviat|better)\s+(?:by|with|after)\s+[\w\s,]+)', text)
    if match:
        result = match.group(0).split('by')[-1].strip() if 'by' in match.group(0) else match.group(0).split('with')[-1].strip()
        return result.replace(' and ', ', ')
    return "Unknown"