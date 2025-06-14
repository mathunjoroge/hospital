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
    if any(term in text for term in ['severe', '8/10', '9/10', '10/10', 'intense']):
        return "Severe"
    elif any(term in text for term in ['moderate', '5/10', '6/10', '7/10']):
        return "Moderate"
    return "Mild"

def extract_location(text: str, symptom: Optional[str] = None) -> str:
    """Extract symptom location from clinical text, optionally using a specific symptom."""
    if not isinstance(text, str):
        logger.error(f"Invalid text type for location: {type(text)}")
        return "Unspecified"
    text = text.lower().strip()
    if symptom and not isinstance(symptom, str):
        logger.warning(f"Invalid symptom type: {type(symptom)}, ignoring symptom")
        symptom = None
    symptom = symptom.lower().strip() if symptom else None

    logger.debug(f"Extracting location for text: {text[:50]}..., symptom: {symptom}")

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
        'fever': 'Systemic',
    }

    # Prioritize symptom-specific mapping if symptom is provided
    if symptom and symptom in symptom_specific:
        logger.debug(f"Matched symptom-specific location: {symptom_specific[symptom]} for symptom: {symptom}")
        return symptom_specific[symptom]

    # Fallback to text-based mapping
    for term, loc in symptom_specific.items():
        if term in text:
            logger.debug(f"Matched text-based location: {loc} for term: {term}")
            return loc

    # General location keywords
    locations = [
        'head', 'chest', 'abdomen', 'back', 'extremity', 'joint', 'neck',
        'hand', 'arm', 'leg', 'knee', 'ankle', 'foot', 'face', 'eyes',
        'cheeks', 'flank', 'epigastric', 'bilateral', 'skin'
    ]
    found = [loc.capitalize() for loc in locations if loc in text]
    result = ", ".join(found) or "Unspecified"
    logger.debug(f"Location result: {result}")
    return result

def extract_aggravating_alleviating(text: str, factor: str) -> str:
    """Extract aggravating or alleviating factors from clinical text."""
    if not isinstance(text, str):
        logger.error(f"Invalid text type for {factor}: {type(text)}")
        return "Unknown"
    if not text.strip():
        logger.debug(f"Empty text provided for {factor} extraction")
        return "Unknown"

    text = text.lower().strip()
    logger.debug(f"Extracting {factor} factors from text: {text[:50]}...")

    if factor == "aggravating":
        patterns = [
            r'(?:(aggravat|worse|exacerbat|trigger)\s+(?:by|with|on|after|during)\s+[\w\s,-]+?)(?=|,|\s*(?:and|or|$))',
            r'(?:worsen(?:s|ed|ing)?\s+(?:by|with|on|after|during)\s+[\w\s,-]+?)(?=|,|\s*(?:and|or|$))',
        ]
    else:
        patterns = [
            r'(?:(alleviat|better|improv|reliev)\s+(?:by|with|after|during)\s+[\w\s,-]+?)(?=|,|\s*(?:and|or|$))',
            r'(?:relief\s+(?:from|by|with|after|during)\s+[\w\s,-]+?)(?=|,|\s*(?:and|or|$))',
        ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            result = match.group(0)
            for prep in ['by', 'with', 'on', 'after', 'during', 'from']:
                if prep in result:
                    result = result.split(prep)[-1].strip()
                    break
            result = re.sub(r'\s+', ' ', result.replace(' and ', ', ')).strip(',.')
            logger.debug(f"Matched {factor} factor: {result}")
            return result if result else "Unknown"

    if len(text.split()) <= 5 and text:
        cleaned = re.sub(r'[^\w\s,]', '', text).strip()
        logger.debug(f"No pattern matched for {factor}, using cleaned text: {cleaned}")
        return cleaned if cleaned else "Unknown"

    logger.debug(f"No {factor} factors found in text")
    return "Unknown"