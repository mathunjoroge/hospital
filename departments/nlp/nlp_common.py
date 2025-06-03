# /home/mathu/projects/hospital/departments/nlp/nlp_common.py
import re
from departments.nlp.logging_setup import get_logger

logger = get_logger(__name__)

# Fallback dictionary for symptoms not in UMLS
FALLBACK_CUI_MAP = {
    "fever": {"umls_cui": "C0018682", "semantic_type": "Sign or Symptom"},
    "fevers": {"umls_cui": "C0018682", "semantic_type": "Sign or Symptom"},
    "pyrexia": {"umls_cui": "C0018682", "semantic_type": "Sign or Symptom"},
    "chills": {"umls_cui": "C0085593", "semantic_type": "Sign or Symptom"},
    "shivering": {"umls_cui": "C0085593", "semantic_type": "Sign or Symptom"},
    "nausea": {"umls_cui": "C0027497", "semantic_type": "Sign or Symptom"},
    "queasiness": {"umls_cui": "C0027497", "semantic_type": "Sign or Symptom"},
    "vomiting": {"umls_cui": "C0042963", "semantic_type": "Sign or Symptom"},
    "loss of appetite": {"umls_cui": "C0234450", "semantic_type": "Sign or Symptom"},
    "anorexia": {"umls_cui": "C0234450", "semantic_type": "Sign or Symptom"},
    "decreased appetite": {"umls_cui": "C0234450", "semantic_type": "Sign or Symptom"},
    "jaundice": {"umls_cui": "C0022346", "semantic_type": "Sign or Symptom"},
    "jaundice in eyes": {"umls_cui": "C0022346", "semantic_type": "Sign or Symptom"},
    "icterus": {"umls_cui": "C0022346", "semantic_type": "Sign or Symptom"},
    "headache": {"umls_cui": "C0018681", "semantic_type": "Sign or Symptom"}
}

def clean_term(term: str) -> str:
    """Clean a medical term for processing."""
    if not isinstance(term, str) or len(term) > 200:
        logger.debug(f"Invalid term for cleaning: {term}")
        return ""
    term = term.strip().lower()
    term = re.sub(r'[^\w\s]', '', term)
    term = ' '.join(term.split())
    return term