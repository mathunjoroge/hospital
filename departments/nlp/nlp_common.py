import re
import os
import json
import time
from typing import Dict, List, Optional
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from departments.nlp.logging_setup import get_logger
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from config import (
    CACHE_DIR,
    MONGO_URI,
    DB_NAME,
    SYMPTOMS_COLLECTION
)

logger = get_logger(__name__)
STOP_TERMS = {
    # your stop terms here
    'the', 'and', 'or', 'is', 'started', 'days', 'weeks', 'months', 'years', 'ago',
    'taking', 'makes', 'alleviates', 'foods', 'fats', 'strong', 'tea', 'licking', 'salt', 'worse'
}

# Expanded fallback map for symptoms with UMLS CUIs and semantic types
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
    "headache": {"umls_cui": "C0018681", "semantic_type": "Sign or Symptom"},
    "cough": {"umls_cui": "C0010200", "semantic_type": "Sign or Symptom"},
    "dry cough": {"umls_cui": "C0010200", "semantic_type": "Sign or Symptom"},
    "productive cough": {"umls_cui": "C0242209", "semantic_type": "Sign or Symptom"},
    "shortness of breath": {"umls_cui": "C0013404", "semantic_type": "Sign or Symptom"},
    "dyspnea": {"umls_cui": "C0013404", "semantic_type": "Sign or Symptom"},
    "chest pain": {"umls_cui": "C0008031", "semantic_type": "Sign or Symptom"},
    "thoracic pain": {"umls_cui": "C0008031", "semantic_type": "Sign or Symptom"},
    "fatigue": {"umls_cui": "C0015672", "semantic_type": "Sign or Symptom"},
    "tiredness": {"umls_cui": "C0015672", "semantic_type": "Sign or Symptom"},
    "abdominal pain": {"umls_cui": "C0000737", "semantic_type": "Sign or Symptom"},
    "stomach pain": {"umls_cui": "C0000737", "semantic_type": "Sign or Symptom"},
    "diarrhea": {"umls_cui": "C0011991", "semantic_type": "Sign or Symptom"},
    "loose stools": {"umls_cui": "C0011991", "semantic_type": "Sign or Symptom"},
    "constipation": {"umls_cui": "C0009806", "semantic_type": "Sign or Symptom"},
    "dizziness": {"umls_cui": "C0012833", "semantic_type": "Sign or Symptom"},
    "vertigo": {"umls_cui": "C0012833", "semantic_type": "Sign or Symptom"},
    "back pain": {"umls_cui": "C0004604", "semantic_type": "Sign or Symptom"},
    "lower back pain": {"umls_cui": "C0004604", "semantic_type": "Sign or Symptom"},
    "joint pain": {"umls_cui": "C0003862", "semantic_type": "Sign or Symptom"},
    "arthralgia": {"umls_cui": "C0003862", "semantic_type": "Sign or Symptom"},
    "sore throat": {"umls_cui": "C0242209", "semantic_type": "Sign or Symptom"},
    "pharyngitis": {"umls_cui": "C0242209", "semantic_type": "Sign or Symptom"},
    "rash": {"umls_cui": "C0015230", "semantic_type": "Sign or Symptom"},
    "skin eruption": {"umls_cui": "C0015230", "semantic_type": "Sign or Symptom"},
    "palpitations": {"umls_cui": "C0030252", "semantic_type": "Sign or Symptom"},
    "facial pain": {"umls_cui": "C0234451", "semantic_type": "Sign or Symptom"},
    "nasal congestion": {"umls_cui": "C0027428", "semantic_type": "Sign or Symptom"},
    "purulent nasal discharge": {"umls_cui": "C0234680", "semantic_type": "Sign or Symptom"},
    "photophobia": {"umls_cui": "C0085636", "semantic_type": "Sign or Symptom"},
    "neck stiffness": {"umls_cui": "C0029100", "semantic_type": "Sign or Symptom"},
    "epigastric pain": {"umls_cui": "C0234453", "semantic_type": "Sign or Symptom"},
    "pain on movement": {"umls_cui": "C0234454", "semantic_type": "Sign or Symptom"},
    "heart palpitations": {"umls_cui": "C0030252", "semantic_type": "Sign or Symptom"},
    "sinusitis": {"umls_cui": "C0037195", "semantic_type": "Disease or Syndrome"},
    "diabetes": {"umls_cui": "C0011849", "semantic_type": "Disease or Syndrome"}
}

def is_negated_term(term: str) -> bool:
    """Detect if a term contains negation words."""
    if not term or not isinstance(term, str):
        return False
    negation_words = {'no', 'not', 'none', 'never', 'without', 'denies', 'negative'}
    term_words = term.lower().split()
    return any(word in negation_words for word in term_words)

def clean_term(term: str) -> str:
    """Clean a medical term for processing.

    - Strips leading/trailing whitespace.
    - Converts to lowercase.
    - Removes non-alphanumeric characters except whitespace.
    - Collapses multiple spaces to a single space.
    - Filters out stop words and negated terms.
    - Returns an empty string if input is invalid, too long, or negated.
    """
    if not isinstance(term, str):
        logger.warning(f"Invalid term for cleaning (not a string): term={term}, type={type(term)}")
        return ""
    term = term.strip()
    if not term:
        logger.warning(f"Empty term after stripping whitespace: original='{term}'")
        return ""
    if len(term) > 100:  # Align with SymptomTracker's limit
        logger.warning(f"Invalid term for cleaning (too long): term={term[:50]}..., length={len(term)}")
        return ""
    try:
        term_lower = term.lower()
        term_clean = re.sub(r'[^\w\s]', '', term_lower)
        term_clean = ' '.join(term_clean.split())
        # Check for stop words or negation
        if term_clean in STOP_TERMS or is_negated_term(term_clean):
            logger.debug(f"Filtered term as stop word or negated: {term_clean}")
            return ""
        logger.debug(f"Cleaned term: {term_clean}")
        return term_clean
    except Exception as e:
        logger.error(f"Error cleaning term '{term}': {e}")
        return ""

@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(2),
    retry=retry_if_exception_type(ConnectionFailure)
)
def initialize_fallback_map():
    """Populate FALLBACK_CUI_MAP from SYMPTOMS_COLLECTION."""
    global FALLBACK_CUI_MAP
    try:
        client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=10000,
            socketTimeoutMS=10000,
           
        )
        client.admin.command('ping')
        db = client[DB_NAME]
        collection = db[SYMPTOMS_COLLECTION]
        cursor = collection.find({"cui": {"$exists": True, "$ne": None}})
        new_entries = {}
        for doc in cursor:
            symptom = doc.get('symptom', '')
            if not symptom or not isinstance(symptom, str):
                logger.warning(f"Skipping MongoDB document with invalid symptom: {doc}")
                continue
            cleaned_symptom = clean_term(symptom)
            if not cleaned_symptom:
                logger.warning(f"Skipping empty symptom after cleaning: {doc.get('symptom', 'unknown')}")
                continue
            cui = doc.get('cui')
            if not cui or not isinstance(cui, str) or not cui.startswith('C'):
                logger.warning(f"Skipping document with invalid CUI: {cui}")
                continue
            semantic_type = doc.get('semantic_type', '')
            if not isinstance(semantic_type, str):
                logger.warning(f"Invalid semantic_type for symptom '{cleaned_symptom}': {semantic_type}, using 'Unknown'")
                semantic_type = 'Unknown'
            new_entries[cleaned_symptom] = {
                'umls_cui': cui,
                'semantic_type': semantic_type
            }
            logger.debug(f"Added symptom '{cleaned_symptom}' to fallback map: CUI={cui}, semantic_type={semantic_type}")
        # Merge new entries, preserving static ones
        for symptom, new_entry in new_entries.items():
            if symptom not in FALLBACK_CUI_MAP:
                FALLBACK_CUI_MAP[symptom] = new_entry
            else:
                logger.debug(f"Preserving static FALLBACK_CUI_MAP entry for '{symptom}'")
        logger.info(f"Initialized FALLBACK_CUI_MAP with {len(FALLBACK_CUI_MAP)} entries (added {len(new_entries)} from MongoDB)")
        save_fallback_map()
        client.close()
    except ConnectionFailure as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize fallback map from MongoDB: {e}")
        raise

def save_fallback_map():
    """Save FALLBACK_CUI_MAP to cache file."""
    if not CACHE_DIR:
        logger.error("CACHE_DIR not configured, skipping fallback map save")
        return
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_file = os.path.join(CACHE_DIR, "fallback_map.json")
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(FALLBACK_CUI_MAP, f, indent=2)
        logger.debug(f"Successfully saved FALLBACK_CUI_MAP to {cache_file}")
    except OSError as e:
        logger.error(f"Failed to save fallback map due to file error: {e}")
    except Exception as e:
        logger.error(f"Failed to save fallback map to {cache_file}: {e}")

def load_fallback_map():
    """Load FALLBACK_CUI_MAP from cache file."""
    global FALLBACK_CUI_MAP
    if not CACHE_DIR:
        logger.error("CACHE_DIR not configured, skipping fallback map load")
        return
    cache_file = os.path.join(CACHE_DIR, "fallback_map.json")
    try:
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    logger.error(f"Invalid format in {cache_file}: expected dict, got {type(data)}")
                    return
                for symptom, entry in data.items():
                    if 'umls_cui' not in entry or 'semantic_type' not in entry:
                        logger.warning(f"Skipping invalid entry in cache for '{symptom}': missing required fields")
                        continue
                    if not isinstance(entry['umls_cui'], str) or not entry['umls_cui'].startswith('C'):
                        logger.warning(f"Skipping invalid CUI in cache for '{symptom}': {entry['umls_cui']}")
                        continue
                    if symptom not in FALLBACK_CUI_MAP:  # Preserve static entries
                        FALLBACK_CUI_MAP[symptom] = entry
                logger.info(f"Loaded {len(FALLBACK_CUI_MAP)} entries from {cache_file}")
        else:
            logger.warning(f"Cache file {cache_file} does not exist")
    except OSError as e:
        logger.error(f"Failed to load fallback map due to file error: {e}")
    except Exception as e:
        logger.error(f"Failed to load fallback map from {cache_file}: {e}")

# Initialize fallback map at module load
try:
    load_fallback_map()
    initialize_fallback_map()
except Exception as e:
    logger.error(f"Failed to initialize module: {e}")
    raise

# Example usage
if __name__ == "__main__":
    test_terms = ["fever", "headache", "back pain", "cough", "", "dizziness", "no cough", "sinusitis", "diabetes"]
    from departments.nlp.nlp_pipeline import search_local_umls_cui
    results = search_local_umls_cui(test_terms)
    for term, cui in results.items():
        if cui:
            logger.info(f"Term: {term}, CUI: {cui}")
            print(f"Term: {term}, CUI: {cui}")
        else:
            logger.warning(f"Term: {term}, No CUI found")
            print(f"Term: {term}, No CUI found")