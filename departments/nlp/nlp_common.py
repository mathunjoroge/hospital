# /home/mathu/projects/hospital/departments/nlp/nlp_common.py
import re
import os
import json
import time
from typing import Dict, List, Optional
import psycopg2
from psycopg2 import sql, pool
from pymongo import MongoClient
from departments.nlp.logging_setup import get_logger
from config import (
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_DB,
    POSTGRES_USER,
    POSTGRES_PASSWORD,
    CACHE_DIR,
    BATCH_SIZE,
    MONGO_URI,
    DB_NAME,
    SYMPTOMS_COLLECTION
)

logger = get_logger(__name__)

# Initialize PostgreSQL connection pool
try:
    db_pool = psycopg2.pool.SimpleConnectionPool(
        1, 20,
        host=POSTGRES_HOST,
        database=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        port=POSTGRES_PORT
    )
    logger.debug("PostgreSQL connection pool initialized")
except Exception as e:
    logger.error(f"Failed to initialize PostgreSQL connection pool: {e}")
    raise

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
        # New entries from warnings
    "facial pain": {"umls_cui": "C0234450", "semantic_type": "Sign or Symptom"},
    "nasal congestion": {"umls_cui": "C0027424", "semantic_type": "Sign or Symptom"},
    "purulent nasal discharge": {"umls_cui": "C0242209", "semantic_type": "Sign or Symptom"},
    "photophobia": {"umls_cui": "C0085636", "semantic_type": "Sign or Symptom"},
    "neck stiffness": {"umls_cui": "C0029101", "semantic_type": "Sign or Symptom"},
    "epigastric pain": {"umls_cui": "C0234451", "semantic_type": "Sign or Symptom"},
    "pain on movement": {"umls_cui": "C0234452", "semantic_type": "Sign or Symptom"},
    "heart palpitations": {"umls_cui": "C0030252", "semantic_type": "Sign or Symptom"}
}

# In-memory cache for CUIs
_cui_cache = {}

def clean_term(term: str) -> str:
    """Clean a medical term for processing."""
    if not isinstance(term, str) or len(term) > 200:
        logger.debug(f"Invalid term for cleaning: {term}")
        return ""
    term = term.strip().lower()
    term = re.sub(r'[^\w\s]', '', term)
    term = ' '.join(term.split())
    return term

def initialize_fallback_map():
    """Populate FALLBACK_CUI_MAP from SYMPTOMS_COLLECTION."""
    global FALLBACK_CUI_MAP
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[SYMPTOMS_COLLECTION]
        cursor = collection.find({"cui": {"$exists": True, "$ne": None}})
        for doc in cursor:
            symptom = clean_term(doc.get('symptom', '')).lower()
            cui = doc.get('cui')
            semantic_type = doc.get('semantic_type', 'Unknown')
            if symptom and cui and cui.startswith('C'):
                FALLBACK_CUI_MAP[symptom] = {
                    'umls_cui': cui,
                    'semantic_type': semantic_type
                }
        logger.info(f"Initialized FALLBACK_CUI_MAP with {len(FALLBACK_CUI_MAP)} symptoms from MongoDB")
        save_fallback_map()
        client.close()
    except Exception as e:
        logger.error(f"Failed to initialize fallback map from MongoDB: {e}")

def save_fallback_map():
    """Save FALLBACK_CUI_MAP to cache file."""
    if not CACHE_DIR:
        logger.debug("CACHE_DIR not set, skipping fallback map save")
        return
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, "fallback_map.json")
    try:
        with open(cache_file, "w") as f:
            json.dump(FALLBACK_CUI_MAP, f)
        logger.debug(f"Saved FALLBACK_CUI_MAP to {cache_file}")
    except Exception as e:
        logger.error(f"Failed to save fallback map: {e}")

def load_fallback_map():
    """Load FALLBACK_CUI_MAP from cache file."""
    global FALLBACK_CUI_MAP
    if not CACHE_DIR:
        logger.debug("CACHE_DIR not set, skipping fallback map load")
        return
    cache_file = os.path.join(CACHE_DIR, "fallback_map.json")
    try:
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                cached = json.load(f)
                FALLBACK_CUI_MAP.update(cached)
            logger.info(f"Loaded {len(FALLBACK_CUI_MAP)} entries from {cache_file}")
    except Exception as e:
        logger.error(f"Failed to load fallback map: {e}")

def save_cui_cache():
    """Save the CUI cache to a file in CACHE_DIR."""
    if not CACHE_DIR:
        logger.debug("CACHE_DIR not set, skipping cache save")
        return
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, "cui_cache.json")
    try:
        with open(cache_file, "w") as f:
            json.dump(_cui_cache, f)
        logger.debug(f"Saved CUI cache to {cache_file}")
    except Exception as e:
        logger.error(f"Failed to save CUI cache: {e}")

def load_cui_cache():
    """Load the CUI cache from a file in CACHE_DIR."""
    if not CACHE_DIR:
        logger.debug("CACHE_DIR not set, skipping cache load")
        return
    cache_file = os.path.join(CACHE_DIR, "cui_cache.json")
    try:
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                _cui_cache.update(json.load(f))
            logger.debug(f"Loaded CUI cache from {cache_file}")
    except Exception as e:
        logger.error(f"Failed to load CUI cache: {e}")

def get_cui_from_db(terms: List[str], max_attempts: int = 3, max_tsquery_bytes: int = 1000000) -> Dict[str, Optional[Dict[str, str]]]:
    """
    Query the umls.MRCONSO table for CUIs of a list of terms, with exact match and full-text search.
    Joins with umls.MRSTY for semantic types if available. Falls back to FALLBACK_CUI_MAP if no match is found.
    """
    start_time = time.time()
    results = {term: None for term in terms}
    terms_to_query = []
    oversized_count = 0

    # Clean terms and check cache/fallback
    for term in terms:
        cleaned_term = clean_term(term)
        if not cleaned_term:
            logger.warning(f"Invalid term after cleaning: {term}")
            continue
        if cleaned_term in _cui_cache:
            logger.debug(f"Found term '{cleaned_term}' in cache")
            results[term] = {
                "umls_cui": _cui_cache[cleaned_term],
                "semantic_type": FALLBACK_CUI_MAP.get(cleaned_term, {}).get("semantic_type", "Unknown")
            }
        elif cleaned_term in FALLBACK_CUI_MAP:
            logger.debug(f"Found term '{cleaned_term}' in fallback map")
            results[term] = FALLBACK_CUI_MAP[cleaned_term]
            _cui_cache[cleaned_term] = FALLBACK_CUI_MAP[cleaned_term]["umls_cui"]
        else:
            terms_to_query.append(cleaned_term)

    if not terms_to_query:
        logger.debug("All terms resolved from cache or fallback")
        save_cui_cache()
        return results

    # Database connection from pool
    conn = None
    cursor = None
    try:
        conn = db_pool.getconn()
        cursor = conn.cursor()

        for attempt in range(max_attempts):
            try:
                # Exact match query
                query_start = time.time()
                query = sql.SQL("""
                    SELECT DISTINCT c.STR, c.CUI, s.STY
                    FROM umls.MRCONSO c
                    LEFT JOIN umls.MRSTY s ON c.CUI = s.CUI
                    WHERE c.SAB = 'SNOMEDCT'
                    AND LOWER(c.STR) IN %s
                    AND c.SUPPRESS = 'N'
                """)
                cursor.execute(query, (tuple(terms_to_query),))
                for row in cursor.fetchall():
                    term = clean_term(row[0])  # STR
                    if term:
                        results[term] = {
                            "umls_cui": row[1],  # CUI
                            "semantic_type": row[2] or FALLBACK_CUI_MAP.get(term, {}).get("semantic_type", "Unknown")  # STY or fallback
                        }
                        _cui_cache[term] = row[1]
                        FALLBACK_CUI_MAP[term] = results[term]  # Update fallback map
                logger.debug(f"Exact match query for {len(terms_to_query)} terms took {time.time() - query_start:.2f} seconds")

                # Full-text search for remaining terms
                remaining = [t for t in terms_to_query if results[t] is None]
                for i in range(0, len(remaining), BATCH_SIZE):
                    batch_terms = remaining[i:i + BATCH_SIZE]
                    tsquery_parts = []
                    current_size = 0
                    batch_oversized = 0

                    for t in batch_terms:
                        byte_len = len(t.encode('utf-8'))
                        if byte_len > max_tsquery_bytes:
                            batch_oversized += 1
                            oversized_count += 1
                            logger.warning(f"Skipping oversized term: {t[:50]}... ({byte_len} bytes)")
                            continue
                        part = f"{' & '.join(t.split())}:*"
                        part_size = len(part.encode('utf-8'))
                        if current_size + part_size + len(' | ') > max_tsquery_bytes:
                            break
                        tsquery_parts.append(part)
                        current_size += part_size

                    if batch_oversized:
                        logger.warning(f"Skipped {batch_oversized} oversized terms in batch {i//BATCH_SIZE + 1}")

                    if not tsquery_parts:
                        logger.warning(f"Skipping batch {i//BATCH_SIZE + 1} due to tsquery size limit")
                        continue

                    tsquery = ' | '.join(tsquery_parts)
                    query_start = time.time()
                    query = sql.SQL("""
                        SELECT DISTINCT c.STR, c.CUI, s.STY
                        FROM umls.MRCONSO c
                        LEFT JOIN umls.MRSTY s ON c.CUI = s.CUI
                        WHERE c.SAB = 'SNOMEDCT'
                        AND to_tsvector('english', c.STR) @@ to_tsquery('english', %s)
                        AND c.SUPPRESS = 'N'
                        AND octet_length(c.STR) <= 1048575
                    """)
                    try:
                        cursor.execute(query, (tsquery,))
                        for row in cursor.fetchall():
                            term = clean_term(row[0])
                            if term:
                                results[term] = {
                                    "umls_cui": row[1],
                                    "semantic_type": row[2] or FALLBACK_CUI_MAP.get(term, {}).get("semantic_type", "Unknown")
                                }
                                _cui_cache[term] = row[1]
                                FALLBACK_CUI_MAP[term] = results[term]  # Update fallback map
                        logger.debug(f"Full-text search for {len(tsquery_parts)} terms took {time.time() - query_start:.3f} seconds")
                    except Exception as e:
                        if "string is too long for tsvector" in str(e):
                            logger.error(f"TSVector overflow despite filters: {e}")
                        else:
                            raise

                conn.commit()
                save_fallback_map()
                break
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{max_attempts} failed: {e}")
                conn.rollback()
                if attempt == max_attempts - 1:
                    logger.error("Max retries reached for UMLS query")
                    break
    except Exception as e:
        logger.error(f"Failed to get database connection: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            db_pool.putconn(conn)
        logger.debug("Returned PostgreSQL connection to pool")

    save_cui_cache()
    if oversized_count:
        logger.warning(f"Skipped {oversized_count} oversized terms during processing")
    logger.info(f"get_cui_from_db processed {len(terms)} terms, took {time.time() - start_time:.3f} seconds")
    return results

# Initialize fallback map and cache at module load
load_fallback_map()
initialize_fallback_map()
load_cui_cache()

# Example usage
if __name__ == "__main__":
    test_terms = ["fever", "headache", "back pain", "cough", "dizziness"]
    results = get_cui_from_db(test_terms)
    for term, result in results.items():
        if result:
            print(f"Term: {term}, CUI: {result['umls_cui']}, Semantic Type: {result['semantic_type']}")
        else:
            print(f"Term: {term}, No CUI found")