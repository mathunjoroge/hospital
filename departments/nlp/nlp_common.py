# /home/mathu/projects/hospital/departments/nlp/nlp_common.py
import re
import os
import json
import time
from typing import Dict, List, Optional
import psycopg2
from psycopg2 import sql, pool
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from departments.nlp.logging_setup import get_logger
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
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
    "facial pain": {"umls_cui": "C0234451", "semantic_type": "Sign or Symptom"},
    "nasal congestion": {"umls_cui": "C0027428", "semantic_type": "Sign or Symptom"},
    "purulent nasal discharge": {"umls_cui": "C0234680", "semantic_type": "Sign or Symptom"},
    "photophobia": {"umls_cui": "C0085636", "semantic_type": "Sign or Symptom"},
    "neck stiffness": {"umls_cui": "C0029100", "semantic_type": "Sign or Symptom"},
    "epigastric pain": {"umls_cui": "C0234453", "semantic_type": "Sign or Symptom"},
    "pain on movement": {"umls_cui": "C0234454", "semantic_type": "Sign or Symptom"},
    "heart palpitations": {"umls_cui": "C0030252", "semantic_type": "Sign or Symptom"}
}

# In-memory cache for CUIs
_cui_cache = {}

def clean_term(term: str) -> str:
    """Clean a medical term for processing.

    - Strips leading/trailing whitespace.
    - Converts to lowercase.
    - Removes non-alphanumeric characters except whitespace.
    - Collapses multiple spaces to a single space.
    - Returns an empty string if input is not a string or is too long.
    """
    if not isinstance(term, str):
        logger.warning(f"Invalid term for cleaning (not a string): term={term}, type={type(term)}")
        return ""
    term = term.strip()
    if not term:
        logger.warning("Empty term after stripping whitespace.")
        return ""
    if len(term) > 200:
        logger.warning(f"Invalid term for cleaning (too long): term={term[:50]}..., length={len(term)}")
        return ""
    try:
        term = term.lower()
        term = re.sub(r'[^\w\s]', '', term)
        term = ' '.join(term.split())
        logger.debug(f"Cleaned term: {term}")
        return term
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
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
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
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if not isinstance(data, dict):
                        logger.error(f"Invalid format in {cache_file}: expected dict, got {type(data)}")
                        return
                    for symptom, entry in data.items():
                        if 'umls_cui' not in entry or 'semantic_type' not in entry:
                            logger.warning(f"Skipping invalid entry in cache for '{symptom}': missing required fields")
                            continue
                        FALLBACK_CUI_MAP[symptom] = entry
                    logger.info(f"Loaded {len(FALLBACK_CUI_MAP)} entries from {cache_file}")
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON format in {cache_file}")
        else:
            logger.warning(f"Cache file {cache_file} does not exist")
    except OSError as e:
        logger.error(f"Failed to load fallback map due to file error: {e}")
    except Exception as e:
        logger.error(f"Failed to load fallback map from {cache_file}: {e}")

def save_cui_cache():
    """Save the CUI cache to a file in CACHE_DIR."""
    if not CACHE_DIR:
        logger.error("CACHE_DIR not configured, skipping cui cache save")
        return
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_file = os.path.join(CACHE_DIR, "cui_cache.json")
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(_cui_cache, f, indent=2)
        logger.debug(f"Successfully saved CUI cache to {cache_file}")
    except OSError as e:
        logger.error(f"Failed to save CUI cache due to file error: {e}")
    except Exception as e:
        logger.error(f"Failed to save CUI cache to {cache_file}: {e}")

def load_cui_cache():
    """Load the CUI cache from a file in CACHE_DIR."""
    if not CACHE_DIR:
        logger.error("CACHE_DIR not configured, skipping cui cache load")
        return
    cache_file = os.path.join(CACHE_DIR, "cui_cache.json")
    try:
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if not isinstance(data, dict):
                        logger.error(f"Invalid data in {cache_file}: expected dict, got {type(data)}")
                        return
                    _cui_cache.update(data)
                    logger.info(f"Loaded {len(_cui_cache)} entries from {cache_file}")
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON format in {cache_file}")
        else:
            logger.warning(f"Cache file {cache_file} does not exist")
    except OSError as e:
        logger.error(f"Failed to load CUI cache due to file error: {e}")
    except Exception as e:
        logger.error(f"Failed to load CUI cache from {cache_file}: {e}")

@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(2),
    retry=retry_if_exception_type(psycopg2.OperationalError)
)
def get_cui_from_db(terms: List[str], max_attempts: int = 3, max_tsquery_bytes: int = 1000000) -> Dict[str, Optional[Dict[str, str]]]:
    """
    Query the umls.MRCONSO table for CUIs of a list of terms, with exact match and full-text search.
    Joins with umls.MRSTY for semantic types if available. Falls back to FALLBACK_CUI_MAP if no match is found.
    """
    start_time = time.time()
    results = {term: None for term in terms}
    terms_to_query = []
    oversized_count = 0

    # Validate and clean terms
    for term in terms:
        cleaned_term = clean_term(term)
        if not cleaned_term:
            logger.warning(f"Skipping invalid term: {term}")
            continue
        if cleaned_term in _cui_cache:
            logger.debug(f"Cache hit for '{cleaned_term}': CUI={_cui_cache[cleaned_term]}")
            results[term] = {
                "umls_cui": _cui_cache[cleaned_term],
                "semantic_type": FALLBACK_CUI_MAP.get(cleaned_term, {}).get('semantic_type', 'Unknown')
            }
        elif cleaned_term in FALLBACK_CUI_MAP:
            logger.debug(f"Fallback hit for '{cleaned_term}': {FALLBACK_CUI_MAP[cleaned_term]}")
            results[term] = FALLBACK_CUI_MAP[cleaned_term]
            _cui_cache[cleaned_term] = FALLBACK_CUI_MAP[cleaned_term]['umls_cui']
            logger.debug(f"Added to cache: {cleaned_term}, CUI={_cui_cache[cleaned_term]}")
        else:
            terms_to_query.append(cleaned_term)
            logger.debug(f"Queueing term for DB query: {cleaned_term}")

    if not terms_to_query:
        logger.info("All terms resolved via cache or fallback")
        save_cui_cache()
        return results

    # Database connection
    conn = None
    cursor = None
    try:
        for conn_attempt in range(max_attempts):
            try:
                conn = db_pool.getconn()
                logger.debug("Acquired PostgreSQL connection from pool")
                cursor = conn.cursor()
                break
            except psycopg2.OperationalError as e:
                logger.error(f"Connection attempt {conn_attempt + 1}/{max_attempts} failed: {e}")
                if conn_attempt == max_attempts - 1:
                    logger.error("Failed to acquire database connection after retries")
                    return results
                time.sleep(1)

        # Batch exact-match queries
        for i in range(0, len(terms_to_query), BATCH_SIZE):
            batch_terms = terms_to_query[i:i + BATCH_SIZE]
            query_start = time.time()
            for attempt in range(max_attempts):
                try:
                    # Exact match query
                    query = sql.SQL("""
                        SELECT DISTINCT c.STR, c.CUI, s.STY
                        FROM umls.MRCONSO c
                        LEFT JOIN umls.MRSTY s ON c.CUI = s.CUI
                        WHERE c.SAB = 'SNOMEDCT'
                        AND c.SUPPRESS = 'N'
                        AND LOWER(c.STR) IN %s
                    """)
                    cursor.execute(query, (tuple(batch_terms),))
                    for row in cursor.fetchall():
                        term = clean_term(row[0])
                        if term:
                            results[term] = {
                                "umls_cui": row[1],
                                "semantic_type": row[2] or FALLBACK_CUI_MAP.get(term, {}).get('semantic_type', 'Unknown')
                            }
                            _cui_cache[term] = row[1]
                            if term not in FALLBACK_CUI_MAP:
                                FALLBACK_CUI_MAP[term] = results[term]
                                logger.debug(f"Updated FALLBACK_CUI_MAP with {term}: {results[term]}")
                    logger.debug(f"Exact match query for batch {i//BATCH_SIZE + 1} ({len(batch_terms)}) took {time.time() - query_start:.2f} seconds")
                    break
                except Exception as e:
                    logger.error(f"Exact match attempt {attempt + 1}/{max_attempts} for batch {i//BATCH_SIZE + 1} failed: {str(e)}")
                    conn.rollback()
                    if attempt == max_attempts - 1:
                        logger.error("Max retries reached for exact match query")
                        continue

            # Full-text search for remaining terms
            remaining = [t for t in batch_terms if results[t] is None]
            if not remaining:
                continue
            tsquery_parts = []
            current_size = 0
            batch_oversized = 0
            for t in remaining:
                byte_len = len(t.encode('utf-8'))
                if byte_len > max_tsquery_bytes:
                    batch_oversized += 1
                    oversized_count += 1
                    logger.warning(f"Skipping oversized term: {t[:50]}... ({byte_len} bytes)")
                    continue
                part = ' & '.join(t.split())
                part += ':*'
                part_size = len(part.encode('utf-8'))
                if current_size + part_size + len(' | ') > max_tsquery_bytes:
                    break
                tsquery_parts.append(part)
                current_size += part_size

            if batch_oversized:
                logger.warning(f"Skipped {batch_oversized} oversized terms for batch {i//BATCH_SIZE + 1}")
            if not tsquery_parts:
                logger.warning(f"Skipping batch {i//BATCH_SIZE + 1} due to tsquery size limit")
                continue

            tsquery = ' | '.join(tsquery_parts)
            query_start = time.time()
            for attempt in range(max_attempts):
                try:
                    query = sql.SQL("""
                        SELECT DISTINCT c.STR, c.CUI, s.STY
                        FROM umls.MRCONSO c
                        LEFT JOIN umls.MRSTY s ON c.CUI = s.CUI
                        WHERE c.SAB = 'SNOMEDCT'
                        AND to_tsvector('english', c.STR) @@ to_tsquery('english', %s)
                        AND c.SUPPRESS = 'N'
                        AND octet_length(c.STR) <= 1048575
                    """)
                    cursor.execute(query, (tsquery,))
                    for row in cursor.fetchall():
                        term = clean_term(row[0])
                        if term:
                            results[term] = {
                                'umls_cui': row[1],
                                "semantic_type": row[2] or FALLBACK_CUI_MAP.get(term, {}).get('semantic_type', 'Unknown')
                            }
                            _cui_cache[term] = row[1]
                            if term not in FALLBACK_CUI_MAP:
                                FALLBACK_CUI_MAP[term] = results[term]
                                logger.debug(f"Updated FALLBACK_CUI_MAP with {term}: {results[term]}")
                    logger.debug(f"Full-text search for {len(tsquery_parts)} terms in batch {i//BATCH_SIZE + 1} took {time.time() - query_start:.2f} seconds")
                    break
                except Exception as e:
                    if "string is too long for tsvector" in str(e):
                        logger.warning(f"TSVector overflow in batch {i//BATCH_SIZE + 1}: {str(e)}")
                        break
                    logger.error(f"Full-text search attempt {attempt + 1}/{max_attempts} for batch {i//BATCH_SIZE + 1} failed: {str(e)}")
                    conn.rollback()
                    if attempt == max_attempts - 1:
                        logger.error("Max retries reached for full-text search")
                        continue

            conn.commit()
            save_fallback_map()

    except Exception as e:
        logger.error(f"Failed to process terms in get_cui_from_db: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            db_pool.putconn(conn)
            logger.debug("Returned PostgreSQL connection to pool")
    try:
        save_cui_cache()
    except Exception as e:
        logger.error(f"Failed to save CUI cache after query: {e}")
    if oversized_count:
        logger.warning(f"Skipped {oversized_count} oversized terms during processing")
    logger.info(f"get_cui_from_db processed {len(terms)} terms in {time.time() - start_time:.2f} seconds")
    return results

# Initialize fallback map and cache at module load
try:
    load_fallback_map()
    initialize_fallback_map()
    load_cui_cache()
except Exception as e:
    logger.error(f"Failed to initialize module: {e}")
    raise

# Example usage
if __name__ == "__main__":
    test_terms = ["fever", "headache", "back pain", "cough", "", "dizziness"]
    results = get_cui_from_db(test_terms)
    for term, result in results.items():
        if result:
            logger.info(f"Term: {term}, CUI: {result['umls_cui']}, Semantic Type: {result['semantic_type']}")
            print(f"Term: {term}, CUI: {result['umls_cui']}, Semantic Type: {result['semantic_type']}")
        else:
            logger.warning(f"Term: {term}, No CUI found")
            print(f"Term: {term}, No CUI found")