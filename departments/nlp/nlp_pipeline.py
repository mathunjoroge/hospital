import spacy
from spacy.language import Language
from medspacy.target_matcher import TargetMatcher, TargetRule
from medspacy.context import ConText
from scispacy.linking import EntityLinker
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
from departments.nlp.logging_setup import get_logger
from departments.nlp.knowledge_base_io import load_knowledge_base, invalidate_cache
from dotenv import load_dotenv
import os
import time
import pickle
from cachetools import LRUCache
import re

# Load environment variables explicitly
load_dotenv()

# Import config after loading .env
from config import (
    MONGO_URI,
    DB_NAME,
    KB_PREFIX,
    SYMPTOMS_COLLECTION,
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_DB,
    POSTGRES_USER,
    POSTGRES_PASSWORD,
    CACHE_DIR
)

logger = get_logger()
_nlp_pipeline = None
_cui_cache = LRUCache(maxsize=10000)  # Thread-safe cache
_cache_file = os.path.join(CACHE_DIR or 'data_cache', "cui_cache.pkl")
_sci_ner = spacy.load("en_core_sci_scibert")

# Load STOP_TERMS from knowledge base
knowledge_base = load_knowledge_base()
STOP_TERMS = knowledge_base.get('medical_stop_words', set())
logger.info(f"Loaded {len(STOP_TERMS)} stop terms from knowledge base")

# Initialize PostgreSQL connection pool
try:
    pool = SimpleConnectionPool(
        minconn=1,
        maxconn=10,
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        cursor_factory=RealDictCursor
    )
    logger.info("Initialized PostgreSQL connection pool")
except Exception as e:
    logger.error(f"Failed to initialize PostgreSQL connection pool: {e}")
    pool = None

def load_cui_cache():
    global _cui_cache
    if os.path.exists(_cache_file):
        try:
            with open(_cache_file, 'rb') as f:
                cached = pickle.load(f)
                _cui_cache.update(cached)
            logger.info(f"Loaded CUI cache with {len(_cui_cache)} entries from {_cache_file}")
        except Exception as e:
            logger.warning(f"Failed to load CUI cache: {e}")

def save_cui_cache():
    try:
        os.makedirs(os.path.dirname(_cache_file), exist_ok=True)
        start_time = time.time()
        with open(_cache_file, 'wb') as f:
            pickle.dump(dict(_cui_cache), f)
        logger.debug(f"Saved CUI cache to {_cache_file}, took {time.time() - start_time:.3f} seconds")
    except Exception as e:
        logger.warning(f"Failed to save CUI cache: {e}")

def get_postgres_connection():
    if pool is None:
        logger.error("Connection pool not initialized")
        return None
    try:
        conn = pool.getconn()
        logger.debug("Connected to PostgreSQL database from pool")
        return conn
    except Exception as e:
        logger.error(f"Failed to get connection from pool: {e}")
        return None

def clean_term(term: str) -> str:
    term = term.strip().lower()
    term = re.sub(r'[^\w\s]', '', term)  # Remove punctuation
    term = ' '.join(term.split())  # Normalize whitespace
    return term

def search_local_umls_cui(term: str, max_attempts=3):
    cleaned_term = clean_term(term)
    logger.debug(f"Starting CUI search for '{cleaned_term}'")
    
    # Check cache
    if cleaned_term in _cui_cache:
        logger.debug(f"Cache hit: {cleaned_term} -> {_cui_cache[cleaned_term]}")
        return _cui_cache[cleaned_term]

    if not cleaned_term or len(cleaned_term) > 100 or cleaned_term in STOP_TERMS:
        logger.warning(f"Skipping non-clinical or blocked term: '{cleaned_term}'")
        _cui_cache[cleaned_term] = None
        save_cui_cache()
        return None

    conn = get_postgres_connection()
    if not conn:
        logger.error(f"Cannot query UMLS for term '{cleaned_term}': No database connection")
        return None

    try:
        cursor = conn.cursor()
        for attempt in range(max_attempts):
            try:
                logger.debug(f"Executing exact match query for '{cleaned_term}'")
                start_time = time.time()
                query = """
                    SELECT DISTINCT c.CUI
                    FROM umls.MRCONSO c
                    WHERE c.SAB = 'SNOMEDCT_US'
                    AND LOWER(c.STR) = %s
                    AND c.SUPPRESS = 'N'
                    LIMIT 1
                """
                cursor.execute(query, (cleaned_term,))
                result = cursor.fetchone()
                logger.debug(f"Exact match result: {result}, took {time.time() - start_time:.3f} seconds")
                if result and result['cui']:
                    cui = result['cui']
                    _cui_cache[cleaned_term] = cui
                    save_cui_cache()
                    return cui

                logger.debug(f"Exact match failed, trying full-text search for '{cleaned_term}'")
                start_time = time.time()
                tsquery = ' & '.join(cleaned_term.split()) + ':*'  # Improve full-text search precision
                query = """
                    SELECT DISTINCT c.CUI
                    FROM umls.MRCONSO c
                    WHERE c.SAB = 'SNOMEDCT_US'
                    AND to_tsvector('english', c.STR) @@ to_tsquery('english', %s)
                    AND c.SUPPRESS = 'N'
                    LIMIT 1
                """
                cursor.execute(query, (tsquery,))
                result = cursor.fetchone()
                logger.debug(f"Full-text search result: {result}, took {time.time() - start_time:.3f} seconds")
                if result and result['cui']:
                    cui = result['cui']
                    _cui_cache[cleaned_term] = cui
                    save_cui_cache()
                    return cui

                logger.warning(f"No CUI found for term '{cleaned_term}'")
                _cui_cache[cleaned_term] = None
                save_cui_cache()
                return None

            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{max_attempts} failed for term '{cleaned_term}': {e}")
                if attempt == max_attempts - 1:
                    logger.error(f"Max retries reached for term '{cleaned_term}'")
                    return None
                continue
    finally:
        if cursor:
            cursor.close()
        if conn:
            pool.putconn(conn)
        logger.debug(f"Returned connection for '{cleaned_term}' to pool")

def extract_clinical_phrases(text):
    doc = _sci_ner(text)
    phrases = [clean_term(ent.text) for ent in doc.ents if len(ent.text.strip()) > 2]
    # Filter out stop terms and limit to 3-word phrases
    filtered_phrases = [p for p in phrases if p not in STOP_TERMS and len(p.split()) <= 3]
    logger.debug(f"Extracted phrases from '{text}': {filtered_phrases}")
    return list(set(filtered_phrases))

def get_nlp() -> Language:
    global _nlp_pipeline
    if _nlp_pipeline is not None and "medspacy_target_matcher" in _nlp_pipeline.pipe_names:
        logger.debug("Returning cached NLP pipeline with medspacy_target_matcher")
        return _nlp_pipeline

    try:
        nlp = spacy.load("en_core_sci_sm", disable=["lemmatizer", "parser", "ner"])
        logger.info("Loaded base spacy model: en_core_sci_sm")

        nlp.add_pipe("sentencizer", first=True)
        logger.info("Added sentencizer to pipeline")

        try:
            nlp.add_pipe("medspacy_pyrush", before="sentencizer")
            logger.info("Added medspacy_pyrush to pipeline")
        except Exception as e:
            logger.warning(f"Failed to add medspacy_pyrush: {str(e)}. Using sentencizer only.")

        nlp.add_pipe("medspacy_target_matcher")
        logger.info("Added medspacy_target_matcher to pipeline")

        try:
            client = MongoClient(MONGO_URI)
            start_time = time.time()
            client.admin.command('ping')
            logger.info(f"MongoDB ping took {time.time() - start_time:.3f} seconds")
            db = client[DB_NAME]
            symptoms_collection = db[SYMPTOMS_COLLECTION]
            # Create indexes
            symptoms_collection.create_index("symptom")
            symptoms_collection.create_index("_id")
            logger.info("Created MongoDB indexes on symptom and _id")

            kb = load_knowledge_base()
            logger.debug(f"Knowledge base version: {kb.get('version', 'Unknown')}, last updated: {kb.get('last_updated', 'Unknown')}")

            symptom_rules = []
            valid_count = 0
            invalid_count = 0
            load_cui_cache()

            for doc in symptoms_collection.find():
                if 'symptom' not in doc or not doc['symptom'] or len(doc['symptom']) < 2:
                    logger.warning(f"Skipping symptom document {doc.get('_id')}: Invalid symptom name")
                    invalid_count += 1
                    continue

                cui = doc.get('umls_cui', 'Unknown')
                if not cui or cui == 'Unknown' or not isinstance(cui, str) or not cui.startswith('C'):
                    logger.debug(f"Extracting phrases from symptom: {doc['symptom']}")
                    terms = extract_clinical_phrases(doc['symptom'])
                    fetched_cui = None
                    for term in terms:
                        start_time = time.time()
                        fetched_cui = search_local_umls_cui(term)
                        logger.debug(f"search_local_umls_cui for '{term}' took {time.time() - start_time:.3f} seconds")
                        if fetched_cui:
                            start_time = time.time()
                            symptoms_collection.update_one(
                                {'_id': doc['_id']},
                                {'$set': {'umls_cui': fetched_cui}},
                                upsert=True
                            )
                            logger.debug(f"Updated MongoDB with CUI {fetched_cui} for symptom '{doc['symptom']}', took {time.time() - start_time:.3f} seconds")
                            break
                    cui = fetched_cui if fetched_cui else 'NO_CUI_' + clean_term(doc['symptom'])

                semantic_type = 'Unknown'
                conn = get_postgres_connection()
                if conn:
                    try:
                        cursor = conn.cursor()
                        start_time = time.time()
                        query = """
                            SELECT DISTINCT sty.TUI, sty.STY
                            FROM umls.MRSTY sty
                            WHERE sty.CUI = %s
                            LIMIT 1
                        """
                        cursor.execute(query, (cui,))
                        result = cursor.fetchone()
                        logger.debug(f"Semantic type query for CUI {cui} took {time.time() - start_time:.3f} seconds")
                        if result and result['sty']:
                            semantic_type = result['sty']
                    except Exception as e:
                        logger.warning(f"Failed to fetch semantic type for CUI {cui}: {e}")
                    finally:
                        if cursor:
                            cursor.close()
                        if conn:
                            pool.putconn(conn)
                        logger.debug(f"Returned connection for semantic type query")

                attributes = {
                    "cui": cui,
                    "category": doc.get('category', 'Unknown'),
                    "semantic_type": semantic_type,
                    "icd10": doc.get('icd10', None)
                }

                if attributes['category'] != 'Unknown' and attributes['category'] not in kb.get('symptoms', {}):
                    logger.warning(f"Invalid category for symptom '{doc['symptom']}': {attributes['category']}")

                symptom_rules.append(
                    TargetRule(
                        literal=clean_term(doc['symptom']),
                        category="SYMPTOM",
                        attributes=attributes
                    )
                )
                valid_count += 1

            logger.info(f"Processed {valid_count} valid and {invalid_count} invalid symptom documents")
            if symptom_rules:
                target_matcher = nlp.get_pipe("medspacy_target_matcher")
                target_matcher.add(symptom_rules)
                logger.info(f"Loaded {len(symptom_rules)} symptom rules from MongoDB")
            else:
                logger.warning("No valid symptom rules found in MongoDB.")
            client.close()
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}. Invalidating cache.")
            invalidate_cache()

        if "medspacy_context" not in nlp.pipe_names:
            nlp.add_pipe("medspacy_context", after="medspacy_target_matcher")
            logger.info("Added medspacy_context to pipeline")

        try:
            if "scispacy_linker" not in nlp.pipe_names:
                nlp.add_pipe("scispacy_linker", config={
                    "resolve_abbreviations": True,
                    "linker_name": "umls",
                    "k": 10,
                    "threshold": 0.7
                })
                logger.info("Added scispacy_linker to pipeline")
        except Exception as e:
            logger.warning(f"Failed to add scispacy_linker: {str(e)}. Continuing without UMLS linking.")

        nlp.initialize()
        logger.info(f"Initialized medspacy pipeline with components: {nlp.pipe_names}")

        _nlp_pipeline = nlp
        return nlp

    except Exception as e:
        logger.error(f"Failed to initialize NLP pipeline: {e}")
        nlp = spacy.load("en_core_sci_sm", disable=["lemmatizer", "parser", "ner"])
        nlp.add_pipe("sentencizer", first=True)
        logger.warning("Using fallback spacy pipeline without parser, ner, or linker")
        _nlp_pipeline = nlp
        return nlp