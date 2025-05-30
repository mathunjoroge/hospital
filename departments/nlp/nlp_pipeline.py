from logging import Logger
import spacy
from spacy.language import Language
from medspacy.target_matcher import TargetMatcher, TargetRule
from medspacy.context import ConText
from scispacy.linking import EntityLinker
from pymongo import MongoClient, UpdateOne
from pymongo.errors import ConnectionFailure, OperationFailure
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

# Check for GPU support
try:
    import torch
    if torch.cuda.is_available():
        spacy.prefer_gpu()
        Logger.info("GPU support enabled for SpaCy")
except ImportError:
    Logger.warning("PyTorch not installed; running on CPU")

# Load environment variables
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
_cui_cache = LRUCache(maxsize=10000)  # Thread-safe cache for CUIs
_phrase_cache = LRUCache(maxsize=1000)  # Cache for clinical phrases
_cache_file = os.path.join(CACHE_DIR or 'data_cache', "cui_cache.pkl")
_sci_ner = spacy.load("en_core_sci_scibert", disable=["lemmatizer"])  # Keep transformer and NER

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

def search_local_umls_cui(terms: list, max_attempts=3):
    cleaned_terms = [clean_term(term) for term in terms]
    results = {term: None for term in cleaned_terms}
    
    # Check cache
    for term in cleaned_terms:
        if term in _cui_cache:
            results[term] = _cui_cache[term]
            logger.debug(f"Cache hit: {term} -> {_cui_cache[term]}")

    # Filter out invalid or cached terms
    terms_to_query = [t for t in cleaned_terms if results[t] is None and t and len(t) <= 100 and t not in STOP_TERMS]
    if not terms_to_query:
        return results

    conn = get_postgres_connection()
    if not conn:
        logger.error("Cannot query UMLS: No database connection")
        return results

    try:
        cursor = conn.cursor()
        for attempt in range(max_attempts):
            try:
                # Exact match
                start_time = time.time()
                query = """
                    SELECT DISTINCT c.STR, c.CUI
                    FROM umls.MRCONSO c
                    WHERE c.SAB = 'SNOMEDCT_US'
                    AND LOWER(c.STR) IN %s
                    AND c.SUPPRESS = 'N'
                """
                cursor.execute(query, (tuple(terms_to_query),))
                for row in cursor.fetchall():
                    term = clean_term(row['str'])
                    results[term] = row['cui']
                    _cui_cache[term] = row['cui']
                logger.debug(f"Exact match query took {time.time() - start_time:.3f} seconds")

                # Full-text search for remaining terms
                remaining = [t for t in terms_to_query if results[t] is None]
                if remaining:
                    tsquery = ' | '.join(f"{' & '.join(t.split())}:*" for t in remaining)
                    start_time = time.time()
                    query = """
                        SELECT DISTINCT c.STR, c.CUI
                        FROM umls.MRCONSO c
                        WHERE c.SAB = 'SNOMEDCT_US'
                        AND to_tsvector('english', c.STR) @@ to_tsquery('english', %s)
                        AND c.SUPPRESS = 'N'
                    """
                    cursor.execute(query, (tsquery,))
                    for row in cursor.fetchall():
                        term = clean_term(row['str'])
                        results[term] = row['cui']
                        _cui_cache[term] = row['cui']
                    logger.debug(f"Full-text search took {time.time() - start_time:.3f} seconds")
                break
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{max_attempts} failed: {e}")
                if attempt == max_attempts - 1:
                    logger.error("Max retries reached for UMLS query")
                    return results
        save_cui_cache()
        return results
    finally:
        if cursor:
            cursor.close()
        if conn:
            pool.putconn(conn)
        logger.debug("Returned PostgreSQL connection to pool")

def extract_clinical_phrases(texts):
    if not isinstance(texts, list):
        texts = [texts]
    
    # Check cache
    cached_results = [(_phrase_cache[text] if text in _phrase_cache else None) for text in texts]
    if all(r is not None for r in cached_results):
        return cached_results if len(texts) > 1 else cached_results[0]
    
    # Process uncached texts
    uncached_texts = [t for t, r in zip(texts, cached_results) if r is None]
    start_time = time.time()
    docs = list(_sci_ner.pipe(uncached_texts))
    logger.debug(f"Processed {len(uncached_texts)} texts with SciBERT, took {time.time() - start_time:.3f} seconds")
    
    results = cached_results[:]
    for i, (text, doc) in enumerate(zip(uncached_texts, docs)):
        phrases = [clean_term(ent.text) for ent in doc.ents if len(ent.text.strip()) > 2]
        filtered_phrases = [p for p in phrases if p not in STOP_TERMS and len(p.split()) <= 3]
        _phrase_cache[text] = list(set(filtered_phrases))
        results[i if len(texts) > 1 else 0] = _phrase_cache[text]
    
    return results if len(texts) > 1 else results[0]

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

        try:
            nlp.add_pipe("medspacy_target_matcher")
            logger.info("Added medspacy_target_matcher to pipeline")
        except Exception as e:
            logger.error(f"Failed to add medspacy_target_matcher: {str(e)}")
            raise

        try:
            client = MongoClient(MONGO_URI)
            start_time = time.time()
            client.admin.command('ping')
            logger.info(f"MongoDB ping took {time.time() - start_time:.3f} seconds")
            db = client[DB_NAME]
            symptoms_collection = db[SYMPTOMS_COLLECTION]

            # Check and create index on 'symptom' only if it doesn't exist
            existing_indexes = symptoms_collection.index_information()
            if 'symptom_1' not in existing_indexes:
                try:
                    symptoms_collection.create_index("symptom", unique=True, name="symptom_1")
                    logger.info("Created MongoDB index on 'symptom'")
                except OperationFailure as e:
                    logger.error(f"Failed to create MongoDB index on 'symptom': {str(e)}")
                    # Check if the error is due to a non-unique index conflict
                    if "IndexKeySpecsConflict" in str(e):
                        logger.warning("Existing non-unique index on 'symptom' detected. Consider dropping it or making it unique.")
            else:
                logger.info("MongoDB index on 'symptom' already exists")

            kb = load_knowledge_base()
            logger.debug(f"Knowledge base version: {kb.get('version', 'Unknown')}, last updated: {kb.get('last_updated', 'Unknown')}")

            symptom_rules = []
            valid_count = 0
            invalid_count = 0
            load_cui_cache()

            # Batch fetch symptoms
            symptom_docs = list(symptoms_collection.find())
            symptom_texts = [doc['symptom'] for doc in symptom_docs if 'symptom' in doc and doc['symptom'] and len(doc['symptom']) >= 2]
            terms_list = extract_clinical_phrases(symptom_texts)
            cui_results = search_local_umls_cui([term for terms in terms_list for term in terms])

            updates = []
            for doc, terms in zip(symptom_docs, terms_list):
                if 'symptom' not in doc or not doc['symptom'] or len(doc['symptom']) < 2:
                    logger.warning(f"Skipping symptom document {doc.get('_id')}: Invalid symptom name")
                    invalid_count += 1
                    continue

                cui = doc.get('umls_cui', 'Unknown')
                if not cui or cui == 'Unknown' or not isinstance(cui, str) or not cui.startswith('C'):
                    fetched_cui = None
                    for term in terms:
                        fetched_cui = cui_results.get(clean_term(term))
                        if fetched_cui:
                            updates.append({
                                'filter': {'_id': doc['_id']},
                                'update': {'$set': {'umls_cui': fetched_cui}}
                            })
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
                        logger.debug("Returned connection for semantic type query")

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

            if updates:
                start_time = time.time()
                symptoms_collection.bulk_write([UpdateOne(u['filter'], u['update'], upsert=True) for u in updates])
                logger.info(f"Updated {len(updates)} MongoDB documents, took {time.time() - start_time:.3f} seconds")

            logger.info(f"Processed {valid_count} valid and {invalid_count} invalid symptom documents")
            if symptom_rules:
                try:
                    target_matcher = nlp.get_pipe("medspacy_target_matcher")
                    target_matcher.add(symptom_rules)
                    logger.info(f"Loaded {len(symptom_rules)} symptom rules from MongoDB")
                except Exception as e:
                    logger.error(f"Failed to add symptom rules: {str(e)}")
                    raise
            else:
                logger.warning("No valid symptom rules found in MongoDB.")
            client.close()
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}. Invalidating cache.")
            invalidate_cache()

        if "medspacy_context" not in nlp.pipe_names:
            try:
                nlp.add_pipe("medspacy_context", after="medspacy_target_matcher")
                logger.info("Added medspacy_context to pipeline")
            except Exception as e:
                logger.error(f"Failed to add medspacy_context: {str(e)}")
                raise

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