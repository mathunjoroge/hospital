import logging
import spacy
from spacy.tokens import Token, Span
from spacy.language import Language
from medspacy.target_matcher import TargetMatcher
from medspacy.target_matcher import TargetRule
from medspacy.context import ConText
from scispacy.linking import EntityLinker
from pymongo import MongoClient, UpdateOne
from pymongo.errors import ConnectionFailure, OperationFailure, DuplicateKeyError
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
from pathlib import Path

# Check for GPU support
try:
    import torch
    if torch.cuda.is_available():
        spacy.prefer_gpu()
        logging.info("GPU support enabled for SpaCy")
except ImportError:
    logging.warning("PyTorch not installed; running on CPU")

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
    CACHE_DIR,
    BATCH_SIZE
)

logger = get_logger(__name__)

# Fallback dictionary for symptoms not in UMLS
FALLBACK_CUI_MAP = {
    "fever": {"cui": "C0018682", "semantic_type": "Sign or Symptom"},
    "fevers": {"cui": "C0018682", "semantic_type": "Sign or Symptom"},
    "pyrexia": {"cui": "C0018682", "semantic_type": "Sign or Symptom"},
    "chills": {"cui": "C0085593", "semantic_type": "Sign or Symptom"},
    "shivering": {"cui": "C0085593", "semantic_type": "Sign or Symptom"},
    "nausea": {"cui": "C0027497", "semantic_type": "Sign or Symptom"},
    "queasiness": {"cui": "C0027497", "semantic_type": "Sign or Symptom"},
    "vomiting": {"cui": "C0042963", "semantic_type": "Sign or Symptom"},
    "loss of appetite": {"cui": "C0234450", "semantic_type": "Sign or Symptom"},
    "anorexia": {"cui": "C0234450", "semantic_type": "Sign or Symptom"},
    "decreased appetite": {"cui": "C0234450", "semantic_type": "Sign or Symptom"},
    "jaundice": {"cui": "C0022346", "semantic_type": "Sign or Symptom"},
    "jaundice in eyes": {"cui": "C0022346", "semantic_type": "Sign or Symptom"},
    "icterus": {"cui": "C0022346", "semantic_type": "Sign or Symptom"},
    "headache": {"cui": "C0018681", "semantic_type": "Sign or Symptom"}
}

_nlp_pipeline = None
_cui_cache = LRUCache(maxsize=10000)
_phrase_cache = LRUCache(maxsize=1000)
_cache_file = os.path.join(CACHE_DIR or 'data_cache', "cui_cache.pkl")

try:
    _sci_ner = spacy.load("en_core_sci_scibert", disable=["lemmatizer"])
except OSError as e:
    logger.error(f"Failed to load en_core_sci_scibert: {e}. Using blank model for NER.")
    _sci_ner = spacy.blank("en")

# Load STOP_TERMS from knowledge base and add narrative words
knowledge_base = load_knowledge_base()
STOP_TERMS = knowledge_base.get('medical_stop_words', set())
STOP_TERMS.update({
    'the', 'and', 'or', 'is', 'started', 'days', 'taking', 'makes', 'alleviates',
    'foods', 'fats', 'strong', 'tea', 'licking', 'salt', 'worse'
})
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
            start_time = time.time()
            with open(_cache_file, 'rb') as f:
                cached = pickle.load(f)
                _cui_cache.update(cached)
            logger.info(f"Loaded CUI cache with {len(_cui_cache)} entries from {_cache_file}, took {time.time() - start_time:.3f} seconds")
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
    if len(term) > 200:
        return None
    term = term.strip().lower()
    term = re.sub(r'[^\w\s]', '', term)
    term = ' '.join(term.split())
    return term

def search_local_umls_cui(terms: list, max_attempts=3, batch_size=100, max_tsquery_bytes=500000):
    start_time = time.time()
    oversized_count = 0
    cleaned_terms = []

    for term in terms:
        cleaned = clean_term(term)
        if cleaned is None:
            oversized_count += 1
            logger.warning(f"Skipped oversized term: {term[:50]}... (length: {len(term)})")
            continue
        cleaned_terms.append(cleaned)

    results = {term: None for term in cleaned_terms}

    # Check fallback dictionary
    for term in cleaned_terms:
        if term in FALLBACK_CUI_MAP:
            results[term] = FALLBACK_CUI_MAP[term]['cui']
            _cui_cache[term] = FALLBACK_CUI_MAP[term]['cui']
            logger.debug(f"Fallback hit: {term} -> {FALLBACK_CUI_MAP[term]['cui']}")

    # Cache lookup
    for term in cleaned_terms:
        if term in _cui_cache:
            results[term] = _cui_cache[term]
            logger.debug(f"Cache hit: {term} -> {_cui_cache[term]}")

    terms_to_query = [t for t in cleaned_terms if results[t] is None and t and len(t) <= 100 and t not in STOP_TERMS]
    logger.info(f"Total terms to query: {len(terms_to_query)}")
    if not terms_to_query:
        if oversized_count:
            logger.warning(f"Skipped {oversized_count} oversized terms")
        logger.debug(f"No terms to query after cache and filtering, took {time.time() - start_time:.3f} seconds")
        return results

    conn = get_postgres_connection()
    if not conn:
        logger.error("Cannot query UMLS: No database connection")
        return results

    try:
        cursor = conn.cursor()
        for attempt in range(max_attempts):
            try:
                # Exact match query
                query_start = time.time()
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
                    if term is not None:
                        results[term] = row['cui']
                        _cui_cache[term] = row['cui']
                logger.debug(f"Exact match query for {len(terms_to_query)} terms took {time.time() - query_start:.3f} seconds")

                # Full-text search in batches
                remaining = [t for t in terms_to_query if results[t] is None]
                logger.info(f"Terms for full-text search: {len(remaining)}")
                for i in range(0, len(remaining), batch_size):
                    batch_terms = remaining[i:i + batch_size]
                    tsquery_parts = []
                    current_size = 0
                    batch_oversized = 0

                    for t in batch_terms:
                        byte_len = len(t.encode('utf-8'))
                        if byte_len > 1000000:
                            batch_oversized += 1
                            oversized_count += 1
                            logger.warning(f"Skipping oversized term: {t[:50]}... ({byte_len} bytes)")
                            continue

                        part = f"{' & '.join(t.split())}:*"
                        part_size = len(part.encode('utf-8'))
                        if current_size + part_size + len(' | ') > max_tsquery_bytes:
                            break
                        tsquery_parts.append(part)
                        current_size += part_size + len(' | ')

                    if batch_oversized:
                        logger.warning(f"Skipped {batch_oversized} oversized terms in batch {i//batch_size + 1}")

                    if not tsquery_parts:
                        logger.warning(f"Skipping batch {i//batch_size + 1} due to tsquery size limit")
                        continue

                    tsquery = ' | '.join(tsquery_parts)
                    query_start = time.time()
                    query = """
                        SELECT DISTINCT c.STR, c.CUI
                        FROM umls.MRCONSO c
                        WHERE c.SAB = 'SNOMEDCT_US'
                        AND to_tsvector('english', c.STR) @@ to_tsquery('english', %s)
                        AND c.SUPPRESS = 'N'
                        AND octet_length(c.STR) <= 1048575
                    """
                    try:
                        cursor.execute(query, (tsquery,))
                        for row in cursor.fetchall():
                            term = clean_term(row['str'])
                            if term is not None:
                                results[term] = row['cui']
                                _cui_cache[term] = row['cui']
                        logger.debug(f"Full-text search for {len(tsquery_parts)} terms took {time.time() - query_start:.3f} seconds")
                    except Exception as e:
                        if "string is too long for tsvector" in str(e):
                            logger.error(f"TSVector overflow despite filters: {e}")
                        else:
                            raise

                conn.commit()
                break
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{max_attempts} failed: {e}")
                conn.rollback()
                if attempt == max_attempts - 1:
                    logger.error("Max retries reached for UMLS query")
                    return results
        save_cui_cache()
        if oversized_count:
            logger.warning(f"Skipped {oversized_count} oversized terms during processing")
        logger.info(f"search_local_umls_cui processed {len(terms)} terms, took {time.time() - start_time:.3f} seconds")
        return results
    finally:
        if cursor:
            cursor.close()
        if conn:
            pool.putconn(conn)
        logger.debug("Returned PostgreSQL connection to pool")

def get_semantic_types(cuis: list) -> dict:
    start_time = time.time()
    results = {cui: 'Unknown' for cui in cuis}

    # Check fallback dictionary
    for cui in cuis:
        for term, data in FALLBACK_CUI_MAP.items():
            if data['cui'] == cui:
                results[cui] = data['semantic_type']
                logger.debug(f"Fallback semantic type: {cui} -> {data['semantic_type']}")

    cuis_to_query = [cui for cui in set(cuis) if results[cui] == 'Unknown' and cui and isinstance(cui, str) and cui.startswith('C')]
    if not cuis_to_query:
        logger.debug(f"No CUIs to query for semantic types, took {time.time() - start_time:.3f} seconds")
        return results

    conn = get_postgres_connection()
    if not conn:
        logger.error("Cannot query UMLS for semantic types: No database connection")
        return results

    try:
        cursor = conn.cursor()
        query = """
            SELECT DISTINCT sty.CUI, sty.STY
            FROM umls.MRSTY sty
            WHERE sty.CUI IN %s
        """
        cursor.execute(query, (tuple(cuis_to_query),))
        for row in cursor.fetchall():
            results[row['cui']] = row['sty'] if row['sty'] else 'Unknown'
        logger.debug(f"Semantic type query for {len(cuis_to_query)} CUIs took {time.time() - start_time:.3f} seconds")
        return results
    except Exception as e:
        logger.warning(f"Failed to fetch semantic types: {e}")
        return results
    finally:
        if cursor:
            cursor.close()
        if conn:
            pool.putconn(conn)
        logger.debug("Returned connection for semantic type query")

def extract_clinical_phrases(texts):
    start_time = time.time()
    if not isinstance(texts, list):
        texts = [texts]

    cached_results = [(_phrase_cache[text] if text in _phrase_cache else None) for text in texts]
    if all(r is not None for r in cached_results):
        logger.debug(f"All {len(texts)} texts found in phrase cache, took {time.time() - start_time:.3f} seconds")
        return cached_results if len(texts) > 1 else cached_results[0]

    uncached_texts = [t for t, r in zip(texts, cached_results) if r is None]
    results = cached_results[:]
    batch_size = BATCH_SIZE
    for i in range(0, len(uncached_texts), batch_size):
        batch = uncached_texts[i:i + batch_size]
        batch_start = time.time()
        docs = list(_sci_ner.pipe(batch))
        logger.debug(f"Processed {len(batch)} texts with SciBERT, took {time.time() - batch_start:.3f} seconds")

        for j, (text, doc) in enumerate(zip(batch, docs)):
            phrases = []
            for ent in doc.ents:
                if len(ent.text.strip()) <= 2:
                    continue
                cleaned = clean_term(ent.text)
                if cleaned and len(cleaned) <= 50 and len(cleaned.split()) <= 5:
                    phrases.append(cleaned)
                if ' and ' in ent.text:
                    phrases.append(cleaned)  # Keep compound phrase
                    sub_phrases = [clean_term(p) for p in ent.text.split(' and ')]
                    phrases.extend(p for p in sub_phrases if p and len(p) <= 50 and len(p.split()) <= 5)
            filtered_phrases = [p for p in phrases if p not in STOP_TERMS][:50]
            _phrase_cache[text] = list(set(filtered_phrases))
            results[i + j if len(texts) > 1 else 0] = _phrase_cache[text]

    logger.info(f"extract_clinical_phrases processed {len(texts)} texts, returned {sum(len(r) for r in results if r)} terms, took {time.time() - start_time:.3f} seconds")
    return results if len(texts) > 1 else results[0]

def get_nlp() -> Language:
    global _nlp_pipeline
    start_time = time.time()
    if _nlp_pipeline is not None:
        logger.debug(f"Returning cached NLP pipeline with components: {_nlp_pipeline.pipe_names}, took {time.time() - start_time:.3f} seconds")
        return _nlp_pipeline

    try:
        try:
            nlp = spacy.load("en_core_sci_sm", disable=["lemmatizer", "parser", "ner"])
            logger.info(f"Loaded base spaCy model: en_core_sci_sm")
        except OSError as e:
            logger.error(f"Failed to load en_core_sci_sm: {e}. Falling back to blank English model.")
            nlp = spacy.blank("en")
            logger.info("Initialized blank English model")

        if not Span.has_extension('cui'):
            Span.set_extension('cui', default=None)
        if not Span.has_extension('category'):
            Span.set_extension('category', default=None)
        if not Span.has_extension('semantic_type'):
            Span.set_extension('semantic_type', default=None)
        if not Span.has_extension('icd10'):
            Span.set_extension('icd10', default=None)

        nlp.add_pipe("sentencizer", first=True)
        logger.info("Added sentencizer to pipeline")

        try:
            nlp.add_pipe("medspacy_target_matcher", name="symptom_matcher")
            logger.info("Added medspacy_target_matcher to pipeline")
        except Exception as e:
            logger.error(f"Failed to add medspacy_target_matcher: {e}")
            raise

        symptom_matcher = nlp.get_pipe("symptom_matcher")
        if hasattr(symptom_matcher, 'initialize'):
            try:
                symptom_matcher.initialize()
                logger.info("Initialized symptom_matcher component")
            except Exception as e:
                logger.error(f"Failed to initialize symptom_matcher: {e}")
                raise

        if "medspacy_context" not in nlp.pipe_names:
            try:
                nlp.add_pipe("medspacy_context", after="symptom_matcher")
                logger.info("Added medspacy_context to pipeline")
            except Exception as e:
                logger.error(f"Failed to add medspacy_context: {e}")
                raise

        try:
            if "scispacy_linker" not in nlp.pipe_names:
                logger.info("Initializing scispacy_linker with UMLS configuration")
                nlp.add_pipe("scispacy_linker", config={
                    "resolve_abbreviations": True,
                    "linker_name": "umls",
                    "k": 10,
                    "threshold": 0.7
                })
                logger.info("Added scispacy_linker to pipeline")
        except Exception as e:
            logger.warning(f"Failed to add scispacy_linker: {e}. Continuing without UMLS linking.")

        try:
            client = MongoClient(MONGO_URI)
            client.admin.command('ping')
            logger.info("MongoDB connection successful")
            db = client[DB_NAME]
            symptoms_collection = db[SYMPTOMS_COLLECTION]

            existing_indexes = symptoms_collection.index_information()
            index_name = "symptom_1"
            if index_name not in existing_indexes:
                try:
                    null_count = symptoms_collection.count_documents({"symptom": None})
                    if null_count > 1:
                        null_doc = symptoms_collection.find_one({"symptom": None})
                        if null_doc:
                            symptoms_collection.delete_many({
                                "symptom": None,
                                "_id": {"$ne": null_doc["_id"]}
                            })
                        else:
                            symptoms_collection.delete_many({"symptom": None})
                    symptoms_collection.create_index("symptom", unique=True, name=index_name)
                    logger.info("Created MongoDB index on 'symptom'")
                except DuplicateKeyError as e:
                    logger.error(f"Duplicate key error during index creation: {e}")
                    raise
            elif not existing_indexes[index_name].get("unique"):
                symptoms_collection.drop_index(index_name)
                symptoms_collection.create_index("symptom", unique=True, name=index_name)
                logger.info("Recreated unique index")

            kb = load_knowledge_base()
            logger.debug(f"Knowledge base version: {kb.get('version', 'Unknown')}")

            symptom_rules = [
                TargetRule(literal="nausea and vomiting", category="SYMPTOM", attributes={"category": "gastrointestinal", "umls_cui": "C0027497"}),
                TargetRule(literal="jaundice in eyes", category="SYMPTOM", attributes={"category": "hepatic", "umls_cui": "C0022346"}),
                TargetRule(literal="fevers", category="SYMPTOM", attributes={"category": "general", "umls_cui": "C0018682"})
            ]

            valid_count = 0
            invalid_count = 0
            load_cui_cache()

            cursor = symptoms_collection.find().batch_size(100)
            symptom_docs = []
            for doc in cursor:
                symptom_docs.append(doc)
                if len(symptom_docs) >= 1000:
                    process_symptom_batch(symptom_docs, symptom_rules, valid_count, invalid_count, kb, symptoms_collection)
                    symptom_docs = []
            if symptom_docs:
                process_symptom_batch(symptom_docs, symptom_rules, valid_count, invalid_count, kb, symptoms_collection)
            cursor.close()

            logger.info(f"Processed {valid_count} valid and {invalid_count} invalid symptom documents")

            if symptom_rules:
                try:
                    symptom_matcher.add(symptom_rules)
                    logger.info(f"Added {len(symptom_rules)} symptom rules to matcher")
                except Exception as e:
                    logger.error(f"Failed to add symptom rules: {e}")
                    raise
            else:
                logger.warning("No valid symptom rules found in MongoDB")

            client.close()
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            invalidate_cache()
            raise

        if hasattr(nlp, 'is_blank') and nlp.is_blank and hasattr(nlp, 'initialize'):
            try:
                nlp.tokenizer.initialize()
                logger.info("Initialized tokenizer for blank model")
            except Exception as e:
                logger.warning(f"Couldn't initialize blank model tokenizer: {e}")

        _nlp_pipeline = nlp
        logger.info(f"Pipeline ready with components: {nlp.pipe_names}, total time: {time.time() - start_time:.3f}s")
        return nlp

    except Exception as e:
        logger.error(f"Pipeline creation failed: {e}", exc_info=True)
        raise

def process_symptom_batch(symptom_docs, symptom_rules, valid_count, invalid_count, kb, symptoms_collection):
    symptom_texts = [doc['symptom'] for doc in symptom_docs if 'symptom' in doc and doc['symptom'] and len(doc['symptom']) >= 2]
    terms_list = extract_clinical_phrases(symptom_texts)
    logger.info(f"Extracted {sum(len(terms) for terms in terms_list)} terms from {len(symptom_texts)} symptoms")
    cui_results = search_local_umls_cui([term for terms in terms_list for term in terms])

    all_cuis = []
    for terms in terms_list:
        for term in terms:
            cui = cui_results.get(clean_term(term))
            if cui and cui.startswith('C'):
                all_cuis.append(cui)
    semantic_types = get_semantic_types(list(set(all_cuis)))

    updates = []
    for doc, terms in zip(symptom_docs, terms_list):
        if 'symptom' not in doc or not doc['symptom'] or len(doc['symptom']) < 2:
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

        semantic_type = semantic_types.get(cui, 'Unknown')

        attributes = {
            "cui": cui,
            "category": doc.get('category', 'Unknown'),
            "semantic_type": semantic_type,
            "icd10": doc.get('icd10', None)
        }

        if attributes['category'] != 'Unknown' and attributes['category'] not in kb.get('symptoms', {}):
            logger.warning(f"Invalid category for symptom '{doc['symptom']}': {attributes['category']}")

        literal = clean_term(doc['symptom'])
        if not literal:
            invalid_count += 1
            continue

        symptom_rules.append(
            TargetRule(
                literal=literal,
                category="SYMPTOM",
                attributes=attributes
            )
        )
        valid_count += 1

    if updates:
        try:
            symptoms_collection.bulk_write([UpdateOne(u['filter'], u['update'], upsert=True) for u in updates])
            logger.info(f"Updated {len(updates)} MongoDB documents in batch")
        except Exception as e:
            logger.error(f"Failed to update MongoDB: {e}")