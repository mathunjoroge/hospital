import logging
import spacy
from spacy.tokens import Token, Span
from spacy.language import Language
from medspacy.target_matcher import TargetMatcher, TargetRule
from medspacy.context import ConText, ConTextRule
from scispacy.linking import EntityLinker
from pymongo import MongoClient, UpdateOne
from pymongo.errors import ConnectionFailure, OperationFailure, DuplicateKeyError
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
from departments.models.pharmacy import Batch
from departments.nlp.logging_setup import get_logger
from departments.nlp.knowledge_base_io import load_knowledge_base, invalidate_cache
from departments.nlp.nlp_utils import preprocess_text,  FALLBACK_CUI_MAP
from departments.nlp.nlp_common import clean_term
from departments.nlp.nlp_utils import search_local_umls_cui
from dotenv import load_dotenv
import os
import time
import pickle
from cachetools import LRUCache
import re
from pathlib import Path

# Load environment variables
load_dotenv()

from config import (
    MONGO_URI,
    DB_NAME,

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

_nlp_pipeline = None
_sci_ner = None
_cui_cache = LRUCache(maxsize=10000)
_phrase_cache = LRUCache(maxsize=1000)
_cache_file = os.path.join(CACHE_DIR or 'data_cache', "cui_cache.pkl")

class SciBERTWrapper:
    def __init__(self, model_name="en_core_sci_sm", disable_linker=True):
        try:
            self.nlp = spacy.load(model_name, disable=["lemmatizer"])
            logger.info(f"Loaded SpaCy model: {model_name}")
            if disable_linker and "entity_linker" in self.nlp.pipe_names:
                self.nlp.remove_pipe("entity_linker")
                logger.info("Removed entity_linker to avoid nmslib dependency.")
        except OSError as e:
            logger.error(f"Failed to load spaCy model {model_name}: {e}. Using blank model.")
            self.nlp = spacy.blank("en")

    def extract_entities(self, text):
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]

    def analyze(self, text):
        doc = self.nlp(text)
        return {
            "tokens": [(token.text, token.pos_, token.dep_) for token in doc],
            "entities": [(ent.text, ent.label_) for ent in doc.ents],
            "sentences": [sent.text for sent in doc.sents],
        }

# Initialize _sci_ner
try:
    _sci_ner = SciBERTWrapper(model_name="en_core_sci_sm", disable_linker=True)
    logger.info("Initialized _sci_ner with en_core_sci_sm")
except Exception as e:
    logger.error(f"Failed to initialize _sci_ner: {e}")
    _sci_ner = SciBERTWrapper(model_name="en", disable_linker=True)

# Load STOP_TERMS from knowledge base and add narrative words
knowledge_base = load_knowledge_base()
STOP_TERMS = knowledge_base.get('medical_stop_words', set())
STOP_TERMS.update({
    'the', 'and', 'or', 'is', 'started', 'days', 'weeks', 'months', 'years', 'ago',
    'taking', 'makes', 'alleviates', 'foods', 'fats', 'strong', 'tea', 'licking', 'salt', 'worse'
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
            results[term] = FALLBACK_CUI_MAP[term]['umls_cui']
            _cui_cache[term] = FALLBACK_CUI_MAP[term]['umls_cui']
            logger.debug(f"Fallback hit: {term} -> {FALLBACK_CUI_MAP[term]['umls_cui']}")

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
                # Exact match query with date cleaning
                query_start = time.time()
                query = """
                    SELECT DISTINCT
                        CASE
                            WHEN c.STR ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2} ' THEN REGEXP_REPLACE(c.STR, '^[0-9]{4}-[0-9]{2}-[0-9]{2}\s*', '')
                            ELSE c.STR
                        END AS cleaned_str,
                        c.CUI
                    FROM umls.MRCONSO c
                    WHERE c.SAB = 'SNOMEDCT_US'
                    AND LOWER(c.STR) IN %s
                """
                cursor.execute(query, (tuple(terms_to_query),))
                for row in cursor.fetchall():
                    if row['cleaned_str'] != row.get('str', row['cleaned_str']):
                        logger.warning(f"Removed date prefix from str: {row.get('str', 'unknown')} -> {row['cleaned_str']}")
                    term = clean_term(row['cleaned_str'])
                    if term is not None:
                        results[term] = row['cui']
                        _cui_cache[term] = row['cui']
                logger.debug(f"Exact match query for {len(terms_to_query)} terms took {time.time() - query_start:.2f} seconds")

                # Full-text search with date cleaning
                remaining = [t for t in terms_to_query if results[t] is None]
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
                            logger.warning(f"Skipping oversized term: {t[:50]}... (term: {byte_len} bytes)")
                            continue

                        part = f"{' & '.join(t.split())}:*"
                        part_size = len(part.encode('utf-8'))
                        if current_size + part_size + len(' | ') > max_tsquery_bytes:
                            break
                        tsquery_parts.append(part)
                        current_size += part_size

                    if batch_oversized:
                        logger.warning(f"Skipped {batch_oversized} oversized terms in batch {i//batch_size + 1}")

                    if not tsquery_parts:
                        logger.warning(f"Skipping batch {i//batch_size + 1} due to tsquery size limit")
                        continue

                    tsquery = ' | '.join(tsquery_parts)
                    query_start = time.time()
                    query = """
                        SELECT DISTINCT
                            CASE
                                WHEN c.STR ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2} ' THEN REGEXP_REPLACE(c.STR, '^[0-9]{4}-[0-9]{2}-[0-9]{2}\s*', '')
                                ELSE c.STR
                            END AS cleaned_str,
                            c.CUI,
                            c.STR AS original_str
                        FROM umls.MRCONSO c
                        WHERE c.SAB = 'SNOMEDCT_US'
                        AND to_tsvector('english', c.STR) @@ to_tsquery('english', %s)
                        AND octet_length(c.STR) <= 1048575
                    """
                    try:
                        cursor.execute(query, (tsquery,))
                        for row in cursor.fetchall():
                            if row['cleaned_str'] != row['original_str']:
                                logger.warning(f"Removed date prefix from str: {row['original_str']} -> {row['cleaned_str']}")
                            term = clean_term(row['cleaned_str'])
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
            if data['umls_cui'] == cui:
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
        return cached_results if len(texts) > 1 else cached_results[0] or []
    uncached_texts = [t for t, r in zip(texts, cached_results) if r is None]
    results = cached_results[:]
    batch_size = BATCH_SIZE
    for i in range(0, len(uncached_texts), batch_size):
        batch = uncached_texts[i:i + batch_size]
        batch_start = time.time()
        if _sci_ner is None:
            logger.error("SciBERT NER model (_sci_ner) not initialized")
            raise ValueError("SciBERT NER model not initialized")
        docs = list(_sci_ner.nlp.pipe(batch))
        logger.debug(f"Processed {len(batch)} texts with SciBERT, took {time.time() - batch_start:.3f} seconds")
        for j, (text, doc) in enumerate(zip(batch, docs)):
            phrases = []
            # Extract noun chunks for multi-word phrases
            for chunk in doc.noun_chunks:
                cleaned = clean_term(chunk.text)
                if cleaned and len(cleaned) <= 50 and len(cleaned.split()) <= 5 and cleaned not in STOP_TERMS:
                    phrases.append(cleaned)
            # Extract entities
            for ent in doc.ents:
                if len(ent.text.strip()) <= 2:
                    continue
                cleaned = clean_term(ent.text)
                if cleaned and len(cleaned) <= 50 and len(cleaned.split()) <= 5 and cleaned not in STOP_TERMS:
                    phrases.append(cleaned)
                if ' and ' in ent.text:
                    sub_phrases = [clean_term(p) for p in ent.text.split(' and ')]
                    phrases.extend(p for p in sub_phrases if p and len(p) <= 50 and len(p.split()) <= 5 and p not in STOP_TERMS)
            # Split long texts into sentences
            for sent in doc.sents:
                if len(sent.text.strip().split()) > 5:
                    for sub_ent in sent.ents:
                        cleaned = clean_term(sub_ent.text)
                        if cleaned and len(cleaned) <= 50 and len(cleaned.split()) <= 5 and cleaned not in STOP_TERMS:
                            phrases.append(cleaned)
            filtered_phrases = list(set(phrases))[:50]
            _phrase_cache[text] = filtered_phrases
            results[i + j if len(texts) > 1 else 0] = filtered_phrases
    logger.info(f"extract_clinical_phrases processed {len(texts)} texts, returned {sum(len(r) for r in results if r)} terms, took {time.time() - start_time:.3f} seconds")
    return results if len(texts) > 1 else results[0] or []

def extract_aggravating_alleviating(text: str, factor_type: str) -> str:
    if not isinstance(text, str) or not text.strip():
        logger.debug(f"Invalid text for {factor_type} factor extraction: {text}")
        return "Unknown"
    if factor_type not in ["aggravating", "alleviating"]:
        logger.error(f"Invalid factor_type: {factor_type}")
        return "Unknown"

    text_clean = preprocess_text(text).lower()
    text_clean = re.sub(r'[;:]', ' ', re.sub(r'\s+', ' ', text_clean)).strip()
    if not text_clean:
        logger.debug(f"Empty text after preprocessing for {factor_type} factor")
        return "Unknown"

    aggravating_patterns = [
        r'\b(makes\s+worse|worsens|aggravates|triggers)\s+([\w\s-]+?)(?=\s*|,\s*|\s+and\b|\s+or\b|$)',
        r'\b(worse\s+with|caused\s+by)\s+([\w\s-]+?)(?=\s*|,\s*|\s+and\b|\s+or\b|$)'
    ]
    alleviating_patterns = [
        r'\b(alleviates|relieves|improves|better\s+with)\s+([\w\s-]+?)(?=\s*|,\s*|\s+and\b|\s+or\b|$)',
        r'\b(reduced\s+by|eased\s+by)\s+([\w\s-]+?)(?=\s*|,\s*|\s+and\b|\s+or\b|$)'
    ]

    patterns = aggravating_patterns if factor_type == "aggravating" else alleviating_patterns
    for pattern in patterns:
        try:
            logger.debug(f"Compiling regex pattern: {pattern}")
            re.compile(pattern)
            match = re.search(pattern, text_clean, re.IGNORECASE)
            if match:
                factor = match.group(2).strip()
                factor_clean = clean_term(factor)
                if factor_clean:
                    logger.debug(f"Extracted {factor_type} factor: '{factor_clean}'")
                    return factor_clean
        except re.error as e:
            logger.error(f"Invalid regex pattern '{pattern}': {str(e)}")
            continue
    logger.debug(f"No {factor_type} factor found in text: {text_clean[:50]}...")
    return "Unknown"

def get_nlp() -> Language:
    global _nlp_pipeline
    start_time = time.time()
    if _nlp_pipeline is not None:
        logger.debug(f"Returning cached NLP pipeline with components: {_nlp_pipeline.pipe_names}, took {time.time() - start_time:.3f} seconds")
        return _nlp_pipeline

    try:
        # Load spaCy model
        try:
            nlp = spacy.load("en_core_sci_sm", disable=["lemmatizer", "parser", "ner"])
            logger.info(f"Loaded base spaCy model: en_core_sci_sm")
        except OSError as e:
            logger.error(f"Failed to load en_core_sci_sm: {e}. Falling back to blank English model.")
            nlp = spacy.blank("en")
            logger.info("Initialized blank English model")

        # Register spaCy extensions
        if not Span.has_extension('umls_cui'):
            Span.set_extension('umls_cui', default=None)
        if not Token.has_extension('umls_cui'):
            Token.set_extension('umls_cui', default=None)
        if not Span.has_extension('category'):
            Span.set_extension('category', default=None)
        if not Span.has_extension('semantic_type'):
            Span.set_extension('semantic_type', default=None)
        if not Span.has_extension('icd10'):
            Span.set_extension('icd10', default=None)

        nlp.add_pipe("sentencizer", first=True)
        logger.info("Added sentencizer to pipeline")

        # Add symptom matcher
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

        # Add context component with enhanced negation rules
        if "medspacy_context" not in nlp.pipe_names:
            try:
                context = ConText(nlp, rules="default")
                context.add([
                    ConTextRule(
                        literal="no|without|denies|not",
                        category="NEGATED",
                        pattern=r"\b(no|without|denies|not)\b",
                        direction="FORWARD"
                    ),
                    ConTextRule(
                        literal="absence of",
                        category="NEGATED",
                        pattern=r"\b(absence\s+of)\b",
                        direction="FORWARD"
                    ),
                    ConTextRule(
                        literal="negative for",
                        category="NEGATED",
                        pattern=r"\b(negative\s+for)\b",
                        direction="FORWARD"
                    ),
                    ConTextRule(
                        literal="no evidence of",
                        category="NEGATED",
                        pattern=r"\b(no\s+evidence\s+of)\b",
                        direction="FORWARD"
                    )
                ])
                nlp.add_pipe("medspacy_context", after="symptom_matcher")
                logger.info("Added medspacy_context with custom negation rules")
            except Exception as e:
                logger.error(f"Failed to add medspacy_context: {e}")
                raise

        # Add scispacy linker
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
            logger.warning(f"Failed to access scispacy_linker: {e}. Continuing without linkage.")

        # MongoDB operations
        client = None
        try:
            client = MongoClient(MONGO_URI)
            client.admin.command('ping')
            logger.info("MongoDB connection successful")
            db = client[DB_NAME]
            symptoms_collection = db[SYMPTOMS_COLLECTION]

            # Ensure unique index on 'symptom'
            existing_indexes = symptoms_collection.index_information()
            index_name = "symptom_1"
            if index_name not in existing_indexes:
                try:
                    null_count = symptoms_collection.count_documents({"symptom": None})
                    if null_count > 1:
                        null_doc = symptoms_collection.find_one({'symptom': 'symptom'})
                        if null_doc:
                            symptoms_collection.delete_many({
                                'symptom': None,
                                "_id": {'$ne': null_doc['_id']}
                            })
                        else:
                            symptoms_collection.delete_many({'symptom': None})
                    symptoms_collection.create_index('symptom', unique=True, name=index_name)
                    logger.info("Created MongoDB index for 'symptom'")
                except DuplicateKeyError as e:
                    logger.error(f"Duplicate key error during index creation: {str(e)}")
                    raise
            elif not existing_indexes[index_name].get("unique"):
                symptoms_collection.drop_index(index_name)
                symptoms_collection.create_index('symptom', unique=True, name=index_name)
                logger.info("Recreated unique index for 'symptom'")

            kb = load_knowledge_base()
            logger.debug(f"Knowledge base loaded, version: {kb.get('version', 'Unknown')}")

            # Define enhanced symptom rules
            symptom_rules = [
                TargetRule(
                    literal="nausea and vomiting",
                    category="SYMPTOM",
                    attributes={"category": "gastrointestinal", "umls_cui": "C0027497", "semantic_type": "Sign or Symptom"}
                ),
                TargetRule(
                    literal="loss of appetite",
                    category="SYMPTOM",
                    attributes={"category": "gastrointestinal", "umls_cui": "C0234450", "semantic_type": "Sign or Symptom"}
                ),
                TargetRule(
                    literal="jaundice",
                    category="SYMPTOM",
                    attributes={"category": "hepatic", "umls_cui": "C0022346", "semantic_type": "Sign or Symptom"}
                ),
                TargetRule(
                    literal="fever",
                    category="SYMPTOM",
                    attributes={"category": "systemic", "umls_cui": "C0015967", "semantic_type": "Sign or Symptom"}
                ),
                TargetRule(
                    literal="chills",
                    category="SYMPTOM",
                    attributes={"category": "systemic", "umls_cui": "C0085593", "semantic_type": "Sign or Symptom"}
                ),
                TargetRule(
                    literal="headache",
                    category="SYMPTOM",
                    attributes={"category": "neurological", "umls_cui": "C0018681", "semantic_type": "Sign or Symptom"}
                ),
                TargetRule(
                    literal="fatigue",
                    category="SYMPTOM",
                    attributes={"category": "systemic", "umls_cui": "C0015672", "semantic_type": "Sign or Symptom"}
                ),
                TargetRule(
                    literal="abdominal pain",
                    category="SYMPTOM",
                    attributes={"category": "gastrointestinal", "umls_cui": "C0000737", "semantic_type": "Sign or Symptom"}
                )
            ]

            # Process MongoDB symptoms
            counts = {"valid": 0, "invalid": 0}
            load_cui_cache()
            cursor = symptoms_collection.find().batch_size(100)
            symptom_data = []
            try:
                for doc_data in cursor:
                    symptom_data.append(doc_data)
                    if len(symptom_data) >= 1000:
                        process_symptom_batch(symptom_data, symptom_rules, counts, kb, symptoms_collection)
                        symptom_data = []
                if symptom_data:
                    process_symptom_batch(symptom_data, symptom_rules, counts, kb, symptoms_collection)
            finally:
                cursor.close()

            logger.info(f"Processed {counts['valid']} valid and {counts['invalid']} invalid symptom documents")

            # Add symptom rules to matcher
            if symptom_rules:
                try:
                    symptom_matcher.add(symptom_rules)
                    logger.info(f"Added {len(symptom_rules)} symptom rules to matcher")
                except Exception as e:
                    logger.error(f"Failed to add symptom rules: {str(e)}")
                    raise
            else:
                logger.warning("No valid symptom rules found")

        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            invalidate_cache()
            raise
        finally:
            if client:
                client.close()
                logger.debug("MongoDB client closed")

        # Initialize tokenizer for blank model
        if hasattr(nlp, 'is_blank') and nlp.is_blank and hasattr(nlp, 'initialize'):
            try:
                nlp.tokenizer.initialize()
                logger.info("Initialized tokenizer for blank model")
            except Exception as e:
                logger.warning(f"Couldn't initialize blank model tokenizer: {e}")

        _nlp_pipeline = nlp
        logger.info(f"Pipeline ready with components: {nlp.pipe_names}, total time: {time.time() - start_time:.3f} seconds")
        return nlp

    except Exception as e:
        logger.error(f"Pipeline creation failed: {str(e)}", exc_info=True)
        raise

def process_symptom_batch(symptom_docs: list, symptom_rules: list, counts: dict, kb: dict, symptoms_collection):
    symptom_texts = [doc['symptom'] for doc in symptom_docs if 'symptom' in doc and doc['symptom'] and len(doc['symptom']) >= 2]
    terms_list = extract_clinical_phrases(symptom_texts)
    logger.debug(f"Extracted {sum(len(terms) for terms in terms_list)} terms from {len(symptom_texts)} symptoms")
    # Call search_local_umls_cui with STOP_TERMS
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
            counts['invalid'] += 1
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
            "umls_cui": cui,
            "category": doc.get('category', 'Unknown'),
            "semantic_type": semantic_type,
            "icd10": doc.get('icd10', None)
        }

        if attributes['category'] != 'Unknown' and attributes['category'] not in kb.get('symptoms', {}):
            logger.warning(f"Invalid category for symptom '{doc['symptom']}': {attributes['category']}")

        literal = clean_term(doc['symptom'])
        if not literal:
            counts['invalid'] += 1
            continue

        symptom_rules.append(
            TargetRule(
                literal=literal,
                category="SYMPTOM",
                attributes=attributes
            )
        )
        counts['valid'] += 1

    if updates:
        try:
            symptoms_collection.bulk_write([UpdateOne(u['filter'], u['update']) for u in updates])
            logger.info(f"Updated {len(updates)} MongoDB documents in batch")
        except Exception as e:
            logger.error(f"Failed to update MongoDB: {str(e)}")