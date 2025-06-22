import pickle
import re
import time
from typing import List, Dict, Optional, Tuple, Set
import numpy as np
import torch
from functools import lru_cache
from datetime import date
import os
import json
import logging
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
from sqlalchemy.exc import SQLAlchemyError
from spacy.language import Language
from cachetools import LRUCache

from departments.nlp.logging_setup import get_logger
from departments.nlp.knowledge_base_io import load_knowledge_base
from departments.nlp.nlp_common import clean_term, FALLBACK_CUI_MAP
from departments.models.records import Patient
from departments.nlp.config import POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD, EMBEDDING_DIM, MAX_LENGTH, BATCH_SIZE, DEVICE

logger = get_logger(__name__)

# Lazy-loaded SciBERT model for embeddings
_scibert_model = None
_scibert_tokenizer = None

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

# CUI cache
_cui_cache = LRUCache(maxsize=10000)
_cache_file = os.path.join('data_cache', "cui_cache.pkl")

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

@lru_cache(maxsize=1000)
def embed_text(text: str) -> np.ndarray:
    global _scibert_model, _scibert_tokenizer
    if not isinstance(text, str) or not text.strip():
        logger.debug(f"Invalid text for embedding: {text}")
        return np.zeros(EMBEDDING_DIM)

    text_clean = preprocess_text(text)
    if not text_clean:
        logger.debug("Empty text after preprocessing")
        return np.zeros(EMBEDDING_DIM)

    if _scibert_model is None or _scibert_tokenizer is None:
        try:
            from transformers import AutoTokenizer, AutoModel
            start_time = time.time()
            _scibert_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
            _scibert_model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
            _scibert_model.to(DEVICE)
            logger.info(f"Loaded SciBERT model and tokenizer, took {time.time() - start_time:.3f} seconds")
        except Exception as e:
            logger.error(f"Failed to load SciBERT model: {e}")
            return np.zeros(EMBEDDING_DIM)

    try:
        inputs = _scibert_tokenizer(
            text_clean,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length"
        ).to(DEVICE)
        with torch.no_grad():
            outputs = _scibert_model(**inputs)
            embeddings = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).expand(embeddings.size()).float()
            sum_embeddings = torch.sum(embeddings * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
        embedding = mean_embeddings.squeeze(0).cpu().numpy()
        logger.debug(f"Generated embedding for text: {text_clean[:50]}... (shape: {embedding.shape})")
        torch.cuda.empty_cache()
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding for text '{text_clean[:50]}...': {e}")
        return np.zeros(EMBEDDING_DIM)

def batch_embed_texts(texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
    global _scibert_model, _scibert_tokenizer
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        logger.error(f"Invalid texts for batch embedding: {type(texts)}")
        return np.zeros((len(texts) if isinstance(texts, list) else 0, EMBEDDING_DIM))

    texts_clean = [preprocess_text(t) for t in texts]
    texts_clean = [t if t else "" for t in texts_clean]

    if _scibert_model is None or _scibert_tokenizer is None:
        try:
            from transformers import AutoTokenizer, AutoModel
            start_time = time.time()
            _scibert_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
            _scibert_model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
            _scibert_model.to(DEVICE)
            logger.info(f"Loaded SciBERT model and tokenizer, took {time.time() - start_time:.3f} seconds")
        except Exception as e:
            logger.error(f"Failed to load SciBERT model: {e}")
            return np.zeros((len(texts), EMBEDDING_DIM))

    try:
        embeddings = []
        for i in range(0, len(texts_clean), batch_size):
            batch = texts_clean[i:i + batch_size]
            inputs = _scibert_tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True
            ).to(DEVICE)
            logger.debug(f"Processing batch {i//batch_size + 1}: {len(batch)} texts")
            with torch.no_grad():
                outputs = _scibert_model(**inputs)
                batch_embeddings = outputs.last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).expand(batch_embeddings.size()).float()
                sum_embeddings = torch.sum(batch_embeddings * mask, dim=1)
                sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
                mean_embeddings = sum_embeddings / sum_mask
            embeddings.append(mean_embeddings.cpu())
            torch.cuda.empty_cache()
        result = torch.cat(embeddings, dim=0).numpy()
        logger.debug(f"Generated batch embeddings shape: {result.shape}")
        return result
    except Exception as e:
        logger.error(f"Batch embedding failed: {e}")
        return np.zeros((len(texts), EMBEDDING_DIM))

@lru_cache(maxsize=1)
def _load_stop_words() -> Set[str]:
    try:
        kb = load_knowledge_base()
        return kb.get('medical_stop_words', set()).union({
            'three', 'ago', 'started', 'days', 'weeks', 'months', 'years'
        })
    except Exception as e:
        logger.error(f"Failed to load stop words: {e}")
        return set()

def preprocess_text(text: str, stop_words: Optional[Set[str]] = None) -> str:
    if not isinstance(text, str):
        logger.error(f"Invalid text type: {type(text)}")
        return ""
    if stop_words is None:
        stop_words = _load_stop_words()
    text = text.lower()
    kb = load_knowledge_base()
    multi_word_terms = {t.lower() for t in kb.get('medical_terms', []) if ' ' in t}
    for term in multi_word_terms:
        placeholder = term.replace(' ', '_')
        text = text.replace(term, placeholder)
    text = re.sub(r'[^\w\s\-/]', ' ', text)
    words = [word.replace('_', ' ') for word in text.split() if word not in stop_words]
    text = ' '.join(words)
    logger.debug(f"Preprocessed text: {text[:100]}...")
    return text

def deduplicate(items: List[str], synonyms: Optional[Dict[str, List[str]]] = None) -> List[str]:
    if not isinstance(items, (list, tuple)):
        logger.error(f"Invalid items type: {type(items)}")
        return []
    if synonyms is None:
        try:
            kb = load_knowledge_base()
            synonyms = kb.get('synonyms', {})
            medical_terms = kb.get('medical_terms', [])
        except Exception as e:
            logger.error(f"Failed to load synonyms: {e}")
            synonyms = {}
            medical_terms = []
    if not isinstance(synonyms, dict):
        logger.error(f"synonyms is not a dict: {type(synonyms)}")
        return list(items)
    seen = set()
    result = []
    for item in items:
        if not isinstance(item, str):
            logger.warning(f"Non-string item in deduplicate: {item}")
            continue
        canonical = item
        cui = None
        for key, aliases in synonyms.items():
            if not isinstance(aliases, list):
                logger.error(f"aliases for key {key} is not a list: {type(aliases)}")
                continue
            if item.lower() in [a.lower() for a in aliases if isinstance(a, str)]:
                canonical = key
                cui = next((t['umls_cui'] for t in medical_terms if t['term'].lower() == key.lower()), None)
                break
        canonical_key = (canonical.lower(), cui) if cui else canonical.lower()
        if canonical_key not in seen:
            seen.add(canonical_key)
            result.append(item)
    logger.debug(f"Deduplicated {len(items)} items to {len(result)} items")
    return result

def get_patient_info(patient_id: str) -> Dict[str, any]:
    try:
        patient = Patient.query.filter_by(patient_id=patient_id).first()
        if not patient:
            logger.warning(f"No patient found with patient_id: {patient_id}")
            return {"sex": "Unknown", "age": None}
        sex = patient.sex or "Unknown"
        dob = patient.date_of_birth
        if dob:
            today = date.today()
            age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        else:
            age = None
        logger.debug(f"Retrieved patient info: sex={sex}, age={age}")
        return {"sex": sex, "age": age}
    except SQLAlchemyError as e:
        logger.error(f"SQLAlchemy error retrieving patient info for {patient_id}: {e}")
        return {"sex": "Unknown", "age": None}
    except Exception as e:
        logger.error(f"Unexpected error retrieving patient info for {patient_id}: {e}")
        return {"sex": "Unknown", "age": None}

def search_local_umls_cui(terms: List[str], max_attempts=3, batch_size=100, max_tsquery_bytes=500000) -> Dict[str, Optional[str]]:
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
    stop_terms = _load_stop_words()

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

    terms_to_query = [t for t in cleaned_terms if results[t] is None and t and len(t) <= 100 and t not in stop_terms]
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

def get_umls_cui(symptom: str) -> Tuple[Optional[str], Optional[str]]:
    symptom_lower = clean_term(symptom)
    if not symptom_lower or symptom_lower in _load_stop_words():
        logger.debug(f"Skipping non-clinical term: {symptom}")
        return None, "Unknown"

    # Check fallback dictionary
    if symptom_lower in FALLBACK_CUI_MAP:
        cui = FALLBACK_CUI_MAP[symptom_lower]["umls_cui"]
        semantic_type = FALLBACK_CUI_MAP[symptom_lower]["semantic_type"]
        logger.debug(f"Fallback CUI for '{symptom_lower}': {cui}")
        return cui, semantic_type

    # Check cache
    if symptom_lower in _cui_cache:
        logger.debug(f"Cache hit for '{symptom_lower}': {_cui_cache[symptom_lower]}")
        conn = get_postgres_connection()
        if conn:
            try:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute("""
                    SELECT sty AS semantic_type
                    FROM umls.MRSTY
                    WHERE cui = %s
                    AND sty IN ('Sign or Symptom', 'Finding', 'Disease or Syndrome')
                    LIMIT 1
                """, (_cui_cache[symptom_lower],))
                row = cursor.fetchone()
                semantic_type = row['semantic_type'] if row else 'Unknown'
                logger.debug(f"Found semantic type for '{symptom_lower}': {semantic_type}")
                return _cui_cache[symptom_lower], semantic_type
            except Exception as e:
                logger.error(f"Semantic type query failed: {e}")
            finally:
                cursor.close()
                pool.putconn(conn)
        return _cui_cache[symptom_lower], 'Unknown'

    # Query UMLS database
    try:
        result = search_local_umls_cui([symptom_lower])
        cui = result.get(symptom_lower)
        if cui:
            conn = get_postgres_connection()
            if conn:
                try:
                    cursor = conn.cursor(cursor_factory=RealDictCursor)
                    cursor.execute("""
                        SELECT sty AS semantic_type
                        FROM umls.MRSTY
                        WHERE cui = %s
                        AND sty IN ('Sign or Symptom', 'Finding', 'Disease or Syndrome')
                        LIMIT 1
                    """, (cui,))
                    row = cursor.fetchone()
                    semantic_type = row['semantic_type'] if row else 'Unknown'
                    logger.debug(f"Found UMLS CUI for '{symptom_lower}': {cui}, Semantic Type: {semantic_type}")
                    return cui, semantic_type
                except Exception as e:
                    logger.error(f"Semantic type query failed: {e}")
                finally:
                    cursor.close()
                    pool.putconn(conn)
            return cui, 'Unknown'
        logger.debug(f"No UMLS match for '{symptom_lower}'")
        return None, 'Unknown'
    except Exception as e:
        logger.error(f"UMLS lookup failed for '{symptom_lower}': {e}")
        return None, 'Unknown'

def get_negated_symptoms(text: str, nlp: Optional[Language] = None) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        logger.debug(f"Invalid text for negated symptoms: {text}")
        return []

    if nlp is None:
        from departments.nlp.nlp_pipeline import get_nlp
        nlp = get_nlp()

    try:
        doc = nlp(text)
        negated = []
        for ent in doc.ents:
            if ent._.is_negated:
                cleaned = clean_term(ent.text)
                if cleaned and len(cleaned) <= 50:
                    negated.append(cleaned)
        logger.debug(f"Found {len(negated)} negated symptoms: {negated}")
        return negated
    except Exception as e:
        logger.error(f"Error detecting negated symptoms in text '{text[:50]}...': {e}")
        return []