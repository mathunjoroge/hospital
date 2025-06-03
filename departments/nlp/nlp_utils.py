# /home/mathu/projects/hospital/departments/nlp/nlp_utils.py
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

from departments.nlp.logging_setup import get_logger
from departments.nlp.knowledge_base_io import load_knowledge_base
from departments.nlp.nlp_common import clean_term, FALLBACK_CUI_MAP
from departments.models.records import Patient
from departments.nlp.config import POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD, EMBEDDING_DIM, MAX_LENGTH, BATCH_SIZE, DEVICE

logger = get_logger(__name__)

# Lazy-loaded SciBERT model for embeddings
_scibert_model = None
_scibert_tokenizer = None

@lru_cache(maxsize=1000)
def embed_text(text: str) -> np.ndarray:
    """Generate a SciBERT embedding for the input text."""
    global _scibert_model, _scibert_tokenizer
    if not isinstance(text, str) or not text.strip():
        logger.debug(f"Invalid text for embedding: {text}")
        return np.zeros(EMBEDDING_DIM)  # SciBERT embedding dimension (768)

    # Preprocess text
    text_clean = preprocess_text(text)
    if not text_clean:
        logger.debug("Empty text after preprocessing")
        return np.zeros(EMBEDDING_DIM)

    # Load SciBERT model and tokenizer if not initialized
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
        # Tokenize and encode text
        inputs = _scibert_tokenizer(
            text_clean,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length"
        ).to(DEVICE)
        # Generate embeddings
        with torch.no_grad():
            outputs = _scibert_model(**inputs)
            embeddings = outputs.last_hidden_state  # Shape: (1, seq_len, 768)
            # Mean pooling
            mask = inputs["attention_mask"].unsqueeze(-1).expand(embeddings.size()).float()
            sum_embeddings = torch.sum(embeddings * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask  # Shape: (1, 768)
        embedding = mean_embeddings.squeeze(0).cpu().numpy()  # Shape: (768,)
        logger.debug(f"Generated embedding for text: {text_clean[:50]}... (shape: {embedding.shape})")
        torch.cuda.empty_cache()
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding for text '{text_clean[:50]}...': {e}")
        return np.zeros(EMBEDDING_DIM)

def batch_embed_texts(texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
    """Batch process text embeddings with memory-efficient batching."""
    global _scibert_model, _scibert_tokenizer
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        logger.error(f"Invalid texts for batch embedding: {type(texts)}")
        return np.zeros((len(texts) if isinstance(texts, list) else 0, EMBEDDING_DIM))

    # Preprocess texts
    texts_clean = [preprocess_text(t) for t in texts]
    texts_clean = [t if t else "" for t in texts_clean]

    # Load SciBERT model and tokenizer if not initialized
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
                batch_embeddings = outputs.last_hidden_state  # Shape: (batch_size, seq_len, 768)
                # Mean pooling
                mask = inputs["attention_mask"].unsqueeze(-1).expand(batch_embeddings.size()).float()
                sum_embeddings = torch.sum(batch_embeddings * mask, dim=1)
                sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
                mean_embeddings = sum_embeddings / sum_mask  # Shape: (batch_size, 768)
            embeddings.append(mean_embeddings.cpu())
            torch.cuda.empty_cache()
        result = torch.cat(embeddings, dim=0).numpy()  # Shape: (len(texts), 768)
        logger.debug(f"Generated batch embeddings shape: {result.shape}")
        return result
    except Exception as e:
        logger.error(f"Batch embedding failed: {e}")
        return np.zeros((len(texts), EMBEDDING_DIM))

@lru_cache(maxsize=1)
def _load_stop_words() -> Set[str]:
    """Load stop words from knowledge base."""
    try:
        kb = load_knowledge_base()
        return kb.get('medical_stop_words', set())
    except Exception as e:
        logger.error(f"Failed to load stop words: {e}")
        return set()

def preprocess_text(text: str, stop_words: Optional[Set[str]] = None) -> str:
    """Clean and standardize medical text."""
    if not isinstance(text, str):
        logger.error(f"Invalid text type: {type(text)}")
        return ""
    if stop_words is None:
        stop_words = _load_stop_words()
    text = text.lower()
    text = re.sub(r'[^\w\s\-/]', ' ', text)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    logger.debug(f"Preprocessed text: {text[:100]}...")
    return text

def deduplicate(items: List[str], synonyms: Optional[Dict[str, List[str]]] = None) -> List[str]:
    """Deduplicate items using synonym mappings and UMLS metadata."""
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
    """Retrieve patient information with robust error handling."""
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

def get_umls_cui(symptom: str) -> Tuple[Optional[str], Optional[str]]:
    """Retrieve UMLS CUI and semantic type from local database or fallback dictionary."""
    symptom_lower = clean_term(symptom)
    if not symptom_lower:
        logger.debug(f"Empty symptom after cleaning: {symptom}")
        return None, "Unknown"

    # Check fallback dictionary
    if symptom_lower in FALLBACK_CUI_MAP:
        cui = FALLBACK_CUI_MAP[symptom_lower]["umls_cui"]
        semantic_type = FALLBACK_CUI_MAP[symptom_lower]["semantic_type"]
        logger.debug(f"Fallback CUI for '{symptom_lower}': {cui}")
        return cui, semantic_type

    # Lazy import to avoid circular dependency
    from departments.nlp.nlp_pipeline import search_local_umls_cui

    # Query UMLS database
    try:
        result = search_local_umls_cui([symptom_lower])
        cui = result.get(symptom_lower)
        if cui:
            pool = SimpleConnectionPool(
                minconn=1, maxconn=10,
                host=POSTGRES_HOST, port=POSTGRES_PORT,
                dbname=POSTGRES_DB, user=POSTGRES_USER, password=POSTGRES_PASSWORD
            )
            try:
                conn = pool.getconn()
                if conn:
                    try:
                        cursor = conn.cursor(cursor_factory=RealDictCursor)
                        cursor.execute("""
                            SELECT sty AS semantic_type
                            FROM umls.MRSTY
                            WHERE cui = %s
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
            finally:
                pool.closeall()
        logger.debug(f"No UMLS match for '{symptom_lower}'")
        return None, 'Unknown'
    except Exception as e:
        logger.error(f"UMLS lookup failed for '{symptom_lower}': {e}")
        return None, 'Unknown'

def get_negated_symptoms(text: str, nlp: Optional[Language] = None) -> List[str]:
    """Identify negated symptoms in text using NLP pipeline."""
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