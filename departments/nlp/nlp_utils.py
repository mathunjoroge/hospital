from functools import lru_cache
from typing import List, Tuple, Dict, Set, Optional
import torch
import numpy as np
import re
from datetime import date
import os
import json
import logging
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
from sqlalchemy.exc import SQLAlchemyError

from departments.nlp.knowledge_base_io import load_knowledge_base
from departments.models.records import Patient
from departments.nlp.logging_setup import get_logger
from departments.nlp.config import MAX_LENGTH, EMBEDDING_DIM, DEVICE, BATCH_SIZE
from departments.nlp.models.transformer_model import model, tokenizer
from departments.nlp.config import FALLBACK_CFG, POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
from departments.nlp.nlp_pipeline import search_local_umls_cui, clean_term

logger = get_logger()

@lru_cache(maxsize=1000)  # Reduced from 10000
def embed_text(text: str) -> torch.Tensor:
    """Generate text embeddings with caching."""
    if not isinstance(text, str) or not text.strip():
        logger.warning(f"Invalid or empty text for embedding: {text}")
        return torch.zeros(EMBEDDING_DIM)
    try:
        model.eval()
        model.to(DEVICE)
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length"
        ).to(DEVICE)
        logger.debug(f"Tokenized input for '{text[:50]}...': {inputs.input_ids.shape}")
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu()
        logger.debug(f"Generated embedding shape: {embedding.shape}")
        torch.cuda.empty_cache()
        return embedding
    except Exception as e:
        logger.error(f"Embedding failed for text '{text[:50]}...': {e}")
        return torch.zeros(EMBEDDING_DIM)

@lru_cache(maxsize=1)  # Cache stop words
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

def deduplicate(items: Tuple[str], synonyms: Optional[Dict[str, List[str]]] = None) -> List[str]:
    """Deduplicate items using synonym mappings and UMLS metadata."""
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

def batch_embed_texts(texts: List[str], batch_size: int = BATCH_SIZE) -> torch.Tensor:
    """Batch process text embeddings with memory-efficient batching."""
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        logger.error(f"Invalid texts for batch embedding: {type(texts)}")
        return torch.zeros(len(texts) if isinstance(texts, list) else 0, EMBEDDING_DIM)
    try:
        model.eval()
        model.to(DEVICE)
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True
            ).to(DEVICE)
            logger.debug(f"Processing batch {i//batch_size + 1}: {len(batch)} texts")
            with torch.no_grad():
                outputs = model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu()
            embeddings.append(batch_embeddings)
            torch.cuda.empty_cache()
        result = torch.cat(embeddings, dim=0)
        logger.debug(f"Generated batch embeddings shape: {result.shape}")
        return result
    except Exception as e:
        logger.error(f"Batch embedding failed: {e}")
        return torch.zeros(len(texts), EMBEDDING_DIM)

def get_patient_info(patient_id: str) -> dict:
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
    """Retrieve UMLS CUI and semantic type from local hospital_umls database or fallback dictionary."""
    symptom_lower = clean_term(symptom)
    if not symptom_lower:
        return None, 'Unknown'

    # Check fallback dictionary
    fallback_dict = {}
    if os.path.exists(FALLBACK_CFG):
        try:
            with open(FALLBACK_CFG, 'r') as f:
                fallback_dict = json.load(f)
            if symptom_lower in fallback_dict:
                cui = fallback_dict[symptom_lower].get('cui')
                semantic_type = fallback_dict[symptom_lower].get('semantic_type', 'Unknown')
                logger.debug(f"Found fallback CUI for '{symptom_lower}': {cui}")
                return cui, semantic_type
        except Exception as e:
            logger.warning(f"Failed to load fallback dictionary: {e}")

    # Query hospital_umls
    pool = SimpleConnectionPool(
        minconn=1, maxconn=10,
        host=POSTGRES_HOST, port=POSTGRES_PORT,
        dbname=POSTGRES_DB, user=POSTGRES_USER, password=POSTGRES_PASSWORD
    )
    try:
        result = search_local_umls_cui([symptom_lower])
        cui = result.get(symptom_lower)
        if cui:
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
                    cursor.close()
                    logger.debug(f"Found UMLS CUI for '{symptom_lower}': {cui}, Semantic Type: {semantic_type}")
                    return cui, semantic_type
                except Exception as e:
                    logger.error(f"Semantic type query failed: {e}")
                finally:
                    pool.putconn(conn)
            return cui, 'Unknown'
        logger.warning(f"No UMLS match for '{symptom_lower}'")
        return None, 'Unknown'
    except Exception as e:
        logger.error(f"UMLS lookup failed for '{symptom_lower}': {e}")
        return None, 'Unknown'
    finally:
        pool.closeall()