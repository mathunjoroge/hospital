import re
import time
from typing import List, Dict, Optional, Tuple, Set
import numpy as np
import torch
from functools import lru_cache
from datetime import date, datetime
import logging
import pkg_resources
import os
from sqlalchemy.exc import SQLAlchemyError
from spacy.language import Language

from departments.nlp.logging_setup import get_logger

from departments.nlp.nlp_common import FALLBACK_CUI_MAP
from departments.nlp.nlp_common import STOP_TERMS
from departments.models.records import Patient
from departments.nlp.config import EMBEDDING_DIM, MAX_LENGTH, BATCH_SIZE, DEVICE

logger = get_logger(__name__)

# Lazy-loaded SciBERT model for embeddings
_scibert_model = None
_scibert_tokenizer = None

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
            # Check if transformers is installed
            try:
                pkg_resources.get_distribution('transformers')
                logger.debug("transformers is installed")
            except pkg_resources.DistributionNotFound:
                logger.error("transformers not installed")
                raise ImportError("transformers is required")
            # Check cache directory permissions
            cache_dir = os.getenv('HF_HOME', os.path.expanduser('~/.cache/huggingface/hub'))
            if not os.access(cache_dir, os.W_OK):
                logger.error(f"No write permission to Hugging Face cache directory: {cache_dir}")
                raise PermissionError(f"Cannot write to {cache_dir}")
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
            # Check if transformers is installed
            try:
                pkg_resources.get_distribution('transformers')
                logger.debug("transformers is installed")
            except pkg_resources.DistributionNotFound:
                logger.error("transformers not installed")
                raise ImportError("transformers is required")
            # Check cache directory permissions
            cache_dir = os.getenv('HF_HOME', os.path.expanduser('~/.cache/huggingface/hub'))
            if not os.access(cache_dir, os.W_OK):
                logger.error(f"No write permission to Hugging Face cache directory: {cache_dir}")
                raise PermissionError(f"Cannot write to {cache_dir}")
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

def preprocess_text(text: str, stop_words: Optional[Set[str]] = None) -> str:
    if not isinstance(text, str):
        logger.error(f"Invalid text type: {type(text)}")
        return ""
    if stop_words is None:
        stop_words = STOP_TERMS
    text = text.lower()
    from departments.nlp.knowledge_base_io import load_knowledge_base
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
            from departments.nlp.knowledge_base_io import load_knowledge_base
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

    # Convert synonyms to a lookup dictionary for efficiency
    synonym_map = {}
    for key, aliases in synonyms.items():
        if not isinstance(aliases, list):
            logger.error(f"aliases for key {key} is not a list: {type(aliases)}")
            continue
        for alias in aliases:
            if isinstance(alias, str):
                synonym_map[alias.lower()] = key
    # Map medical terms to CUIs
    cui_map = {t['term'].lower(): t['umls_cui'] for t in medical_terms if isinstance(t, dict) and 'term' in t and 'umls_cui' in t}

    seen = set()
    result = []
    for item in items:
        if not isinstance(item, str):
            logger.warning(f"Non-string item in deduplicate: {item}")
            continue
        item_lower = item.lower()
        canonical = synonym_map.get(item_lower, item_lower)
        cui = cui_map.get(canonical)
        canonical_key = (canonical, cui) if cui else canonical
        if canonical_key not in seen:
            seen.add(canonical_key)
            result.append(item)
    logger.debug(f"Deduplicated {len(items)} items to {len(result)} items")
    return result

def parse_date(date_str):
    try:
        return datetime.fromisoformat(date_str)
    except Exception:
        return datetime.now()

def get_patient_info(patient_id: str) -> Dict[str, any]:
    try:
        patient = Patient.query.filter_by(patient_id=patient_id).first()
        if not patient:
            logger.warning(f"No patient found with patient_id: {patient_id}")
            return {"sex": "Unknown", "age": None}
        sex = patient.sex or "Unknown"
        dob = patient.date_of_birth
        age = None
        if dob:
            today = date.today()
            age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        logger.debug(f"Retrieved patient info: sex={sex}, age={age}")
        return {"sex": sex, "age": age}
    except SQLAlchemyError as e:
        logger.error(f"SQLAlchemy error retrieving patient info for {patient_id}: {e}")
        return {"sex": "Unknown", "age": None}
    except Exception as e:
        logger.error(f"Unexpected error retrieving patient info for {patient_id}: {e}")
        return {"sex": "Unknown", "age": None}

def get_umls_cui(symptom: str) -> Tuple[Optional[str], Optional[str]]:
    from departments.nlp.nlp_common import clean_term
    symptom_lower = clean_term(symptom)
    if not symptom_lower or symptom_lower in STOP_TERMS:
        logger.debug(f"Skipping non-clinical term: {symptom}")
        return None, "Unknown"

    # Check fallback dictionary
    if symptom_lower in FALLBACK_CUI_MAP:
        cui = FALLBACK_CUI_MAP[symptom_lower]["umls_cui"]
        semantic_type = FALLBACK_CUI_MAP[symptom_lower]["semantic_type"]
        logger.debug(f"Fallback CUI for '{symptom_lower}': {cui}")
        return cui, semantic_type

    # Query UMLS and semantic types
    try:
        from departments.nlp.nlp_pipeline import search_local_umls_cui, get_semantic_types
        result = search_local_umls_cui([symptom_lower])
        cui = result.get(symptom_lower)
        if cui:
            semantic_types = get_semantic_types([cui])
            semantic_type = semantic_types.get(cui, 'Unknown')
            if isinstance(semantic_type, list):
                semantic_type = semantic_type[0] if semantic_type else 'Unknown'
            logger.debug(f"Found UMLS CUI for '{symptom_lower}': {cui}, Semantic Type: {semantic_type}")
            return cui, semantic_type
        logger.debug(f"No UMLS match for '{symptom_lower}'")
        return None, 'Unknown'
    except Exception as e:
        logger.error(f"UMLS lookup failed for '{symptom_lower}': {e}")
        return None, 'Unknown'

def get_negated_symptoms(text: str, nlp: Optional[Language] = None, max_attempts: int = 3) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        logger.debug(f"Invalid text for negated symptoms: {text}")
        return []

    if nlp is None:
        from departments.nlp.nlp_pipeline import get_nlp
        for attempt in range(max_attempts):
            try:
                nlp = get_nlp()
                break
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{max_attempts} failed to initialize NLP pipeline: {e}")
                if attempt == max_attempts - 1:
                    logger.error("Max retries reached for NLP pipeline initialization")
                    return []

    try:
        doc = nlp(text)
        negated = []
        for ent in doc.ents:
            if ent._.is_negated:
                from departments.nlp.nlp_common import clean_term
                cleaned = clean_term(ent.text)
                if cleaned and len(cleaned) <= 50:
                    negated.append(cleaned)
        logger.debug(f"Found {len(negated)} negated symptoms: {negated}")
        return negated
    except Exception as e:
        logger.error(f"Error detecting negated symptoms in text '{text[:50]}...': {e}")
        return []