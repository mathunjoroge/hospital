from functools import lru_cache
from typing import List, Tuple, Dict, Set, Optional
import torch
import re
from datetime import date
from departments.models.records import Patient
from departments.nlp.logging_setup import logger
from departments.nlp.config import MAX_LENGTH, EMBEDDING_DIM, DEVICE
from departments.nlp.models.transformer_model import tokenizer, model

@lru_cache(maxsize=10000)
def embed_text(text: str) -> torch.Tensor:
    """Generate text embeddings with caching."""
    if not isinstance(text, str) or not text.strip():
        logger.warning(f"Invalid or empty text for embedding: {text}")
        return torch.zeros(EMBEDDING_DIM)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    ).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().cpu()

def preprocess_text(text: str, stop_words: Set[str]) -> str:
    """Clean and standardize medical text."""
    if not isinstance(text, str):
        logger.error(f"Invalid text type: {type(text)}")
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s\-/]', ' ', text)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

def deduplicate(items: Tuple[str], synonyms: Dict[str, List[str]]) -> List[str]:
    """Deduplicate items using synonym mappings."""
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
        for key, aliases in synonyms.items():
            if not isinstance(aliases, list):
                logger.error(f"aliases for key {key} is not a list: {type(aliases)}")
                continue
                if item.lower() in [a.lower() for a in aliases if isinstance(a, str)]:
                    canonical = key
                    break
        if canonical.lower() not in seen:
            seen.add(canonical.lower())
            result.append(item)
    return result

def parse_conditional_workup(workup: str, symptoms: List[Dict]) -> Optional[str]:
    """Parse conditional workup statements."""
    if not isinstance(workup, str):
        logger.error(f"Invalid workup type: {type(workup)}")
        return None
    if " if " in workup.lower():
        test, condition = workup.lower().split(" if ")
        symptom_descriptions = {s.get('description', '').lower() for s in symptoms if isinstance(s, dict)}
        symptom_severities = {s.get('severity', '').lower() for s in symptoms if isinstance(s, dict)}
        locations = {s.get('location', '').lower() for s in symptoms if isinstance(s, dict)}
        if condition in symptom_descriptions or condition in symptom_severities or condition in locations:
            return test.strip()
        return None
    return workup.strip() if workup.lower() not in ["none", ""] else None

def batch_embed_texts(texts: List[str]) -> torch.Tensor:
    """Batch process text embeddings."""
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        logger.error(f"Invalid texts for batch embedding: {type(texts)}")
        return torch.zeros(len(texts) if isinstance(texts, list) else 0, EMBEDDING_DIM)
    try:
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True
        ).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            return outputs.last_hidden_state[:, 0, :].cpu()
    except Exception as e:
        logger.error(f"Batch embedding failed: {str(e)}")
        return torch.zeros(len(texts), EMBEDDING_DIM)
    
def get_patient_info(patient_id: str) -> dict:
    try:
        patient = Patient.query.filter_by(patient_id=patient_id).first()
        if not patient:
            logger.error(f"No patient found with patient_id: {patient_id}")
            return {"sex": "Unknown", "age": None}
        sex = patient.sex or "Unknown"
        dob = patient.date_of_birth
        if dob:
            today = date(2025, 5, 4)  # Update to current date
            age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        else:
            age = None
        logger.debug(f"Retrieved patient info: sex={sex}, age={age}")
        return {"sex": sex, "age": age}
    except Exception as e:
        logger.error(f"Error retrieving patient info for {patient_id}: {str(e)}")
        return {"sex": "Unknown", "age": None}  