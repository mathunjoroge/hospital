from typing import Dict, List
import re
import spacy
import json
from medspacy.context import ConText
from medspacy.ner import TargetMatcher, TargetRule
from departments.models.medicine import SOAPNote
from departments.nlp.logging_setup import get_logger
from departments.nlp.nlp_utils import get_umls_cui
from departments.nlp.knowledge_base_io import load_knowledge_base, invalidate_cache
import psycopg2
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from concurrent.futures import ThreadPoolExecutor
from config import (
    MONGO_URI, DB_NAME, SYMPTOMS_COLLECTION,
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD,
    CACHE_DIR, BATCH_SIZE, KB_PREFIX
)

# Load environment variables
load_dotenv()

logger = get_logger(__name__)

# Initialize MedSpacy
nlp = spacy.load("en_core_sci_sm")
context = ConText(nlp, rules="default")
target_matcher = TargetMatcher(nlp)
nlp.add_pipe("medspacy_context")
nlp.add_pipe("medspacy_target_matcher")

# Define expanded symptom rules for common clinical terms
symptom_rules = [
    TargetRule("facial pain", "SYMPTOM"),
    TargetRule("nasal congestion", "SYMPTOM"),
    TargetRule("fever", "SYMPTOM"),
    TargetRule("purulent nasal discharge", "SYMPTOM"),
    TargetRule("headache", "SYMPTOM"),
    TargetRule("back pain", "SYMPTOM"),
    TargetRule("nausea", "SYMPTOM"),
    TargetRule("vomiting", "SYMPTOM"),
    TargetRule("jaundice", "SYMPTOM"),
    TargetRule("loss of appetite", "SYMPTOM"),
    TargetRule("chills", "SYMPTOM"),
    TargetRule("sore throat", "SYMPTOM"),
    TargetRule("cough", "SYMPTOM"),
    TargetRule("obesity", "SYMPTOM"),
    TargetRule("eyes", "SYMPTOM"),
    TargetRule("fatigue", "SYMPTOM"),
    TargetRule("dizziness", "SYMPTOM"),
]
target_matcher.add(symptom_rules)

def preprocess_clinical_text(text: str) -> List[Dict]:
    """Preprocess clinical text into individual symptoms with negation status using TargetMatcher and en_core_sci_sm NER."""
    try:
        symptoms = []
        doc = nlp(text)
        
        # Use spaCy's sentence boundary detection
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue
            
            # Split long sentences on conjunctions
            phrases = re.split(r'\s*(?:and|or|with|,)\s+', sent_text, flags=re.IGNORECASE)
            for phrase in phrases:
                phrase = phrase.strip()
                if not phrase or len(phrase) < 2:
                    continue
                
                # Process each phrase with MedSpacy's TargetMatcher
                phrase_doc = nlp(phrase)
                symptom_found = False
                for ent in phrase_doc.ents:
                    if ent.label_ == "SYMPTOM":  # From TargetMatcher
                        symptom_text = ent.text.strip()
                        if len(symptom_text) > 200:
                            logger.warning(f"Symptom '{symptom_text[:50]}...' exceeds 200 chars, truncating")
                            symptom_text = symptom_text[:200]
                        elif len(symptom_text) > 100:
                            logger.info(f"Long symptom '{symptom_text}' (length={len(symptom_text)}), review recommended")
                        symptoms.append({
                            "symptom": symptom_text,
                            "description": sent_text,
                            "negated": getattr(ent._, "is_negated", False),
                            "source": "TargetMatcher"
                        })
                        symptom_found = True
                
                # Fallback: Use en_core_sci_sm NER for entities not matched by TargetMatcher
                if not symptom_found:
                    for ent in phrase_doc.ents:
                        if ent.label_ == "ENTITY":  # From en_core_sci_sm NER
                            symptom_text = ent.text.strip()
                            # Filter for likely symptoms using a simple keyword check
                            if re.search(r'\b(pain|ache|fever|congestion|nausea|vomiting|jaundice|chills|fatigue|dizziness|cough|throat)\b', 
                                       symptom_text, re.IGNORECASE):
                                if len(symptom_text) > 200:
                                    logger.warning(f"NER symptom '{symptom_text[:50]}...' exceeds 200 chars, truncating")
                                    symptom_text = symptom_text[:200]
                                elif len(symptom_text) > 100:
                                    logger.info(f"Long NER symptom '{symptom_text}' (length={len(symptom_text)}), review recommended")
                                symptoms.append({
                                    "symptom": symptom_text,
                                    "description": sent_text,
                                    "negated": getattr(ent._, "is_negated", False),
                                    "source": "en_core_sci_sm"
                                })
                                symptom_found = True
                
                # Final fallback: Use phrase if it contains symptom-like keywords
                if not symptom_found and re.search(r'\b(pain|ache|fever|congestion|nausea|vomiting|jaundice|chills|fatigue|dizziness|cough|throat)\b', 
                                                phrase, re.IGNORECASE):
                    if len(phrase) > 200:
                        logger.warning(f"Fallback symptom '{phrase[:50]}...' exceeds 200 chars, truncating")
                        phrase = phrase[:200]
                    symptoms.append({
                        "symptom": phrase,
                        "description": sent_text,
                        "negated": False,
                        "source": "regex_fallback"
                    })
        
        if not symptoms:
            logger.warning(f"No symptoms extracted from: {text[:100]}...")
            symptom_text = text.strip()[:200]
            if symptom_text and re.search(r'\b(pain|ache|fever|congestion|nausea|vomiting|jaundice|chills|fatigue|dizziness|cough|throat)\b', 
                                       symptom_text, re.IGNORECASE):
                symptoms.append({
                    "symptom": symptom_text,
                    "description": text.strip(),
                    "negated": False,
                    "source": "full_text_fallback"
                })
        
        logger.debug(f"Extracted {len(symptoms)} symptoms from text (Sources: {set(s['source'] for s in symptoms)})")
        return symptoms
    except Exception as e:
        logger.error(f"Error preprocessing text: {str(e)}")
        symptom_text = text.strip()[:200]
        return [{
            "symptom": symptom_text,
            "description": text.strip(),
            "negated": False,
            "source": "error_fallback"
        }]

def normalize_symptom(symptom: str, kb: Dict) -> str:
    """Normalize symptom using synonyms from knowledge base."""
    if not isinstance(symptom, str):
        symptom = str(symptom)
    
    symptom_lower = symptom.lower().strip()
    symptom_clean = re.sub(r'^(presents with |patient complains of |complains of |reports |reports of )\b', 
                          '', symptom_lower, flags=re.IGNORECASE)
    
    variations = {
        "backpain": "back pain",
        "back pain for two": "back pain",
        "head ache": "headache",
        "pyrexia": "fever",
        "stuffy nose": "nasal congestion",
        "purulent nasal": "purulent nasal discharge",
        "jaundice in eyes": "jaundice",
        "headache headache headache": "headache"
    }
    symptom_clean = variations.get(symptom_clean, symptom_clean)
    
    # Split complex terms
    if len(symptom_clean.split()) > 3:
        parts = symptom_clean.split()
        symptom_clean = ' '.join(parts[:3])
        logger.info(f"Simplified complex term '{symptom_lower}' to '{symptom_clean}'")
    
    if len(symptom_clean) > 200:
        logger.warning(f"Normalized symptom '{symptom_clean}' exceeds 200 chars, truncating")
        symptom_clean = symptom_clean[:200]
    
    for canonical, aliases in kb.get('synonyms', {}).items():
        canonical_lower = canonical.lower()
        aliases_lower = [a.lower() for a in aliases]
        if symptom_clean == canonical_lower or symptom_clean in aliases_lower:
            logger.debug(f"Normalized '{symptom_clean}' to '{canonical_lower}'")
            return canonical_lower if len(canonical_lower) <= 200 else canonical_lower[:200]
    
    logger.debug(f"No normalization for '{symptom_clean}', using original")
    return symptom_clean

def validate_symptom(symptom: Dict, kb: Dict) -> bool:
    """Validate symptom data against knowledge base and table constraints."""
    sym = symptom.get('symptom', '').lower().strip()
    desc = symptom.get('description', '').strip()
    category = symptom.get('category', '').lower().strip()
    cui = symptom.get('umls_cui', None)
    semantic_type = symptom.get('semantic_type', '')
    note_id = symptom.get('note_id', None)

    if not sym or len(sym) < 2:
        logger.warning(f"Invalid symptom name: {sym}")
        return False
    if len(sym) > 200:
        logger.warning(f"Symptom name too long: {sym[:50]}... (truncated)")
        symptom['symptom'] = sym[:200]

    allowed_categories = {
        'cardiovascular', 'general', 'gastrointestinal', 'infectious',
        'musculoskeletal', 'neurological', 'pain', 'ent', 'hepatobiliary', 'hepatic', 'pulmonary'
    }
    
    # Fallback to 'general' if category is invalid or 'Uncategorized'
    if not category or category == 'uncategorized' or category not in allowed_categories:
        rule_based_categories = {
            "facial pain": "ent",
            "nasal congestion": "ent",
            "purulent nasal discharge": "ent",
            "fever": "general",
            "headache": "neurological",
            "back pain": "musculoskeletal",
            "nausea": "gastrointestinal",
            "vomiting": "gastrointestinal",
            "jaundice": "hepatobiliary",
            "loss of appetite": "gastrointestinal",
            "chills": "general",
            "sore throat": "ent",
            "cough": "pulmonary",
            "obesity": "general",
            "eyes": "general",
            "fatigue": "general",
            "dizziness": "neurological"
        }
        symptom['category'] = rule_based_categories.get(sym, 'general')
        logger.warning(f"Invalid category '{category}' for symptom '{sym}', assigned '{symptom['category']}'")
    
    if len(category) > 50:
        logger.warning(f"Category '{category}' exceeds 50 chars, setting to 'general'")
        symptom['category'] = 'general'

    if cui and (not cui.startswith('C') or len(cui) > 10):
        logger.warning(f"Invalid UMLS CUI for symptom '{sym}': {cui}")
        symptom['umls_cui'] = None

    valid_semantic_types = {'Sign or Symptom', 'Disease or Syndrome', 'Finding', 'Symptom', 'Unknown'}
    if semantic_type and semantic_type not in valid_semantic_types:
        logger.warning(f"Unusual semantic type for symptom '{sym}': {semantic_type}, setting to 'Unknown'")
        symptom['semantic_type'] = 'Unknown'
    if len(semantic_type) > 50:
        logger.warning(f"Semantic type '{semantic_type}' exceeds 50 chars, setting to 'Unknown'")
        symptom['semantic_type'] = 'Unknown'
    
    if note_id is not None and not isinstance(note_id, int):
        logger.warning(f"Invalid note_id for symptom '{sym}': {note_id}, setting to None")
        symptom['note_id'] = None
    
    logger.debug(f"Validated symptom: {sym}")
    return True

def map_to_umls(term: str, analyzer=None) -> Dict:
    """Map term to UMLS CUI with caching and fallback."""
    try:
        # Check cache first
        cache_file = os.path.join(CACHE_DIR, "umls_cache.json")
        cache = {}
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cache = json.load(f)
            if term in cache:
                return cache[term]
        
        cui, semantic_type = get_umls_cui(term)
        if cui:
            result = {"umls_cui": cui, "semantic_type": semantic_type}
            cache[term] = result
            with open(cache_file, 'w') as f:
                json.dump(cache, f)
            logger.debug(f"Mapped '{term}' to CUI: {cui}, Semantic Type: {semantic_type}")
            return result
        
        # Fallback: split term and try mapping components
        terms = term.split()
        if len(terms) > 1:
            for sub_term in terms:
                cui, semantic_type = get_umls_cui(sub_term)
                if cui:
                    result = {"umls_cui": cui, "semantic_type": semantic_type}
                    cache[sub_term] = result
                    with open(cache_file, 'w') as f:
                        json.dump(cache, f)
                    logger.debug(f"Fallback mapped '{sub_term}' to CUI: {cui}")
                    return result
        
        logger.warning(f"No UMLS CUI for term '{term}'")
        unmapped_file = os.path.join(CACHE_DIR, "unmapped_terms.txt")
        with open(unmapped_file, "a") as f:
            f.write(f"{term}\n")
        return {"umls_cui": None, "semantic_type": "Unknown"}
    except Exception as e:
        logger.error(f"UMLS mapping error for '{term}': {str(e)}")
        return {"umls_cui": None, "semantic_type": "Unknown"}

def store_symptoms(symptoms: List[Dict], note_id: int) -> None:
    """Store validated symptoms in PostgreSQL and MongoDB using batch processing."""
    try:
        # PostgreSQL connection
        pg_conn = psycopg2.connect(
            host=POSTGRES_HOST, port=POSTGRES_PORT, dbname=POSTGRES_DB,
            user=POSTGRES_USER, password=POSTGRES_PASSWORD, sslmode="require"
        )
        pg_cursor = pg_conn.cursor()
        
        # MongoDB connection
        mongo_client = MongoClient(MONGO_URI)
        mongo_db = mongo_client[DB_NAME]
        mongo_collection = mongo_db[SYMPTOMS_COLLECTION]
        
        # Batch processing for PostgreSQL
        pg_batch = []
        for symptom in symptoms:
            sym = symptom.get('symptom', '')[:200]
            if not sym or len(sym.strip()) < 2:
                logger.warning(f"Skipping invalid symptom '{sym}' for note_id {note_id}")
                continue
            desc = symptom.get('description', '')[:1000]
            category = symptom.get('category', 'general')[:50]
            cui = symptom.get('umls_cui', None)
            semantic_type = symptom.get('semantic_type', 'Unknown')[:50]
            
            pg_batch.append((note_id, sym, cui, category, desc, semantic_type))
            
            if len(pg_batch) >= BATCH_SIZE:
                try:
                    pg_cursor.executemany(
                        """
                        INSERT INTO symptoms (note_id, symptom, umls_cui, category, description, semantic_type)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT ON CONSTRAINT symptoms_symptom_key DO UPDATE
                        SET note_id = EXCLUDED.note_id,
                            umls_cui = EXCLUDED.umls_cui,
                            category = EXCLUDED.category,
                            description = EXCLUDED.description,
                            semantic_type = EXCLUDED.semantic_type
                        """,
                        pg_batch
                    )
                    pg_conn.commit()
                    logger.debug(f"Inserted/updated batch of {len(pg_batch)} symptoms for note ID {note_id}")
                except psycopg2.errors.UniqueViolation as e:
                    logger.warning(f"Batch insert conflict for note_id {note_id}: {str(e)}")
                    pg_conn.rollback()
                    for record in pg_batch:
                        try:
                            pg_cursor.execute(
                                """
                                UPDATE symptoms
                                SET note_id = %s, umls_cui = %s, category = %s, description = %s, semantic_type = %s
                                WHERE symptom = %s
                                """,
                                (record[0], record[2], record[3], record[4], record[5], record[1])
                            )
                            pg_conn.commit()
                        except Exception as e:
                            logger.error(f"Error updating symptom '{record[1]}' for note_id {note_id}: {str(e)}")
                            pg_conn.rollback()
                pg_batch = []
        
        # Insert remaining PostgreSQL records
        if pg_batch:
            try:
                pg_cursor.executemany(
                    """
                    INSERT INTO symptoms (note_id, symptom, umls_cui, category, description, semantic_type)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT ON CONSTRAINT symptoms_symptom_key DO UPDATE
                    SET note_id = EXCLUDED.note_id,
                        umls_cui = EXCLUDED.umls_cui,
                        category = EXCLUDED.category,
                        description = EXCLUDED.description,
                        semantic_type = EXCLUDED.semantic_type
                    """,
                    pg_batch
                )
                pg_conn.commit()
                logger.debug(f"Inserted/updated final batch of {len(pg_batch)} symptoms for note ID {note_id}")
            except psycopg2.errors.UniqueViolation as e:
                logger.warning(f"Final batch insert conflict for note_id {note_id}: {str(e)}")
                pg_conn.rollback()
                for record in pg_batch:
                    try:
                        pg_cursor.execute(
                            """
                            UPDATE symptoms
                            SET note_id = %s, umls_cui = %s, category = %s, description = %s, semantic_type = %s
                            WHERE symptom = %s
                            """,
                            (record[0], record[2], record[3], record[4], record[5], record[1])
                        )
                        pg_conn.commit()
                    except Exception as e:
                        logger.error(f"Error updating symptom '{record[1]}' for note_id {note_id}: {str(e)}")
                        pg_conn.rollback()
        
        # Batch processing for MongoDB
        mongo_batch = []
        for symptom in symptoms:
            sym = symptom.get('symptom', '')[:200]
            if not sym or len(sym.strip()) < 2:
                logger.warning(f"Skipping invalid symptom '{sym}' for note_id {note_id}")
                continue
            desc = symptom.get('description', '')[:1000]
            category = symptom.get('category', 'general')[:50]
            cui = symptom.get('umls_cui', None)
            semantic_type = symptom.get('semantic_type', 'Unknown')[:50]
            
            mongo_batch.append({
                "update_one": {
                    "filter": {"note_id": note_id, "symptom": sym},
                    "update": {
                        "$set": {
                            "category": category,
                            "umls_cui": cui,
                            "description": desc,
                            "semantic_type": semantic_type
                        }
                    },
                    "upsert": True
                }
            })
            
            if len(mongo_batch) >= BATCH_SIZE:
                for attempt in range(3):
                    try:
                        mongo_collection.bulk_write([op["update_one"] for op in mongo_batch], ordered=False)
                        logger.debug(f"Inserted/updated batch of {len(mongo_batch)} symptoms in MongoDB for note ID {note_id}")
                        break
                    except Exception as e:
                        logger.warning(f"MongoDB batch write attempt {attempt + 1} failed for note_id {note_id}: {str(e)}")
                        if attempt == 2:
                            for op in mongo_batch:
                                try:
                                    mongo_collection.update_one(
                                        op["update_one"]["filter"],
                                        op["update_one"]["update"],
                                        upsert=True
                                    )
                                except Exception as e:
                                    logger.error(f"MongoDB individual write error for symptom '{op['update_one']['filter']['symptom']}': {str(e)}")
                mongo_batch = []
        
        # Insert remaining MongoDB records
        if mongo_batch:
            for attempt in range(3):
                try:
                    mongo_collection.bulk_write([op["update_one"] for op in mongo_batch], ordered=False)
                    logger.debug(f"Inserted/updated final batch of {len(mongo_batch)} symptoms in MongoDB for note ID {note_id}")
                    break
                except Exception as e:
                    logger.warning(f"MongoDB final batch write attempt {attempt + 1} failed for note_id {note_id}: {str(e)}")
                    if attempt == 2:
                        for op in mongo_batch:
                            try:
                                mongo_collection.update_one(
                                    op["update_one"]["filter"],
                                    op["update_one"]["update"],
                                    upsert=True
                                )
                            except Exception as e:
                                logger.error(f"MongoDB individual write error for symptom '{op['update_one']['filter']['symptom']}': {str(e)}")
        
        logger.info(f"Stored {len(symptoms)} symptoms for note ID {note_id}")
        
    except Exception as e:
        logger.error(f"Error storing symptoms for note ID {note_id}: {str(e)}")
        if 'pg_conn' in locals():
            pg_conn.rollback()
    finally:
        if 'pg_cursor' in locals():
            pg_cursor.close()
        if 'pg_conn' in locals():
            pg_conn.close()
        if 'mongo_client' in locals():
            mongo_client.close()

def process_symptom(symptom: Dict, kb: Dict, note_id: int) -> Dict:
    """Process a single symptom for normalization and UMLS mapping."""
    s_norm = normalize_symptom(symptom["symptom"], kb)
    symptom_dict = {
        "symptom": s_norm,
        "description": symptom["description"],
        "category": "general",
        "umls_cui": None,
        "semantic_type": "Unknown",
        "negated": symptom.get("negated", False),
        "note_id": note_id
    }
    
    symptom_dict.update(map_to_umls(s_norm))
    
    if validate_symptom(symptom_dict, kb):
        return symptom_dict
    logger.warning(f"Invalid symptom for note ID {note_id}: {symptom_dict.get('symptom', 'unknown')}")
    return None

def generate_ai_summary(note: SOAPNote, analyzer=None) -> str:
    """Generate an AI summary for a SOAP note, including symptoms with UMLS metadata."""
    logger.debug(f"Starting generate_ai_summary for note ID {getattr(note, 'id', 'unknown')}")
    
    if not isinstance(note, SOAPNote):
        logger.error(f"Invalid input: note must be a SOAPNote object, got {type(note)}")
        raise TypeError(f"Expected SOAPNote object, got {type(note)}")

    try:
        note_id = getattr(note, 'id', 0)
        
        logger.debug("Generating summary without ClinicalAnalyzer")

        # Load knowledge base with prefix
        kb_file = os.path.join(CACHE_DIR, f"{KB_PREFIX}knowledge_base.json")
        kb = load_knowledge_base()
        if not isinstance(kb.get('symptoms', {}), dict) or not isinstance(kb.get('synonyms', {}), dict):
            logger.error(f"Invalid knowledge base structure for note ID {note_id}")
            invalidate_cache()
            return "Summary unavailable"
        
        summary_parts = []
        situation = getattr(note, 'situation', '') or ''
        if not isinstance(situation, str):
            situation = str(situation)

        preprocessed_terms = preprocess_clinical_text(situation)
        chief_complaint = ""
        if preprocessed_terms:
            chief_complaint = normalize_symptom(preprocessed_terms[0]["symptom"], kb)
            summary_parts.append(f"Chief Complaint: {chief_complaint}")
        else:
            chief_complaint = normalize_symptom(situation.strip(), kb)
            summary_parts.append(f"Chief Complaint: {chief_complaint}")

        full_text = f"{situation}. {getattr(note, 'hpi', '')}. {getattr(note, 'assessment', '')}"
        preprocessed_symptoms = preprocess_clinical_text(full_text)
        
        # Parallelize symptom processing
        with ThreadPoolExecutor(max_workers=4) as executor:
            enriched_symptoms = list(
                filter(None, executor.map(lambda s: process_symptom(s, kb, note_id), preprocessed_symptoms))
            )
        
        # Update symptoms with knowledge base data
        for symptom_dict in enriched_symptoms:
            for category, symptoms in kb.get('symptoms', {}).items():
                for kb_symptom, data in symptoms.items():
                    if symptom_dict['symptom'].lower() == kb_symptom.lower():
                        symptom_dict.update({
                            "symptom": data.get('description', symptom_dict['symptom'])[:200],
                            "umls_cui": data.get('umls_cui', symptom_dict.get('umls_cui')),
                            "semantic_type": data.get('semantic_type', symptom_dict.get('semantic_type'))[:50],
                            "category": category[:50],
                            "icd10": data.get('icd10', None),
                            "note_id": note_id
                        })
                        break
                else:
                    continue
                break
        
        store_symptoms(enriched_symptoms, note_id)
        
        if enriched_symptoms:
            symptom_text = ["Extracted Symptoms:"]
            for s in enriched_symptoms:
                symptom_line = f"- {s.get('symptom', '')} (Category: {s.get('category', 'general')}"
                if s.get('umls_cui'):
                    symptom_line += f", CUI: {s['umls_cui']}"
                if s.get('semantic_type'):
                    symptom_line += f", Semantic Type: {s['semantic_type']}"
                if s.get('icd10'):
                    symptom_line += f", ICD-10: {s['icd10']}"
                if s.get('negated'):
                    symptom_line += ", Negated: True"
                symptom_line += ")"
                symptom_text.append(symptom_line)
            summary_parts.append('\n'.join(symptom_text))

        hpi = getattr(note, 'hpi', '') or ''
        if hpi:
            summary_parts.append(f"HPI: {hpi.strip()}")

        medication_history = getattr(note, 'medication_history', '') or ''
        if medication_history:
            summary_parts.append(f"Medications: {medication_history.strip()}")

        assessment = getattr(note, 'assessment', '') or ''
        if assessment:
            primary_dx = re.search(r"Primary Assessment: (.*?)(?:\.|$)", assessment, re.DOTALL)
            if primary_dx:
                summary_parts.append(f"Assessment: {primary_dx.group(1).strip()}")
            else:
                summary_parts.append(f"Assessment: {assessment.strip()}")

        if not summary_parts:
            logger.warning(f"No valid fields found for summary for note ID {note_id}")
            return "Summary unavailable"

        summary_parts.append(
            f"\nKnowledge Base: Version {kb.get('version', 'unknown')}, Last Updated: {kb.get('last_updated', 'unknown')}"
        )

        summary = '\n'.join(summary_parts)
        logger.info(f"Generated summary for note ID {note_id}: {len(summary)} characters")
        return summary
    except Exception as e:
        logger.error(f"Error in generate_ai_summary for note ID {note_id}: {str(e)}")
        invalidate_cache()
        raise