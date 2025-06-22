from typing import Dict, List
import re
import spacy
from medspacy.context import ConText  # Fix import
from medspacy.ner import TargetMatcher, TargetRule
from departments.models.medicine import SOAPNote
from departments.nlp.logging_setup import get_logger
from departments.nlp.clinical_analyzer import ClinicalAnalyzer
from departments.nlp.knowledge_base_io import load_knowledge_base, invalidate_cache
import psycopg2
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from config import (
    MONGO_URI, DB_NAME, KB_PREFIX, SYMPTOMS_COLLECTION,
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD,
    CACHE_DIR, BATCH_SIZE
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

# Define symptom rules for common clinical terms
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
]
target_matcher.add(symptom_rules)

def preprocess_clinical_text(text: str) -> List[Dict]:
    """Preprocess clinical text into individual symptoms with negation status."""
    try:
        # Split on period, semicolon, or newlines for better sentence segmentation
        sentences = re.split(r'[.;\n]\s*', text.strip())
        symptoms = []
        for sentence in sentences:
            if not sentence.strip():
                continue
            doc = nlp(sentence)
            # Use medspacy TargetMatcher for symptom extraction
            for ent in doc.ents:
                # Only include entities labeled as SYMPTOM (from TargetMatcher)
                if hasattr(ent, "label_") and ent.label_ == "SYMPTOM":
                    symptom_text = ent.text.strip()
                    if len(symptom_text) > 100:
                        logger.warning(f"Symptom '{symptom_text}' exceeds 100 chars, truncating")
                        symptom_text = symptom_text[:100]
                    symptoms.append({
                        "symptom": symptom_text,
                        "description": sentence.strip(),
                        "negated": getattr(ent._, "is_negated", False)
                    })
        # Fallback: if no symptoms found, treat the whole text as one symptom
        if not symptoms:
            logger.warning(f"No symptoms extracted from: {text[:100]}...")
            symptom_text = text.strip()[:100]
            symptoms.append({
                "symptom": symptom_text,
                "description": text.strip(),
                "negated": False
            })
        logger.debug(f"Extracted {len(symptoms)} symptoms from text")
        return symptoms
    except Exception as e:
        logger.error(f"Error preprocessing text: {str(e)}")
        symptom_text = text.strip()[:100]
        return [{
            "symptom": symptom_text,
            "description": text.strip(),
            "negated": False
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
        "head ache": "headache",
        "pyrexia": "fever",
        "stuffy nose": "nasal congestion",
        "purulent nasal": "purulent nasal discharge"
    }
    symptom_clean = variations.get(symptom_clean, symptom_clean)
    
    if len(symptom_clean) > 100:
        logger.warning(f"Normalized symptom '{symptom_clean}' exceeds 100 chars, truncating")
        symptom_clean = symptom_clean[:100]
    
    for canonical, aliases in kb.get('synonyms', {}).items():
        canonical_lower = canonical.lower()
        aliases_lower = [a.lower() for a in aliases]
        if symptom_clean == canonical_lower or symptom_clean in aliases_lower:
            logger.debug(f"Normalized '{symptom_clean}' to '{canonical_lower}'")
            return canonical_lower if len(canonical_lower) <= 100 else canonical_lower[:100]
    
    logger.debug(f"No normalization for '{symptom_clean}', using original")
    return symptom_clean

def validate_symptom(symptom: Dict, kb: Dict) -> bool:
    """Validate symptom data against knowledge base and table constraints."""
    sym = symptom.get('symptom', '').lower().strip()
    desc = symptom.get('description', '').strip()
    category = symptom.get('category', '')
    cui = symptom.get('umls_cui', None)
    semantic_type = symptom.get('semantic_type', '')
    note_id = symptom.get('note_id', None)

    if not sym or len(sym) < 2:
        logger.warning(f"Invalid symptom name: {sym}")
        return False
    if len(sym) > 100:
        logger.warning(f"Symptom name too long: {sym[:50]}... (truncated)")
        symptom['symptom'] = sym[:100]

    rule_based_categories = {
        "facial pain": "ENT",
        "nasal congestion": "ENT",
        "purulent nasal discharge": "ENT",
        "fever": "general",
        "headache": "neurological",
        "back pain": "musculoskeletal",
        "nausea": "gastrointestinal",
        "vomiting": "gastrointestinal",
        "jaundice": "hepatobiliary",
        "loss of appetite": "gastrointestinal",
        "chills": "general",
        "sore throat": "ENT",
        "cough": "pulmonary",
        "obesity": "general"
    }
    
    valid_categories = {k for k in kb.get('symptoms', {}).keys()}
    if valid_categories and category and category not in valid_categories:
        symptom['category'] = rule_based_categories.get(sym, 'general')
        logger.warning(f"Category '{category}' not in knowledge base for symptom '{sym}', assigned '{symptom['category']}'")
    
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

def map_to_umls(term: str, analyzer: ClinicalAnalyzer) -> Dict:
    """Map term to UMLS CUI with fallback."""
    try:
        result = analyzer.search_local_umls_cui([term])
        if result.get(term):
            logger.debug(f"Mapped '{term}' to {result[term]}")
            return result[term]
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
            sym = symptom.get('symptom', '')[:100]
            desc = symptom.get('description', '')
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
            sym = symptom.get('symptom', '')[:100]
            desc = symptom.get('description', '')
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
                try:
                    mongo_collection.bulk_write([op["update_one"] for op in mongo_batch], ordered=False)
                    logger.debug(f"Inserted/updated batch of {len(mongo_batch)} symptoms in MongoDB for note ID {note_id}")
                except Exception as e:
                    logger.error(f"MongoDB batch write error for note_id {note_id}: {str(e)}")
                mongo_batch = []
        
        # Insert remaining MongoDB records
        if mongo_batch:
            try:
                mongo_collection.bulk_write([op["update_one"] for op in mongo_batch], ordered=False)
                logger.debug(f"Inserted/updated final batch of {len(mongo_batch)} symptoms in MongoDB for note ID {note_id}")
            except Exception as e:
                logger.error(f"MongoDB final batch write error for note_id {note_id}: {str(e)}")
        
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

def generate_ai_summary(note: SOAPNote, analyzer: ClinicalAnalyzer = None) -> str:
    """Generate an AI summary for a SOAP note, including symptoms with UMLS metadata."""
    logger.debug(f"Starting generate_ai_summary for note ID {getattr(note, 'id', 'unknown')}")
    
    if not isinstance(note, SOAPNote):
        logger.error(f"Invalid input: note must be a SOAPNote object, got {type(note)}")
        raise TypeError(f"Expected SOAPNote object, got {type(note)}")

    try:
        # Define note_id early to avoid UnboundLocalError
        note_id = getattr(note, 'id', 0)
        
        analyzer = analyzer or ClinicalAnalyzer()
        logger.debug("Initialized ClinicalAnalyzer instance")

        # Load knowledge base without prefix (fix TypeError)
        kb = load_knowledge_base()
        if KB_PREFIX:
            logger.warning(f"KB_PREFIX '{KB_PREFIX}' is defined but not used in load_knowledge_base")
        
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
        
        enriched_symptoms = []
        for symptom in preprocessed_symptoms:
            if not isinstance(symptom, dict):
                logger.warning(f"Skipping invalid symptom format for note ID {note_id}: {symptom}")
                continue
            
            s_norm = normalize_symptom(symptom["symptom"], kb)
            symptom_dict = {
                "symptom": s_norm,
                "description": symptom["description"],
                "category": "general",
                "umls_cui": None,
                "semantic_type": "Unknown",
                "negated": symptom.get("negated", False),
                "note_id": note_id  # Add note_id to align with symptoms table
            }
            
            umls_data = map_to_umls(s_norm, analyzer)
            symptom_dict.update(umls_data)
            
            for category, symptoms in kb.get('symptoms', {}).items():
                for kb_symptom, data in symptoms.items():
                    if s_norm == kb_symptom.lower():
                        symptom_dict.update({
                            "symptom": data.get('description', s_norm)[:100],
                            "umls_cui": data.get('umls_cui', symptom_dict.get('umls_cui')),
                            "semantic_type": data.get('semantic_type', symptom_dict.get('semantic_type'))[:50],
                            "category": category[:50],  # Use KB category
                            "icd10": data.get('icd10', None),
                            "note_id": note_id
                        })
                        break
                else:
                    continue
                break
            
            if validate_symptom(symptom_dict, kb):
                enriched_symptoms.append(symptom_dict)
            else:
                logger.warning(f"Invalid symptom for note ID {note_id}: {symptom_dict.get('symptom', 'unknown')}")

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