import json
from pathlib import Path
from pymongo import MongoClient
from psycopg2.extras import execute_batch
from departments.nlp.logging_setup import get_logger
from departments.nlp.config import MONGO_URI, DB_NAME
from departments.nlp.knowledge_base_io import get_postgres_connection

logger = get_logger()

def migrate_json_to_db(kb_dir: str = None):
    """Migrate JSON files to PostgreSQL and MongoDB."""
    # Default to knowledge_base directory relative to this script
    if kb_dir is None:
        kb_dir = Path(__file__).parent / "knowledge_base"
    else:
        kb_dir = Path(kb_dir)
    
    if not kb_dir.exists():
        logger.error(f"Knowledge base directory not found: {kb_dir}")
        return
    
    logger.info(f"Migrating JSON files from {kb_dir}")

    # PostgreSQL connection
    conn = get_postgres_connection()
    if not conn:
        logger.error("Failed to connect to PostgreSQL")
        return
    cursor = conn.cursor()

    # MongoDB connection
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        cursor.close()
        conn.close()
        return

    try:
        # Migrate symptoms.json to PostgreSQL
        symptoms_file = kb_dir / "symptoms.json"
        try:
            if symptoms_file.exists():
                with symptoms_file.open() as f:
                    symptoms = json.load(f)
                symptom_records = [
                    (symptom, category, info.get("description"), info.get("umls_cui"), info.get("semantic_type"))
                    for category, symptom_dict in symptoms.items()
                    for symptom, info in symptom_dict.items()
                    if isinstance(info, dict)
                ]
                execute_batch(cursor, """
                    INSERT INTO symptoms (symptom, category, description, umls_cui, semantic_type)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (symptom) DO NOTHING
                """, symptom_records)
                logger.info(f"Migrated {len(symptom_records)} symptoms to PostgreSQL")
            else:
                logger.warning(f"symptoms.json not found in {kb_dir}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse symptoms.json: {e}")
        except Exception as e:
            logger.error(f"Error migrating symptoms.json: {e}")

        # Migrate medical_stop_words.json to PostgreSQL
        stop_words_file = kb_dir / "medical_stop_words.json"
        try:
            if stop_words_file.exists():
                with stop_words_file.open() as f:
                    stop_words = json.load(f)
                execute_batch(cursor, """
                    INSERT INTO medical_stop_words (word)
                    VALUES (%s)
                    ON CONFLICT (word) DO NOTHING
                """, [(word,) for word in stop_words if isinstance(word, str)])
                logger.info(f"Migrated {len(stop_words)} medical_stop_words to PostgreSQL")
            else:
                logger.warning(f"medical_stop_words.json not found in {kb_dir}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse medical_stop_words.json: {e}")
        except Exception as e:
            logger.error(f"Error migrating medical_stop_words.json: {e}")

        # Migrate medical_terms.json to PostgreSQL
        terms_file = kb_dir / "medical_terms.json"
        try:
            if terms_file.exists():
                with terms_file.open() as f:
                    terms = json.load(f)
                term_records = [
                    (t["term"], t.get("category", "unknown"), t.get("umls_cui"), t.get("semantic_type", "Unknown"))
                    for t in terms if isinstance(t, dict) and "term" in t
                ]
                execute_batch(cursor, """
                    INSERT INTO medical_terms (term, category, umls_cui, semantic_type)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (term) DO NOTHING
                """, term_records)
                logger.info(f"Migrated {len(term_records)} medical_terms from JSON to PostgreSQL")
            else:
                logger.warning(f"medical_terms.json not found in {kb_dir}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse medical_terms.json: {e}")
        except Exception as e:
            logger.error(f"Error migrating medical_terms.json: {e}")

        # Migrate medical_terms from MongoDB to PostgreSQL
        try:
            mongo_terms = list(db.medical_terms.find())
            if mongo_terms:
                mongo_term_records = [
                    (t["term"], t.get("category", "unknown"), t.get("umls_cui"), t.get("semantic_type", "Unknown"))
                    for t in mongo_terms if isinstance(t, dict) and "term" in t
                ]
                execute_batch(cursor, """
                    INSERT INTO medical_terms (term, category, umls_cui, semantic_type)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (term) DO NOTHING
                """, mongo_term_records)
                logger.info(f"Migrated {len(mongo_term_records)} medical_terms from MongoDB to PostgreSQL")
                # Optional: Remove MongoDB medical_terms collection
                db.medical_terms.drop()
                logger.info("Cleared medical_terms collection from MongoDB")
            else:
                logger.info("No medical_terms found in MongoDB")
        except Exception as e:
            logger.error(f"Error migrating medical_terms from MongoDB: {e}")

        # Migrate synonyms.json to MongoDB
        synonyms_file = kb_dir / "synonyms.json"
        try:
            if synonyms_file.exists():
                with synonyms_file.open() as f:
                    synonyms = json.load(f)
                db.synonyms.drop()
                db.synonyms.insert_many([{"term": term, "aliases": aliases}
                                        for term, aliases in synonyms.items()
                                        if isinstance(aliases, list)])
                logger.info(f"Migrated {len(synonyms)} synonyms to MongoDB")
            else:
                logger.warning(f"synonyms.json not found in {kb_dir}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse synonyms.json: {e}")
        except Exception as e:
            logger.error(f"Error migrating synonyms.json: {e}")

        # Migrate clinical_pathways.json to MongoDB
        pathways_file = kb_dir / "clinical_pathways.json"
        try:
            if pathways_file.exists():
                with pathways_file.open() as f:
                    pathways = json.load(f)
                db.clinical_pathways.drop()
                db.clinical_pathways.insert_many([{"category": cat, "paths": paths}
                                                 for cat, paths in pathways.items()
                                                 if isinstance(paths, dict)])
                logger.info(f"Migrated {len(pathways)} clinical_pathways to MongoDB")
            else:
                logger.warning(f"clinical_pathways.json not found in {kb_dir}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse clinical_pathways.json: {e}")
        except Exception as e:
            logger.error(f"Error migrating clinical_pathways.json: {e}")

        # Migrate diagnosis_relevance.json to MongoDB
        diag_relevance_file = kb_dir / "diagnosis_relevance.json"
        try:
            if diag_relevance_file.exists():
                with diag_relevance_file.open() as f:
                    diag_relevance = json.load(f)
                if isinstance(diag_relevance, dict):
                    valid_items = [
                        {"diagnosis": diagnosis, "relevance": symptoms, "category": "unknown"}
                        for diagnosis, symptoms in diag_relevance.items()
                        if isinstance(symptoms, list)
                    ]
                    if valid_items:
                        db.diagnosis_relevance.drop()
                        db.diagnosis_relevance.insert_many(valid_items)
                        logger.info(f"Migrated {len(valid_items)} diagnosis_relevance entries to MongoDB")
                    else:
                        logger.warning(f"No valid items in diagnosis_relevance.json: {list(diag_relevance.items())[:2]}")
                elif isinstance(diag_relevance, list) and diag_relevance:
                    valid_items = [
                        {"diagnosis": item["diagnosis"], "relevance": item["relevance"], "category": item.get("category", "unknown")}
                        for item in diag_relevance
                        if isinstance(item, dict) and "diagnosis" in item and "relevance" in item
                    ]
                    if valid_items:
                        db.diagnosis_relevance.drop()
                        db.diagnosis_relevance.insert_many(valid_items)
                        logger.info(f"Migrated {len(valid_items)} diagnosis_relevance entries to MongoDB")
                    else:
                        logger.warning(f"No valid items in diagnosis_relevance.json: {diag_relevance[:2]}")
                else:
                    logger.warning(f"diagnosis_relevance.json is not a valid format: {diag_relevance}")
            else:
                logger.warning(f"diagnosis_relevance.json not found in {kb_dir}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse diagnosis_relevance.json: {e}")
        except Exception as e:
            logger.error(f"Error migrating diagnosis_relevance.json: {e}")

        # Migrate history_diagnoses.json to MongoDB
        history_file = kb_dir / "history_diagnoses.json"
        try:
            if history_file.exists():
                with history_file.open() as f:
                    history = json.load(f)
                db.history_diagnoses.drop()
                db.history_diagnoses.insert_many([{"key": k, "value": v}
                                                 for k, v in history.items()])
                logger.info(f"Migrated {len(history)} history_diagnoses to MongoDB")
            else:
                logger.warning(f"history_diagnoses.json not found in {kb_dir}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse history_diagnoses.json: {e}")
        except Exception as e:
            logger.error(f"Error migrating history_diagnoses.json: {e}")

        # Migrate management_config.json to MongoDB
        config_file = kb_dir / "management_config.json"
        try:
            if config_file.exists():
                with config_file.open() as f:
                    config = json.load(f)
                db.management_config.drop()
                db.management_config.insert_many([{"key": k, "value": v}
                                                 for k, v in config.items()])
                logger.info(f"Migrated {len(config)} management_config entries to MongoDB")
            else:
                logger.warning(f"management_config.json not found in {kb_dir}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse management_config.json: {e}")
        except Exception as e:
            logger.error(f"Error migrating management_config.json: {e}")

        # Migrate clinical_relevance.json to MongoDB (placeholder)
# ... (previous imports and code unchanged)

        # Migrate clinical_relevance.json to MongoDB (placeholder)
        clinical_relevance_file = kb_dir / "clinical_relevance.json"
        try:
            if clinical_relevance_file.exists():
                logger.warning("clinical_relevance.json not in resources. Migrating to MongoDB as kb_clinical_relevance.")
                with clinical_relevance_file.open() as f:
                    clinical_relevance = json.load(f)
                if isinstance(clinical_relevance, list):
                    if clinical_relevance:
                        db.kb_clinical_relevance.drop()
                        db.kb_clinical_relevance.insert_many(clinical_relevance)
                        logger.info(f"Migrated {len(clinical_relevance)} clinical_relevance entries to MongoDB")
                    else:
                        logger.warning("clinical_relevance.json is empty, skipping migration")
                else:
                    logger.warning(f"clinical_relevance.json is not a list: {clinical_relevance}")
            else:
                logger.warning(f"clinical_relevance.json not found in {kb_dir}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse clinical_relevance.json: {e}")
        except Exception as e:
            logger.error(f"Error migrating clinical_relevance.json: {e}")

        # Migrate diagnosis_treatments.json to MongoDB (placeholder)
        treatments_file = kb_dir / "diagnosis_treatments.json"
        try:
            if treatments_file.exists():
                logger.warning("diagnosis_treatments.json not in resources. Migrating to MongoDB as kb_diagnosis_treatments.")
                with treatments_file.open() as f:
                    treatments = json.load(f)
                if isinstance(treatments, list):
                    if treatments:
                        db.kb_diagnosis_treatments.drop()
                        db.kb_diagnosis_treatments.insert_many(treatments)
                        logger.info(f"Migrated {len(treatments)} diagnosis_treatments to MongoDB")
                    else:
                        logger.warning("diagnosis_treatments.json is empty, skipping migration")
                else:
                    logger.warning(f"diagnosis_treatments.json is not a list: {treatments}")
            else:
                logger.warning(f"diagnosis_treatments.json not found in {kb_dir}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse diagnosis_treatments.json: {e}")
        except Exception as e:
            logger.error(f"Error migrating diagnosis_treatments.json: {e}")

# ... (rest of the script unchanged)

        conn.commit()
    finally:
        cursor.close()
        conn.close()
        client.close()
        logger.info("Closed database connections")

if __name__ == "__main__":
    migrate_json_to_db()