from typing import Tuple, List, Dict, Optional
from flask import current_app
from tqdm import tqdm
import time
import psutil
from retry import retry
from departments.models.medicine import SOAPNote
from departments.models.records import Patient
from extensions import db
from departments.nlp.logging_setup import get_logger
from departments.nlp.config import BATCH_SIZE, KB_PREFIX, SYMPTOMS_COLLECTION
from departments.nlp.knowledge_base_io import load_knowledge_base, save_knowledge_base, REQUIRED_CATEGORIES
from departments.nlp.note_processing import generate_ai_summary
from departments.nlp.ai_summary import generate_ai_analysis
from departments.nlp.clinical_analyzer import ClinicalAnalyzer

logger = get_logger(__name__)

@retry(tries=3, delay=1, backoff=2, exceptions=(Exception,), logger=logger)
def update_single_soap_note(note_id: int, analyzer: ClinicalAnalyzer) -> None:
    """Update AI-generated summary, analysis, and symptoms for a single SOAP note."""
    logger.info(f"Updating AI notes for SOAP note ID {note_id}")
    try:
        with db.session.no_autoflush:
            note = db.session.query(SOAPNote).get(note_id)
            if not note:
                logger.error(f"SOAP note ID {note_id} not found")
                return
            if not isinstance(note, SOAPNote):
                logger.error(f"Invalid note type for ID {note_id}: {type(note)}")
                return
            patient = db.session.query(Patient).get(note.patient_id)
            if not patient:
                logger.warning(f"Patient not found for SOAP note ID {note_id}")

            # Generate AI summary
            if note.ai_notes is None:
                ai_summary = generate_ai_summary(note, analyzer)
                if not isinstance(ai_summary, str):
                    logger.error(f"generate_ai_summary returned non-string: {type(ai_summary)}")
                    ai_summary = "Summary unavailable"
                note.ai_notes = ai_summary

            # Generate AI analysis with UMLS mappings from ai_summary.py
            if note.ai_analysis is None:
                ai_analysis = generate_ai_analysis(note, patient, symptom_collection=KB_PREFIX + SYMPTOMS_COLLECTION)
                if not isinstance(ai_analysis, str):
                    logger.error(f"generate_ai_analysis returned non-string: {type(ai_analysis)}")
                    ai_analysis = "Analysis unavailable"
                note.ai_analysis = ai_analysis

            # Validate symptom categories
            features = analyzer.extract_clinical_features(note)
            symptoms = features.get('symptoms', [])
            kb = load_knowledge_base()
            for symptom in symptoms:
                if not isinstance(symptom, dict) or not symptom.get('description'):
                    continue
                if symptom.get('category', 'uncategorized') not in REQUIRED_CATEGORIES:
                    symptom['category'] = 'uncategorized'
                    logger.warning(f"Invalid category for symptom '{symptom.get('description')}' in note {note_id}. Set to 'uncategorized'.")

            # Update knowledge base with new symptoms
            if symptoms:
                symptom_descriptions = [s['description'] for s in symptoms if s.get('description')]
                logger.debug(f"Extracted symptoms for note ID {note_id}: {symptom_descriptions}")
                for s in symptoms:
                    desc = s.get('description', '').lower()
                    category = s.get('category', 'uncategorized')
                    cui = s.get('umls_cui', 'None')
                    if desc and cui != 'None':
                        if category not in kb.get('symptoms', {}):
                            kb['symptoms'][category] = {}
                        kb['symptoms'][category][desc] = {
                            'description': desc,
                            'umls_cui': cui,
                            'semantic_type': s.get('semantic_type', 'Sign or Symptom')
                        }
                try:
                    save_knowledge_base(kb)
                    logger.debug(f"Updated knowledge base with symptoms from note {note_id}")
                except Exception as e:
                    logger.error(f"Failed to update knowledge base for note {note_id}: {e}")

            db.session.add(note)
            db.session.commit()
            logger.info(f"Successfully updated AI notes for SOAP note ID {note_id}")
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error updating AI notes for SOAP note ID {note_id}: {e}", exc_info=True)
        raise

@retry(tries=3, delay=1, backoff=2, exceptions=(Exception,), logger=logger)
def update_all_ai_notes(batch_size: int = BATCH_SIZE) -> Tuple[int, int]:
    """Process all notes needing AI updates, including symptom extraction."""
    with current_app.app_context():
        try:
            # Adjust batch size based on available memory
            available_memory = psutil.virtual_memory().available / (1024 ** 2)  # MB
            memory_per_note = 100  # Conservative estimate
            adjusted_batch_size = min(batch_size, max(10, int(available_memory // memory_per_note)))
            if available_memory < 1000:
                logger.warning(f"Low memory ({available_memory:.2f} MB). Using batch size {adjusted_batch_size}.")
            logger.info(f"Adjusted batch size to {adjusted_batch_size} based on {available_memory:.2f} MB available")

            # Query notes needing updates
            base_query = db.session.query(SOAPNote, Patient).outerjoin(
                Patient, SOAPNote.patient_id == Patient.patient_id
            ).filter(
                (SOAPNote.ai_notes.is_(None)) |
                (SOAPNote.ai_analysis.is_(None))
            )
            total = base_query.count()
            logger.info(f"Total notes to process: {total}")
            if total == 0:
                return 0, 0

            success = error = 0
            failed_notes = []
            try:
                analyzer = ClinicalAnalyzer()
                logger.info("ClinicalAnalyzer initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize ClinicalAnalyzer: {e}", exc_info=True)
                raise RuntimeError("Cannot initialize NLP pipeline for batch processing") from e

            start_time = time.time()
            with tqdm(total=total, desc="Processing SOAP notes", mininterval=1.0, disable=False) as pbar:
                for offset in range(0, total, adjusted_batch_size):
                    batch_start = time.time()
                    try:
                        batch = base_query.offset(offset).limit(adjusted_batch_size).all()
                        for note, patient in batch:
                            try:
                                update_single_soap_note(note.id, analyzer)
                                success += 1
                            except Exception as e:
                                error += 1
                                failed_notes.append(note.id)
                                logger.error(f"Failed to process note {note.id}: {e}")
                        db.session.commit()
                        logger.info(f"Processed batch {offset//adjusted_batch_size + 1} in {time.time() - batch_start:.2f} seconds")
                    except Exception as e:
                        db.session.rollback()
                        logger.error(f"Batch {offset//adjusted_batch_size + 1} failed: {e}", exc_info=True)
                        error += len(batch)
                        failed_notes.extend([note.id for note, _ in batch])
                    finally:
                        db.session.expunge_all()
                    pbar.update(len(batch))

            total_time = time.time() - start_time
            logger.info(f"Batch processing complete: {success} successes, {error} errors in {total_time:.2f} seconds")
            if failed_notes:
                logger.info(f"Failed notes: {failed_notes}")
            return success, error
        except Exception as e:
            db.session.rollback()
            logger.error(f"Batch processing failed: {e}", exc_info=True)
            raise RuntimeError("Batch update failed") from e
        finally:
            db.session.close()