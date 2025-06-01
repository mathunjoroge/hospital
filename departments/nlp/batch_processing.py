from typing import Tuple, List, Dict, Optional
from flask import current_app
from tqdm import tqdm
import time
import psutil
from typing import Tuple, List, Dict, Optional
from departments.models.medicine import SOAPNote
from departments.models.records import Patient
from extensions import db
from departments.nlp.logging_setup import get_logger
from departments.nlp.config import BATCH_SIZE
from departments.nlp.knowledge_base_io import load_knowledge_base, save_knowledge_base
from departments.nlp.note_processing import generate_ai_summary
from departments.nlp.ai_summary import generate_ai_analysis
from departments.nlp.clinical_analyzer import ClinicalAnalyzer

logger = get_logger()

def update_single_soap_note(note_id: int, analyzer: ClinicalAnalyzer) -> None:
    """Update AI-generated summary, analysis, and symptoms for a single SOAP note with UMLS integration."""
    logger.info(f"Updating AI notes for SOAP note ID {note_id}")
    try:
        note = db.session.query(SOAPNote).get(note_id)
        if not note:
            logger.error(f"SOAP note ID {note_id} not found")
            return
        patient = db.session.query(Patient).get(note.patient_id)
        if not patient:
            logger.warning(f"Patient not found for SOAP note ID {note.id}")
            patient = None

        # Extract clinical features and update with UMLS data
        features = analyzer.extract_clinical_features(note)
        symptoms = features.get('symptoms', [])
        if symptoms:
            symptom_descriptions = [s['description'] for s in symptoms if s.get('description')]
            logger.info(f"Extracted symptoms for note ID {note.id}: {symptom_descriptions}")
            for symptom in symptom_descriptions:
                for s in symptoms:
                    if s['description'].lower() == symptom.lower():
                        if not s.get('umls_cui') or s.get('umls_cui') == 'Unknown':
                            cui, semantic_type = analyzer._get_umls_cui(symptom)
                            s['umls_cui'] = cui
                            s['semantic_type'] = semantic_type
        else:
            logger.debug(f"No symptoms extracted for note ID {note.id}")

        # Generate AI summary
        if note.ai_notes is None:
            ai_summary = generate_ai_summary(note, analyzer)
            if not isinstance(ai_summary, str):
                logger.error(f"generate_ai_summary returned non-string: {type(ai_summary)}")
                ai_summary = "Summary unavailable"
            if symptoms:
                symptom_text = "\nExtracted Symptoms with UMLS Data:\n" + "\n".join(
                    f"- {s['description']} (CUI: {s['umls_cui']}, Semantic Type: {s['semantic_type']})"
                    for s in symptoms if s.get('umls_cui')
                )
                ai_summary += symptom_text
            note.ai_notes = ai_summary

        # Generate AI analysis
        if note.ai_analysis is None:
            ai_analysis = generate_ai_analysis(note, patient)
            if not isinstance(ai_analysis, str):
                logger.error(f"generate_ai_analysis returned non-string: {type(ai_analysis)}")
                ai_analysis = "Analysis unavailable"
            if symptoms and patient:
                differentials = analyzer.generate_differential_dx(features, patient)
                if differentials and differentials[0][0] != "Undetermined":
                    diff_text = "\nDifferential Diagnoses: " + ", ".join(f"{dx[0]} ({dx[1]:.2f})" for dx in differentials)
                    ai_analysis += diff_text
                if symptoms:
                    umls_text = f"\nUMLS Mappings:\n" + "\n".join(
                        f"- {s['description']} (CUI: {s['umls_cui']})" for s in symptoms if s.get('umls_cui')
                    )
                    ai_analysis += umls_text
            note.ai_analysis = ai_analysis

        db.session.commit()
        logger.info(f"Successfully updated AI notes for SOAP note ID {note.id}")
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error updating AI notes for SOAP note ID {note_id}: {e}", exc_info=True)
        raise

def update_all_ai_notes(batch_size: int = BATCH_SIZE) -> Tuple[int, int]:
    """Process all notes needing AI updates, including symptom extraction and UMLS integration."""
    with current_app.app_context():
        try:
            # Adjust batch size based on available memory
            available_memory = psutil.virtual_memory().available / (1024 ** 2)  # MB
            memory_per_note = 50  # Estimated MB per note
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
            # Initialize ClinicalAnalyzer
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
                    batch = base_query.offset(offset).limit(adjusted_batch_size).all()
                    for note, patient in batch:
                        try:
                            if not isinstance(note, SOAPNote):
                                logger.error(f"Invalid note type for ID {note.id}: {type(note)}")
                                error += 1
                                failed_notes.append(note.id)
                                continue
                            features = analyzer.extract_clinical_features(note)
                            symptoms = features.get('symptoms', [])
                            if symptoms:
                                symptom_descriptions = [s['description'] for s in symptoms if s.get('description')]
                                if not symptom_descriptions:
                                    logger.warning(f"Empty symptom descriptions for note ID {note.id}")
                                    continue
                                logger.debug(f"Extracted symptoms for note ID {note.id}: {symptom_descriptions}")
                                for symptom in symptom_descriptions:
                                    for s in symptoms:
                                        if s['description'].lower() == symptom.lower():
                                            if not s.get('umls_cui') or s.get('umls_cui') == 'Unknown':
                                                cui, semantic_type = analyzer._get_umls_cui(symptom)
                                                s['umls_cui'] = cui
                                                s['semantic_type'] = semantic_type
                            if note.ai_notes is None:
                                ai_summary = generate_ai_summary(note, analyzer)
                                if not isinstance(ai_summary, str):
                                    logger.error(f"generate_ai_summary returned non-string: {type(ai_summary)}")
                                    ai_summary = "Summary unavailable"
                                if symptoms:
                                    symptom_text = "\nExtracted Symptoms with UMLS Data:\n" + "\n".join(
                                        f"- {s['description']} (CUI: {s['umls_cui']}, Semantic Type: {s['semantic_type']})"
                                        for s in symptoms if s.get('umls_cui')
                                    )
                                    ai_summary += symptom_text
                                note.ai_notes = ai_summary
                            if note.ai_analysis is None:
                                ai_analysis = generate_ai_analysis(note, patient)
                                if not isinstance(ai_analysis, str):
                                    logger.error(f"generate_ai_analysis returned non-string: {type(ai_analysis)}")
                                if symptoms and patient:
                                    differentials = analyzer.generate_differential_dx(features, patient)
                                    if differentials and differentials[0][0] != "Undetermined":
                                        diff_text = "\nDifferential Diagnoses: " + ", ".join(f"{dx[0]} ({dx[1]:.2f})" for dx in differentials)
                                        ai_analysis += diff_text
                                    if symptoms:
                                        umls_text = f"\nUMLS Mappings:\n" + "\n".join(
                                            f"- {s['description']} (CUI: {s['umls_cui']})" for s in symptoms if s.get('umls_cui')
                                        )
                                        ai_analysis += umls_text
                                note.ai_analysis = ai_analysis
                            db.session.add(note)
                            success += 1
                            logger.debug(f"Processed note {note.id} successfully")
                        except Exception as e:
                            error += 1
                            logger.error(f"Error processing note {note.id}: {e}", exc_info=True)
                            failed_notes.append(note.id)
                    db.session.commit()
                    db.session.expunge_all()
                    batch_time = time.time() - batch_start
                    logger.info(f"Processed batch {offset//adjusted_batch_size + 1} in {batch_time:.2f} seconds")
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