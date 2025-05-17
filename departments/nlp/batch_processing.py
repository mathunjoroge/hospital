# departments/nlp/batch_processing.py
from typing import Tuple
from departments.models.medicine import SOAPNote
from departments.models.records import Patient
from extensions import db
from departments.nlp.logging_setup import logger
from departments.nlp.config import BATCH_SIZE
from departments.nlp.note_processing import generate_ai_summary
from departments.nlp.ai_summary import generate_ai_analysis
from departments.nlp.clinical_analyzer import ClinicalAnalyzer
from tqdm import tqdm

def update_single_soap_note(note_id: int) -> None:
    """Update AI-generated summary, analysis, and symptoms for a single SOAP note."""
    logger.info(f"Updating AI notes for SOAP note ID {note_id}")
    try:
        note = SOAPNote.query.get(note_id)
        if not note:
            logger.error(f"SOAP note ID {note_id} not found")
            return
        patient = Patient.query.get(note.patient_id)
        if not patient:
            logger.warning(f"Patient not found for SOAP note ID {note.id}")
            patient = None

        # Initialize ClinicalAnalyzer
        analyzer = ClinicalAnalyzer()

        # Extract clinical features and update knowledge base
        features = analyzer.extract_clinical_features(note)
        symptoms = features.get('symptoms', [])
        if symptoms:
            symptom_descriptions = [s['description'] for s in symptoms]
            logger.info(f"Extracted symptoms for note ID {note.id}: {symptom_descriptions}")
        else:
            logger.debug(f"No symptoms extracted for note ID {note.id}")

        # Generate AI summary
        if note.ai_notes is None:
            ai_summary = generate_ai_summary(note)
            if not isinstance(ai_summary, str):
                logger.error(f"generate_ai_summary returned non-string: {type(ai_summary)}")
                ai_summary = "Summary unavailable"
            # Append symptoms to summary
            if symptoms:
                symptom_text = f"\nExtracted Symptoms: {', '.join(s['description'] for s in symptoms)}"
                ai_summary += symptom_text
            note.ai_notes = ai_summary

        # Generate AI analysis
        if note.ai_analysis is None:
            ai_analysis = generate_ai_analysis(note, patient)
            if not isinstance(ai_analysis, str):
                logger.error(f"generate_ai_analysis returned non-string: {type(ai_analysis)}")
                ai_analysis = "Analysis unavailable"
            # Append differential diagnoses if available
            if symptoms and patient:
                differentials = analyzer.generate_differential_dx(features, patient)
                if differentials and differentials[0][0] != "Undetermined":
                    diff_text = f"\nDifferential Diagnoses: {', '.join(f'{dx} ({score:.2f})' for dx, score, _ in differentials)}"
                    ai_analysis += diff_text
            note.ai_analysis = ai_analysis

        db.session.commit()
        logger.info(f"Successfully updated AI notes for SOAP note ID {note.id}")
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error updating AI notes for SOAP note ID {note_id}: {str(e)}")
        raise

def update_all_ai_notes(batch_size: int = BATCH_SIZE) -> Tuple[int, int]:
    """Process all notes needing AI updates, including symptom extraction."""
    from flask import current_app
    with current_app.app_context():
        try:
            base_query = db.session.query(SOAPNote, Patient).filter(
                (SOAPNote.ai_notes.is_(None)) |
                (SOAPNote.ai_analysis.is_(None)),
                SOAPNote.patient_id == Patient.patient_id
            )
            total = base_query.count()
            logger.debug(f"Total notes to process: {total}")
            if total == 0:
                return 0, 0
            success = error = 0
            failed_notes = []
            # Initialize ClinicalAnalyzer
            analyzer = ClinicalAnalyzer()
            with tqdm(total=total, desc="Processing SOAP notes") as pbar:
                for offset in range(0, total, batch_size):
                    batch = base_query.offset(offset).limit(batch_size).all()
                    for note, patient in batch:
                        try:
                            if not isinstance(note, SOAPNote):
                                logger.error(f"Invalid note type for ID {note.id}: {type(note)}")
                                error += 1
                                failed_notes.append(note.id)
                                continue
                            # Extract clinical features
                            features = analyzer.extract_clinical_features(note)
                            symptoms = features.get('symptoms', [])
                            if symptoms:
                                symptom_descriptions = [s['description'] for s in symptoms]
                                logger.debug(f"Extracted symptoms for note ID {note.id}: {symptom_descriptions}")
                            # Generate AI summary
                            if note.ai_notes is None:
                                ai_summary = generate_ai_summary(note)
                                if not isinstance(ai_summary, str):
                                    logger.error(f"generate_ai_summary returned non-string: {type(ai_summary)}")
                                    ai_summary = "Summary unavailable"
                                # Append symptoms to summary
                                if symptoms:
                                    symptom_text = f"\nExtracted Symptoms: {', '.join(s['description'] for s in symptoms)}"
                                    ai_summary += symptom_text
                                note.ai_notes = ai_summary
                            # Generate AI analysis
                            if note.ai_analysis is None:
                                ai_analysis = generate_ai_analysis(note, patient)
                                if not isinstance(ai_analysis, str):
                                    logger.error(f"generate_ai_analysis returned non-string: {type(ai_analysis)}")
                                    ai_analysis = "Analysis unavailable"
                                # Append differential diagnoses
                                if symptoms:
                                    differentials = analyzer.generate_differential_dx(features, patient)
                                    if differentials and differentials[0][0] != "Undetermined":
                                        diff_text = f"\nDifferential Diagnoses: {', '.join(f'{dx} ({score:.2f})' for dx, score, _ in differentials)}"
                                        ai_analysis += diff_text
                                note.ai_analysis = ai_analysis
                            db.session.add(note)
                            success += 1
                            logger.debug(f"Processed note {note.id} successfully")
                        except Exception as e:
                            error += 1
                            logger.error(f"Error processing note {note.id}: {str(e)}")
                            failed_notes.append(note.id)
                    db.session.commit()
                    pbar.update(len(batch))
            if failed_notes:
                logger.info(f"Failed notes: {failed_notes}")
            logger.info(f"Batch processing complete: {success} successes, {error} errors")
            return success, error
        except Exception as e:
            db.session.rollback()
            logger.error(f"Batch processing failed: {str(e)}")
            raise RuntimeError("Batch update failed")