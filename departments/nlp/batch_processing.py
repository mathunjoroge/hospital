from typing import Tuple
from departments.models.medicine import SOAPNote
from departments.models.records import Patient
from extensions import db
from departments.nlp.logging_setup import logger
from departments.nlp.config import BATCH_SIZE
from departments.nlp.note_processing import generate_ai_summary
from departments.nlp.ai_summary import generate_ai_analysis
from tqdm import tqdm

def update_all_ai_notes(batch_size: int = BATCH_SIZE) -> Tuple[int, int]:
    """Process all notes needing AI updates."""
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
                            if note.ai_notes is None:
                                note.ai_notes = generate_ai_summary(note)
                            if note.ai_analysis is None:
                                note.ai_analysis = generate_ai_analysis(note, patient)
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