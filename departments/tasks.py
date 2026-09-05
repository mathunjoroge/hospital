try:
    from celery import shared_task
except ImportError:
    def shared_task(func):
        return func
import logging

from flask import current_app

from departments.models.medicine import SOAPNote
from departments.models.records import Patient
from departments.nlp.summarizer import ClinicalSummarizer
from extensions import db

logger = logging.getLogger(__name__)

@shared_task
def update_ai_note(note_id, patient_id):
    with current_app.app_context():
        try:
            note = SOAPNote.query.get(note_id)
            Patient.query.get(patient_id)
            if note:
                text_content = f"{note.situation or ''} {note.hpi or ''} {note.assessment or ''} {note.recommendation or ''}".strip()
                if text_content:
                    summarizer = ClinicalSummarizer()
                    summary_result = summarizer.summarize(text_content)
                    if note.ai_notes is None:
                        note.ai_notes = summary_result
                    if note.ai_analysis is None:
                        note.ai_analysis = summary_result
                    db.session.commit()
                    logger.info(f"Updated AI note for SOAP note ID {note_id}")
        except Exception as e:
            logger.error(f"Failed to update AI note for SOAP note ID {note_id}: {str(e)}")
            db.session.rollback()
