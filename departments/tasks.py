# departments/tasks.py
from celery import shared_task
from flask import current_app
from departments.models.medicine import SOAPNote
from departments.models.records import Patient
from departments.nlp.note_processing import generate_ai_summary
from departments.nlp.ai_summary import generate_ai_analysis
from extensions import db
import logging

logger = logging.getLogger(__name__)

@shared_task
def update_ai_note(note_id, patient_id):
    with current_app.app_context():
        try:
            note = SOAPNote.query.get(note_id)
            patient = Patient.query.get(patient_id)
            if note.ai_notes is None:
                note.ai_notes = generate_ai_summary(note)
            if note.ai_analysis is None:
                note.ai_analysis = generate_ai_analysis(note, patient)
            db.session.commit()
            logger.info(f"Updated AI note for SOAP note ID {note_id}")
        except Exception as e:
            logger.error(f"Failed to update AI note for SOAP note ID {note_id}: {str(e)}")
            db.session.rollback()