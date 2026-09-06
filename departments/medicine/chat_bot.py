import csv
import os
import re
import time
from uuid import uuid4

import bleach

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    from docx import Document
except ImportError:
    Document = None

from flask import (
    Response,
    current_app,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    stream_with_context,
)
from flask_login import login_required
from flask_wtf import FlaskForm
from flask_wtf.csrf import CSRFError
from PIL import Image
from werkzeug.utils import secure_filename

from departments.tasks import process_clinical_chatbot_task

from departments.api.ai_audit import (
    AIInputValidationError,
    AIMode,
    AITimer,
    log_ai_call,
    validate_ai_input,
)
from departments.api.audit import log_audit_event
from departments.models.compliance import has_ai_consent
from departments.nlp.chatbot import UniversalClinicalSummarizer
from departments.nlp.logging_setup import get_logger

from . import bp

logger = get_logger()

# Instantiate the summarizer for use in chatbot_interface

gemini_api_key = os.environ.get("GEMINI_API_KEY")
nvidia_api_key = os.environ.get("NVIDIA_API_KEY")
Summarizer = UniversalClinicalSummarizer(gemini_api_key=gemini_api_key, nvidia_api_key=nvidia_api_key)



ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf', 'txt', 'csv', 'docx'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB max file size


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_file_content(file):
    """Extract content from uploaded file based on its type."""
    filename = secure_filename(file.filename)
    extension = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''

    try:
        if extension in ['png', 'jpg', 'jpeg', 'gif']:
            # Extract text from images using OCR
            if not pytesseract:
                return "OCR capability (pytesseract) is not installed."
            img = Image.open(file.stream)
            text = pytesseract.image_to_string(img)
            return text.strip() or "No text could be extracted from the image."

        elif extension == 'pdf':
            # Extract text from PDF
            if not PyPDF2:
                return "PDF extraction capability (PyPDF2) is not installed."
            reader = PyPDF2.PdfReader(file.stream)
            text = ''
            for page in reader.pages:
                text += page.extract_text() or ''
            return text.strip() or "No text could be extracted from the PDF."

        elif extension == 'txt':
            # Read plain text
            text = file.stream.read().decode('utf-8')
            return text.strip()

        elif extension == 'csv':
            # Read CSV content
            text = ''
            reader = csv.reader(file.stream.read().decode('utf-8').splitlines())
            for row in reader:
                text += ' '.join(row) + '\n'
            return text.strip()

        elif extension == 'docx':
            # Extract text from DOCX
            if not Document:
                return "DOCX extraction capability (docx) is not installed."
            doc = Document(file.stream)
            text = '\n'.join([para.text for para in doc.paragraphs])
            return text.strip() or "No text could be extracted from the DOCX."

        else:
            return "Unsupported file type."

    except Exception as e:
        logger.error(f"Error extracting content from file {filename}: {e}")
        return f"Error processing file: {str(e)}"

def persist_session(app, sess):
    """
    Force Flask to persist session data during streaming responses.
    Works safely for both Redis and filesystem session backends.
    """
    try:
        session_interface = app.session_interface

        if not hasattr(session_interface, 'save_session'):
            logger.warning("Session interface does not implement save_session().")
            return

        dummy_response = app.make_response('')
        session_interface.save_session(app, sess, dummy_response)

        if app.config.get('SESSION_TYPE') == 'filesystem':
            session_dir = app.config.get('SESSION_FILE_DIR')
            if session_dir:
                os.makedirs(session_dir, exist_ok=True)
                if not os.access(session_dir, os.W_OK):
                    logger.warning(f"Session directory is not writable: {session_dir}")
                else:
                    logger.debug(f"Session directory verified: {session_dir}")

        logger.debug("Session persisted successfully during stream.")

    except Exception as e:
        logger.error(f"Failed to persist session during stream: {e}", exc_info=True)

@bp.route('/chatbot', methods=['GET', 'POST'])
@login_required
def chatbot_interface():
    """
    Handles displaying the chatbot form (GET) and processing submitted notes and files (POST).
    Maintains conversation history in the session using role/content pairs for multi-turn coherence.
    """
    session_id = session.sid if hasattr(session, 'sid') else str(uuid4())

    # Initialize session conversation list if missing
    if 'conversation' not in session:
        session['conversation'] = []
        session.modified = True
        logger.debug(f"Initialized new conversation for session {session_id}")

    current_history = session.get('conversation', [])
    logger.debug(f"Session ID: {session_id}, Current conversation length: {len(current_history)}")

    # --- POST: Handle submitted note and/or file ---
    if request.method == 'POST':
        try:
            input_note = request.form.get('clinical_note', '').strip()
            file = request.files.get('attachment')
            file_content = None
            file_name = None

            # Validate and process file if uploaded
            if file and file.filename:
                if not allowed_file(file.filename):
                    logger.warning(f"Invalid file type uploaded: {file.filename}")
                    return Response(
                        Summarizer._format_output(
                            "Invalid file type. Allowed types: images, PDF, TXT, CSV, DOCX.",
                            is_error=True
                        ).encode('utf-8'),
                        status=400
                    )

                # Check file size
                file.stream.seek(0, os.SEEK_END)
                file_size = file.stream.tell()
                file.stream.seek(0)
                if file_size > MAX_FILE_SIZE:
                    logger.warning(f"File too large: {file.filename}, size: {file_size} bytes")
                    return Response(
                        Summarizer._format_output(
                            f"File size exceeds limit of {MAX_FILE_SIZE // (1024 * 1024)}MB.",
                            is_error=True
                        ).encode('utf-8'),
                        status=400
                    )

                file_name = secure_filename(file.filename)
                file_content = extract_file_content(file)
                logger.debug(f"Extracted content from file {file_name}: {file_content[:80]}...")

            if not input_note and not file_content:
                logger.warning("No input note or valid file content provided.")
                return Response(
                    Summarizer._format_output(
                        "Please provide a clinical note or a valid file.", is_error=True
                    ).encode('utf-8'),
                    status=400
                )

            # Validate combined input length before processing
            combined_check = (input_note or '') + (file_content or '')
            try:
                validate_ai_input(combined_check, feature='clinical_chatbot')
            except AIInputValidationError as ve:
                logger.warning(f"Chatbot input validation failed: {ve}")
                return Response(
                    Summarizer._format_output(str(ve), is_error=True).encode('utf-8'),
                    status=400
                )

            # AI Consent check if patient_id is provided
            patient_id = (request.form.get('patient_id') or request.args.get('patient_id') or '').strip()
            if patient_id:
                if not has_ai_consent(patient_id):
                    logger.warning(f"AI consent check failed for patient_id={patient_id}")
                    log_audit_event(
                        action='AI_CONSENT_REFUSED',
                        resource_type='Patient',
                        resource_id=patient_id,
                        details={'feature': 'clinical_chatbot', 'reason': 'Missing or revoked ai_diagnosis consent'}
                    )
                    return Response(
                        Summarizer._format_output(
                            "AI-assisted summary unavailable: patient has not consented to AI processing of clinical notes.",
                            is_error=True
                        ).encode('utf-8'),
                        status=403
                    )
                else:
                    log_audit_event(
                        action='AI_CONSENT_GRANTED',
                        resource_type='Patient',
                        resource_id=patient_id,
                        details={'feature': 'clinical_chatbot'}
                    )

            # Assemble full context
            conversation_context = session.get('conversation', [])[:]
            combined_input = input_note
            if file_content:
                combined_input += (
                    f"\n\n[Attachment: {file_name}]\n{file_content}"
                    if input_note
                    else f"[Attachment: {file_name}]\n{file_content}"
                )

            # Dispatch Celery task asynchronously
            task = process_clinical_chatbot_task.delay(combined_input, conversation_context, patient_id)
            logger.info(f"Dispatched chatbot task {task.id} via Celery for session {session_id}")

            persist_session(current_app, session)

            return jsonify({
                'status': 'PROCESSING',
                'task_id': task.id,
                'message': 'Clinical note analysis queued via Celery task worker.',
                'status_url': f'/medicine/chatbot/status/{task.id}'
            }), 202

        except CSRFError as e:
            logger.error(f"CSRF validation failed: {e}")
            return Response(
                Summarizer._format_output(
                    "CSRF validation failed. Please refresh and try again.", is_error=True
                ).encode('utf-8'),
                status=403
            )

    # --- GET: Render chat page ---
    class ChatForm(FlaskForm):
        pass

    class ClearForm(FlaskForm):
        pass

    logger.debug(f"Rendering chatbot template for session {session_id}")
    return render_template(
        'medicine/chat_bot.html',
        conversation=[{'question': entry['content'], 'response': Summarizer._format_output(entry['content'])}
                      for entry in current_history if entry['role'] == 'user'],
        input_note='',
        form=ChatForm(),
        clear_form=ClearForm()
    )

@bp.route('/clear_conversation', methods=['POST'])
@login_required
def clear_conversation():
    """
    Clears conversation history from the session.
    """
    session_id = session.sid if hasattr(session, 'sid') else str(uuid4())
    logger.debug(f"Clearing conversation for session: {session_id}")
    session.pop('conversation', None)
    session.modified = True
    persist_session(current_app, session)
    logger.info(f"Conversation cleared for session {session_id}")
    return redirect('/medicine/chatbot')

@bp.route('/chatbot/status/<task_id>', methods=['GET'])
@login_required
def chatbot_task_status(task_id):
    """
    Polling endpoint for checking clinical chatbot Celery task execution status.
    """
    try:
        try:
            from celery.result import AsyncResult
            task_result = AsyncResult(task_id)
            state = task_result.state
            res = task_result.result or {}
        except ImportError:
            state = 'SUCCESS'
            res = {'status': 'SUCCESS', 'raw_text': 'Analysis completed', 'summary_html': '<div>Analysis completed</div>'}
    except Exception as exc:
        logger.error(f"Error checking status for Celery task {task_id}: {exc}")
        return jsonify({'status': 'ERROR', 'error': str(exc)}), 500

    if state in ('PENDING', 'RECEIVED', 'STARTED'):
        return jsonify({'status': 'PROCESSING', 'state': state, 'task_id': task_id}), 200
    elif state == 'SUCCESS':
        input_text = res.get('input_note', '')
        raw_text = res.get('raw_text', '')
        summary_html = res.get('summary_html', '')

        if 'conversation' in session:
            conv = session.get('conversation', [])
            if input_text and not any(turn.get('content') == input_text for turn in conv):
                session['conversation'].append({'role': 'user', 'content': input_text})
                session['conversation'].append({'role': 'model', 'content': raw_text})
                if len(session['conversation']) > 10:
                    session['conversation'] = session['conversation'][-10:]
                session.modified = True
                persist_session(current_app, session)

        return jsonify({
            'status': 'SUCCESS',
            'state': state,
            'task_id': task_id,
            'summary_html': summary_html,
            'raw_text': raw_text
        }), 200
    else:
        return jsonify({
            'status': 'FAILURE',
            'state': state,
            'task_id': task_id,
            'error': str(task_result.info if 'task_result' in locals() else 'Task failed')
        }), 500
