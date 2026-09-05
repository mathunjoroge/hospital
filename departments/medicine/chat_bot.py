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

from departments.api.ai_audit import (
    AIInputValidationError,
    AIMode,
    AITimer,
    log_ai_call,
    validate_ai_input,
)
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

            def generate():
                try:
                    # Assemble full context (past turns + new input + file content)
                    conversation_context = session.get('conversation', [])[:]
                    combined_input = input_note
                    if file_content:
                        combined_input += f"\n\n[Attachment: {file_name}]\n{file_content}" if input_note else f"[Attachment: {file_name}]\n{file_content}"
                    conversation_context.append({'role': 'user', 'content': combined_input})

                    logger.info(f"Processing input ({len(combined_input)} chars) for session {session_id}")

                    # Generate AI summary with latency tracking
                    with AITimer() as timer:
                        summary_html = Summarizer.answer(combined_input, conversation_history=conversation_context)

                    # Determine mode based on API key availability
                    ai_mode = AIMode.LIVE_LLM if (gemini_api_key or nvidia_api_key) else AIMode.OFFLINE_FALLBACK

                    # Extract plain text for storage and audit
                    raw_text_response = bleach.clean(summary_html, tags=[], strip=True)
                    raw_text_response = re.sub(r'Response generated on.*', '', raw_text_response, flags=re.DOTALL)
                    raw_text_response = re.sub(r'Powered by Gemini AI.*', '', raw_text_response, flags=re.DOTALL)
                    raw_text_response = re.sub(r'\s{2,}', ' ', raw_text_response).strip()

                    # Audit log this AI call
                    log_ai_call(
                        feature='clinical_chatbot',
                        mode=ai_mode,
                        input_summary=combined_input[:200],
                        output_summary=raw_text_response[:200],
                        latency_ms=timer.elapsed_ms
                    )

                    # Store both user and model messages
                    session['conversation'].append({'role': 'user', 'content': combined_input})
                    session['conversation'].append({'role': 'model', 'content': raw_text_response})

                    # Limit stored turns
                    if len(session['conversation']) > 10:
                        session['conversation'] = session['conversation'][-10:]

                    # Mark session as modified and persist
                    session.modified = True
                    session['_last_save'] = time.time()
                    persist_session(current_app, session)
                    logger.debug(f"Saved conversation (total turns: {len(session['conversation'])})")

                    # Stream response in chunks
                    chunk_size = 50
                    for i in range(0, len(summary_html), chunk_size):
                        yield summary_html[i:i + chunk_size].encode('utf-8')
                        time.sleep(0.05)

                except Exception as e:
                    logger.error(f"Error generating AI response: {e}", exc_info=True)
                    log_ai_call(
                        feature='clinical_chatbot',
                        mode=AIMode.OFFLINE_FALLBACK,
                        input_summary=(input_note or '')[:200],
                        output_summary='',
                        error='Internal error during inference'
                    )
                    yield Summarizer._format_output(
                        "An error occurred while processing your request. Please try again.",
                        is_error=True
                    ).encode('utf-8')

            # Persist session before streaming starts
            persist_session(current_app, session)
            return Response(stream_with_context(generate()), content_type='text/html; charset=utf-8')

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


