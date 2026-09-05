import requests 
import pickle
import re
from flask import render_template, redirect, url_for, request, flash, jsonify,session
from flask_wtf import FlaskForm
from sqlalchemy import func
from wtforms import SelectField
from wtforms.validators import DataRequired
from sqlalchemy import text
import json
from contextlib import contextmanager
from typing import Optional, List, Dict, Any
import psycopg2
from datetime import date
from psycopg2.extras import RealDictCursor
from flask import current_app
from flask_login import login_required, current_user
from departments.rbac import roles_required, get_effective_role
from flask_wtf.csrf import CSRFProtect,CSRFError
from scipy.spatial.distance import cosine
from extensions import db
from flask import session
from flask_socketio import SocketIO
import uuid
from uuid import uuid4
from sqlalchemy.orm import joinedload
import bleach 
from . import bp
from departments.forms import PatientSearchForm, OncoPatientForm, OncologyNoteForm, AdmitPatientForm
import os
from datetime import datetime
from departments.models.laboratory import LabResult,LabResultTemplate
from departments.models.records import PatientWaitingList, Patient
from departments.models.medicine import (
    SOAPNote, LabTest, Imaging, Medicine, PrescribedMedicine, RequestedLab, 
    RequestedImage, UnmatchedImagingRequest, TheatreProcedure, TheatreList, 
    Ward, AdmittedPatient, SpecialWarning,RegimenDrugAssociation, 
    OncologyBooking, OncoDrugCategory, RegimenCategory, 
    WardBedHistory, WardRoom, Bed, WardRound,Disease, 
    DiseaseManagementPlan, DiseaseLab, OncoPatient, 
    OncologyDrug, OncologyRegimen, OncoPrescription, 
    OncoTreatmentRecord,PrescriptionDrugDetail,OncologyNote,
    CancerType, CancerStage, CancerTypeStage, CancerDetail
)
from departments.nlp.chatbot import UniversalClinicalSummarizer
import logging
import json
from flask import Response, stream_with_context, request
import time
from departments.nlp.logging_setup import get_logger
from flask.sessions import SecureCookieSessionInterface
logger = get_logger()
import PyPDF2  # For PDF processing
from docx import Document  # For DOCX processing
import pytesseract  # For OCR on images
from PIL import Image  # For image handling
import csv  #
from werkzeug.utils import secure_filename

# Instantiate the summarizer for use in chatbot_interface


from extensions import csrf

gemini_api_key = os.environ.get("GEMINI_API_KEY")
nvidia_api_key = os.environ.get("NVIDIA_API_KEY")
Summarizer = UniversalClinicalSummarizer(gemini_api_key=gemini_api_key, nvidia_api_key=nvidia_api_key)


from flask import make_response

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
            img = Image.open(file.stream)
            text = pytesseract.image_to_string(img)
            return text.strip() or "No text could be extracted from the image."
        
        elif extension == 'pdf':
            # Extract text from PDF
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

            def generate():
                try:
                    # Assemble full context (past turns + new input + file content)
                    conversation_context = session.get('conversation', [])[:]
                    combined_input = input_note
                    if file_content:
                        combined_input += f"\n\n[Attachment: {file_name}]\n{file_content}" if input_note else f"[Attachment: {file_name}]\n{file_content}"
                    conversation_context.append({'role': 'user', 'content': combined_input})

                    logger.info(f"Processing input ({len(combined_input)} chars) for session {session_id}")

                    # Generate AI summary
                    summary_html = Summarizer.answer(combined_input, conversation_history=conversation_context)

                    # Extract plain text for storage
                    raw_text_response = bleach.clean(summary_html, tags=[], strip=True)
                    raw_text_response = re.sub(r'Response generated on.*', '', raw_text_response, flags=re.DOTALL)
                    raw_text_response = re.sub(r'Powered by Gemini AI.*', '', raw_text_response, flags=re.DOTALL)
                    raw_text_response = re.sub(r'\s{2,}', ' ', raw_text_response).strip()

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


