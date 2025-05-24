from typing import Tuple, List, Dict, Optional
from departments.models.medicine import SOAPNote
from departments.models.records import Patient
from extensions import db
from departments.nlp.logging_setup import get_logger
from departments.nlp.config import BATCH_SIZE, UTS_API_KEY, UTS_BASE_URL
from departments.nlp.knowledge_base_io import load_knowledge_base, save_knowledge_base
from departments.nlp.note_processing import generate_ai_summary
from departments.nlp.ai_summary import generate_ai_analysis
from departments.nlp.clinical_analyzer import ClinicalAnalyzer
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from tqdm import tqdm
import requests
import json
import time
import psutil

logger = get_logger()

def query_uts_api(symptoms: List[str], analyzer: ClinicalAnalyzer, mongo_uri: str = 'mongodb://localhost:27017', db_name: str = 'clinical_db', retries: int = 3, delay: float = 1.0) -> List[Dict]:
    """Batch query UMLS CUIs and semantic types using ClinicalAnalyzer, with MongoDB caching and retries."""
    results = []
    cache = {}
    
    try:
        client = MongoClient(mongo_uri)
        db = client[db_name]
        cache_collection = db['umls_cache']
        
        for symptom in symptoms:
            symptom_lower = symptom.lower().strip()
            # Check in-memory cache
            if symptom_lower in cache:
                logger.debug(f"Retrieved cached UMLS data for '{symptom_lower}' from in-memory cache")
                results.append(cache[symptom_lower])
                continue
                
            # Query UMLS via ClinicalAnalyzer
            for attempt in range(retries):
                try:
                    cui, semantic_type = analyzer._get_umls_cui(symptom_lower, cache_collection='umls_cache')
                    result = {
                        'cui': cui,
                        'semantic_type': semantic_type,
                        'preferred_term': symptom
                    }
                    cache[symptom_lower] = result
                    results.append(result)
                    logger.debug(f"Retrieved UMLS data for '{symptom_lower}': CUI={cui}, Semantic Type={semantic_type}")
                    break
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1}/{retries} failed for '{symptom_lower}': {str(e)}")
                    if attempt < retries - 1:
                        time.sleep(delay)
                    else:
                        logger.error(f"Failed to retrieve UMLS data for '{symptom_lower}' after {retries} attempts")
                        results.append({
                            'cui': None,
                            'semantic_type': 'Unknown',
                            'preferred_term': symptom
                        })
        client.close()
    except Exception as e:
        logger.error(f"Error accessing MongoDB for UMLS cache: {str(e)}")
        for symptom in symptoms:
            symptom_lower = symptom.lower().strip()
            if symptom_lower in cache:
                results.append(cache[symptom_lower])
            else:
                results.append({
                    'cui': None,
                    'semantic_type': 'Unknown',
                    'preferred_term': symptom
                })
    
    return results

def mock_uts_symptom(symptom: str) -> dict:
    """Mock UTS API for testing."""
    mock_data = {
        "fatigue": {"cui": "C0013144", "semantic_type": "Sign or Symptom", "preferred_term": "Fatigue"},
        "cough": {"cui": "C0010200", "semantic_type": "Sign or Symptom", "preferred_term": "Cough"},
        "headache": {"cui": "C0018681", "semantic_type": "Sign or Symptom", "preferred_term": "Headache"},
        "fever": {"cui": "C0015967", "semantic_type": "Sign or Symptom", "preferred_term": "Fever"},
        "chest tightness": {"cui": "C0242209", "semantic_type": "Sign or Symptom", "preferred_term": "Chest Tightness"},
        "back pain": {"cui": "C0004604", "semantic_type": "Sign or Symptom", "preferred_term": "Back Pain"},
        "obesity": {"cui": "C0028754", "semantic_type": "Disease or Syndrome", "preferred_term": "Obesity"}
    }
    return mock_data.get(symptom.lower(), {'cui': None, 'semantic_type': 'Unknown', 'preferred_term': symptom})

def update_single_soap_note(note_id: int, analyzer: Optional[ClinicalAnalyzer] = None) -> None:
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

        # Initialize ClinicalAnalyzer if not provided
        if analyzer is None:
            analyzer = ClinicalAnalyzer()

        # Extract clinical features and update with UMLS data
        features = analyzer.extract_clinical_features(note)
        symptoms = features.get('symptoms', [])
        umls_data = []
        if symptoms:
            symptom_descriptions = [s['description'] for s in symptoms]
            logger.info(f"Extracted symptoms for note ID {note.id}: {symptom_descriptions}")
            umls_data = query_uts_api(symptom_descriptions, analyzer)
            for symptom, umls_result in zip(symptom_descriptions, umls_data):
                for s in symptoms:
                    if s['description'].lower() == symptom.lower():
                        # Update only if CUI is missing or invalid
                        if not s.get('umls_cui') or s.get('umls_cui') == 'Unknown':
                            s['umls_cui'] = umls_result['cui']
                            s['semantic_type'] = umls_result['semantic_type']
        else:
            logger.debug(f"No symptoms extracted for note ID {note.id}")

        # Generate AI summary
        if note.ai_notes is None:
            ai_summary = generate_ai_summary(note)
            if not isinstance(ai_summary, str):
                logger.error(f"generate_ai_summary returned non-string: {type(ai_summary)}")
                ai_summary = "Summary unavailable"
            if umls_data:
                symptom_text = "\nExtracted Symptoms with UMLS Data:\n" + "\n".join(
                    f"- {d['preferred_term']} (CUI: {d['cui']}, Semantic Type: {d['semantic_type']})"
                    for d in umls_data if d['cui']
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
                differentials = analyzer.generate_differential_dx(features['symptoms'], features['chief_complaint'])
                if differentials and differentials[0]['diagnosis'] != "Undetermined":
                    diff_text = "\nDifferential Diagnoses: " + ", ".join(f"{dx['diagnosis']} ({dx['confidence']:.2f})" for dx in differentials)
                    ai_analysis += diff_text
                if umls_data:
                    umls_text = f"\nUMLS Mappings:\n" + "\n".join(
                        f"- {d['preferred_term']} (CUI: {d['cui']})" for d in umls_data if d['cui']
                    )
                    ai_analysis += umls_text
            note.ai_analysis = ai_analysis

        db.session.commit()
        logger.info(f"Successfully updated AI notes for SOAP note ID {note.id}")
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error updating AI notes for SOAP note ID {note_id}: {str(e)}")
        raise

def update_all_ai_notes(batch_size: int = BATCH_SIZE) -> Tuple[int, int]:
    """Process all notes needing AI updates, including symptom extraction and UMLS integration."""
    from flask import current_app
    with current_app.app_context():
        try:
            # Adjust batch size based on available memory
            available_memory = psutil.virtual_memory().available / (1024 ** 2)  # MB
            adjusted_batch_size = min(batch_size, max(10, int(available_memory // 100)))  # ~100MB per note
            logger.info(f"Adjusted batch size to {adjusted_batch_size} based on {available_memory:.2f} MB available")

            # Query notes needing updates
            base_query = db.session.query(SOAPNote, Patient).outerjoin(
                Patient, SOAPNote.patient_id == Patient.patient_id
            ).filter(
                (SOAPNote.ai_notes.is_(None)) |
                (SOAPNote.ai_analysis.is_(None))
            )
            total = base_query.count()
            logger.debug(f"Total notes to process: {total}")
            if total == 0:
                return 0, 0

            success = error = 0
            failed_notes = []
            analyzer = ClinicalAnalyzer()

            with tqdm(total=total, desc="Processing SOAP notes", mininterval=1.0, disable=False) as pbar:
                for offset in range(0, total, adjusted_batch_size):
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
                            umls_data = []
                            if symptoms:
                                symptom_descriptions = [s['description'] for s in symptoms]
                                logger.debug(f"Extracted symptoms for note ID {note.id}: {symptom_descriptions}")
                                umls_data = query_uts_api(symptom_descriptions, analyzer)
                                for symptom, umls_result in zip(symptom_descriptions, umls_data):
                                    for s in symptoms:
                                        if s['description'].lower() == symptom.lower():
                                            if not s.get('umls_cui') or s.get('umls_cui') == 'Unknown':
                                                s['umls_cui'] = umls_result['cui']
                                                s['semantic_type'] = umls_result['semantic_type']
                            if note.ai_notes is None:
                                ai_summary = generate_ai_summary(note)
                                if not isinstance(ai_summary, str):
                                    logger.error(f"generate_ai_summary returned non-string: {type(ai_summary)}")
                                    ai_summary = "Summary unavailable"
                                if umls_data:
                                    symptom_text = "\nExtracted Symptoms with UMLS Data:\n" + "\n".join(
                                        f"- {d['preferred_term']} (CUI: {d['cui']}, Semantic Type: {d['semantic_type']})"
                                        for d in umls_data if d['cui']
                                    )
                                    ai_summary += symptom_text
                                note.ai_notes = ai_summary
                            if note.ai_analysis is None:
                                ai_analysis = generate_ai_analysis(note, patient)
                                if not isinstance(ai_analysis, str):
                                    logger.error(f"generate_ai_analysis returned non-string: {type(ai_analysis)}")
                                if symptoms:
                                    differentials = analyzer.generate_differential_dx(features['symptoms'], features['chief_complaint'])
                                    if differentials and differentials[0]['diagnosis'] != "Undetermined":
                                        diff_text = "\nDifferential Diagnoses: " + ", ".join(f"{dx['diagnosis']} ({dx['confidence']:.2f})" for dx in differentials)
                                        ai_analysis += diff_text
                                    if umls_data:
                                        umls_text = f"\nUMLS Mappings:\n" + "\n".join(
                                            f"- {d['preferred_term']} (CUI: {d['cui']})" for d in umls_data if d['cui']
                                        )
                                        ai_analysis += umls_text
                                note.ai_analysis = ai_analysis
                            db.session.add(note)
                            success += 1
                            logger.debug(f"Processed note {note.id} successfully")
                        except Exception as e:
                            error += 1
                            logger.error(f"Error processing note {note.id}: {str(e)}")
                            failed_notes.append(note.id)
                    db.session.commit()
                    db.session.expunge_all()
                    pbar.update(len(batch))
            if failed_notes:
                logger.info(f"Failed notes: {failed_notes}")
            logger.info(f"Batch processing complete: {success} successes, {error} errors")
            return success, error
        except Exception as e:
            db.session.rollback()
            logger.error(f"Batch processing failed: {str(e)}")
            raise RuntimeError("Batch update failed")