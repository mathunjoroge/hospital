# departments/nlp/batch_processing.py
from typing import Tuple
from departments.models.medicine import SOAPNote
from departments.models.records import Patient
from extensions import db
from departments.nlp.logging_setup import logger
from departments.nlp.config import BATCH_SIZE, UTS_API_KEY, UTS_BASE_URL
from departments.nlp.note_processing import generate_ai_summary
from departments.nlp.ai_summary import generate_ai_analysis
from departments.nlp.clinical_analyzer import ClinicalAnalyzer
from tqdm import tqdm
import requests

def query_uts_api(symptom: str, api_key: str = UTS_API_KEY, base_url: str = UTS_BASE_URL) -> dict:
    """Query UTS API to retrieve UMLS CUI and semantic type for a symptom."""
    try:
        if api_key == "mock_api_key":
            return mock_uts_symptom(symptom)
        # Obtain a single-use ticket
        ticket_url = f"{base_url}/authentication/ticket"
        response = requests.post(ticket_url, data={'apiKey': api_key})
        response.raise_for_status()
        ticket = response.text

        # Search for CUI
        url = f"{base_url}/search/current"
        params = {
            'string': symptom.lower().strip(),
            'ticket': ticket,
            'searchType': 'exact',
            'sabs': 'SNOMEDCT_US'  # Limit to SNOMED CT for clinical terms
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if not data.get('result', {}).get('results'):
            logger.debug(f"No UMLS data found for symptom: {symptom}")
            return {'cui': None, 'semantic_type': 'Unknown', 'preferred_term': symptom}

        cui = data['result']['results'][0]['ui']
        # Get semantic type
        concept_url = f"{base_url}/content/current/CUI/{cui}"
        concept_response = requests.get(concept_url, params={'ticket': ticket})
        concept_response.raise_for_status()
        concept_data = concept_response.json()
        semantic_type = concept_data['result'].get('semanticTypes', [{}])[0].get('name', 'Unknown')
        preferred_term = data['result']['results'][0].get('name', symptom)
        
        logger.debug(f"UMLS data for '{symptom}': CUI={cui}, Semantic Type={semantic_type}, Preferred Term={preferred_term}")
        return {'cui': cui, 'semantic_type': semantic_type, 'preferred_term': preferred_term}
    except Exception as e:
        logger.error(f"UMLS query failed for '{symptom}': {str(e)}")
        return {'cui': None, 'semantic_type': 'Unknown', 'preferred_term': symptom}

def mock_uts_symptom(symptom: str) -> dict:
    """Mock UTS API for testing."""
    mock_data = {
        "fatigue": {"cui": "C0013144", "semantic_type": "Sign or Symptom", "preferred_term": "Fatigue"},
        "cough": {"cui": "C0010200", "semantic_type": "Sign or Symptom", "preferred_term": "Cough"},
        "headache": {"cui": "C0018681", "semantic_type": "Sign or Symptom", "preferred_term": "Headache"},
        "fever": {"cui": "C0015967", "semantic_type": "Sign or Symptom", "preferred_term": "Fever"}
    }
    return mock_data.get(symptom.lower(), {'cui': None, 'semantic_type': 'Unknown', 'preferred_term': symptom})

def update_single_soap_note(note_id: int) -> None:
    """Update AI-generated summary, analysis, and symptoms for a single SOAP note with UMLS integration."""
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

        # Extract clinical features and update with UMLS data
        features = analyzer.extract_clinical_features(note)
        symptoms = features.get('symptoms', [])
        umls_data = []
        if symptoms:
            symptom_descriptions = [s['description'] for s in symptoms]
            logger.info(f"Extracted symptoms for note ID {note.id}: {symptom_descriptions}")
            for symptom in symptom_descriptions:
                umls_result = query_uts_api(symptom)
                umls_data.append({
                    'symptom': symptom,
                    'cui': umls_result['cui'],
                    'semantic_type': umls_result['semantic_type'],
                    'preferred_term': umls_result['preferred_term']
                })
        else:
            logger.debug(f"No symptoms extracted for note ID {note.id}")

        # Generate AI summary
        if note.ai_notes is None:
            ai_summary = generate_ai_summary(note)
            if not isinstance(ai_summary, str):
                logger.error(f"generate_ai_summary returned non-string: {type(ai_summary)}")
                ai_summary = "Summary unavailable"
            # Append symptoms and UMLS data to summary
            if umls_data:
                symptom_text = "\nExtracted Symptoms with UMLS Data:\n" + "\n".join(
                    f"- {d['preferred_term']} (CUI: {d['cui']}, Semantic Type: {d['semantic_type']})"
                    for d in umls_data
                )
                ai_summary += symptom_text
            note.ai_notes = ai_summary

        # Generate AI analysis
        if note.ai_analysis is None:
            ai_analysis = generate_ai_analysis(note, patient)
            if not isinstance(ai_analysis, str):
                logger.error(f"generate_ai_analysis returned non-string: {type(ai_analysis)}")
                ai_analysis = "Analysis unavailable"
            # Append differential diagnoses and UMLS data
            if symptoms and patient:
                differentials = analyzer.generate_differential_dx(features, patient)
                if differentials and differentials[0][0] != "Undetermined":
                    diff_text = f"\nDifferential Diagnoses: {', '.join(f'{dx} ({score:.2f})' for dx, score, _ in differentials)}"
                    ai_analysis += diff_text
                if umls_data:
                    umls_text = f"\nUMLS Mappings:\n" + "\n".join(
                        f"- {d['preferred_term']} (CUI: {d['cui']})" for d in umls_data
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
                            # Extract clinical features and update with UMLS data
                            features = analyzer.extract_clinical_features(note)
                            symptoms = features.get('symptoms', [])
                            umls_data = []
                            if symptoms:
                                symptom_descriptions = [s['description'] for s in symptoms]
                                logger.debug(f"Extracted symptoms for note ID {note.id}: {symptom_descriptions}")
                                for symptom in symptom_descriptions:
                                    umls_result = query_uts_api(symptom)
                                    umls_data.append({
                                        'symptom': symptom,
                                        'cui': umls_result['cui'],
                                        'semantic_type': umls_result['semantic_type'],
                                        'preferred_term': umls_result['preferred_term']
                                    })
                            # Generate AI summary
                            if note.ai_notes is None:
                                ai_summary = generate_ai_summary(note)
                                if not isinstance(ai_summary, str):
                                    logger.error(f"generate_ai_summary returned non-string: {type(ai_summary)}")
                                    ai_summary = "Summary unavailable"
                                # Append symptoms and UMLS data
                                if umls_data:
                                    symptom_text = "\nExtracted Symptoms with UMLS Data:\n" + "\n".join(
                                        f"- {d['preferred_term']} (CUI: {d['cui']}, Semantic Type: {d['semantic_type']})"
                                        for d in umls_data
                                    )
                                    ai_summary += symptom_text
                                note.ai_notes = ai_summary
                            # Generate AI analysis
                            if note.ai_analysis is None:
                                ai_analysis = generate_ai_analysis(note, patient)
                                if not isinstance(ai_analysis, str):
                                    logger.error(f"generate_ai_analysis returned non-string: {type(ai_analysis)}")
                                    ai_analysis = "Analysis unavailable"
                                # Append differential diagnoses and UMLS data
                                if symptoms:
                                    differentials = analyzer.generate_differential_dx(features, patient)
                                    if differentials and differentials[0][0] != "Undetermined":
                                        diff_text = f"\nDifferential Diagnoses: {', '.join(f'{dx} ({score:.2f})' for dx, score, _ in differentials)}"
                                        ai_analysis += diff_text
                                    if umls_data:
                                        umls_text = f"\nUMLS Mappings:\n" + "\n".join(
                                            f"- {d['preferred_term']} (CUI: {d['cui']})" for d in umls_data
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
                    pbar.update(len(batch))
            if failed_notes:
                logger.info(f"Failed notes: {failed_notes}")
            logger.info(f"Batch processing complete: {success} successes, {error} errors")
            return success, error
        except Exception as e:
            db.session.rollback()
            logger.error(f"Batch processing failed: {str(e)}")
            raise RuntimeError("Batch update failed")