# nlp_processing.py
from .database import fetch_soap_notes, update_ai_analysis
from .clinical_ner import ClinicalNER
from .disease_predictor import DiseasePredictor

def prepare_note_for_nlp(note: dict) -> str:
    """Combine relevant fields into a single text"""
    fields = [
        note.get('situation', ''),
        note.get('hpi', ''),
        note.get('symptoms', ''),
        note.get('assessment', ''),
        note.get('additional_notes', '')
    ]
    return ' '.join(filter(None, fields)).strip()

def extract_keywords_and_cuis(note: dict, ner: ClinicalNER) -> tuple:
    """Extract keywords and CUIs from a SOAP note"""
    text = prepare_note_for_nlp(note)
    entities = ner.extract_entities(text)
    keywords = {ent[0].lower() for ent in entities}
    
    # Map to known disease CUIs
    disease_map = {
        "pneumonia": "C0032285",
        "myocardial infarction": "C0027051",
        "stroke": "C0038454",
        "diabetes": "C0011849"
    }
    
    cuis = []
    for keyword in keywords:
        if keyword in disease_map:
            cuis.append(disease_map[keyword])
    
    return list(keywords), cuis

def process_soap_note(note: dict):
    """Full processing pipeline for a SOAP note"""
    ner = ClinicalNER()
    predictor = DiseasePredictor(ner)
    
    text = prepare_note_for_nlp(note)
    if not text:
        return None
        
    keywords, cuis = extract_keywords_and_cuis(note, ner)
    diseases = predictor.predict(text, keywords, cuis)
    
    return {
        "diseases": diseases,
        "keywords": keywords,
        "cuis": cuis,
        "note_id": note["id"]
    }