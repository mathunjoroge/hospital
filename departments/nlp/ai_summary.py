from transformers import AutoTokenizer, AutoModel
import torch
from departments.models.medicine import SOAPNote
from departments.models.records import Patient
from extensions import db
from sqlalchemy.orm import load_only
import logging
from typing import List, Tuple, Optional, Dict, Set
import re
import os
import json
from functools import lru_cache
from tqdm import tqdm
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('clinical_ai.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 512
BATCH_SIZE = 10
EMBEDDING_DIM = 768
SIMILARITY_THRESHOLD = 0.5

def load_medical_stop_words() -> Set[str]:
    """Load medical stop words from an external JSON file"""
    knowledge_base_dir = os.path.join(os.path.dirname(__file__), "knowledge_base")
    stop_words_path = os.path.join(knowledge_base_dir, "medical_stop_words.json")
    
    try:
        with open(stop_words_path, 'r') as f:
            return set(json.load(f))
    except FileNotFoundError:
        logger.error(f"Stop words file not found: {stop_words_path}")
        raise RuntimeError("Stop words loading failed")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in stop words: {str(e)}")
        raise RuntimeError("Stop words parsing failed")

def load_medical_terms() -> Set[str]:
    """Load medical terms from an external JSON file"""
    knowledge_base_dir = os.path.join(os.path.dirname(__file__), "knowledge_base")
    medical_terms_path = os.path.join(knowledge_base_dir, "medical_terms.json")
    
    try:
        with open(medical_terms_path, 'r') as f:
            return set(json.load(f))
    except FileNotFoundError:
        logger.error(f"Medical terms file not found: {medical_terms_path}")
        raise RuntimeError("Medical terms loading failed")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in medical terms: {str(e)}")
        raise RuntimeError("Medical terms parsing failed")

def load_synonyms() -> Dict[str, List[str]]:
    """Load synonym mappings from an external JSON file"""
    knowledge_base_dir = os.path.join(os.path.dirname(__file__), "knowledge_base")
    synonyms_path = os.path.join(knowledge_base_dir, "synonyms.json")
    
    try:
        with open(synonyms_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Synonyms file not found: {synonyms_path}")
        raise RuntimeError("Synonyms loading failed")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in synonyms: {str(e)}")
        raise RuntimeError("Synonyms parsing failed")

def load_knowledge_base() -> Dict:
    """Load clinical pathways from an external JSON file"""
    knowledge_base_dir = os.path.join(os.path.dirname(__file__), "knowledge_base")
    knowledge_base_path = os.path.join(knowledge_base_dir, "clinical_pathways.json")
    
    try:
        with open(knowledge_base_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Knowledge base file not found: {knowledge_base_path}")
        raise RuntimeError("Knowledge base loading failed")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in knowledge base: {str(e)}")
        raise RuntimeError("Knowledge base parsing failed")

def load_history_diagnoses() -> Dict[str, str]:
    """Load history-to-diagnosis mappings from an external JSON file"""
    knowledge_base_dir = os.path.join(os.path.dirname(__file__), "knowledge_base")
    history_diagnoses_path = os.path.join(knowledge_base_dir, "history_diagnoses.json")
    
    try:
        with open(history_diagnoses_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"History diagnoses file not found: {history_diagnoses_path}")
        raise RuntimeError("History diagnoses loading failed")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in history diagnoses: {str(e)}")
        raise RuntimeError("History diagnoses parsing failed")

def load_diagnosis_relevance() -> Dict[str, List[str]]:
    """Load diagnosis relevance rules from an external JSON file"""
    knowledge_base_dir = os.path.join(os.path.dirname(__file__), "knowledge_base")
    relevance_path = os.path.join(knowledge_base_dir, "diagnosis_relevance.json")
    
    try:
        with open(relevance_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Diagnosis relevance file not found: {relevance_path}")
        raise RuntimeError("Diagnosis relevance loading failed")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in diagnosis relevance: {str(e)}")
        raise RuntimeError("Diagnosis relevance parsing failed")

def load_management_config() -> Dict[str, List[str]]:
    """Load management configuration (medications, specialties, tests) from an external JSON file"""
    knowledge_base_dir = os.path.join(os.path.dirname(__file__), "knowledge_base")
    config_path = os.path.join(knowledge_base_dir, "management_config.json")
    
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Management config file not found: {config_path}")
        raise RuntimeError("Management config loading failed")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in management config: {str(e)}")
        raise RuntimeError("Management config parsing failed")

def load_diagnosis_treatments() -> Dict[str, Dict]:
    """Load diagnosis-to-treatment mappings from an external JSON file"""
    knowledge_base_dir = os.path.join(os.path.dirname(__file__), "knowledge_base")
    treatments_path = os.path.join(knowledge_base_dir, "diagnosis_treatments.json")
    
    try:
        with open(treatments_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Diagnosis treatments file not found: {treatments_path}")
        raise RuntimeError("Diagnosis treatments loading failed")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in diagnosis treatments: {str(e)}")
        raise RuntimeError("Diagnosis treatments parsing failed")

# Initialize model and tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    logger.info(f"Model loaded: {MODEL_NAME} on {DEVICE}")
except Exception as e:
    logger.error(f"Model loading failed: {str(e)}")
    raise RuntimeError("Model initialization failed")

# Core NLP Functions
@lru_cache(maxsize=10000)
def embed_text(text: str) -> torch.Tensor:
    """Generate text embeddings with caching"""
    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length"
        ).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            return outputs.last_hidden_state[:, 0, :].squeeze().cpu()
    except Exception as e:
        logger.error(f"Embedding failed: {str(e)}")
        return torch.zeros(EMBEDDING_DIM)

def preprocess_text(text: str, stop_words: Set[str]) -> str:
    """Clean and standardize medical text for analysis"""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s\-/]', ' ', text)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

def deduplicate(items: List[str], synonyms: Dict[str, List[str]]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        canonical = item
        for key, aliases in synonyms.items():
            if item.lower() in [a.lower() for a in aliases]:
                canonical = key
                break
        if canonical.lower() not in seen:
            seen.add(canonical.lower())
            result.append(item)
    return result

# Clinical Analysis Engine
class ClinicalAnalyzer:
    def __init__(self):
        self.model = model
        self.tokenizer = tokenizer
        self.medical_stop_words = load_medical_stop_words()
        self.medical_terms = load_medical_terms()
        self.synonyms = load_synonyms()
        self.clinical_pathways = load_knowledge_base()
        self.history_diagnoses = load_history_diagnoses()
        self.diagnosis_relevance = load_diagnosis_relevance()
        self.management_config = load_management_config()
        self.diagnosis_treatments = load_diagnosis_treatments()
        
        # Warn if any resource is empty
        for name, resource in [
            ("Medical stop words", self.medical_stop_words),
            ("Medical terms", self.medical_terms),
            ("Synonyms", self.synonyms),
            ("Clinical pathways", self.clinical_pathways),
            ("History diagnoses", self.history_diagnoses),
            ("Diagnosis relevance", self.diagnosis_relevance),
            ("Management config", self.management_config),
            ("Diagnosis treatments", self.diagnosis_treatments)
        ]:
            if not resource:
                logger.warning(f"{name} is empty")

    def extract_clinical_features(self, note: SOAPNote) -> Dict:
        """Extract structured clinical features from SOAP note"""
        features = {
            'chief_complaint': note.situation.replace("Patient presents with", "").replace("Patient experiencing", "").strip() if note.situation else "",
            'hpi': note.hpi or "",
            'history': note.medical_history or "",
            'medications': note.medication_history or "",
            'assessment': note.assessment or "",
            'recommendation': note.recommendation or "",
            'additional_notes': note.additional_notes or ""
        }
        features['symptoms'] = []
        if features['chief_complaint']:
            features['symptoms'].append({
                'description': preprocess_text(features['chief_complaint'], self.medical_stop_words),
                'duration': extract_duration(features['hpi']),
                'severity': classify_severity(features['hpi']),
                'location': extract_location(features['chief_complaint'] + " " + features['hpi'])
            })
        text = f"{features['chief_complaint']} {features['hpi']} {features['additional_notes']}"
        symptom_candidates = [word for word in preprocess_text(text, self.medical_stop_words).split() if word in self.medical_terms]
        symptom_candidates = list(set(symptom_candidates))
        clinical_embedding = embed_text("clinical symptom")
        for term in symptom_candidates:
            term_embedding = embed_text(term)
            similarity = torch.cosine_similarity(term_embedding.unsqueeze(0), clinical_embedding.unsqueeze(0)).item()
            if similarity > SIMILARITY_THRESHOLD:
                features['symptoms'].append({
                    'description': term,
                    'duration': 'Unknown',
                    'severity': 'Unknown',
                    'location': extract_location(text)
                })
        return features

    def generate_differential_dx(self, features: Dict) -> List[str]:
        """Generate prioritized differential diagnoses"""
        dx = set()
        
        # Extract from assessment
        assessment = features.get('assessment', '').lower()
        if assessment:
            for term in assessment.split():
                if (term.endswith("itis") or 
                    term in ["migraine", "angina", "asthma", "gerd", "neuropathy", "postnasal", "eczema", "dermatitis"]):
                    dx.add(term.capitalize())
        
        # Match symptoms and locations
        symptoms = features.get('symptoms', [])
        for symptom in symptoms:
            symptom_type = symptom.get('description', '').lower()
            location = symptom.get('location', '').lower()
            if symptom_type or location:
                for category, pathways in self.clinical_pathways.items():
                    for key, path in pathways.items():
                        key_lower = key.lower()
                        if (symptom_type and symptom_type in key_lower) or (location and location in key_lower):
                            dx.update(path.get('differentials', []))
        
        # Check medical history
        history = features.get('history', '').lower()
        for condition, diag in self.history_diagnoses.items():
            if condition in history:
                dx.add(diag)
        
        # Contextual clues from additional notes
        additional_notes = features.get('additional_notes', '').lower()
        if 'travel' in additional_notes:
            dx.add('Traveler’s diarrhea')
        if 'pet' in additional_notes:
            dx.add('Allergic reaction')
        
        # Ensure minimum diagnoses using embeddings
        if len(dx) < 3:
            common_diagnoses = ['Viral infection', 'Dehydration', 'Medication side effect']
            text = f"{features.get('chief_complaint', '')} {features.get('hpi', '')} {additional_notes}"
            text_embedding = embed_text(text)
            diagnosis_scores = []
            for diag in common_diagnoses:
                diag_embedding = embed_text(diag)
                similarity = torch.cosine_similarity(text_embedding.unsqueeze(0), diag_embedding.unsqueeze(0)).item()
                if similarity > SIMILARITY_THRESHOLD:
                    diagnosis_scores.append((diag, similarity))
            diagnosis_scores.sort(key=lambda x: x[1], reverse=True)
            dx.update(diag for diag, _ in diagnosis_scores[:5 - len(dx)])
        
        # Filter irrelevant diagnoses
        def is_relevant(diag: str, symptoms: List[Dict], history: str) -> bool:
            diag_lower = diag.lower()
            symptom_words = {s.get('description', '').lower() for s in symptoms}
            locations = {s.get('location', '').lower() for s in symptoms}
            
            for condition, required in self.diagnosis_relevance.items():
                if diag_lower == condition:
                    if condition == 'appendicitis':
                        return any(loc in required for loc in locations)
                    return any(word in symptom_words for word in required)
            return True
        
        dx = {d for d in dx if is_relevant(d, symptoms, history)}
        
        # Return deduplicated and sorted list
        return deduplicate(sorted(dx)[:5], self.synonyms)

    def generate_management_plan(self, features: Dict, differentials: List[str]) -> Dict:
        """Generate comprehensive management plan"""
        plan = {
            'workup': {'urgent': [], 'routine': []},
            'treatment': {'symptomatic': [], 'definitive': []},
            'follow_up': ['Follow-up in 1-2 weeks']
        }
        
        # Process recommendations
        recommendation = features.get('recommendation', '').lower()
        if recommendation:
            if 'prescribe' in recommendation:
                for med in self.management_config.get('medications', []):
                    if med in recommendation:
                        plan['treatment']['definitive'].append(f"{med.capitalize()} as prescribed")
            if 'refer' in recommendation:
                for spec in self.management_config.get('specialties', []):
                    if spec in recommendation:
                        plan['treatment']['definitive'].append(f"{spec.capitalize()} referral")
                        plan['workup']['routine'].append(f"{spec.capitalize()} consult")
            for test in self.management_config.get('tests', []):
                if test in recommendation:
                    plan['workup']['urgent'].append(test.upper())
        
        # Process symptoms and clinical pathways
        for symptom in features.get('symptoms', []):
            symptom_type = symptom.get('description', '').lower()
            location = symptom.get('location', '').lower()
            for category, pathways in self.clinical_pathways.items():
                for key, path in pathways.items():
                    if symptom_type in key.lower() or (location and location in key.lower()):
                        plan['workup']['urgent'].extend(path.get('workup', {}).get('urgent', []))
                        plan['workup']['routine'].extend(path.get('workup', {}).get('routine', []))
                        plan['treatment']['symptomatic'].extend(path.get('management', {}).get('symptomatic', []))
                        plan['treatment']['definitive'].extend(path.get('management', {}).get('definitive', []))
        
        # Process differential diagnoses
        for dx in differentials:
            dx_lower = dx.lower()
            for diag_key, mappings in self.diagnosis_treatments.items():
                if diag_key in dx_lower:
                    plan['treatment']['definitive'].extend(mappings.get('treatment', {}).get('definitive', []))
                    plan['workup']['urgent'].extend(mappings.get('workup', {}).get('urgent', []))
                    plan['workup']['routine'].extend(mappings.get('workup', {}).get('routine', []))
        
        # Deduplicate and sort
        for key in plan['workup']:
            plan['workup'][key] = deduplicate(sorted(set(plan['workup'][key])), self.synonyms)
            if key == 'routine':
                plan['workup'][key] = [item for item in plan['workup'][key] if item not in plan['workup']['urgent']]
        for key in plan['treatment']:
            plan['treatment'][key] = deduplicate(sorted(set(plan['treatment'][key])), self.synonyms)
        
        return plan

# SOAP Note Processing
def build_note_text(note: SOAPNote) -> str:
    """Construct full note text from components"""
    sections = [
        ("Chief Complaint", note.situation),
        ("HPI", note.hpi),
        ("Medical History", note.medical_history),
        ("Medications", note.medication_history),
        ("Assessment", note.assessment),
        ("Plan", note.recommendation),
        ("Additional Notes", note.additional_notes)
    ]
    return "\n".join(f"{k}: {v}" for k, v in sections if v)

def generate_ai_summary(note: SOAPNote) -> str:
    """Generate concise clinical summary with newline-separated statements"""
    try:
        sections = [
            ("Chief Complaint", note.situation),
            ("HPI", note.hpi),
            ("Medications", note.medication_history),
            ("Assessment", note.assessment)
        ]
        selected = []
        for key, value in sections:
            if value:
                prefix = key + ": "
                if key == "Chief Complaint" and not value.startswith("Patient"):
                    value = "Patient experiencing " + value
                selected.append(f"{prefix}{value}")
        return "\n".join(selected)
    except Exception as e:
        logger.error(f"Summary failed: {str(e)}")
        return "Summary unavailable"

def generate_ai_analysis(note: SOAPNote, patient: Patient = None) -> str:
    """Generate clinical analysis with semicolon-separated lists"""
    try:
        analyzer = ClinicalAnalyzer()
        features = analyzer.extract_clinical_features(note)
        differentials = analyzer.generate_differential_dx(features)
        plan = analyzer.generate_management_plan(features, differentials)
        analysis_output = f"""
=== AI CLINICAL ANALYSIS ===
[CHIEF CONCERN]
{features['chief_complaint'].lower()}
[DIFFERENTIAL DIAGNOSIS]
{";".join(differentials)}
[RECOMMENDED WORKUP]
■ Urgent: {";".join(sorted(plan['workup']['urgent'])) or "None"}
■ Routine: {";".join(sorted(plan['workup']['routine'])) or "None"}
[TREATMENT OPTIONS]
▲ Symptomatic: {";".join(sorted(plan['treatment']['symptomatic'])) or "None"}
● Definitive: {";".join(sorted(plan['treatment']['definitive'])) or "Pending diagnosis"}
[FOLLOW-UP]
{";".join(plan['follow_up'])}
DISCLAIMER: This AI-generated analysis requires clinical correlation.
"""
        return analysis_output.strip()
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return """
=== AI CLINICAL ANALYSIS ===
[CHIEF CONCERN]
Unknown
[DIFFERENTIAL DIAGNOSIS]
Undetermined
[RECOMMENDED WORKUP]
■ Urgent: None
■ Routine: CBC;Basic metabolic panel
[TREATMENT OPTIONS]
▲ Symptomatic: None
● Definitive: Pending diagnosis
[FOLLOW-UP]
As needed
DISCLAIMER: This AI-generated analysis requires clinical correlation.
"""

# Batch Processing
def update_all_ai_notes(batch_size: int = BATCH_SIZE) -> Tuple[int, int]:
    """Process all notes needing AI updates"""
    from flask import current_app
    analyzer = ClinicalAnalyzer()
    with current_app.app_context():
        try:
            base_query = db.session.query(SOAPNote, Patient).filter(
                (SOAPNote.ai_notes.is_(None)) | 
                (SOAPNote.ai_analysis.is_(None)),
                SOAPNote.patient_id == Patient.patient_id
            )
            total = base_query.count()
            if total == 0:
                return 0, 0
            success = error = 0
            with tqdm(total=total) as pbar:
                for offset in range(0, total, batch_size):
                    batch = base_query.offset(offset).limit(batch_size).all()
                    for note, patient in batch:
                        try:
                            if note.ai_notes is None:
                                note.ai_notes = generate_ai_summary(note)
                            if note.ai_analysis is None:
                                note.ai_analysis = generate_ai_analysis(note, patient)
                            success += 1
                        except Exception as e:
                            error += 1
                            logger.error(f"Error processing note {note.id}: {str(e)}")
                    db.session.commit()
                    pbar.update(len(batch))
            return success, error
        except Exception as e:
            db.session.rollback()
            logger.error(f"Batch processing failed: {str(e)}")
            raise RuntimeError("Batch update failed")

# Helper Functions
def calculate_age(dob: datetime.date) -> Optional[int]:
    """Calculate age from date of birth"""
    if not dob:
        return None
    today = datetime.now().date()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

def extract_duration(text: str) -> str:
    """Extract duration from clinical text"""
    match = re.search(r'(\d+\s*(day|week|month|year)s?)', text.lower())
    return match.group(0) if match else "Unknown"

def classify_severity(text: str) -> str:
    """Classify symptom severity"""
    text = text.lower()
    if 'severe' in text:
        return "Severe"
    elif 'moderate' in text:
        return "Moderate"
    return "Mild"

def extract_location(text: str) -> str:
    """Extract symptom location"""
    locations = ['head', 'chest', 'abdomen', 'back', 'extremity', 'joint', 'neck', 'hand', 'arm', 'leg', 'knee']
    text = text.lower()
    for loc in locations:
        if loc in text:
            return loc.capitalize()
    return "Unspecified"

def batch_embed_texts(texts: List[str]) -> torch.Tensor:
    """Batch process text embeddings"""
    try:
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            add_special_tokens=True
        ).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            return outputs.last_hidden_state[:, 0, :].cpu()
    except Exception as e:
        logger.error(f"Batch embedding failed: {str(e)}")
        return torch.zeros(len(texts), EMBEDDING_DIM)