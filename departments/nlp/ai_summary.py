from transformers import AutoTokenizer, AutoModel
import torch
from departments.models.medicine import SOAPNote
from departments.models.records import Patient
from extensions import db
from sqlalchemy.orm import load_only
import logging
from typing import List, Tuple, Optional, Dict, Set
import re
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

# Medical stop words
MEDICAL_STOP_WORDS = {
    'patient', 'history', 'present', 'illness', 'denies', 'reports',
    'without', 'with', 'this', 'that', 'these', 'those', 'they', 'them',
    'their', 'have', 'has', 'had', 'been', 'being', 'other', 'associated'
}

# Medical terms for symptom extraction
MEDICAL_TERMS = {
    'pain', 'headache', 'numbness', 'tingling', 'cough', 'wheezing', 'rash',
    'swelling', 'diarrhea', 'nausea', 'vomiting', 'dizziness', 'stiffness',
    'shortness of breath', 'palpitations', 'tremors', 'fatigue', 'constipation'
}

# Synonym mapping for deduplication
SYNONYMS = {
    'GERD': ['Gerd', 'GERD exacerbation'],
    'Peripheral neuropathy': ['Neuropathy'],
    'PPI trial': ['PPI for gastritis', 'PPI for GERD'],
    'ECG': ['EKG'],
    'Carpal tunnel syndrome': ['Carpal tunnel'],
    'Postnasal drip': ['Postnasal']
}

# Clinical Decision Support Knowledge
CLINICAL_PATHWAYS = {
    'pain': {
        'head': {
            'differentials': ['Migraine', 'Tension headache', 'Cluster headache', 'Sinusitis'],
            'workup': {'urgent': ['Head CT if red flags'], 'routine': ['Neurology consult if persistent', 'CBC', 'BMP']},
            'management': {'symptomatic': ['Hydration'], 'definitive': ['Triptans for migraine', 'NSAIDs for tension-type', 'Antibiotics for sinusitis']}
        },
        'chest': {
            'differentials': ['Angina', 'GERD', 'Musculoskeletal pain', 'Pulmonary embolism'],
            'workup': {'urgent': ['ECG', 'Cardiac enzymes', 'Chest X-ray', 'D-dimer'], 'routine': ['CBC', 'BMP']},
            'management': {'symptomatic': [], 'definitive': ['Aspirin for cardiac', 'PPI trial for GERD', 'NSAIDs for musculoskeletal']}
        },
        'abdomen': {
            'differentials': ['Gastritis', 'GERD', 'Cholecystitis', 'Pancreatitis'],
            'workup': {'urgent': ['Abdominal ultrasound', 'LFTs', 'Lipase'], 'routine': ['CBC', 'BMP', 'Upper endoscopy if chronic']},
            'management': {'symptomatic': ['Antiemetics'], 'definitive': ['PPI for gastritis', 'PPI trial for GERD', 'Surgical consult for cholecystitis']}
        },
        'joint': {
            'differentials': ['Osteoarthritis', 'Meniscal injury', 'Gout', 'Rheumatoid arthritis'],
            'workup': {'urgent': ['Knee X-ray'], 'routine': ['ESR', 'Uric acid', 'Rheumatoid factor', 'CBC', 'BMP']},
            'management': {'symptomatic': ['Ice', 'Elevation'], 'definitive': ['NSAIDs', 'Physical therapy', 'Colchicine for gout']}
        },
        'back': {
            'differentials': ['Mechanical back pain', 'Disc herniation', 'Spinal stenosis'],
            'workup': {'urgent': ['Lumbar X-ray if trauma'], 'routine': ['CBC', 'BMP', 'MRI if neurological signs']},
            'management': {'symptomatic': ['Heat'], 'definitive': ['NSAIDs', 'Physical therapy', 'Muscle relaxants']}
        },
        'neck': {
            'differentials': ['Cervical strain', 'Cervical spondylosis', 'Meningitis'],
            'workup': {'urgent': ['Cervical X-ray', 'Head CT if fever'], 'routine': ['CBC', 'BMP']},
            'management': {'symptomatic': ['Massage'], 'definitive': ['NSAIDs', 'Physical therapy']}
        }
    },
    'respiratory': {
        'cough': {
            'differentials': ['Allergic rhinitis', 'Postnasal drip', 'GERD', 'Chronic bronchitis'],
            'workup': {'urgent': ['Chest X-ray if febrile'], 'routine': ['CBC', 'BMP', 'Allergy testing']},
            'management': {'symptomatic': ['Antitussives'], 'definitive': ['Antihistamines for allergies', 'PPI trial for GERD', 'Inhaled steroids for bronchitis']}
        },
        'shortness of breath': {
            'differentials': ['Asthma', 'COPD', 'Heart failure', 'Pulmonary embolism'],
            'workup': {'urgent': ['Chest X-ray', 'Pulmonary function test', 'D-dimer', 'BNP'], 'routine': ['CBC', 'BMP']},
            'management': {'symptomatic': [], 'definitive': ['Bronchodilators', 'Diuretics for heart failure', 'Anticoagulation for embolism']}
        }
    },
    'swelling': {
        'extremity': {
            'differentials': ['Venous insufficiency', 'Heart failure', 'DVT', 'Lymphedema'],
            'workup': {'urgent': ['Doppler ultrasound', 'BNP', 'D-dimer'], 'routine': ['CBC', 'BMP']},
            'management': {'symptomatic': ['Elevation'], 'definitive': ['Compression therapy', 'Diuretics', 'Anticoagulation if DVT']}
        }
    },
    'neurological': {
        'numbness': {
            'differentials': ['Peripheral neuropathy', 'Carpal tunnel syndrome', 'Stroke', 'Multiple sclerosis'],
            'workup': {'urgent': ['Nerve conduction study', 'Blood glucose'], 'routine': ['CBC', 'BMP', 'B12 level', 'Neurology consult']},
            'management': {'symptomatic': [], 'definitive': ['Gabapentin', 'Neurology referral', 'Wrist splints for carpal tunnel']}
        },
        'dizziness': {
            'differentials': ['BPPV', 'Orthostatic hypotension', 'Dehydration', 'Labyrinthitis'],
            'workup': {'urgent': ['Orthostatic vitals'], 'routine': ['CBC', 'BMP', 'Head CT if focal signs']},
            'management': {'symptomatic': ['Meclizine'], 'definitive': ['Hydration', 'Epley maneuver for BPPV']}
        }
    },
    'gastrointestinal': {
        'nausea': {
            'differentials': ['Gastritis', 'Vestibular disturbance', 'Medication side effect'],
            'workup': {'urgent': ['Electrolytes'], 'routine': ['CBC', 'BMP', 'Upper endoscopy if chronic']},
            'management': {'symptomatic': ['Antiemetics'], 'definitive': ['PPI trial', 'Discontinue offending medication']}
        },
        'diarrhea': {
            'differentials': ['Viral gastroenteritis', 'Lactose intolerance', 'Bacterial gastroenteritis', 'IBD'],
            'workup': {'urgent': ['Stool studies', 'Electrolytes'], 'routine': ['CBC', 'BMP']},
            'management': {'symptomatic': ['Hydration'], 'definitive': ['Antidiarrheals', 'Antibiotics if bacterial', 'Dietary modification for lactose intolerance']}
        }
    },
    'skin': {
        'rash': {
            'differentials': ['Eczema', 'Contact dermatitis', 'Psoriasis', 'Drug reaction'],
            'workup': {'urgent': [], 'routine': ['CBC', 'BMP', 'Skin biopsy if persistent', 'Allergy testing']},
            'management': {'symptomatic': ['Cool compresses'], 'definitive': ['Topical steroids', 'Antihistamines', 'Discontinue offending drug']}
        }
    }
}

# Initialize model
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

def preprocess_text(text: str) -> str:
    """Clean and standardize medical text for analysis"""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s\-/]', ' ', text)
    text = ' '.join(word for word in text.split() if word not in MEDICAL_STOP_WORDS)
    return text

def deduplicate(items: List[str], synonyms: Dict[str, List[str]]) -> List[str]:
    """Deduplicate items using synonym mapping"""
    seen = set()
    result = []
    for item in items:
        canonical = item
        for key, aliases in synonyms.items():
            if item in aliases:
                canonical = key
                break
        if canonical not in seen:
            seen.add(canonical)
            result.append(item)
    return result

# Clinical Analysis Engine
def extract_clinical_features(note: SOAPNote) -> Dict:
    """Extract structured clinical features from SOAP note"""
    features = {
        'chief_complaint': note.situation.replace("Patient presents with", "").replace("Patient experiencing", "").strip(),
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
            'description': preprocess_text(features['chief_complaint']),
            'duration': extract_duration(features['hpi']),
            'severity': classify_severity(features['hpi']),
            'location': extract_location(features['chief_complaint'] + " " + features['hpi'])
        })
    
    text = f"{features['chief_complaint']} {features['hpi']} {features['additional_notes']}"
    symptom_candidates = [word for word in preprocess_text(text).split() if word in MEDICAL_TERMS]
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

def generate_differential_dx(features: Dict) -> List[str]:
    """Generate prioritized differential diagnoses"""
    dx = set()
    
    if features['assessment']:
        for term in features['assessment'].lower().split():
            if term.endswith("itis") or term in ["migraine", "angina", "asthma", "gerd", "neuropathy", "postnasal", "eczema", "dermatitis"]:
                dx.add(term.capitalize())
    
    for symptom in features['symptoms']:
        symptom_type = symptom['description'].lower()
        location = symptom.get('location', '').lower()
        
        for category, pathways in CLINICAL_PATHWAYS.items():
            for key, path in pathways.items():
                if symptom_type in key or (location and location in key):
                    dx.update(path['differentials'])
    
    history_checks = {
        'hypertension': 'Hypertensive complication',
        'diabetes': 'Peripheral neuropathy',
        'allergies': 'Allergic rhinitis',
        'gerd': 'GERD exacerbation',
        'asthma': 'Asthma exacerbation',
        'eczema': 'Eczema flare',
        'lactose intolerance': 'Lactose intolerance'
    }
    for condition, diag in history_checks.items():
        if condition in features['history'].lower():
            dx.add(diag)
    
    if 'travel' in features['additional_notes'].lower():
        dx.add('Traveler’s diarrhea')
    if 'pet' in features['additional_notes'].lower():
        dx.add('Allergic reaction')
    
    if len(dx) < 3:
        common_diagnoses = ['Viral infection', 'Dehydration', 'Medication side effect']
        text = f"{features['chief_complaint']} {features['hpi']} {features['additional_notes']}"
        text_embedding = embed_text(text)
        diagnosis_scores = []
        for diag in common_diagnoses:
            diag_embedding = embed_text(diag)
            similarity = torch.cosine_similarity(text_embedding.unsqueeze(0), diag_embedding.unsqueeze(0)).item()
            if similarity > SIMILARITY_THRESHOLD:
                diagnosis_scores.append((diag, similarity))
        diagnosis_scores.sort(key=lambda x: x[1], reverse=True)
        dx.update(diag for diag, _ in diagnosis_scores[:5 - len(dx)])
    
    def is_relevant(dx, symptoms, history):
        symptom_words = {s['description'].lower() for s in symptoms}
        if dx.lower() in ['copd', 'bronchitis'] and not any(w in ['cough', 'wheezing'] for w in symptom_words):
            return False
        if dx.lower() in ['anemia', 'uti'] and not any(w in ['fatigue', 'dysuria'] for w in symptom_words):
            return False
        if dx.lower() == 'appendicitis' and not 'abdomen' in [s.get('location', '').lower() for s in symptoms]:
            return False
        return True
    
    dx = {d for d in dx if is_relevant(d, features['symptoms'], features['history'])}
    return deduplicate(sorted(dx)[:5], SYNONYMS)

def generate_management_plan(features: Dict, differentials: List[str]) -> Dict:
    """Generate comprehensive management plan"""
    plan = {
        'workup': {'urgent': [], 'routine': []},
        'treatment': {'symptomatic': [], 'definitive': []},
        'follow_up': ['Follow-up in 1-2 weeks']
    }
    
    recommendation = features['recommendation'].lower()
    if recommendation:
        if 'prescribe' in recommendation:
            for med in ['sumatriptan', 'omeprazole', 'albuterol', 'gabapentin']:
                if med in recommendation:
                    plan['treatment']['definitive'].append(f"{med.capitalize()} as prescribed")
        if 'refer' in recommendation:
            for spec in ['neurology', 'cardiology', 'surgical', 'orthopedic', 'dermatology']:
                if spec in recommendation:
                    plan['treatment']['definitive'].append(f"{spec.capitalize()} referral")
                    plan['workup']['routine'].append(f"{spec.capitalize()} consult")
        for test in ['ecg', 'x-ray', 'ultrasound', 'ct', 'mri']:
            if test in recommendation:
                plan['workup']['urgent'].append(test.upper())
    
    for symptom in features['symptoms']:
        symptom_type = symptom['description'].lower()
        location = symptom.get('location', '').lower()
        
        for category, pathways in CLINICAL_PATHWAYS.items():
            for key, path in pathways.items():
                if symptom_type in key or (location and location in key):
                    plan['workup']['urgent'].extend(path['workup']['urgent'])
                    plan['workup']['routine'].extend(path['workup']['routine'])
                    plan['treatment']['symptomatic'].extend(path['management']['symptomatic'])
                    plan['treatment']['definitive'].extend(path['management']['definitive'])
    
    for dx in differentials:
        dx_lower = dx.lower()
        if 'migraine' in dx_lower:
            plan['treatment']['definitive'].append('Triptans for migraine')
            plan['workup']['routine'].append('Neurology consult')
        elif 'angina' in dx_lower:
            plan['treatment']['definitive'].append('Aspirin for cardiac')
            plan['workup']['urgent'].append('Cardiac enzymes')
        elif 'asthma' in dx_lower or 'copd' in dx_lower:
            plan['treatment']['definitive'].append('Bronchodilators')
            plan['workup']['urgent'].append('Pulmonary function test')
        elif 'gerd' in dx_lower:
            plan['treatment']['definitive'].append('PPI trial for GERD')
        elif 'eczema' in dx_lower or 'dermatitis' in dx_lower:
            plan['treatment']['definitive'].append('Topical steroids')
            plan['workup']['routine'].append('Dermatology consult')
        elif 'osteoarthritis' in dx_lower or 'meniscal' in dx_lower:
            plan['treatment']['definitive'].append('Physical therapy')
            plan['workup']['routine'].append('Orthopedic consult')
        elif 'cervical strain' in dx_lower:
            plan['treatment']['definitive'].append('Physical therapy')
            plan['workup']['routine'].append('Cervical X-ray')
        elif 'lactose intolerance' in dx_lower:
            plan['treatment']['definitive'].append('Dietary modification for lactose intolerance')
    
    for key in plan['workup']:
        plan['workup'][key] = deduplicate(sorted(set(plan['workup'][key])), SYNONYMS)
        if key == 'routine':
            plan['workup'][key] = [item for item in plan['workup'][key] if item not in plan['workup']['urgent']]
    for key in plan['treatment']:
        plan['treatment'][key] = deduplicate(sorted(set(plan['treatment'][key])), SYNONYMS)
    
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
        return "\n".join(selected)  # MODIFIED: Newline separator
    except Exception as e:
        logger.error(f"Summary failed: {str(e)}")
        return "Summary unavailable"

def generate_ai_analysis(note: SOAPNote, patient: Patient = None) -> str:
    """Generate clinical analysis with semicolon-separated lists"""
    try:
        features = extract_clinical_features(note)
        differentials = generate_differential_dx(features)
        plan = generate_management_plan(features, differentials)
        
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