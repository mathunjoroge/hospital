from transformers import AutoTokenizer, AutoModel
import torch
from departments.models.medicine import SOAPNote
from departments.models.records import Patient
from extensions import db
import logging
from typing import List, Tuple, Optional, Dict, Set
import re
import os
import json
from functools import lru_cache
from tqdm import tqdm
from datetime import datetime

# Configure logging
def setup_logging():
    """Set up logging with separate handlers for detailed and error logs."""
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    clinical_log_path = os.path.join(log_dir, 'clinical_ai.log')
    error_log_path = os.path.join(log_dir, 'error_log.txt')

    detailed_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    simple_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    clinical_file_handler = logging.FileHandler(clinical_log_path)
    clinical_file_handler.setLevel(logging.DEBUG)
    clinical_file_handler.setFormatter(detailed_formatter)

    error_file_handler = logging.FileHandler(error_log_path)
    error_file_handler.setLevel(logging.WARNING)
    error_file_handler.setFormatter(simple_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(detailed_formatter)

    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[clinical_file_handler, error_file_handler, console_handler]
    )

    logger = logging.getLogger(__name__)
    logger.info("Logging configuration set up successfully")
    return logger

logger = setup_logging()

# Constants
MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 512
BATCH_SIZE = 16
EMBEDDING_DIM = 768
SIMILARITY_THRESHOLD = 0.9
CONFIDENCE_THRESHOLD = 0.85

def load_knowledge_base() -> Dict:
    """Load knowledge base resources with validation and fallback."""
    knowledge_base_dir = os.path.join(os.path.dirname(__file__), "knowledge_base")
    resources = {
        "medical_stop_words": "medical_stop_words.json",
        "medical_terms": "medical_terms.json",
        "synonyms": "synonyms.json",
        "clinical_pathways": "clinical_pathways.json",
        "history_diagnoses": "history_diagnoses.json",
        "diagnosis_relevance": "diagnosis_relevance.json",
        "management_config": "management_config.json",
        "diagnosis_treatments": "diagnosis_treatments.json"
    }
    knowledge = {}
    default_stop_words = [
        "patient", "history", "present", "illness", "denies", "reports",
        "without", "with", "this", "that", "these", "those", "they", "them",
        "their", "have", "has", "had", "been", "being", "other", "associated",
        "complains", "noted", "states", "observed", "left", "right", "ago",
        "since", "recently", "following", "during", "upon", "after"
    ]
    fallback_clinical_pathways = {
        "neurological": {
            "headache": {
                "differentials": ["Migraine", "Tension headache", "Cluster headache"],
                "workup": {"urgent": ["CT head if thunderclap"], "routine": ["CBC", "ESR"]},
                "management": {"symptomatic": ["Ibuprofen 400 mg", "Hydration"], "definitive": ["Sumatriptan 50 mg"]}
            },
            "photophobia": {
                "differentials": ["Migraine", "Meningitis"],
                "workup": {"urgent": ["Lumbar puncture if fever"], "routine": []},
                "management": {"symptomatic": ["Dark room rest"], "definitive": []}
            }
        },
        "cardiovascular": {
            "chest pain": {
                "differentials": ["Angina", "Myocardial infarction", "Musculoskeletal pain"],
                "workup": {"urgent": ["ECG", "Troponin"], "routine": ["Lipid panel", "Stress test"]},
                "management": {"symptomatic": ["Nitroglycerin 0.4 mg SL"], "definitive": ["Aspirin 81 mg daily"]}
            },
            "shortness of breath": {
                "differentials": ["Angina", "Pulmonary embolism", "COPD"],
                "workup": {"urgent": ["D-dimer if acute"], "routine": ["Pulmonary function test"]},
                "management": {"symptomatic": ["Oxygen therapy"], "definitive": []}
            },
            "edema": {
                "differentials": ["Heart failure", "Venous insufficiency", "Nephrotic syndrome"],
                "workup": {"urgent": ["BNP"], "routine": ["Renal panel", "Ultrasound Doppler"]},
                "management": {"symptomatic": ["Leg elevation"], "definitive": ["Furosemide 20 mg"]}
            }
        },
        "gastrointestinal": {
            "epigastric pain": {
                "differentials": ["GERD", "Peptic ulcer", "Pancreatitis"],
                "workup": {"urgent": ["Lipase if severe"], "routine": ["H. pylori test"]},
                "management": {"symptomatic": ["Antacids"], "definitive": ["Omeprazole 20 mg daily"]}
            },
            "nausea": {
                "differentials": ["GERD", "Gastritis"],
                "workup": {"urgent": [], "routine": ["Upper endoscopy if persistent"]},
                "management": {"symptomatic": ["Ondansetron 4 mg"], "definitive": []}
            },
            "diarrhea": {
                "differentials": ["Viral gastroenteritis", "Lactose intolerance", "Traveler’s diarrhea"],
                "workup": {"urgent": ["Stool culture if bloody"], "routine": ["Electrolytes"]},
                "management": {"symptomatic": ["Loperamide 2 mg"], "definitive": ["Hydration"]}
            }
        },
        "musculoskeletal": {
            "knee pain": {
                "differentials": ["Osteoarthritis", "Meniscal injury", "Bursitis"],
                "workup": {"urgent": [], "routine": ["Knee X-ray", "MRI if locking"]},
                "management": {"symptomatic": ["Ibuprofen 600 mg", "Ice"], "definitive": ["Physical therapy"]}
            },
            "swelling": {
                "differentials": ["Osteoarthritis", "Gout"],
                "workup": {"urgent": ["Joint aspiration if acute"], "routine": ["Uric acid level"]},
                "management": {"symptomatic": ["Elevation"], "definitive": []}
            },
            "back pain": {
                "differentials": ["Mechanical low back pain", "Herniated disc", "Spondylosis"],
                "workup": {"urgent": ["MRI if neurological symptoms"], "routine": ["Lumbar X-ray"]},
                "management": {"symptomatic": ["Acetaminophen 500 mg", "Heat"], "definitive": ["Physical therapy"]}
            }
        },
        "respiratory": {
            "cough": {
                "differentials": ["Postnasal drip", "Allergic cough", "Chronic bronchitis"],
                "workup": {"urgent": [], "routine": ["Chest X-ray", "Allergy testing"]},
                "management": {"symptomatic": ["Dextromethorphan 20 mg"], "definitive": ["Intranasal steroids"]}
            }
        },
        "dermatological": {
            "rash": {
                "differentials": ["Eczema flare", "Contact dermatitis", "Drug reaction"],
                "workup": {"urgent": [], "routine": ["Skin patch test"]},
                "management": {"symptomatic": ["Hydrocortisone 1% cream"], "definitive": ["Avoid triggers"]}
            }
        }
    }

    for key, filename in resources.items():
        file_path = os.path.join(knowledge_base_dir, filename)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if key == "medical_stop_words":
                    if not isinstance(data, list):
                        logger.error(f"Expected list for {filename}, got {type(data)}")
                        data = default_stop_words
                    data = set(data).union(default_stop_words)
                elif key == "clinical_pathways":
                    if not isinstance(data, dict):
                        logger.error(f"Expected dict for {filename}, got {type(data)}")
                        data = fallback_clinical_pathways
                    else:
                        valid_data = {}
                        for cat, paths in data.items():
                            if not isinstance(paths, dict):
                                logger.warning(f"Skipping invalid category {cat}: {type(paths)}")
                                continue
                            valid_paths = {}
                            for pkey, path in paths.items():
                                if not isinstance(path, dict):
                                    logger.warning(f"Skipping invalid path {pkey}: {type(path)}")
                                    continue
                                valid_paths[pkey] = path
                            if valid_paths:
                                valid_data[cat] = valid_paths
                        data = valid_data or fallback_clinical_pathways
                        logger.info(f"Loaded {len(data)} clinical pathway categories with {sum(len(paths) for paths in data.values())} total pathways")
                else:
                    if not isinstance(data, dict):
                        logger.error(f"Expected dict for {filename}, got {type(data)}")
                        data = {}
                knowledge[key] = data
                if not data:
                    logger.warning(f"Empty resource loaded: {filename}")
                else:
                    logger.info(f"Loaded {len(data)} entries for {key}")
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            if key == "medical_stop_words":
                knowledge[key] = set(default_stop_words)
            elif key == "clinical_pathways":
                knowledge[key] = fallback_clinical_pathways
            else:
                knowledge[key] = {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {file_path}: {str(e)}")
            if key == "medical_stop_words":
                knowledge[key] = set(default_stop_words)
            elif key == "clinical_pathways":
                knowledge[key] = fallback_clinical_pathways
            else:
                knowledge[key] = {}
    return knowledge

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
    """Generate text embeddings with caching."""
    if not isinstance(text, str) or not text.strip():
        logger.warning(f"Invalid or empty text for embedding: {text}")
        return torch.zeros(EMBEDDING_DIM)
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

def preprocess_text(text: str, stop_words: Set[str]) -> str:
    """Clean and standardize medical text."""
    if not isinstance(text, str):
        logger.error(f"Invalid text type: {type(text)}")
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s\-/]', ' ', text)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

def deduplicate(items: Tuple[str], synonyms: Dict[str, List[str]]) -> List[str]:
    """Deduplicate items using synonym mappings."""
    if not isinstance(synonyms, dict):
        logger.error(f"synonyms is not a dict: {type(synonyms)}")
        return list(items)
    seen = set()
    result = []
    for item in items:
        if not isinstance(item, str):
            logger.warning(f"Non-string item in deduplicate: {item}")
            continue
        canonical = item
        for key, aliases in synonyms.items():
            if not isinstance(aliases, list):
                logger.error(f"aliases for key {key} is not a list: {type(aliases)}")
                continue
            if item.lower() in [a.lower() for a in aliases if isinstance(a, str)]:
                canonical = key
                break
        if canonical.lower() not in seen:
            seen.add(canonical.lower())
            result.append(item)
    return result

def parse_conditional_workup(workup: str, symptoms: List[Dict]) -> Optional[str]:
    """Parse conditional workup statements."""
    if not isinstance(workup, str):
        logger.error(f"Invalid workup type: {type(workup)}")
        return None
    if " if " in workup.lower():
        test, condition = workup.lower().split(" if ")
        symptom_descriptions = {s.get('description', '').lower() for s in symptoms if isinstance(s, dict)}
        symptom_severities = {s.get('severity', '').lower() for s in symptoms if isinstance(s, dict)}
        locations = {s.get('location', '').lower() for s in symptoms if isinstance(s, dict)}
        if condition in symptom_descriptions or condition in symptom_severities or condition in locations:
            return test.strip()
        return None
    return workup.strip() if workup.lower() not in ["none", ""] else None

# Clinical Analysis Engine
class ClinicalAnalyzer:
    def __init__(self):
        self.model = model
        self.tokenizer = tokenizer
        self.knowledge = load_knowledge_base()
        self.medical_stop_words = self.knowledge.get("medical_stop_words", set())
        self.medical_terms = self.knowledge.get("medical_terms", set())
        self.synonyms = self.knowledge.get("synonyms", {})
        self.clinical_pathways = self.knowledge.get("clinical_pathways", {})
        self.history_diagnoses = self.knowledge.get("history_diagnoses", {})
        self.diagnosis_relevance = self.knowledge.get("diagnosis_relevance", {})
        self.management_config = self.knowledge.get("management_config", {})
        self.diagnosis_treatments = self.knowledge.get("diagnosis_treatments", {})
        
        # Cache diagnoses list
        self.diagnoses_list = set()
        if isinstance(self.clinical_pathways, dict):
            for category, pathways in self.clinical_pathways.items():
                if isinstance(pathways, dict):
                    for key, path in pathways.items():
                        if isinstance(path, dict):
                            differentials = path.get('differentials', [])
                            if isinstance(differentials, list):
                                self.diagnoses_list.update(d.lower() for d in differentials)
        
        # Common symptoms
        self.common_symptoms = {
            'headache', 'photophobia', 'nausea', 'chest pain', 'shortness of breath',
            'epigastric pain', 'vomiting', 'knee pain', 'swelling', 'numbness',
            'tingling', 'cough', 'rash', 'back pain', 'diarrhea', 'cramping',
            'wheezing', 'constipation', 'fatigue', 'edema', 'abdominal discomfort'
        }

    def extract_clinical_features(self, note: SOAPNote) -> Dict:
        """Extract structured clinical features from SOAP note."""
        logger.debug(f"Extracting features for note {note.id}")
        features = {
            'chief_complaint': "",
            'hpi': note.hpi or "",
            'history': note.medical_history or "",
            'medications': note.medication_history or "",
            'assessment': note.assessment or "",
            'recommendation': note.recommendation or "",
            'additional_notes': note.additional_notes or "",
            'symptoms': [],
            'aggravating_factors': note.aggravating_factors or "",
            'alleviating_factors': note.alleviating_factors or ""
        }

        # Set chief complaint
        if hasattr(note, 'situation') and note.situation:
            features['chief_complaint'] = note.situation.replace("Patient presents with", "").replace("Patient reports", "").replace("Patient experiencing", "").strip()
            logger.debug(f"Chief complaint set: {features['chief_complaint']}")
        else:
            logger.warning(f"No situation for note {note.id}")

        # Extract symptoms
        text = f"{features['chief_complaint']} {features['hpi']} {features['additional_notes']}"
        # Improved negation detection: capture multi-word phrases and filter by medical terms
        negated_terms = set()
        for match in re.finditer(r'\b(?:no|denies|without)\s+([\w\s]+?)(?:\s|$)', text.lower()):
            term = match.group(1).strip()
            if term in self.common_symptoms or term in self.medical_terms:
                negated_terms.add(term)
        logger.debug(f"Negated terms: {negated_terms}")

        if features['chief_complaint']:
            chief_symptom = preprocess_text(features['chief_complaint'], self.medical_stop_words)
            if chief_symptom and chief_symptom not in negated_terms:
                symptom_dict = {
                    'description': chief_symptom,
                    'duration': extract_duration(text),
                    'severity': classify_severity(text),
                    'location': extract_location(features['chief_complaint'] + " " + features['hpi']),
                    'aggravating': features['aggravating_factors'] or extract_aggravating_alleviating(text, "aggravating"),
                    'alleviating': features['alleviating_factors'] or extract_aggravating_alleviating(text, "alleviating")
                }
                features['symptoms'].append(symptom_dict)
                logger.debug(f"Added chief symptom: {symptom_dict}")
            # Split composite chief complaints
            if ' and ' in features['chief_complaint']:
                for term in features['chief_complaint'].split(' and '):
                    term = term.strip()
                    if not term or term in negated_terms:
                        continue
                    if any(word in self.common_symptoms or word in self.medical_terms for word in term.split()):
                        symptom_dict = {
                            'description': term,
                            'duration': extract_duration(text),
                            'severity': classify_severity(text),
                            'location': extract_location(term + " " + text),
                            'aggravating': features['aggravating_factors'] or extract_aggravating_alleviating(text, "aggravating"),
                            'alleviating': features['alleviating_factors'] or extract_aggravating_alleviating(text, "alleviating")
                        }
                        features['symptoms'].append(symptom_dict)
                        logger.debug(f"Added split symptom: {symptom_dict}")

        # Rule-based symptom extraction
        symptom_candidates = set(preprocess_text(text, self.medical_stop_words).split())
        logger.debug(f"Symptom candidates: {symptom_candidates}")
        for term in symptom_candidates:
            if not isinstance(term, str):
                logger.warning(f"Non-string symptom candidate: {term}")
                continue
            if (term in self.common_symptoms or term in self.medical_terms) and term not in negated_terms:
                symptom_dict = {
                    'description': term,
                    'duration': extract_duration(text),
                    'severity': classify_severity(text),
                    'location': extract_location(term + " " + text),
                    'aggravating': features['aggravating_factors'] or extract_aggravating_alleviating(text, "aggravating"),
                    'alleviating': features['alleviating_factors'] or extract_aggravating_alleviating(text, "alleviating")
                }
                features['symptoms'].append(symptom_dict)
                logger.debug(f"Added rule-based symptom: {symptom_dict}")

        # Embedding-based symptom validation
        clinical_embedding = embed_text("clinical symptom")
        expanded_candidates = set()
        for term in symptom_candidates:
            if not isinstance(term, str):
                logger.warning(f"Skipping invalid term (non-string): {term}")
                continue
            if term in negated_terms:
                logger.debug(f"Skipping negated term: {term}")
                continue
            if term not in self.common_symptoms and term not in self.medical_terms:
                continue
            expanded_candidates.add(term)
            if isinstance(self.synonyms, dict):
                for key, aliases in self.synonyms.items():
                    if not isinstance(aliases, list):
                        logger.warning(f"Invalid aliases for {key}: {aliases}")
                        continue
                    if term.lower() in [a.lower() for a in aliases if isinstance(a, str)]:
                        if key.lower() not in self.diagnoses_list:
                            expanded_candidates.add(key.lower())
                            expanded_candidates.update(a.lower() for a in aliases if isinstance(a, str))

        for term in expanded_candidates:
            if not isinstance(term, str):
                logger.warning(f"Non-string expanded candidate: {term}")
                continue
            if term in self.medical_terms and term not in self.diagnoses_list:
                try:
                    term_embedding = embed_text(term)
                    location = extract_location(term + " " + text)
                    context_term = f"{term} {location.lower()}" if location != "Unspecified" else term
                    context_embedding = embed_text(context_term)
                    similarity = torch.cosine_similarity(context_embedding.unsqueeze(0), clinical_embedding.unsqueeze(0)).item()
                    if similarity > SIMILARITY_THRESHOLD:
                        symptom_dict = {
                            'description': term,
                            'duration': extract_duration(text),
                            'severity': classify_severity(text),
                            'location': location,
                            'aggravating': features['aggravating_factors'] or extract_aggravating_alleviating(text, "aggravating"),
                            'alleviating': features['alleviating_factors'] or extract_aggravating_alleviating(text, "alleviating")
                        }
                        features['symptoms'].append(symptom_dict)
                        logger.debug(f"Added embedding-based symptom: {symptom_dict}, similarity: {similarity}")
                except Exception as e:
                    logger.warning(f"Embedding failed for term {term}: {str(e)}")

        # Deduplicate symptoms
        original_symptoms = features['symptoms'].copy()
        symptom_descriptions = [s.get('description', '') for s in original_symptoms if isinstance(s, dict)]
        deduped_descriptions = deduplicate(tuple(symptom_descriptions), self.synonyms)
        features['symptoms'] = []
        seen = set()
        for desc in deduped_descriptions:
            if not isinstance(desc, str):
                logger.warning(f"Non-string description in deduplication: {desc}")
                continue
            desc_lower = desc.lower()
            if desc_lower not in seen:
                seen.add(desc_lower)
                for symptom in original_symptoms:
                    if not isinstance(symptom, dict):
                        logger.warning(f"Non-dict symptom in deduplication: {symptom}")
                        continue
                    if symptom.get('description', '').lower() == desc_lower:
                        features['symptoms'].append(symptom)
                        break
        logger.debug(f"Final symptoms: {features['symptoms']}")
        return features

    def generate_differential_dx(self, features: Dict) -> List[Tuple[str, float, str]]:
        """Generate ranked differential diagnoses."""
        logger.debug(f"Generating differentials for chief complaint: {features.get('chief_complaint')}, symptoms: {features.get('symptoms', [])}")
        dx_scores = {}
        symptoms = features.get('symptoms', [])
        history = features.get('history', '').lower()
        additional_notes = features.get('additional_notes', '').lower()
        text = f"{features.get('chief_complaint', '')} {features.get('hpi', '')} {additional_notes}"
        text_embedding = embed_text(text)
        primary_dx = features.get('assessment', '').lower()
        chief_complaint = features.get('chief_complaint', '').lower()

        # Assessment-based differential
        if primary_dx:
            clean_assessment = primary_dx.replace("possible ", "").strip()
            logger.debug(f"Primary diagnosis: {clean_assessment}")

        # Symptom and location matching
        for symptom in symptoms:
            if not isinstance(symptom, dict):
                logger.warning(f"Invalid symptom format: {symptom}")
                continue
            symptom_type = symptom.get('description', '').lower()
            location = symptom.get('location', '').lower()
            aggravating = symptom.get('aggravating', '').lower()
            alleviating = symptom.get('alleviating', '').lower()
            if not isinstance(self.clinical_pathways, dict):
                logger.error(f"clinical_pathways not a dict: {type(self.clinical_pathways)}")
                continue
            for category, pathways in self.clinical_pathways.items():
                if not isinstance(pathways, dict):
                    logger.error(f"pathways not a dict for {category}: {type(pathways)}")
                    continue
                for key, path in pathways.items():
                    if not isinstance(path, dict):
                        logger.error(f"path not a dict for {key}: {type(path)}")
                        continue
                    key_lower = key.lower()
                    synonyms = self.synonyms.get(symptom_type, [])
                    if (symptom_type == key_lower or location == key_lower or symptom_type in synonyms):
                        differentials = path.get('differentials', [])
                        if not isinstance(differentials, list):
                            logger.error(f"differentials not a list for {key}: {type(differentials)}")
                            continue
                        for diff in differentials:
                            if not isinstance(diff, str):
                                logger.warning(f"Non-string differential for {key}: {diff}")
                                continue
                            if diff.lower() != primary_dx:
                                score = 0.7
                                if symptom_type in chief_complaint:
                                    score += 0.2
                                reasoning = f"Matches symptom: {symptom_type} in {location}"
                                if aggravating and alleviating:
                                    reasoning += f"; influenced by {aggravating}/{alleviating}"
                                dx_scores[diff] = (score, reasoning)
                                logger.debug(f"Added symptom-based dx: {diff}")

        # History-based differentials
        if isinstance(self.history_diagnoses, dict):
            for condition, aliases in self.history_diagnoses.items():
                if not isinstance(aliases, list):
                    logger.error(f"aliases not a list for {condition}: {type(aliases)}")
                    continue
                if any(alias.lower() in history for alias in aliases):
                    if condition.lower() != primary_dx:
                        dx_scores[condition] = (0.75, f"Supported by medical history: {condition}")
                        logger.debug(f"Added history-based dx: {condition}")

        # Contextual clues
        if 'new pet' in additional_notes:
            dx_scores['Allergic cough'] = (0.75, "Supported by new pet exposure")
        if 'new medication' in additional_notes:
            dx_scores['Drug reaction'] = (0.75, "Suggested by new medication")
        if 'travel' in additional_notes and 'diarrhea' in chief_complaint:
            dx_scores['Traveler’s diarrhea'] = (0.75, "Suggested by recent travel")
        if 'sedentary job' in history:
            dx_scores['Mechanical low back pain'] = (0.75, "Supported by sedentary lifestyle")
        if 'eczema' in history:
            dx_scores['Eczema flare'] = (0.75, "Supported by eczema history")
        if 'lactose intolerance' in history:
            dx_scores['Lactose intolerance'] = (0.75, "Supported by lactose intolerance history")
        if "no weight loss" in text.lower():
            dx_scores.pop("Malignancy", None)
            logger.debug("Removed Malignancy due to no weight loss")

        # Embedding-based scoring
        for dx in dx_scores:
            try:
                dx_embedding = embed_text(dx)
                similarity = torch.cosine_similarity(text_embedding.unsqueeze(0), dx_embedding.unsqueeze(0)).item()
                old_score, reasoning = dx_scores[dx]
                dx_scores[dx] = (min(old_score + similarity * 0.1, 0.9), reasoning)
            except Exception as e:
                logger.warning(f"Similarity failed for dx {dx}: {str(e)}")

        # Filter irrelevant diagnoses with relaxed criteria
        def is_relevant(dx: str) -> bool:
            dx_lower = dx.lower()
            symptom_words = {s.get('description', '').lower() for s in symptoms if isinstance(s, dict)}
            locations = {s.get('location', '').lower() for s in symptoms if isinstance(s, dict)}
            if isinstance(self.diagnosis_relevance, dict):
                for condition, required in self.diagnosis_relevance.items():
                    if dx_lower == condition.lower():
                        matches = sum(1 for word in required if word in symptom_words or word in locations)
                        return matches >= len(required) * 0.1 or any(s in chief_complaint for s in required)
            return True

        dx_scores = {dx: score for dx, score in dx_scores.items() if is_relevant(dx)}
        logger.debug(f"Filtered dx: {dx_scores.keys()}")

        # Normalize scores
        if dx_scores:
            total_score = sum(score for score, _ in dx_scores.values())
            if total_score > 0:
                dx_scores = {dx: (score / total_score * 0.9, reason) for dx, (score, reason) in dx_scores.items()}

        ranked_dx = []
        logger.debug(f"dx_scores before sorting: {dx_scores}")
        try:
            ranked_dx = [(dx, score, reason) for dx, (score, reason) in sorted(dx_scores.items(), key=lambda x: x[1][0], reverse=True)[:5]]
        except ValueError as e:
            logger.error(f"Error sorting differentials: {str(e)}")
            ranked_dx = []
            for dx, value in dx_scores.items():
                if not isinstance(value, tuple) or len(value) != 2:
                    logger.warning(f"Invalid dx_scores entry for {dx}: {value}")
                    continue
                ranked_dx.append((dx, value[0], value[1]))
            ranked_dx = sorted(ranked_dx, key=lambda x: x[1], reverse=True)[:5]

        if not ranked_dx:
            ranked_dx = [("Undetermined", 0.1, "Insufficient data")]
            logger.warning(f"No differentials generated for chief complaint: {features.get('chief_complaint', 'None')}, symptoms: {features.get('symptoms', [])}")
        logger.debug(f"Returning differentials: {ranked_dx}")
        return ranked_dx

    def generate_management_plan(self, features: Dict, differentials: List[Tuple[str, float, str]]) -> Dict:
        """Generate tailored management plan."""
        logger.debug(f"Generating management plan for {features.get('chief_complaint')}")
        plan = {
            'workup': {'urgent': [], 'routine': []},
            'treatment': {'symptomatic': [], 'definitive': []},
            'follow_up': []
        }
        symptoms = features.get('symptoms', [])
        symptom_descriptions = {s.get('description', '').lower() for s in symptoms if isinstance(s, dict)}
        primary_dx = features.get('assessment', '').lower()
        filtered_dx = set()
        high_risk = False

        # Validate differentials
        validated_differentials = []
        for diff in differentials:
            if not isinstance(diff, tuple) or len(diff) != 3:
                logger.warning(f"Invalid differential format: {diff}. Expected tuple (diagnosis, score, reasoning)")
                if isinstance(diff, str):
                    # Convert string to tuple with default values
                    validated_differentials.append((diff, 0.5, "Unknown reasoning"))
                    filtered_dx.add(diff.lower())
                continue
            dx, score, reason = diff
            if not isinstance(dx, str) or not isinstance(score, (int, float)) or not isinstance(reason, str):
                logger.warning(f"Invalid differential components: {diff}")
                continue
            validated_differentials.append(diff)
            filtered_dx.add(dx.lower())
            if score >= CONFIDENCE_THRESHOLD:
                high_risk = True

        # Primary diagnosis-based plan
        if primary_dx and isinstance(self.clinical_pathways, dict):
            for category, pathways in self.clinical_pathways.items():
                if not isinstance(pathways, dict):
                    logger.error(f"pathways not a dict for {category}: {type(pathways)}")
                    continue
                for key, path in pathways.items():
                    if not isinstance(path, dict):
                        logger.error(f"path not a dict for {key}: {type(path)}")
                        continue
                    differentials = path.get('differentials', [])
                    if not isinstance(differentials, list):
                        logger.error(f"differentials not a list for {key}: {type(differentials)}")
                        continue
                    if any(d.lower() in primary_dx for d in differentials):
                        workup = path.get('workup', {})
                        if not isinstance(workup, dict):
                            logger.error(f"workup not a dict for {key}: {type(workup)}")
                            continue
                        for w in workup.get('urgent', []):
                            parsed = parse_conditional_workup(w, symptoms)
                            if parsed:
                                plan['workup']['urgent'].append(parsed)
                        for w in workup.get('routine', []):
                            parsed = parse_conditional_workup(w, symptoms)
                            if parsed:
                                plan['workup']['routine'].append(parsed)
                        management = path.get('management', {})
                        if not isinstance(management, dict):
                            logger.error(f"management not a dict for {key}: {type(management)}")
                            continue
                        plan['treatment']['symptomatic'].extend(management.get('symptomatic', []))
                        plan['treatment']['definitive'].extend(management.get('definitive', []))
                        logger.debug(f"Added primary dx-based plan for {key}")

        # Symptom-based pathways
        for symptom in symptoms:
            if not isinstance(symptom, dict):
                logger.warning(f"Invalid symptom format: {symptom}")
                continue
            symptom_type = symptom.get('description', '').lower()
            location = symptom.get('location', '').lower()
            if not isinstance(self.clinical_pathways, dict):
                logger.error(f"clinical_pathways not a dict: {type(self.clinical_pathways)}")
                continue
            for category, pathways in self.clinical_pathways.items():
                if not isinstance(pathways, dict):
                    logger.error(f"pathways not a dict for {category}: {type(pathways)}")
                    continue
                for key, path in pathways.items():
                    if not isinstance(path, dict):
                        logger.error(f"path not a dict for {key}: {type(path)}")
                        continue
                    if symptom_type == key.lower() or location == key.lower():
                        differentials = path.get('differentials', [])
                        if not isinstance(differentials, list):
                            logger.error(f"differentials not a list for {key}: {type(differentials)}")
                            continue
                        for diff in differentials:
                            if not isinstance(diff, str):
                                logger.warning(f"Non-string differential for {key}: {diff}")
                                continue
                            if diff.lower() == primary_dx or diff.lower() not in filtered_dx:
                                continue
                            workup = path.get('workup', {})
                            if not isinstance(workup, dict):
                                logger.error(f"workup not a dict for {key}: {type(workup)}")
                                continue
                            for w in workup.get('urgent', []):
                                parsed = parse_conditional_workup(w, symptoms)
                                if parsed:
                                    plan['workup']['urgent'].append(parsed)
                            for w in workup.get('routine', []):
                                parsed = parse_conditional_workup(w, symptoms)
                                if parsed:
                                    plan['workup']['routine'].append(parsed)
                            management = path.get('management', {})
                            if not isinstance(management, dict):
                                logger.error(f"management not a dict for {key}: {type(management)}")
                                continue
                            plan['treatment']['symptomatic'].extend(management.get('symptomatic', []))
                            plan['treatment']['definitive'].extend(management.get('definitive', []))
                            logger.debug(f"Added differential-based plan for {key}")

        # Differential-based management
        if isinstance(self.diagnosis_treatments, dict):
            for diff in validated_differentials:
                dx, _, _ = diff
                if not isinstance(dx, str):
                    logger.warning(f"Non-string differential: {dx}")
                    continue
                for diag_key, mappings in self.diagnosis_treatments.items():
                    if not isinstance(mappings, dict):
                        logger.error(f"mappings not a dict for {diag_key}: {type(mappings)}")
                        continue
                    if diag_key.lower() in dx.lower():
                        workup = mappings.get('workup', {})
                        if not isinstance(workup, dict):
                            logger.error(f"workup not a dict for {diag_key}: {type(workup)}")
                            continue
                        for w in workup.get('urgent', []):
                            parsed = parse_conditional_workup(w, symptoms)
                            if parsed:
                                plan['workup']['urgent'].append(parsed)
                        for w in workup.get('routine', []):
                            parsed = parse_conditional_workup(w, symptoms)
                            if parsed:
                                plan['workup']['routine'].append(parsed)
                        treatment = mappings.get('treatment', {})
                        if not isinstance(treatment, dict):
                            logger.error(f"treatment not a dict for {diag_key}: {type(treatment)}")
                        plan['treatment']['definitive'].extend(treatment.get('definitive', []))
                        logger.debug(f"Added dx-based plan for {dx}")

        # Contextual adjustments
        additional_notes = features.get('additional_notes', '').lower()
        if 'new pet' in additional_notes:
            plan['workup']['routine'].append("Allergy testing")
        if 'new medication' in additional_notes:
            plan['workup']['routine'].append("Medication history review")
        if 'travel' in additional_notes and 'diarrhea' in symptom_descriptions:
            plan['workup']['routine'].append("Stool culture")
        if 'sedentary job' in additional_notes:
            plan['treatment']['definitive'].append("Ergonomic counseling")

        # Follow-up customization
        if high_risk:
            plan['follow_up'] = ["Follow-up in 3-5 days or sooner if symptoms worsen"]
        else:
            plan['follow_up'] = ["Follow-up in 1-2 weeks"]

        # Deduplicate and filter
        for key in plan['workup']:
            plan['workup'][key] = deduplicate(tuple(sorted(set(plan['workup'][key]))), self.synonyms)
            if key == 'routine':
                plan['workup'][key] = [item for item in plan['workup'][key] if item not in plan['workup']['urgent']]
        for key in plan['treatment']:
            plan['treatment'][key] = deduplicate(tuple(sorted(set(plan['treatment'][key]))), self.synonyms)
        logger.debug(f"Final plan: {plan}")
        return plan

# SOAP Note Processing
def build_note_text(note: SOAPNote) -> str:
    """Construct full note text."""
    if not isinstance(note, SOAPNote):
        logger.error(f"Invalid note type: {type(note)}")
        return ""
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
    """Generate concise clinical summary."""
    if not isinstance(note, SOAPNote):
        logger.error(f"Invalid note type: {type(note)}")
        return "Summary unavailable"
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
        logger.error(f"Summary failed for note {note.id}: {str(e)}")
        return "Summary unavailable"

def generate_ai_analysis(note: SOAPNote, patient: Patient = None) -> str:
    """Generate clinical analysis with ranked differentials and reasoning."""
    logger.debug(f"Generating analysis for note {note.id}, situation: {note.situation}")
    if not isinstance(note, SOAPNote):
        logger.error(f"Invalid note type: {type(note)}")
        return """
[=== AI CLINICAL ANALYSIS ===]
[CHIEF CONCERN]
Unknown
[DIAGNOSIS]
Not specified
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
    try:
        analyzer = ClinicalAnalyzer()
        features = analyzer.extract_clinical_features(note)
        logger.debug(f"Features for note {note.id}: {features}")
        if not features['symptoms']:
            logger.warning(f"No symptoms extracted for note {note.id}")
        differentials = analyzer.generate_differential_dx(features)
        plan = analyzer.generate_management_plan(features, differentials)
        high_risk = any(len(diff) >= 2 and diff[1] >= CONFIDENCE_THRESHOLD for diff in differentials if isinstance(diff, tuple))
        analysis_output = f"""
[=== AI CLINICAL ANALYSIS ===]
[CHIEF CONCERN]
{features['chief_complaint'].lower() or 'unknown'}
[DIAGNOSIS]
{features['assessment'] or 'Not specified'}
[DIFFERENTIAL DIAGNOSIS]
{';'.join(f'{dx} ({score:.2%}): {reason}' for dx, score, reason in differentials if isinstance(dx, str) and isinstance(score, (int, float))) or 'Undetermined'}
[RECOMMENDED WORKUP]
■ Urgent: {';'.join(sorted(plan['workup']['urgent'])) or 'None'}
■ Routine: {';'.join(sorted(plan['workup']['routine'])) or 'None'}
[TREATMENT OPTIONS]
▲ Symptomatic: {';'.join(sorted(plan['treatment']['symptomatic'])) or 'None'}
● Definitive: {';'.join(sorted(plan['treatment']['definitive'])) or 'Pending diagnosis'}
[FOLLOW-UP]
{';'.join(plan['follow_up']) or 'As needed'}
DISCLAIMER: This AI-generated analysis requires clinical correlation. {'High-risk conditions detected; urgent review recommended.' if high_risk else ''}
"""
        logger.debug(f"Analysis output for note {note.id}: {analysis_output}")
        return analysis_output.strip()
    except Exception as e:
        logger.error(f"Analysis failed for note {note.id}: {str(e)}")
        return """
[=== AI CLINICAL ANALYSIS ===]
[CHIEF CONCERN]
Unknown
[DIAGNOSIS]
Not specified
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
    """Process all notes needing AI updates."""
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
                            if note.ai_notes is None:
                                note.ai_notes = generate_ai_summary(note)
                            if note.ai_analysis is None:
                                note.ai_analysis = generate_ai_analysis(note, patient)
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

# Helper Functions
def calculate_age(dob: datetime.date) -> Optional[int]:
    """Calculate age from date of birth."""
    if not isinstance(dob, datetime.date):
        logger.warning(f"Invalid dob type: {type(dob)}")
        return None
    today = datetime.now().date()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

def extract_duration(text: str) -> str:
    """Extract duration from clinical text."""
    if not isinstance(text, str):
        logger.error(f"Invalid text type for duration: {type(text)}")
        return "Unknown"
    match = re.search(r'(\d+\s*(day|week|month|year)s?)', text.lower())
    return match.group(0) if match else "Unknown"

def classify_severity(text: str) -> str:
    """Classify symptom severity."""
    if not isinstance(text, str):
        logger.error(f"Invalid text type for severity: {type(text)}")
        return "Mild"
    text = text.lower()
    if 'severe' in text or '8/10' in text:
        return "Severe"
    elif 'moderate' in text:
        return "Moderate"
    return "Mild"

def extract_location(text: str) -> str:
    """Extract symptom location."""
    if not isinstance(text, str):
        logger.error(f"Invalid text type for location: {type(text)}")
        return "Unspecified"
    symptom_specific = {
        'headache': 'Head',
        'photophobia': 'Head',
        'chest pain': 'Chest',
        'shortness of breath': 'Chest',
        'epigastric pain': 'Abdomen',
        'nausea': 'Abdomen',
        'knee pain': 'Knee',
        'swelling': 'Knee',
        'wheezing': 'Chest',
        'cough': 'Chest',
        'rash': 'Skin',
        'back pain': 'Back',
        'diarrhea': 'Abdomen',
        'cramping': 'Abdomen',
        'constipation': 'Abdomen',
        'abdominal discomfort': 'Abdomen',
        'fatigue': 'Generalized'
    }
    text = text.lower()
    for term, loc in symptom_specific.items():
        if term in text:
            return loc
    locations = [
        'head', 'chest', 'abdomen', 'back', 'extremity', 'joint', 'neck',
        'hand', 'arm', 'leg', 'knee', 'ankle', 'foot', 'face', 'eyes',
        'cheeks', 'flank', 'epigastric', 'bilateral', 'skin'
    ]
    found = [loc.capitalize() for loc in locations if loc in text]
    return ", ".join(found) or "Unspecified"

def extract_aggravating_alleviating(text: str, factor: str) -> str:
    """Extract aggravating or alleviating factors."""
    if not isinstance(text, str):
        logger.error(f"Invalid text type for {factor}: {type(text)}")
        return "Unknown"
    text = text.lower()
    if factor == "aggravating":
        match = re.search(r'(?:(aggravat|worse)\s+(?:by|with|on|after)\s+[\w\s,]+)', text)
    else:
        match = re.search(r'(?:(alleviat|better)\s+(?:by|with|after)\s+[\w\s,]+)', text)
    if match:
        result = match.group(0).split('by')[-1].strip() if 'by' in match.group(0) else match.group(0).split('with')[-1].strip()
        return result.replace(' and ', ', ')
    return "Unknown"

def batch_embed_texts(texts: List[str]) -> torch.Tensor:
    """Batch process text embeddings."""
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        logger.error(f"Invalid texts for batch embedding: {type(texts)}")
        return torch.zeros(len(texts) if isinstance(texts, list) else 0, EMBEDDING_DIM)
    try:
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True
        ).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            return outputs.last_hidden_state[:, 0, :].cpu()
    except Exception as e:
        logger.error(f"Batch embedding failed: {str(e)}")
        return torch.zeros(len(texts), EMBEDDING_DIM)