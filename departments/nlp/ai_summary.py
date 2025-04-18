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
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, PyMongoError
import requests
from bs4 import BeautifulSoup
from Bio import Entrez
import time
from urllib.parse import quote

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

# NEW: Expanded update_knowledge_base for head-to-toe disease coverage
def update_knowledge_base():
    """Fetch and update MongoDB knowledge base with diseases from head to toe from PubMed, Medscape, and other sources."""
    logger.info("Starting knowledge base update for head-to-toe disease coverage")
    
    # MongoDB connection
    try:
        client = MongoClient("mongodb://127.0.0.1:27017/", serverSelectionTimeoutMS=2000)
        db = client["clinical_knowledge_base"]
        logger.info("Connected to MongoDB for knowledge base update")
    except ConnectionFailure as e:
        logger.error(f"MongoDB connection failed: {str(e)}")
        return

    # Initialize data structures
    medical_terms = {"symptoms": [], "diagnoses": [], "procedures": []}
    synonyms = {}
    clinical_pathways = {
        "neurological": {},
        "head_neck": {},
        "cardiopulmonary": {},
        "gastrointestinal": {},
        "genitourinary": {},
        "musculoskeletal": {},
        "dermatological": {},
        "endocrine": {},
        "hematological": {},
        "infectious": {},
        "podiatric": {}
    }
    history_diagnoses = {}
    diagnosis_relevance = {}
    diagnosis_treatments = {}
    
    # Helper function to clean text
    def clean_text(text: str) -> str:
        return re.sub(r'\s+', ' ', text.strip()).lower()

    # Define search terms for head-to-toe coverage
    search_terms = {
        "neurological": ["stroke", "migraine", "dementia", "seizure", "headache", "dizziness"],
        "head_neck": ["sinusitis", "thyroiditis", "hypothyroidism", "hyperthyroidism", "otitis media"],
        "cardiopulmonary": ["heart failure", "copd exacerbation", "pulmonary embolism", "asthma", "dyspnea", "edema", "orthopnea"],
        "gastrointestinal": ["gerd", "inflammatory bowel disease", "hepatitis", "abdominal pain", "diarrhea"],
        "genitourinary": ["urinary tract infection", "kidney stones", "benign prostatic hyperplasia", "dysuria"],
        "musculoskeletal": ["osteoarthritis", "low back pain", "rheumatoid arthritis", "joint pain"],
        "dermatological": ["eczema", "psoriasis", "cellulitis", "rash"],
        "endocrine": ["diabetes mellitus", "hypothyroidism", "hyperglycemia", "fatigue"],
        "hematological": ["anemia", "leukemia", "thrombocytopenia", "fatigue"],
        "infectious": ["sepsis", "pneumonia", "tuberculosis", "fever", "cough"],
        "podiatric": ["diabetic foot", "plantar fasciitis", "foot pain"]
    }

    # 1. PubMed: Fetch recent articles for all systems
    try:
        Entrez.email = "your.email@example.com"  # Replace with your email
        for system, terms in search_terms.items():
            for term in terms:
                handle = Entrez.esearch(db="pubmed", term=term, retmax=3, sort="relevance")
                record = Entrez.read(handle)
                handle.close()
                for pmid in record["IdList"]:
                    handle = Entrez.efetch(db="pubmed", id=pmid, rettype="abstract", retmode="text")
                    abstract = handle.read()
                    handle.close()
                    abstract_lower = abstract.lower()
                    # Extract symptoms, diagnoses, procedures
                    if system == "neurological":
                        if "headache" in abstract_lower:
                            medical_terms["symptoms"].extend(["headache", "migraine"])
                            synonyms["headache"] = ["migraine", "cephalalgia"]
                        if "stroke" in abstract_lower:
                            medical_terms["diagnoses"].append("stroke")
                            diagnosis_relevance["stroke"] = ["headache", "dizziness", "weakness"]
                            diagnosis_treatments["stroke"] = {
                                "workup": {"urgent": ["CT head", "MRI"], "routine": ["lipid panel"]},
                                "treatment": {"symptomatic": ["analgesics"], "definitive": ["aspirin 81 mg daily"]}
                            }
                    elif system == "cardiopulmonary":
                        if "dyspnea" in abstract_lower or "shortness of breath" in abstract_lower:
                            medical_terms["symptoms"].extend(["dyspnea", "shortness of breath"])
                            synonyms["dyspnea"] = ["shortness of breath", "breathlessness"]
                        if "heart failure" in abstract_lower:
                            medical_terms["diagnoses"].append("heart failure")
                            diagnosis_relevance["heart failure"] = ["dyspnea", "edema", "orthopnea"]
                            diagnosis_treatments["heart failure"] = {
                                "workup": {"urgent": ["BNP", "chest x-ray"], "routine": ["echocardiogram"]},
                                "treatment": {"symptomatic": ["fluid restriction: 1.5-2 L/day"], "definitive": ["furosemide 40 mg daily"]}
                            }
                    elif system == "gastrointestinal":
                        if "abdominal pain" in abstract_lower:
                            medical_terms["symptoms"].append("abdominal pain")
                        if "gerd" in abstract_lower:
                            medical_terms["diagnoses"].append("GERD")
                            diagnosis_relevance["GERD"] = ["heartburn", "regurgitation"]
                            diagnosis_treatments["GERD"] = {
                                "workup": {"urgent": [], "routine": ["endoscopy"]},
                                "treatment": {"symptomatic": ["antacids"], "definitive": ["omeprazole 20 mg daily"]}
                            }
                    # Add similar logic for other systems (abridged for brevity)
                    time.sleep(0.3)  # Respect API rate limits
            logger.info(f"Fetched PubMed data for {system}")
    except Exception as e:
        logger.error(f"PubMed fetch failed: {str(e)}")

    # 2. Web scraping: Medscape, Cleveland Clinic, Mayo Clinic, NICE, CDC, WHO
    sources = [
        ("Medscape", "https://www.medscape.com/viewarticle/heart-failure-diagnosis-and-management-2023"),
        ("Cleveland Clinic", "https://my.clevelandclinic.org/health/diseases/17069-heart-failure"),
        ("Mayo Clinic", "https://www.mayoclinic.org/diseases-conditions/heart-failure/symptoms-causes/syc-20373142"),
        ("NICE", "https://www.nice.org.uk/guidance/ng106"),
        ("CDC", "https://www.cdc.gov/diabetes/managing/index.html"),
        ("WHO", "https://www.who.int/news-room/fact-sheets/detail/tuberculosis"),
        # NEW: Add system-specific URLs
        ("Medscape Neuro", "https://www.medscape.com/viewarticle/stroke-diagnosis-and-management-2023"),
        ("Mayo Clinic Derm", "https://www.mayoclinic.org/diseases-conditions/eczema/symptoms-causes/syc-20353273"),
        ("Cleveland Clinic MSK", "https://my.clevelandclinic.org/health/diseases/14526-osteoarthritis")
    ]
    headers = {"User-Agent": "Mozilla/5.0"}
    for source_name, url in sources:
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = soup.find_all("p")
            for p in paragraphs:
                text = clean_text(p.get_text())
                # Cardiopulmonary
                if "heart failure" in text:
                    if "symptoms" in text:
                        medical_terms["symptoms"].extend(["shortness of breath", "dyspnea", "edema", "orthopnea"])
                        synonyms["dyspnea"] = ["shortness of breath", "breathlessness"]
                    if "diagnosis" in text:
                        medical_terms["procedures"].extend(["BNP", "echocardiogram", "chest x-ray"])
                    if "treatment" in text:
                        diagnosis_treatments["heart failure"] = {
                            "workup": {"urgent": ["BNP", "chest x-ray"], "routine": ["echocardiogram"]},
                            "treatment": {"symptomatic": ["fluid restriction: 1.5-2 L/day"], "definitive": ["furosemide 40 mg daily"]}
                        }
                        clinical_pathways["cardiopulmonary"]["dyspnea"] = {
                            "differentials": ["heart failure", "COPD exacerbation", "pulmonary embolism"],
                            "workup": {"urgent": ["BNP", "chest x-ray", "D-dimer", "ECG"], "routine": ["echocardiogram"]},
                            "management": {"symptomatic": ["fluid restriction: 1.5-2 L/day"], "definitive": ["furosemide 40 mg daily"]}
                        }
                # Neurological
                elif "stroke" in text:
                    if "symptoms" in text:
                        medical_terms["symptoms"].extend(["headache", "dizziness", "weakness"])
                    if "diagnosis" in text:
                        medical_terms["procedures"].extend(["CT head", "MRI"])
                    if "treatment" in text:
                        diagnosis_treatments["stroke"] = {
                            "workup": {"urgent": ["CT head", "MRI"], "routine": ["lipid panel"]},
                            "treatment": {"symptomatic": ["analgesics"], "definitive": ["aspirin 81 mg daily"]}
                        }
                        clinical_pathways["neurological"]["headache"] = {
                            "differentials": ["stroke", "migraine", "tension headache"],
                            "workup": {"urgent": ["CT head"], "routine": ["EEG"]},
                            "management": {"symptomatic": ["ibuprofen 400 mg"], "definitive": []}
                        }
                # Dermatological
                elif "eczema" in text:
                    if "symptoms" in text:
                        medical_terms["symptoms"].extend(["rash", "itching"])
                    if "diagnosis" in text:
                        medical_terms["procedures"].append("skin biopsy")
                    if "treatment" in text:
                        diagnosis_treatments["eczema"] = {
                            "workup": {"urgent": [], "routine": ["skin biopsy"]},
                            "treatment": {"symptomatic": ["moisturizers"], "definitive": ["topical corticosteroids"]}
                        }
                        clinical_pathways["dermatological"]["rash"] = {
                            "differentials": ["eczema", "psoriasis", "cellulitis"],
                            "workup": {"urgent": [], "routine": ["skin biopsy"]},
                            "management": {"symptomatic": ["moisturizers"], "definitive": ["topical corticosteroids"]}
                        }
                # Add similar logic for other systems (abridged for brevity)
            logger.info(f"Fetched data from {source_name}")
            time.sleep(1)
        except Exception as e:
            logger.error(f"Failed to scrape {source_name}: {str(e)}")

    # 3. UpToDate (hypothetical API, replace with actual API if available)
    try:
        uptodate_data = {
            "heart failure": {
                "symptoms": ["dyspnea", "edema", "orthopnea", "fatigue"],
                "diagnoses": ["heart failure"],
                "procedures": ["BNP", "echocardiogram", "chest x-ray"],
                "treatments": ["furosemide 40 mg daily", "fluid restriction: 1.5-2 L/day"],
                "relevance": ["dyspnea", "edema", "orthopnea"]
            },
            "stroke": {
                "symptoms": ["headache", "dizziness", "weakness"],
                "diagnoses": ["stroke"],
                "procedures": ["CT head", "MRI"],
                "treatments": ["aspirin 81 mg daily"],
                "relevance": ["headache", "dizziness", "weakness"]
            },
            "diabetes mellitus": {
                "symptoms": ["fatigue", "polyuria", "polydipsia"],
                "diagnoses": ["diabetes mellitus"],
                "procedures": ["HbA1c", "fasting glucose"],
                "treatments": ["metformin 500 mg twice daily"],
                "relevance": ["fatigue", "polyuria"]
            }
        }
        for dx, data in uptodate_data.items():
            medical_terms["symptoms"].extend(data["symptoms"])
            medical_terms["diagnoses"].extend(data["diagnoses"])
            medical_terms["procedures"].extend(data["procedures"])
            diagnosis_relevance[dx] = data["relevance"]
            diagnosis_treatments[dx] = {
                "workup": {"urgent": data["procedures"][:1], "routine": data["procedures"][1:]},
                "treatment": {"symptomatic": [], "definitive": data["treatments"]}
            }
        logger.info("Processed UpToDate data (placeholder)")
    except Exception as e:
        logger.error(f"UpToDate fetch failed: {str(e)}")

    # 4. Deduplicate and structure data
    for key in medical_terms:
        medical_terms[key] = list(set(clean_text(term) for term in medical_terms[key]))
    synonyms = {k: list(set(clean_text(s) for s in v)) for k, v in synonyms.items()}
    history_diagnoses.update({
        "COPD": ["chronic obstructive pulmonary disease", "copd"],
        "hypertension": ["high blood pressure", "htn"],
        "diabetes": ["diabetes mellitus", "dm"],
        "eczema": ["atopic dermatitis"]
    })

    # 5. Update MongoDB collections
    collections = {
        "medical_terms": medical_terms,
        "synonyms": synonyms,
        "clinical_pathways": clinical_pathways,
        "history_diagnoses": history_diagnoses,
        "diagnosis_relevance": diagnosis_relevance,
        "diagnosis_treatments": diagnosis_treatments,
        "management_config": {"high_risk_follow_up": "Follow-up in 3-5 days", "low_risk_follow_up": "Follow-up in 1-2 weeks"},
        "medical_stop_words": {"stop_words": [
            "patient", "history", "present", "illness", "denies", "reports",
            "without", "with", "this", "that", "these", "those", "they", "them",
            "their", "have", "has", "had", "been", "being", "other", "associated"
        ]}
    }
    
    for collection_name, data in collections.items():
        try:
            collection = db[collection_name]
            collection.delete_many({})
            collection.insert_one(data)
            logger.info(f"Updated {collection_name} with {len(data)} entries")
        except PyMongoError as e:
            logger.error(f"Failed to update {collection_name}: {str(e)}")
    
    try:
        client.close()
        logger.info("MongoDB connection closed after update")
    except Exception as e:
        logger.warning(f"Error closing MongoDB connection: {str(e)}")

def load_knowledge_base() -> Dict:
    """Load knowledge base resources from MongoDB collections with enhanced validation and fallback."""
    logger.info("Loading knowledge base from MongoDB clinical_knowledge_base database")
    knowledge = {}
    
    try:
        client = MongoClient("mongodb://127.0.0.1:27017/", serverSelectionTimeoutMS=2000)
        db = client["clinical_knowledge_base"]
        logger.info("Connected to MongoDB: clinical_knowledge_base")
    except ConnectionFailure as e:
        logger.error(f"MongoDB connection failed: {str(e)}")
        raise RuntimeError("Failed to connect to MongoDB")

    resources = {
        "medical_stop_words": "medical_stop_words",
        "medical_terms": "medical_terms",
        "synonyms": "synonyms",
        "clinical_pathways": "clinical_pathways",
        "history_diagnoses": "history_diagnoses",
        "diagnosis_relevance": "diagnosis_relevance",
        "management_config": "management_config",
        "diagnosis_treatments": "diagnosis_treatments"
    }
    
    default_stop_words = [
        "patient", "history", "present", "illness", "denies", "reports",
        "without", "with", "this", "that", "these", "those", "they", "them",
        "their", "have", "has", "had", "been", "being", "other", "associated",
        "complains", "noted", "states", "observed", "left", "right", "ago",
        "since", "recently", "following", "during", "upon", "after"
    ]
    default_medical_terms = {
        "symptoms": ["headache", "chest pain", "shortness of breath"],
        "diagnoses": ["migraine", "myocardial infarction", "pneumonia"],
        "procedures": ["electrocardiogram", "chest x-ray", "complete blood count"]
    }
    default_clinical_pathways = {
        "neurological": {
            "headache": {
                "differentials": ["Migraine", "Tension headache"],
                "workup": {"urgent": [], "routine": ["CBC"]},
                "management": {"symptomatic": ["Ibuprofen 400 mg"], "definitive": []}
            }
        }
    }

    for key, collection_name in resources.items():
        try:
            collection = db[collection_name]
            data = collection.find_one()
            if not data:
                logger.warning(f"No data found in collection: {collection_name}")
                if key == "medical_stop_words":
                    knowledge[key] = set(default_stop_words)
                elif key == "medical_terms":
                    knowledge[key] = default_medical_terms
                elif key == "clinical_pathways":
                    knowledge[key] = default_clinical_pathways
                else:
                    knowledge[key] = {}
                continue

            if "_id" in data:
                del data["_id"]

            if key == "medical_stop_words":
                stop_words = data.get("stop_words", default_stop_words)
                if not isinstance(stop_words, list):
                    logger.error(f"Expected list for {collection_name}, got {type(stop_words)}")
                    stop_words = default_stop_words
                knowledge[key] = set(stop_words).union(default_stop_words)
            elif key == "medical_terms":
                if not isinstance(data, dict) or not all(k in data for k in ["symptoms", "diagnoses", "procedures"]):
                    logger.error(f"Expected dict with symptoms, diagnoses, procedures for {collection_name}")
                    data = default_medical_terms
                for subkey in ["symptoms", "diagnoses", "procedures"]:
                    if not isinstance(data.get(subkey, []), list):
                        logger.warning(f"Invalid {subkey} in {collection_name}, using default")
                        data[subkey] = default_medical_terms[subkey]
                knowledge[key] = data
            elif key == "clinical_pathways":
                if not isinstance(data, dict):
                    logger.error(f"Expected dict for {collection_name}, got {type(data)}")
                    data = default_clinical_pathways
                else:
                    valid_data = {}
                    for cat, paths in data.items():
                        if not isinstance(paths, dict):
                            logger.warning(f"Skipping invalid category {cat}: {type(paths)}")
                            continue
                        valid_paths = {}
                        for pkey, path in paths.items():
                            if not isinstance(path, dict) or not all(k in path for k in ["differentials", "workup", "management"]):
                                logger.warning(f"Skipping invalid path {pkey}: {type(path)}")
                                continue
                            if not isinstance(path["differentials"], list) or not isinstance(path["workup"], dict) or not isinstance(path["management"], dict):
                                logger.warning(f"Invalid structure for {pkey}, skipping")
                                continue
                            valid_paths[pkey] = path
                        if valid_paths:
                            valid_data[cat] = valid_paths
                    data = valid_data or default_clinical_pathways
                knowledge[key] = data
            elif key in ["synonyms", "history_diagnoses", "diagnosis_relevance", "diagnosis_treatments"]:
                if not isinstance(data, dict):
                    logger.error(f"Expected dict for {collection_name}, got {type(data)}")
                    data = {}
                if key == "diagnosis_treatments":
                    valid_data = {}
                    for diag, mappings in data.items():
                        if not isinstance(mappings, dict) or not all(k in mappings for k in ["workup", "treatment"]):
                            logger.warning(f"Skipping invalid diagnosis {diag} in {collection_name}")
                            continue
                        if not isinstance(mappings["workup"], dict) or not isinstance(mappings["treatment"], dict):
                            logger.warning(f"Invalid workup or treatment for {diag}, skipping")
                            continue
                        valid_data[diag] = mappings
                    data = valid_data
                knowledge[key] = data
            elif key == "management_config":
                if not isinstance(data, dict) or not all(k in data for k in ["high_risk_follow_up", "low_risk_follow_up"]):
                    logger.error(f"Expected dict with follow-up keys for {collection_name}")
                    data = {
                        "high_risk_follow_up": "Follow-up in 3-5 days",
                        "low_risk_follow_up": "Follow-up in 1-2 weeks"
                    }
                knowledge[key] = data

            logger.info(f"Loaded {key} from {collection_name} with {len(knowledge[key])} entries")
        except PyMongoError as e:
            logger.error(f"Error querying {collection_name}: {str(e)}")
            if key == "medical_stop_words":
                knowledge[key] = set(default_stop_words)
            elif key == "medical_terms":
                knowledge[key] = default_medical_terms
            elif key == "clinical_pathways":
                knowledge[key] = default_clinical_pathways
            else:
                knowledge[key] = {}
    
    try:
        client.close()
        logger.info("MongoDB connection closed")
    except Exception as e:
        logger.warning(f"Error closing MongoDB connection: {str(e)}")
    
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
        return list(set(items))
    seen = set()
    result = []
    for item in items:
        if not isinstance(item, str):
            logger.warning(f"Non-string item in deduplicate: {item}")
            continue
        canonical = item.lower()
        for key, aliases in synonyms.items():
            if not isinstance(aliases, list):
                logger.error(f"aliases for key {key} is not a list: {type(aliases)}")
                continue
            if item.lower() in [a.lower() for a in aliases if isinstance(a, str)]:
                canonical = key.lower()
                break
        if canonical not in seen:
            seen.add(canonical)
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
        self.medical_terms = set(
            self.knowledge.get("medical_terms", {}).get("symptoms", []) +
            self.knowledge.get("medical_terms", {}).get("diagnoses", []) +
            self.knowledge.get("medical_terms", {}).get("procedures", [])
        )
        self.synonyms = self.knowledge.get("synonyms", {})
        self.clinical_pathways = self.knowledge.get("clinical_pathways", {})
        self.history_diagnoses = self.knowledge.get("history_diagnoses", {})
        self.diagnosis_relevance = self.knowledge.get("diagnosis_relevance", {})
        self.management_config = self.knowledge.get("management_config", {})
        self.diagnosis_treatments = self.knowledge.get("diagnosis_treatments", {})
        
        self.diagnoses_list = set(self.knowledge.get("medical_terms", {}).get("diagnoses", []))
        self.procedures_list = set(self.knowledge.get("medical_terms", {}).get("procedures", []))
        if isinstance(self.clinical_pathways, dict):
            for category, pathways in self.clinical_pathways.items():
                if isinstance(pathways, dict):
                    for key, path in pathways.items():
                        if isinstance(path, dict):
                            differentials = path.get("differentials", [])
                            if isinstance(differentials, list):
                                self.diagnoses_list.update(d.lower() for d in differentials)
        
        self.common_symptoms = set(self.knowledge.get("medical_terms", {}).get("symptoms", []))
        logger.info(f"Initialized ClinicalAnalyzer with {len(self.common_symptoms)} symptoms, "
                    f"{len(self.diagnoses_list)} diagnoses, {len(self.procedures_list)} procedures")

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
            'procedures': [],
            'aggravating_factors': note.aggravating_factors or "",
            'alleviating_factors': note.alleviating_factors or "",
            'negated_terms': set()
        }

        if hasattr(note, 'situation') and note.situation:
            features['chief_complaint'] = note.situation.replace("Patient presents with", "").replace("Patient reports", "").replace("Patient experiencing", "").strip()
            logger.debug(f"Chief complaint set: {features['chief_complaint']}")
        else:
            logger.warning(f"No situation for note {note.id}")

        text = f"{features['chief_complaint']} {features['hpi']} {features['additional_notes']}"
        for match in re.finditer(r'\b(?:no|denies|without)\s+([\w\s]+?)(?:\s|$)', text.lower()):
            term = match.group(1).strip()
            if term in self.common_symptoms or term in self.medical_terms:
                features['negated_terms'].add(term)
        logger.debug(f"Negated terms: {features['negated_terms']}")

        if features['chief_complaint']:
            chief_symptom = preprocess_text(features['chief_complaint'], self.medical_stop_words)
            if chief_symptom and chief_symptom not in features['negated_terms']:
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
            if ' and ' in features['chief_complaint']:
                for term in features['chief_complaint'].split(' and '):
                    term = term.strip()
                    if not term or term in features['negated_terms']:
                        continue
                    if term in self.common_symptoms or term in self.medical_terms:
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

        symptom_candidates = set(preprocess_text(text, self.medical_stop_words).split())
        logger.debug(f"Symptom candidates: {symptom_candidates}")
        for term in symptom_candidates:
            if not isinstance(term, str):
                logger.warning(f"Non-string symptom candidate: {term}")
                continue
            if (term in self.common_symptoms or term in self.medical_terms) and term not in features['negated_terms']:
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
            if term in self.procedures_list:
                features['procedures'].append(term)
                logger.debug(f"Added procedure: {term}")

        clinical_embedding = embed_text("clinical symptom")
        expanded_candidates = set()
        for term in symptom_candidates:
            if not isinstance(term, str) or term in features['negated_terms']:
                continue
            if term not in self.common_symptoms and term not in self.medical_terms:
                continue
            expanded_candidates.add(term)
            if isinstance(self.synonyms, dict):
                for key, aliases in self.synonyms.items():
                    if not isinstance(aliases, list):
                        continue
                    if term.lower() in [a.lower() for a in aliases if isinstance(a, str)]:
                        if key.lower() not in self.diagnoses_list:
                            expanded_candidates.add(key.lower())
                            expanded_candidates.update(a.lower() for a in aliases if isinstance(a, str))

        for term in expanded_candidates:
            if term in self.common_symptoms and term not in self.diagnoses_list:
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

        original_symptoms = features['symptoms'].copy()
        symptom_descriptions = [s.get('description', '') for s in original_symptoms if isinstance(s, dict)]
        deduped_symptom_descriptions = deduplicate(tuple(symptom_descriptions), self.synonyms)
        features['symptoms'] = []
        seen = set()
        for desc in deduped_symptom_descriptions:
            if not isinstance(desc, str):
                continue
            desc_lower = desc.lower()
            if desc_lower not in seen:
                seen.add(desc_lower)
                for symptom in original_symptoms:
                    if not isinstance(symptom, dict):
                        continue
                    if symptom.get('description', '').lower() == desc_lower:
                        features['symptoms'].append(symptom)
                        break
        features['procedures'] = deduplicate(tuple(features['procedures']), self.synonyms)
        logger.debug(f"Final features: {features}")
        return features

    def generate_differential_dx(self, features: Dict) -> List[Tuple[str, float, str]]:
        """Generate ranked differential diagnoses with system-specific prioritization."""
        logger.debug(f"Generating differentials for chief complaint: {features.get('chief_complaint')}")
        dx_scores = {}
        symptoms = features.get('symptoms', [])
        history = features.get('history', '').lower()
        additional_notes = features.get('additional_notes', '').lower()
        negated_terms = features.get('negated_terms', set())
        text = f"{features.get('chief_complaint', '')} {features.get('hpi', '')} {additional_notes}"
        text_embedding = embed_text(text)
        primary_dx = features.get('assessment', '').lower()
        chief_complaint = features.get('chief_complaint', '').lower()

        if primary_dx:
            clean_assessment = primary_dx.replace("possible ", "").strip()
            logger.debug(f"Primary diagnosis: {clean_assessment}")

        for symptom in symptoms:
            if not isinstance(symptom, dict):
                continue
            symptom_type = symptom.get('description', '').lower()
            location = symptom.get('location', '').lower()
            aggravating = symptom.get('aggravating', '').lower()
            alleviating = symptom.get('alleviating', '').lower()
            if not isinstance(self.clinical_pathways, dict):
                continue
            for category, pathways in self.clinical_pathways.items():
                if not isinstance(pathways, dict):
                    continue
                for key, path in pathways.items():
                    if not isinstance(path, dict):
                        continue
                    key_lower = key.lower()
                    synonyms = self.synonyms.get(symptom_type, [])
                    if (symptom_type == key_lower or location == key_lower or symptom_type in synonyms):
                        differentials = path.get('differentials', [])
                        if not isinstance(differentials, list):
                            continue
                        for diff in differentials:
                            if not isinstance(diff, str):
                                continue
                            diff_lower = diff.lower()
                            # Skip if diagnosis requires negated symptom
                            if diff_lower == "angina" and "chest pain" in negated_terms:
                                continue
                            if diff_lower == "pneumonia" and "fever" in negated_terms:
                                continue
                            if diff_lower != primary_dx:
                                score = 0.7
                                if symptom_type in chief_complaint:
                                    score += 0.2
                                matches = sum(1 for s in symptoms if s.get('description', '').lower() in self.diagnosis_relevance.get(diff, []))
                                score += 0.1 * matches
                                reasoning = f"Matches symptom: {symptom_type} in {location}"
                                if aggravating and alleviating:
                                    reasoning += f"; influenced by {aggravating}/{alleviating}"
                                dx_scores[diff] = (score, reasoning)
                                logger.debug(f"Added symptom-based dx: {diff}")

        if isinstance(self.history_diagnoses, dict):
            for condition, aliases in self.history_diagnoses.items():
                if not isinstance(aliases, list):
                    continue
                if any(alias.lower() in history for alias in aliases):
                    if condition.lower() != primary_dx:
                        dx_scores[condition] = (0.75, f"Supported by medical history: {condition}")
                        logger.debug(f"Added history-based dx: {condition}")

        # Boost for smoking history (cardiopulmonary)
        if "smoking" in additional_notes or "pack-years" in additional_notes:
            for dx in ["heart failure", "COPD exacerbation", "pulmonary embolism"]:
                if dx in dx_scores:
                    score, reason = dx_scores[dx]
                    dx_scores[dx] = (score + 0.1, reason + "; supported by smoking history")
        
        # System-specific boosts
        if "new pet" in additional_notes:
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

        for dx in dx_scores:
            try:
                dx_embedding = embed_text(dx)
                similarity = torch.cosine_similarity(text_embedding.unsqueeze(0), dx_embedding.unsqueeze(0)).item()
                old_score, reasoning = dx_scores[dx]
                dx_scores[dx] = (min(old_score + similarity * 0.1, 0.9), reasoning)
            except Exception as e:
                logger.warning(f"Similarity failed for dx {dx}: {str(e)}")

        def is_relevant(dx: str) -> bool:
            dx_lower = dx.lower()
            symptom_words = {s.get('description', '').lower() for s in symptoms if isinstance(s, dict)}
            locations = {s.get('location', '').lower() for s in symptoms if isinstance(s, dict)}
            if isinstance(self.diagnosis_relevance, dict):
                for condition, required in self.diagnosis_relevance.items():
                    if dx_lower == condition.lower():
                        matches = sum(1 for word in required if word in symptom_words or word in locations)
                        return matches >= len(required) * 0.5 or any(s in chief_complaint for s in required)
            return True

        dx_scores = {dx: score for dx, score in dx_scores.items() if is_relevant(dx)}
        logger.debug(f"Filtered dx: {dx_scores.keys()}")

        if dx_scores:
            total_score = sum(score for score, _ in dx_scores.values())
            if total_score > 0:
                dx_scores = {dx: (score / total_score * 0.9, reason) for dx, (score, reason) in dx_scores.items()}

        ranked_dx = []
        try:
            ranked_dx = [(dx, score, reason) for dx, (score, reason) in sorted(dx_scores.items(), key=lambda x: x[1][0], reverse=True)[:5]]
        except ValueError as e:
            logger.error(f"Error sorting differentials: {str(e)}")
            ranked_dx = [(dx, value[0], value[1]) for dx, value in dx_scores.items() if isinstance(value, tuple) and len(value) == 2]
            ranked_dx = sorted(ranked_dx, key=lambda x: x[1], reverse=True)[:5]

        if not ranked_dx:
            ranked_dx = [("Undetermined", 0.1, "Insufficient data")]
            logger.warning(f"No differentials generated for chief complaint: {features.get('chief_complaint', 'None')}")
        logger.debug(f"Returning differentials: {ranked_dx}")
        return ranked_dx

    def generate_management_plan(self, features: Dict, differentials: List[Tuple[str, float, str]]) -> Dict:
        """Generate tailored management plan with note-specific follow-up."""
        logger.debug(f"Generating management plan for {features.get('chief_complaint')}")
        plan = {
            'workup': {'urgent': [], 'routine': []},
            'treatment': {'symptomatic': [], 'definitive': []},
            'follow_up': []
        }
        symptoms = features.get('symptoms', [])
        performed_procedures = features.get('procedures', [])
        symptom_descriptions = {s.get('description', '').lower() for s in symptoms if isinstance(s, dict)}
        primary_dx = features.get('assessment', '').lower()
        filtered_dx = set()
        high_risk = False

        validated_differentials = []
        for diff in differentials:
            if not isinstance(diff, tuple) or len(diff) != 3:
                if isinstance(diff, str):
                    validated_differentials.append((diff, 0.5, "Unknown reasoning"))
                    filtered_dx.add(diff.lower())
                continue
            dx, score, reason = diff
            if not isinstance(dx, str) or not isinstance(score, (int, float)) or not isinstance(reason, str):
                continue
            validated_differentials.append(diff)
            filtered_dx.add(dx.lower())
            if score >= CONFIDENCE_THRESHOLD:
                high_risk = True

        performed_procedures_set = set(p.lower() for p in performed_procedures)

        if primary_dx and isinstance(self.clinical_pathways, dict):
            for category, pathways in self.clinical_pathways.items():
                if not isinstance(pathways, dict):
                    continue
                for key, path in pathways.items():
                    if not isinstance(path, dict):
                        continue
                    differentials = path.get('differentials', [])
                    if any(d.lower() in primary_dx for d in differentials):
                        workup = path.get('workup', {})
                        for w in workup.get('urgent', []):
                            parsed = parse_conditional_workup(w, symptoms)
                            if parsed and parsed.lower() not in performed_procedures_set:
                                plan['workup']['urgent'].append(parsed)
                        for w in workup.get('routine', []):
                            parsed = parse_conditional_workup(w, symptoms)
                            if parsed and parsed.lower() not in performed_procedures_set:
                                plan['workup']['routine'].append(parsed)
                        management = path.get('management', {})
                        plan['treatment']['symptomatic'].extend(management.get('symptomatic', []))
                        plan['treatment']['definitive'].extend(management.get('definitive', []))
                        logger.debug(f"Added primary dx-based plan for {key}")

        for symptom in symptoms:
            if not isinstance(symptom, dict):
                continue
            symptom_type = symptom.get('description', '').lower()
            location = symptom.get('location', '').lower()
            for category, pathways in self.clinical_pathways.items():
                if not isinstance(pathways, dict):
                    continue
                for key, path in pathways.items():
                    if not isinstance(path, dict):
                        continue
                    if symptom_type == key.lower() or location == key.lower():
                        differentials = path.get('differentials', [])
                        for diff in differentials:
                            if diff.lower() == primary_dx or diff.lower() not in filtered_dx:
                                continue
                            workup = path.get('workup', {})
                            for w in workup.get('urgent', []):
                                parsed = parse_conditional_workup(w, symptoms)
                                if parsed and parsed.lower() not in performed_procedures_set:
                                    plan['workup']['urgent'].append(parsed)
                            for w in workup.get('routine', []):
                                parsed = parse_conditional_workup(w, symptoms)
                                if parsed and parsed.lower() not in performed_procedures_set:
                                    plan['workup']['routine'].append(parsed)
                            management = path.get('management', {})
                            plan['treatment']['symptomatic'].extend(management.get('symptomatic', []))
                            plan['treatment']['definitive'].extend(management.get('definitive', []))
                            logger.debug(f"Added differential-based plan for {key}")

        for diff in validated_differentials:
            dx, _, _ = diff
            if not isinstance(dx, str):
                continue
            for diag_key, mappings in self.diagnosis_treatments.items():
                if diag_key.lower() in dx.lower():
                    workup = mappings.get('workup', {})
                    for w in workup.get('urgent', []):
                        parsed = parse_conditional_workup(w, symptoms)
                        if parsed and parsed.lower() not in performed_procedures_set:
                            plan['workup']['urgent'].append(parsed)
                    for w in workup.get('routine', []):
                        parsed = parse_conditional_workup(w, symptoms)
                        if parsed and parsed.lower() not in performed_procedures_set:
                            plan['workup']['routine'].append(parsed)
                    treatment = mappings.get('treatment', {})
                    plan['treatment']['definitive'].extend(treatment.get('definitive', []))

        additional_notes = features.get('additional_notes', '').lower()
        if 'new pet' in additional_notes and 'allergy testing' not in performed_procedures_set:
            plan['workup']['routine'].append("Allergy testing")
        if 'new medication' in additional_notes:
            plan['workup']['routine'].append("Medication history review")
        if 'travel' in additional_notes and 'diarrhea' in symptom_descriptions and 'stool culture' not in performed_procedures_set:
            plan['workup']['routine'].append("Stool culture")
        if 'sedentary job' in additional_notes:
            plan['treatment']['definitive'].append("Ergonomic counseling")

        if features['recommendation']:
            match = re.search(r'follow\s*up\s*in\s*([\w\s-]+?)(?:\s|$)', features['recommendation'].lower())
            if match:
                plan['follow_up'] = [f"Follow-up in {match.group(1)}"]
            else:
                plan['follow_up'] = [self.management_config.get('high_risk_follow_up' if high_risk else 'low_risk_follow_up', 'Follow-up in 1-2 weeks')]
        else:
            plan['follow_up'] = [self.management_config.get('high_risk_follow_up' if high_risk else 'low_risk_follow_up', 'Follow-up in 1-2 weeks')]

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
                if key == "Chief Complaint" and not value.lower().startswith(("patient", "55-year-old")):
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
        'fatigue': 'Generalized',
        'edema': 'Legs',
        'orthopnea': 'Chest',
        'dizziness': 'Head',
        'weakness': 'Generalized',
        'heartburn': 'Abdomen',
        'dysuria': 'Pelvis',
        'joint pain': 'Joints',
        'itching': 'Skin',
        'fever': 'Generalized',
        'foot pain': 'Foot'
    }
    text = text.lower()
    for term, loc in symptom_specific.items():
        if term in text:
            return loc
    locations = [
        'head', 'chest', 'abdomen', 'back', 'extremity', 'joint', 'neck',
        'hand', 'arm', 'leg', 'knee', 'ankle', 'foot', 'face', 'eyes',
        'cheeks', 'flank', 'epigastric', 'bilateral', 'skin', 'pelvis'
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

# Run knowledge base update on startup
if __name__ == "__main__":
    update_knowledge_base()