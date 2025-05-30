import os
import json
from datetime import datetime
from typing import Dict, Optional, Set, List
from pathlib import Path
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, PyMongoError
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel, Field, ValidationError
from departments.nlp.logging_setup import get_logger
from departments.nlp.knowledge_base_init import initialize_knowledge_files
from departments.nlp.config import MONGO_URI, DB_NAME

logger = get_logger()

# Singleton cache
_knowledge_base_cache: Optional[Dict] = None
_cache_mtime: Optional[float] = None

# Pydantic models for validation
class Symptom(BaseModel):
    description: str
    umls_cui: Optional[str] = None
    semantic_type: str = "Unknown"

class MedicalTerm(BaseModel):
    term: str
    category: str = "unknown"
    umls_cui: Optional[str] = None
    semantic_type: str = "Unknown"

class ClinicalPath(BaseModel):
    differentials: List[str] = Field(default_factory=lambda: ["Undetermined"])
    contextual_triggers: List[str] = Field(default_factory=list)
    management: Dict = Field(default_factory=lambda: {"symptomatic": ["Symptomatic relief pending"]})
    workup: Dict = Field(default_factory=lambda: {"routine": ["Diagnostic evaluation pending"]})
    references: List[str] = Field(default_factory=lambda: ["None specified"])
    metadata: Dict = Field(default_factory=lambda: {"source": "Unknown", "last_updated": datetime.now().strftime("%Y-%m-%d")})
    follow_up: List[str] = Field(default_factory=lambda: ["Follow-up in 2 weeks"])

class KnowledgeBase(BaseModel):
    version: str = "1.1.0"
    last_updated: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    source: str = "MongoDB"
    symptoms: Dict[str, Dict[str, Symptom]] = {}
    medical_stop_words: Set[str] = set()
    medical_terms: List[MedicalTerm] = []
    synonyms: Dict[str, List[str]] = {}
    clinical_pathways: Dict[str, Dict[str, ClinicalPath]] = {}
    history_diagnoses: Dict = {}
    diagnosis_relevance: Dict = {}
    management_config: Dict = {}

# Resource configuration
RESOURCES = {
    "symptoms": "symptoms.json",
    "medical_stop_words": "medical_stop_words.json",
    "medical_terms": "medical_terms.json",
    "synonyms": "synonyms.json",
    "clinical_pathways": "clinical_pathways.json",
    "history_diagnoses": "history_diagnoses.json",
    "diagnosis_relevance": "diagnosis_relevance.json",
    "management_config": "management_config.json"
}
REQUIRED_CATEGORIES = {'respiratory', 'neurological', 'cardiovascular', 'gastrointestinal', 'musculoskeletal', 'infectious'}
HIGH_RISK_CONDITIONS = {'pulmonary embolism', 'myocardial infarction', 'meningitis', 'malaria', 'dengue'}
STRICT_VALIDATION = os.getenv('STRICT_KB_VALIDATION', 'false').lower() == 'true'

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def connect_mongodb() -> MongoClient:
    """Connect to MongoDB with retry logic."""
    client = MongoClient(MONGO_URI)
    client.admin.command('ping')
    return client

def validate_clinical_path(data: Dict, category: str, path_key: str) -> Optional[ClinicalPath]:
    """Validate and normalize a clinical path."""
    try:
        path = ClinicalPath(**data)
        if STRICT_VALIDATION:
            if not path.differentials:
                logger.error(f"Path {path_key}: missing differentials")
                return None
            if any(dx.lower() in HIGH_RISK_CONDITIONS for dx in path.differentials) and not path.contextual_triggers:
                logger.error(f"Path {path_key}: high-risk differentials missing contextual_triggers")
                return None
        return path
    except ValidationError as e:
        logger.warning(f"Invalid clinical path {path_key} in {category}: {e}")
        return None

def load_from_mongodb() -> Dict[str, any]:
    """Load knowledge base from MongoDB."""
    knowledge = KnowledgeBase().dict()
    try:
        client = connect_mongodb()
        db = client[DB_NAME]
        for key in RESOURCES:
            collection = db[key]
            data = list(collection.find())
            if not data:
                logger.warning(f"No data in MongoDB collection {key}")
                continue
            if key == "symptoms":
                valid_data = {}
                for doc in data:
                    category = doc.get('category')
                    symptom = doc.get('symptom')
                    if not category or not symptom:
                        logger.warning(f"Invalid symptom document: {doc}")
                        continue
                    if category not in valid_data:
                        valid_data[category] = {}
                    try:
                        valid_data[category][symptom] = Symptom(
                            description=doc.get('description', f"UMLS-derived: {symptom}"),
                            umls_cui=doc.get('umls_cui'),
                            semantic_type=doc.get('semantic_type', 'Unknown')
                        ).dict()
                    except ValidationError as e:
                        logger.warning(f"Invalid symptom {symptom}: {e}")
                knowledge[key] = valid_data
            elif key == "medical_stop_words":
                knowledge[key] = set(doc['word'] for doc in data if 'word' in doc)
            elif key == "medical_terms":
                knowledge[key] = [MedicalTerm(**doc).dict() for doc in data if doc.get("term")]
            elif key == "synonyms":
                knowledge[key] = {doc['term']: doc['aliases'] for doc in data if 'term' in doc and isinstance(doc.get('aliases'), list)}
            elif key == "clinical_pathways":
                valid_data = {}
                for doc in data:
                    category = doc.get('category')
                    paths = doc.get('paths', {})
                    if not category or not isinstance(paths, dict):
                        logger.warning(f"Invalid clinical pathway document: {doc}")
                        continue
                    valid_paths = {}
                    for pkey, path in paths.items():
                        validated_path = validate_clinical_path(path, category, pkey)
                        if validated_path:
                            valid_paths[pkey] = validated_path.dict()
                    if valid_paths:
                        valid_data[category] = valid_paths
                knowledge[key] = valid_data
                if REQUIRED_CATEGORIES - set(valid_data.keys()):
                    logger.warning(f"Missing required categories in clinical_pathways: {REQUIRED_CATEGORIES - set(valid_data.keys())}")
            else:
                knowledge[key] = {doc['key']: doc['value'] for doc in data if 'key' in doc and 'value' in doc}
            logger.info(f"Loaded {len(data)} entries for {key} from MongoDB")
        client.close()
        knowledge['source'] = "MongoDB"
    except ConnectionFailure as e:
        logger.error(f"MongoDB connection failed: {e}. Falling back to JSON.")
        knowledge['source'] = "JSON"
    return knowledge

def load_from_json(knowledge_base_dir: Path) -> Dict[str, any]:
    """Load knowledge base from JSON files."""
    knowledge = KnowledgeBase().dict()
    for key, filename in RESOURCES.items():
        if knowledge[key] and key != 'medical_stop_words':
            continue
        file_path = knowledge_base_dir / filename
        try:
            with file_path.open('r') as f:
                data = json.load(f)
                if key == "symptoms":
                    valid_data = {}
                    for category, symptoms in data.items():
                        if not isinstance(symptoms, dict):
                            logger.warning(f"Invalid category {category}: {type(symptoms)}")
                            continue
                        valid_data[category] = {}
                        for symptom, info in symptoms.items():
                            try:
                                valid_data[category][symptom] = Symptom(**info).dict()
                            except ValidationError as e:
                                logger.warning(f"Invalid symptom {symptom}: {e}")
                    knowledge[key] = valid_data
                elif key == "medical_stop_words":
                    knowledge[key] = set(data) if isinstance(data, list) and all(isinstance(w, str) for w in data) else set()
                elif key == "medical_terms":
                    knowledge[key] = [MedicalTerm(**t).dict() for t in data if isinstance(t, dict) and t.get("term")]
                elif key == "synonyms":
                    knowledge[key] = {k: v for k, v in data.items() if isinstance(v, list) and all(isinstance(a, str) for a in v)}
                elif key == "clinical_pathways":
                    valid_data = {}
                    for cat, paths in data.items():
                        if not isinstance(paths, dict):
                            logger.warning(f"Invalid category {cat}: {type(paths)}")
                            continue
                        valid_paths = {}
                        for pkey, path in paths.items():
                            validated_path = validate_clinical_path(path, cat, pkey)
                            if validated_path:
                                valid_paths[pkey] = validated_path.dict()
                        if valid_paths:
                            valid_data[cat] = valid_paths
                    knowledge[key] = valid_data
                    if REQUIRED_CATEGORIES - set(valid_data.keys()):
                        logger.warning(f"Missing required categories in clinical_pathways: {REQUIRED_CATEGORIES - set(valid_data.keys())}")
                elif key == "diagnosis_relevance":
                    if isinstance(data, list):
                        data = {
                            item['diagnosis']: {
                                'relevance': item.get('relevance', []),
                                'category': item.get('category', 'unknown')
                            } for item in data if isinstance(item, dict) and 'diagnosis' in item
                        }
                    knowledge[key] = {
                        k: v for k, v in data.items()
                        if isinstance(v, dict) and 'relevance' in v and isinstance(v['relevance'], list)
                    }
                else:
                    knowledge[key] = data if isinstance(data, dict) else {}
                logger.info(f"Loaded {len(data)} entries for {key} from JSON")
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}. Using default data.")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {file_path}: {e}. Using default data.")
    return knowledge

def cross_reference_umls(knowledge: Dict) -> Dict:
    """Cross-reference UMLS CUIs for synonyms."""
    for term in knowledge['medical_terms']:
        term_lower = term['term'].lower()
        if term_lower in knowledge['synonyms'] and term.get('umls_cui'):
            for synonym in knowledge['synonyms'][term_lower]:
                if not any(t['term'].lower() == synonym.lower() and t.get('umls_cui') for t in knowledge['medical_terms']):
                    knowledge['medical_terms'].append({
                        'term': synonym,
                        'category': term['category'],
                        'umls_cui': term['umls_cui'],
                        'semantic_type': term['semantic_type']
                    })
                    logger.debug(f"Added UMLS-linked synonym {synonym} for {term_lower}")
    return knowledge

def load_knowledge_base(knowledge_base_path: str = None, force_reload: bool = False) -> Dict:
    """Load knowledge base from MongoDB (primary) or JSON files (fallback) with validation and UMLS metadata."""
    global _knowledge_base_cache, _cache_mtime
    knowledge_base_dir = Path(knowledge_base_path or os.path.join(os.path.dirname(__file__), "knowledge_base"))

    # Check cache validity
    if _knowledge_base_cache and not force_reload:
        if knowledge_base_dir.exists():
            try:
                mtime = max(f.stat().st_mtime for f in knowledge_base_dir.glob("*.json"))
                if _cache_mtime and mtime <= _cache_mtime:
                    logger.debug("Returning cached knowledge base")
                    return _knowledge_base_cache
            except ValueError:
                logger.debug("No JSON files found, using cached knowledge base")
                return _knowledge_base_cache

    initialize_knowledge_files()
    knowledge = load_from_mongodb()
    if knowledge['source'] == "JSON" or not all(knowledge[key] for key in RESOURCES):
        knowledge = load_from_json(knowledge_base_dir)
    knowledge = cross_reference_umls(knowledge)

    try:
        _knowledge_base_cache = KnowledgeBase(**knowledge).dict()
        if knowledge_base_dir.exists():
            _cache_mtime = max(f.stat().st_mtime for f in knowledge_base_dir.glob("*.json"))
        logger.info("Knowledge base loaded successfully")
        return _knowledge_base_cache
    except ValidationError as e:
        logger.error(f"Knowledge base validation failed: {e}. Returning unvalidated data.")
        return knowledge

def save_knowledge_base(kb: Dict, knowledge_base_path: str = None) -> bool:
    """Save knowledge base to MongoDB (primary) and JSON files (fallback)."""
    global _knowledge_base_cache, _cache_mtime
    knowledge_base_dir = Path(knowledge_base_path or os.path.join(os.path.dirname(__file__), "knowledge_base"))
    knowledge_base_dir.mkdir(exist_ok=True)

    try:
        kb_validated = KnowledgeBase(**kb).dict()
    except ValidationError as e:
        logger.error(f"Invalid knowledge base data: {e}. Attempting to save unvalidated data.")
        kb_validated = kb

    # Save to MongoDB
    try:
        client = connect_mongodb()
        db = client[DB_NAME]
        bulk_ops = {key: [] for key in RESOURCES}
        for key in RESOURCES:
            collection = db[key]
            collection.drop()
            data = kb_validated.get(key, {})
            if key == "symptoms":
                for category, symptoms in data.items():
                    for symptom, info in symptoms.items():
                        bulk_ops[key].append({
                            'category': category,
                            'symptom': symptom,
                            'description': info['description'],
                            'umls_cui': info.get('umls_cui'),
                            'semantic_type': info.get('semantic_type', 'Unknown')
                        })
            elif key == "medical_stop_words":
                bulk_ops[key] = [{'word': word} for word in data]
            elif key == "medical_terms":
                bulk_ops[key] = [term for term in data]
            elif key == "synonyms":
                bulk_ops[key] = [{'term': term, 'aliases': aliases} for term, aliases in data.items()]
            elif key == "clinical_pathways":
                bulk_ops[key] = [{'category': category, 'paths': paths} for category, paths in data.items()]
            else:
                bulk_ops[key] = [{'key': k, 'value': v} for k, v in data.items()]
            if bulk_ops[key]:
                collection.insert_many(bulk_ops[key])
            logger.info(f"Saved {len(bulk_ops[key])} entries for {key} to MongoDB")
        client.close()
    except PyMongoError as e:
        logger.error(f"Error saving to MongoDB: {e}. Saving to JSON only.")

    # Save to JSON
    try:
        for key, filename in RESOURCES.items():
            file_path = knowledge_base_dir / filename
            data = kb_validated.get(key, {})
            if key == "medical_stop_words":
                data = list(data)
            with file_path.open('w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {key} to {file_path}")
        _knowledge_base_cache = kb_validated
        _cache_mtime = max(f.stat().st_mtime for f in knowledge_base_dir.glob("*.json"))
        logger.info("Knowledge base saved to MongoDB and JSON files.")
        return True
    except Exception as e:
        logger.error(f"Error saving knowledge base to JSON: {e}")
        return False

def invalidate_cache():
    """Invalidate the knowledge base cache."""
    global _knowledge_base_cache, _cache_mtime
    _knowledge_base_cache = None
    _cache_mtime = None
    logger.info("Knowledge base cache invalidated")