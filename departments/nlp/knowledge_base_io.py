from datetime import datetime
import os
from typing import Dict, Optional, Set, List
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, PyMongoError
from psycopg2.extras import RealDictCursor, execute_batch
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel, Field, ValidationError
from departments.nlp.logging_setup import get_logger
from departments.nlp.knowledge_base_init import initialize_knowledge_files
from departments.nlp.nlp_pipeline import get_postgres_connection
from departments.nlp.nlp_utils import parse_date
from departments.nlp.nlp_common import FALLBACK_CUI_MAP
from departments.nlp.config import MONGO_URI, DB_NAME, KB_PREFIX, CACHE_DIR

logger = get_logger(__name__)

# Singleton cache
_knowledge_base_cache: Optional[Dict] = None

# Pydantic models
class Symptom(BaseModel):
    description: str
    umls_cui: Optional[str] = None
    semantic_type: str = "unknown"

class MedicalTerm(BaseModel):
    term: str
    category: str = "unknown"
    umls_cui: Optional[str] = None
    semantic_type: str = "unknown"

class ClinicalPath(BaseModel):
    differentials: List[str] = Field(default_factory=lambda: ["Undetermined"])
    contextual_triggers: List[str] = Field(default_factory=list)
    required_symptoms: List[str] = Field(default_factory=list)
    exclusion_criteria: List[str] = Field(default_factory=list)
    management: Dict = Field(default_factory=lambda: {"symptomatic": [], "definitive": [], "lifestyle": []})
    workup: Dict = Field(default_factory=lambda: {"urgent": [], "routine": []})
    references: List[str] = Field(default_factory=lambda: ["Clinical guidelines pending"])
    metadata: Dict = Field(default_factory=lambda: {
        "source": ["Automated update"],
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    follow_up: List[str] = Field(default_factory=lambda: ["Follow-up in 1-2 weeks"])

class DiagnosisRelevance(BaseModel):
    diagnosis: str
    relevance: List[Dict] = Field(default_factory=list)
    category: str = "unknown"

class HistoryDiagnosis(BaseModel):
    synonyms: List[str] = Field(default_factory=list)
    umls_cui: Optional[str] = None
    semantic_type: str = "unknown"

class DiagnosisTreatment(BaseModel):
    treatments: Dict = Field(default_factory=lambda: {"symptomatic": [], "definitive": [], "lifestyle": []})

class ManagementConfig(BaseModel):
    follow_up_default: str = "Follow-up in 2 weeks"
    follow_up_urgent: str = "Follow-up in 3-5 days or sooner if symptoms worsen"
    urgent_threshold: float = 0.9
    min_symptom_match: float = 0.7

class Version(BaseModel):
    version: str = "1.1.0"
    last_updated: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    updated_collections: Dict = Field(default_factory=dict)

class KnowledgeBase(BaseModel):
    version: str = "1.1.0"
    last_updated: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    source: str = "PostgreSQL/MongoDB"
    symptoms: Dict[str, Dict[str, Symptom]] = Field(default_factory=dict)
    medical_stop_words: Set[str] = Field(default_factory=set)
    medical_terms: List[MedicalTerm] = Field(default_factory=list)
    synonyms: Dict[str, List[str]] = Field(default_factory=dict)
    clinical_pathways: Dict[str, Dict[str, ClinicalPath]] = Field(default_factory=dict)
    history_diagnoses: Dict[str, HistoryDiagnosis] = Field(default_factory=dict)
    diagnosis_relevance: List[DiagnosisRelevance] = Field(default_factory=list)
    management_config: Dict[str, ManagementConfig] = Field(default_factory=dict)
    diagnosis_treatments: Dict[str, DiagnosisTreatment] = Field(default_factory=dict)
    versions: Dict[str, Version] = Field(default_factory=dict)

# Resource configuration
RESOURCES = {
    "symptoms": "postgresql",
    "medical_stop_words": "postgresql",
    "medical_terms": "postgresql",
    "synonyms": "mongodb",
    "clinical_pathways": "mongodb",
    "history_diagnoses": "mongodb",
    "diagnosis_relevance": "mongodb",
    "management_config": "mongodb",
    "diagnosis_treatments": "mongodb",
    "versions": "mongodb"
}
REQUIRED_CATEGORIES = {
    'respiratory', 'neurological', 'cardiovascular', 'gastrointestinal',
    'musculoskeletal', 'infectious', 'hepatic'
}
HIGH_RISK_CONDITIONS = {
    'pulmonary embolism', 'myocardial infarction', 'meningitis', 'malaria', 'dengue'
}
STRICT_VALIDATION = os.getenv('STRICT_KB_VALIDATION', 'false').lower() == 'true'

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def connect_mongodb() -> MongoClient:
    """Connect to MongoDB with retry logic."""
    try:
        client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=10000,
            socketTimeoutMS=10000,
        )
        client.admin.command('ping')
        logger.info("MongoDB connection successful")
        return client
    except ConnectionFailure as e:
        logger.error(f"MongoDB connection failed: {str(e)}")
        raise

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
            try:
                with get_postgres_connection() as cursor:
                    if not cursor:
                        logger.error(f"No database cursor available to validate clinical path {path_key}")
                        return None
                    for symptom in path.required_symptoms:
                        cursor.execute("SELECT 1 FROM symptoms WHERE symptom = %s", (symptom.lower(),))
                        if not cursor.fetchone():
                            logger.warning(f"Required symptom '{symptom}' not found in symptoms table")
                            return None
                    for criterion in path.exclusion_criteria:
                        cursor.execute("SELECT 1 FROM symptoms WHERE symptom = %s", (criterion.lower(),))
                        if not cursor.fetchone():
                            logger.warning(f"Exclusion criterion '{criterion}' not found in symptoms table")
                            return None
            except Exception as e:
                logger.error(f"Error validating symptoms for path {path_key}: {str(e)}")
                return None
        return path
    except ValidationError as e:
        logger.warning(f"Invalid clinical path {path_key} in {category}: {e}")
        return None

def load_from_postgresql() -> Dict[str, any]:
    """Load knowledge base from PostgreSQL."""
    knowledge = KnowledgeBase().dict()
    with get_postgres_connection() as cursor:
        if not cursor:
            logger.error("Failed to get PostgreSQL cursor")
            return knowledge
        try:
            # Load medical_stop_words
            cursor.execute("SELECT word FROM medical_stop_words")
            knowledge['medical_stop_words'] = {row['word'].lower() for row in cursor.fetchall()}
            logger.info(f"Loaded {len(knowledge['medical_stop_words'])} medical_stop_words from PostgreSQL")

            # Load medical_terms
            cursor.execute("SELECT term, category, umls_cui, semantic_type FROM medical_terms")
            knowledge['medical_terms'] = [
                MedicalTerm(
                    term=row['term'],
                    category=row['category'] if row['category'] in REQUIRED_CATEGORIES else "unknown",
                    umls_cui=row['umls_cui'],
                    semantic_type=row['semantic_type'] or "unknown"
                ).dict() for row in cursor.fetchall()
            ]
            logger.info(f"Loaded {len(knowledge['medical_terms'])} medical_terms from PostgreSQL")

            # Load symptoms
            cursor.execute("SELECT symptom, category, description, umls_cui, semantic_type FROM symptoms")
            valid_data = {}
            for row in cursor.fetchall():
                category = row['category'] if row['category'] in REQUIRED_CATEGORIES else "unknown"
                symptom = row['symptom']
                if not symptom:
                    logger.warning(f"Invalid symptom: {row}")
                    continue
                if category not in valid_data:
                    valid_data[category] = {}
                try:
                    umls_cui = row['umls_cui']
                    semantic_type = row['semantic_type'] or "unknown"
                    if not umls_cui and symptom.lower() in FALLBACK_CUI_MAP:
                        umls_cui = FALLBACK_CUI_MAP[symptom.lower()]['umls_cui']
                        semantic_type = FALLBACK_CUI_MAP[symptom.lower()]['semantic_type']
                        logger.debug(f"Applied fallback for symptom '{symptom}': CUI={umls_cui}, SemanticType={semantic_type}")
                    valid_data[category][symptom] = Symptom(
                        description=row['description'] or f"Description for {symptom}",
                        umls_cui=umls_cui,
                        semantic_type=semantic_type
                    ).dict()
                except ValidationError as e:
                    logger.warning(f"Invalid symptom {symptom}: {e}")
            knowledge['symptoms'] = valid_data
            logger.info(f"Loaded {sum(len(s) for s in valid_data.values())} symptoms from PostgreSQL")

            # Load metadata
            cursor.execute("SELECT version, last_updated FROM knowledge_base_metadata WHERE key = 'knowledge_base'")
            metadata = cursor.fetchone()
            if metadata:
                knowledge['version'] = metadata['version']
                knowledge['last_updated'] = parse_date(metadata['last_updated']).strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            logger.error(f"Failed to load from PostgreSQL: {str(e)}")
    return knowledge

def load_from_mongodb() -> Dict[str, any]:
    """Load knowledge base from MongoDB."""
    knowledge = KnowledgeBase().dict()
    try:
        client = connect_mongodb()
        db = client[DB_NAME]
        kb_prefix = KB_PREFIX or 'kb_'
        for key in RESOURCES:
            if RESOURCES[key] != "mongodb":
                continue
            collection = db[f'{kb_prefix}{key}']
            data = list(collection.find())
            if not data and key != "clinical_pathways":
                logger.warning(f"No data in MongoDB collection {key}")
                continue
            if key == "synonyms":
                knowledge[key] = {
                    doc['term'].lower(): doc['aliases']
                    for doc in data if 'term' in doc and isinstance(doc.get('aliases'), list)
                }
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
                        if 'metadata' in path and 'last_updated' in path['metadata']:
                            path['metadata']['last_updated'] = parse_date(path['metadata']['last_updated']).strftime("%Y-%m-%d %H:%M:%S")
                        validated_path = validate_clinical_path(path, category, pkey)
                        if validated_path:
                            valid_paths[pkey] = validated_path.dict()
                    if valid_paths:
                        valid_data[category] = valid_paths
                knowledge[key] = valid_data
                missing_categories = REQUIRED_CATEGORIES - set(valid_data.keys())
                if missing_categories:
                    logger.warning(f"Missing required categories in clinical_pathways: {missing_categories}")
                    for category in missing_categories:
                        knowledge[key][category] = {}
                        collection.update_one(
                            {'category': category},
                            {'$set': {'category': category, 'paths': {}}},
                            upsert=True
                        )
                        logger.info(f"Saved empty clinical pathway for category: {category} to MongoDB")
            elif key == "diagnosis_relevance":
                knowledge[key] = [
                    DiagnosisRelevance(
                        diagnosis=doc['diagnosis'],
                        relevance=doc.get('relevance', []),
                        category=doc.get('category', 'unknown')
                    ).dict() for doc in data if 'diagnosis' in doc and isinstance(doc.get('relevance'), list)
                ]
            elif key == "diagnosis_treatments":
                knowledge[key] = {
                    doc['key'].lower(): DiagnosisTreatment(**doc['value']).dict()
                    for doc in data if 'key' in doc and 'value' in doc
                }
            elif key == "versions":
                knowledge[key] = {
                    doc['version']: Version(**doc).dict()
                    for doc in data if 'version' in doc
                }
            elif key == "history_diagnoses":
                knowledge[key] = {
                    doc['key'].lower(): HistoryDiagnosis(**doc['value']).dict()
                    for doc in data if 'key' in doc and 'value' in doc
                }
            elif key == "management_config":
                knowledge[key] = {}
                for doc in data:
                    if 'key' in doc and 'value' in doc:
                        try:
                            if isinstance(doc['value'], dict):
                                knowledge[key][doc['key'].lower()] = ManagementConfig(**doc['value']).dict()
                            else:
                                logger.warning(f"Invalid ManagementConfig for key '{doc['key']}': value is {type(doc['value'])}, expected dict. Using default.")
                                knowledge[key][doc['key'].lower()] = ManagementConfig().dict()
                        except ValidationError as e:
                            logger.error(f"Failed to parse ManagementConfig for key '{doc['key']}': {e}. Using default.")
                            knowledge[key][doc['key'].lower()] = ManagementConfig().dict()
                    else:
                        logger.warning(f"Skipping invalid management_config document: {doc}")
            logger.info(f"Loaded {len(data)} entries for {key} from MongoDB")
        client.close()
    except ConnectionFailure as e:
        logger.error(f"MongoDB connection failed: {e}")
    except PyMongoError as e:
        logger.error(f"MongoDB load failed: {str(e)}")
    return knowledge

def cross_reference_umls(knowledge: Dict) -> Dict:
    """Cross-reference UMLS CUIs for synonyms."""
    with get_postgres_connection() as cursor:
        if not cursor:
            logger.error("Failed to get PostgreSQL cursor for UMLS cross-referencing")
            return knowledge
        try:
            existing_terms = set(t['term'].lower() for t in knowledge['medical_terms'] if t.get('umls_cui'))
            new_terms = []
            for term in knowledge['medical_terms']:
                term_lower = term['term'].lower()
                if term_lower in knowledge['synonyms'] and term.get('umls_cui'):
                    for synonym in knowledge['synonyms'][term_lower]:
                        synonym_lower = synonym.lower()
                        if synonym_lower in existing_terms:
                            continue
                        cui = FALLBACK_CUI_MAP.get(synonym_lower, {}).get('umls_cui')
                        semantic_type = FALLBACK_CUI_MAP.get(synonym_lower, {}).get('semantic_type', term['semantic_type'])
                        if not cui:
                            cursor.execute("""
                                SELECT CUI FROM umls.MRCONSO
                                WHERE LOWER(STR) = %s AND SAB = 'SNOMEDCT_US' AND SUPPRESS = 'N'
                            """, (synonym_lower,))
                            result = cursor.fetchone()
                            cui = result['CUI'] if result else term['umls_cui']
                        new_terms.append({
                            'term': synonym,
                            'category': term['category'] if term['category'] in REQUIRED_CATEGORIES else 'unknown',
                            'umls_cui': cui,
                            'semantic_type': semantic_type
                        })
                        existing_terms.add(synonym_lower)
                        logger.debug(f"Added UMLS-linked synonym '{synonym}' for '{term_lower}' with CUI {cui}")
            knowledge['medical_terms'].extend(new_terms)
        except Exception as e:
            logger.error(f"Failed to cross-reference UMLS: {str(e)}")
    return knowledge

def load_knowledge_base(force_reload: bool = False) -> Dict:
    """Load knowledge base from PostgreSQL and MongoDB."""
    global _knowledge_base_cache
    if _knowledge_base_cache and not force_reload:
        logger.debug("Returning cached knowledge base")
        return _knowledge_base_cache

    initialize_knowledge_files()
    knowledge = load_from_postgresql()
    mongo_data = load_from_mongodb()
    knowledge.update({k: v for k, v in mongo_data.items() if v})
    knowledge = cross_reference_umls(knowledge)

    try:
        _knowledge_base_cache = KnowledgeBase(**knowledge).dict()
        logger.info("Knowledge base loaded successfully")
        return _knowledge_base_cache
    except ValidationError as e:
        logger.error(f"Knowledge base validation failed: {e}. Returning unvalidated data.")
        return knowledge

def save_knowledge_base(kb: Dict) -> bool:
    """Save knowledge base to PostgreSQL and MongoDB."""
    global _knowledge_base_cache
    try:
        kb_validated = KnowledgeBase(**kb).dict()
    except ValidationError as e:
        logger.error(f"Invalid knowledge base data: {e}. Attempting to save unvalidated data.")
        kb_validated = kb.copy()

    # Save to PostgreSQL
    with get_postgres_connection(readonly=False) as cursor:
        if not cursor:
            logger.error("Failed to get PostgreSQL cursor")
            return False
        try:
            cursor.execute("DELETE FROM medical_stop_words")
            execute_batch(cursor, "INSERT INTO medical_stop_words (word) VALUES (%s)",
                          [(word,) for word in kb_validated.get('medical_stop_words', [])])
            logger.info(f"Inserted {len(kb_validated.get('medical_stop_words', []))} medical_stop_words into table")

            cursor.execute("DELETE FROM medical_terms")
            execute_batch(cursor, """
                INSERT INTO medical_terms (term, category, umls_cui, semantic_type)
                VALUES (%s, %s, %s, %s)
            """, [(t['term'], t['category'] if t['category'] in REQUIRED_CATEGORIES else 'unknown',
                   t.get('umls_cui'), t.get('semantic_type'))
                  for t in kb_validated.get('medical_terms', [])])
            logger.info(f"Inserted {len(kb_validated.get('medical_terms', []))} medical_terms into table")

            cursor.execute("DELETE FROM symptoms")
            symptom_data = []
            for cat, symptoms in kb_validated.get('symptoms', {}).items():
                if cat not in REQUIRED_CATEGORIES:
                    logger.warning(f"Invalid symptom category '{cat}', skipping")
                    continue
                for s, info in symptoms.items():
                    umls_cui = info.get('umls_cui')
                    semantic_type = info.get('semantic_type', 'unknown')
                    if not umls_cui and s.lower() in FALLBACK_CUI_MAP:
                        umls_cui = FALLBACK_CUI_MAP[s.lower()]['umls_cui']
                        semantic_type = FALLBACK_CUI_MAP[s.lower()]['semantic_type']
                        logger.debug(f"Applied fallback for symptom '{s}' during save: CUI={umls_cui}, SemanticType={semantic_type}")
                    symptom_data.append((s, cat, info['description'], umls_cui, semantic_type))
            if symptom_data:
                execute_batch(cursor, """
                    INSERT INTO symptoms (symptom, category, description, umls_cui, semantic_type)
                    VALUES (%s, %s, %s, %s, %s)
                """, symptom_data)
            logger.info(f"Inserted {len(symptom_data)} symptoms into table")

            cursor.execute("""
                INSERT INTO knowledge_base_metadata (key, version, last_updated)
                VALUES (%s, %s, %s) ON CONFLICT (key) DO UPDATE
                SET version = EXCLUDED.version, last_updated = EXCLUDED.last_updated
            """, ('knowledge_base', kb_validated.get('version', '1.1.0'), parse_date(kb_validated.get('last_updated', datetime.now())).strftime("%Y-%m-%d %H:%M:%S")))
            cursor.connection.commit()
        except Exception as e:
            logger.error(f"PostgreSQL save failed: {str(e)}")
            cursor.connection.rollback()
            return False

    # Save to MongoDB
    try:
        client = connect_mongodb()
        db = client[DB_NAME]
        kb_prefix = KB_PREFIX or 'kb_'
        for key in RESOURCES:
            if RESOURCES[key] != "mongodb":
                continue
            collection = db[f'{kb_prefix}{key}']
            data = kb_validated.get(key, {})
            if key == "synonyms":
                for term, aliases in data.items():
                    collection.update_one(
                        {'term': term.lower()},
                        {'$set': {'term': term.lower(), 'aliases': aliases}},
                        upsert=True
                    )
            elif key == "clinical_pathways":
                for category, paths in data.items():
                    if category not in REQUIRED_CATEGORIES:
                        logger.warning(f"Invalid clinical pathway category '{category}', skipping")
                        continue
                    formatted_paths = {}
                    for pkey, path in paths.items():
                        formatted_path = path.copy()
                        if 'metadata' in formatted_path and 'last_updated' in formatted_path['metadata']:
                            formatted_path['metadata']['last_updated'] = parse_date(formatted_path['metadata']['last_updated']).strftime("%Y-%m-%d %H:%M:%S")
                        validated_path = validate_clinical_path(formatted_path, category, pkey)
                        if validated_path:
                            formatted_paths[pkey] = validated_path.dict()
                    collection.update_one(
                        {'category': category},
                        {'$set': {'category': category, 'paths': formatted_paths}},
                        upsert=True
                    )
            elif key == "diagnosis_relevance":
                for item in data:
                    collection.update_one(
                        {'diagnosis': item['diagnosis'].lower()},
                        {'$set': {
                            'diagnosis': item['diagnosis'].lower(),
                            'relevance': item['relevance'],
                            'category': item['category']
                        }},
                        upsert=True
                    )
            elif key == "diagnosis_treatments":
                for k, v in data.items():
                    collection.update_one(
                        {'key': k.lower()},
                        {'$set': {'key': k.lower(), 'value': DiagnosisTreatment(**v).dict()}},
                        upsert=True
                    )
            elif key == "versions":
                for k, v in data.items():
                    collection.update_one(
                        {'version': k},
                        {'$set': {'version': k, **Version(**v).dict()}},
                        upsert=True
                    )
            elif key == "history_diagnoses":
                for k, v in data.items():
                    collection.update_one(
                        {'key': k.lower()},
                        {'$set': {'key': k.lower(), 'value': HistoryDiagnosis(**v).dict()}},
                        upsert=True
                    )
            elif key == "management_config":
                for k, v in data.items():
                    try:
                        collection.update_one(
                            {'key': k.lower()},
                            {'$set': {'key': k.lower(), 'value': ManagementConfig(**v).dict()}},
                            upsert=True
                        )
                    except ValidationError as e:
                        logger.error(f"Failed to save ManagementConfig for key '{k}': {e}. Using default.")
                        collection.update_one(
                            {'key': k.lower()},
                            {'$set': {'key': k.lower(), 'value': ManagementConfig().dict()}},
                            upsert=True
                        )
            logger.info(f"Saved {len(data)} entries for {key} to MongoDB")
        client.close()
    except PyMongoError as e:
        logger.error(f"MongoDB save failed: {str(e)}")
        return False

    _knowledge_base_cache = kb_validated
    logger.info("Knowledge base saved to PostgreSQL and MongoDB")
    return True

def invalidate_cache():
    """Invalidate the knowledge base cache."""
    global _knowledge_base_cache
    _knowledge_base_cache = None
    cache_file = os.path.join(CACHE_DIR, f"{KB_PREFIX}knowledge_base.json")
    if os.path.exists(cache_file):
        try:
            os.remove(cache_file)
            logger.info(f"Invalidated knowledge base cache file: {cache_file}")
        except Exception as e:
            logger.error(f"Failed to invalidate cache file: {e}")
    logger.info("Knowledge base cache invalidated")