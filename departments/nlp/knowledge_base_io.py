# knowledge_base_io.py
from datetime import datetime
import os
from typing import Dict, Optional, Set, List
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, PyMongoError
from psycopg2.extras import RealDictCursor, execute_batch
from psycopg2.pool import SimpleConnectionPool
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel, Field, ValidationError
from departments.nlp.logging_setup import get_logger
from departments.nlp.knowledge_base_init import initialize_knowledge_files
from departments.nlp.config import (
    MONGO_URI, DB_NAME, POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
)
from departments.nlp.nlp_common import FALLBACK_CUI_MAP  # Import from nlp_common.py

logger = get_logger(__name__)

# Singleton cache
_knowledge_base_cache: Optional[Dict] = None

# Pydantic models
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
    references: List[str] = Field(default_factory=list)
    metadata: Dict = Field(default_factory=lambda: {
        "source": "Unknown",
        "last_updated": datetime.now().strftime("%Y-%m-%d")
    })
    follow_up: List[str] = Field(default_factory=lambda: ["Follow-up in 2 weeks"])

class KnowledgeBase(BaseModel):
    version: str = "1.1.0"
    last_updated: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    source: str = "PostgreSQL/MongoDB"
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
    "symptoms": "postgresql",
    "medical_stop_words": "postgresql",
    "medical_terms": "postgresql",
    "synonyms": "mongodb",
    "clinical_pathways": "mongodb",
    "history_diagnoses": "mongodb",
    "diagnosis_relevance": "mongodb",
    "management_config": "mongodb"
}
REQUIRED_CATEGORIES = {
    'respiratory', 'neurological', 'cardiovascular', 'gastrointestinal',
    'musculoskeletal', 'infectious', 'hepatic'
}
HIGH_RISK_CONDITIONS = {
    'pulmonary embolism', 'myocardial infarction', 'meningitis', 'malaria', 'dengue'
}
STRICT_VALIDATION = os.getenv('STRICT_KB_VALIDATION', 'false').lower() == 'true'

# PostgreSQL connection pool
try:
    pool = SimpleConnectionPool(
        minconn=1, maxconn=10, host=POSTGRES_HOST, port=POSTGRES_PORT,
        dbname=POSTGRES_DB, user=POSTGRES_USER, password=POSTGRES_PASSWORD,
        cursor_factory=RealDictCursor
    )
    logger.info("Initialized PostgreSQL connection pool")
except Exception as e:
    logger.error(f"Failed to initialize database connection pool: {e}")
    pool = None

def get_postgres_connection():
    """Get a PostgreSQL connection from the pool."""
    if pool is None:
        logger.error("Connection pool not initialized")
        return None
    try:
        conn = pool.getconn()
        logger.debug("Connected to PostgreSQL database from pool")
        return conn
    except Exception as e:
        logger.error(f"Failed to get connection from pool: {e}")
        return None

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

def load_from_postgresql() -> Dict[str, any]:
    """Load knowledge base from PostgreSQL."""
    knowledge = KnowledgeBase().dict()
    conn = get_postgres_connection()
    if not conn:
        logger.error("Failed to connect to PostgreSQL")
        return knowledge
    try:
        cursor = conn.cursor()
        # Load medical_stop_words
        cursor.execute("SELECT word FROM medical_stop_words")
        knowledge['medical_stop_words'] = {row['word'].lower() for row in cursor.fetchall()}
        logger.info(f"Loaded {len(knowledge['medical_stop_words'])} medical_stop_words from PostgreSQL")

        # Load medical_terms
        cursor.execute("SELECT term, category, umls_cui, semantic_type FROM medical_terms")
        knowledge['medical_terms'] = [
            MedicalTerm(
                term=row['term'],
                category=row['category'] or "unknown",
                umls_cui=row['umls_cui'],
                semantic_type=row['semantic_type'] or "Unknown"
            ).dict() for row in cursor.fetchall()
        ]
        logger.info(f"Loaded {len(knowledge['medical_terms'])} medical_terms from PostgreSQL")

        # Load symptoms
        cursor.execute("SELECT symptom, category, description, umls_cui, semantic_type FROM symptoms")
        valid_data = {}
        for row in cursor.fetchall():
            category = row['category'] or "general"
            symptom = row['symptom']
            if not symptom:
                logger.warning(f"Invalid symptom: {row}")
                continue
            if category not in valid_data:
                valid_data[category] = {}
            try:
                # Apply fallback if no UMLS CUI
                umls_cui = row['umls_cui']
                semantic_type = row['semantic_type'] or "Unknown"
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
            knowledge['last_updated'] = metadata['last_updated'].strftime("%Y-%m-%d %H:%M:%S") if metadata['last_updated'] else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        logger.error(f"Failed to load from PostgreSQL: {str(e)}")
    finally:
        cursor.close()
        pool.putconn(conn)
    return knowledge

def load_from_mongodb() -> Dict[str, any]:
    """Load knowledge base from MongoDB."""
    knowledge = KnowledgeBase().dict()
    try:
        client = connect_mongodb()
        db = client[DB_NAME]
        for key in RESOURCES:
            if RESOURCES[key] != "mongodb":
                continue
            collection = db[key]
            data = list(collection.find())
            if not data:
                logger.warning(f"No data in MongoDB collection {key}")
                continue
            if key == "synonyms":
                knowledge[key] = {doc['term'].lower(): doc['aliases'] for doc in data if 'term' in doc and isinstance(doc.get('aliases'), list)}
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
            elif key == "diagnosis_relevance":
                knowledge[key] = {
                    doc['diagnosis'].lower(): {
                        'relevance': doc.get('relevance', []),
                        'category': doc.get('category', 'unknown')
                    } for doc in data if 'diagnosis' in doc
                }
            else:
                knowledge[key] = {doc['key'].lower(): doc['value'] for doc in data if 'key' in doc and 'value' in doc}
            logger.info(f"Loaded {len(data)} entries for {key} from MongoDB")
        client.close()
    except ConnectionFailure as e:
        logger.error(f"MongoDB connection failed: {e}")
    return knowledge

def cross_reference_umls(knowledge: Dict) -> Dict:
    """Cross-reference UMLS CUIs for synonyms."""
    conn = get_postgres_connection()
    if not conn:
        logger.error("Failed to connect to PostgreSQL for UMLS cross-referencing")
        return knowledge
    try:
        cursor = conn.cursor()
        for term in knowledge['medical_terms']:
            term_lower = term['term'].lower()
            if term_lower in knowledge['synonyms'] and term.get('umls_cui'):
                for synonym in knowledge['synonyms'][term_lower]:
                    synonym_lower = synonym.lower()
                    if not any(t['term'].lower() == synonym_lower and t.get('umls_cui') for t in knowledge['medical_terms']):
                        # Check fallback dictionary
                        cui = FALLBACK_CUI_MAP.get(synonym_lower, {}).get('umls_cui')
                        semantic_type = FALLBACK_CUI_MAP.get(synonym_lower, {}).get('semantic_type', term['semantic_type'])
                        if not cui:
                            # Verify synonym in UMLS
                            cursor.execute("""
                                SELECT CUI FROM umls.MRCONSO
                                WHERE LOWER(STR) = %s AND SAB = 'SNOMEDCT_US' AND SUPPRESS = 'N'
                            """, (synonym_lower,))
                            result = cursor.fetchone()
                            cui = result['CUI'] if result else term['umls_cui']
                        knowledge['medical_terms'].append({
                            'term': synonym,
                            'category': term['category'],
                            'umls_cui': cui,
                            'semantic_type': semantic_type
                        })
                        logger.debug(f"Added UMLS-linked synonym '{synonym}' for '{term_lower}' with CUI {cui}")
    except Exception as e:
        logger.error(f"Failed to cross-reference UMLS: {str(e)}")
    finally:
        cursor.close()
        pool.putconn(conn)
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
    knowledge.update({k: v for k, v in mongo_data.items() if v})  # Merge MongoDB data
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
        kb_validated = kb

    # Save to PostgreSQL
    conn = get_postgres_connection()
    if not conn:
        logger.error("Failed to connect to PostgreSQL")
        return False
    try:
        cursor = conn.cursor()
        # Save medical_stop_words
        cursor.execute("DELETE FROM medical_stop_words")
        execute_batch(cursor, "INSERT INTO medical_stop_words (word) VALUES (%s)",
                      [(word,) for word in kb_validated.get('medical_stop_words', [])])
        logger.info(f"Saved {len(kb_validated.get('medical_stop_words', []))} medical_stop_words to PostgreSQL")

        # Save medical_terms
        cursor.execute("DELETE FROM medical_terms")
        execute_batch(cursor, """
            INSERT INTO medical_terms (term, category, umls_cui, semantic_type)
            VALUES (%s, %s, %s, %s)
        """, [(t['term'], t['category'], t['umls_cui'], t['semantic_type'])
              for t in kb_validated.get('medical_terms', [])])
        logger.info(f"Saved {len(kb_validated.get('medical_terms', []))} medical_terms to PostgreSQL")

        # Save symptoms
        cursor.execute("DELETE FROM symptoms")
        symptom_data = []
        for cat, symptoms in kb_validated.get('symptoms', {}).items():
            for s, info in symptoms.items():
                # Apply fallback if no UMLS CUI
                umls_cui = info.get('umls_cui')
                semantic_type = info.get('semantic_type', 'Unknown')
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
        logger.info(f"Saved {len(symptom_data)} symptoms to PostgreSQL")

        # Update metadata
        cursor.execute("""
            INSERT INTO knowledge_base_metadata (key, version, last_updated)
            VALUES (%s, %s, %s) ON CONFLICT (key) DO UPDATE
            SET version = EXCLUDED.version, last_updated = EXCLUDED.last_updated
        """, ('knowledge_base', kb_validated.get('version', '1.1.0'), kb_validated.get('last_updated', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))))
        conn.commit()
    except Exception as e:
        logger.error(f"PostgreSQL save failed: {str(e)}")
        conn.rollback()
        return False
    finally:
        cursor.close()
        pool.putconn(conn)

    # Save to MongoDB
    try:
        client = connect_mongodb()
        db = client[DB_NAME]
        for key in RESOURCES:
            if RESOURCES[key] != "mongodb":
                continue
            collection = db[key]
            collection.drop()
            data = kb_validated.get(key, {})
            if key == "synonyms":
                collection.insert_many([{'term': term.lower(), 'aliases': aliases} for term, aliases in data.items()])
            elif key == "clinical_pathways":
                collection.insert_many([{'category': category, 'paths': paths} for category, paths in data.items()])
            elif key == "diagnosis_relevance":
                collection.insert_many([
                    {'diagnosis': diagnosis.lower(), 'relevance': info['relevance'], 'category': info['category']}
                    for diagnosis, info in data.items()
                ])
            else:
                collection.insert_many([{'key': k.lower(), 'value': v} for k, v in data.items()])
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
    logger.info("Knowledge base cache invalidated")