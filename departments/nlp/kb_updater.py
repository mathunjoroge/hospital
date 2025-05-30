import time
from typing import List, Dict, Optional
from enum import Enum
from pathlib import Path
from datetime import datetime
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, PyMongoError
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel, Field, ValidationError
from departments.nlp.logging_setup import get_logger
from departments.nlp.knowledge_base_io import load_knowledge_base, save_knowledge_base
from departments.nlp.nlp_utils import embed_text
from departments.nlp.models.transformer_model import model, tokenizer
import torch
from concurrent.futures import ThreadPoolExecutor
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
from departments.nlp.config import (
    MONGO_URI,
    DB_NAME,
    KB_PREFIX,
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_DB,
    POSTGRES_USER,
    POSTGRES_PASSWORD
)

logger = get_logger()

# Enums and Pydantic models
class Category(str, Enum):
    MUSCULOSKELETAL = "musculoskeletal"
    RESPIRATORY = "respiratory"
    GASTROINTESTINAL = "gastrointestinal"
    CARDIOVASCULAR = "cardiovascular"
    NEUROLOGICAL = "neurological"
    DERMATOLOGICAL = "dermatological"
    SENSORY = "sensory"
    HEMATOLOGIC = "hematologic"
    ENDOCRINE = "endocrine"
    GENITOURINARY = "genitourinary"
    PSYCHIATRIC = "psychiatric"
    GENERAL = "general"
    INFECTIOUS = "infectious"

class SemanticType(str, Enum):
    SIGN_OR_SYMPTOM = "Sign or Symptom"
    DISEASE_OR_SYNDROME = "Disease or Syndrome"
    FINDING = "Finding"
    UNKNOWN = "Unknown"

class UmlsMetadata(BaseModel):
    cui: Optional[str] = None
    semantic_type: SemanticType = SemanticType.UNKNOWN
    icd10: Optional[str] = None

class SymptomData(BaseModel):
    term: str
    category: Category = Category.GENERAL
    synonyms: List[str] = Field(default_factory=list)
    umls_metadata: UmlsMetadata = Field(default_factory=UmlsMetadata)
    description: Optional[str] = None

class ClinicalPath(BaseModel):
    differentials: List[str] = Field(default_factory=lambda: ["Undetermined"])
    contextual_triggers: List[str] = Field(default_factory=list)
    required_symptoms: List[str] = Field(default_factory=list)
    exclusion_criteria: List[str] = Field(default_factory=list)
    workup: Dict = Field(default_factory=lambda: {"urgent": [], "routine": []})
    management: Dict = Field(default_factory=lambda: {"symptoms": [], "definitive": [], "lifestyle": []})
    follow_up: List[str] = Field(default_factory=lambda: ["Follow-up in 1-2 weeks"])
    references: List[str] = Field(default_factory=lambda: ["Clinical guidelines pending"])
    metadata: Dict = Field(default_factory=lambda: {"source": ["Automated update"], "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

# Initialize PostgreSQL connection pool
try:
    pool = SimpleConnectionPool(
        minconn=1,
        maxconn=10,
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        cursor_factory=RealDictCursor
    )
    logger.info("Initialized PostgreSQL connection pool")
except Exception as e:
    logger.error(f"Failed to initialize PostgreSQL connection pool: {e}")
    pool = None

def get_postgres_connection():
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

class KnowledgeBaseUpdater:
    def __init__(self, mongo_uri: str = None, db_name: str = None, kb_prefix: str = 'kb_', model=None, tokenizer=None):
        self.mongo_uri = mongo_uri or MONGO_URI or 'mongodb://localhost:27017'
        self.db_name = db_name or DB_NAME or 'clinical_db'
        self.kb_prefix = kb_prefix
        self.model = model or globals().get('model')
        self.tokenizer = tokenizer or globals().get('tokenizer')
        self.knowledge_base = load_knowledge_base()
        self.version = self.knowledge_base.get('version', '1.1.0')

        # Initialize MongoDB with indexes
        try:
            self.client = self._connect_mongodb()
            self.db = self.client[self.db_name]
            self.medical_terms_collection = self.db[f'{self.kb_prefix}medical_terms']
            self.synonyms_collection = self.db[f'{self.kb_prefix}synonyms']
            self.pathways_collection = self.db[f'{self.kb_prefix}clinical_pathways']
            self.symptoms_collection = self.db[f'{self.kb_prefix}symptoms']
            self.umls_cache = self.db['umls_cache']
            self.versions_collection = self.db[f'{self.kb_prefix}versions']
            self._create_indexes()
            logger.info("Connected to MongoDB for knowledge base updates.")
        except ConnectionFailure as e:
            logger.error(f"MongoDB connection failed: {e}. Falling back to JSON.")
            self.client = None
            self.db = None
            self.medical_terms_collection = None
            self.synonyms_collection = None
            self.pathways_collection = None
            self.symptoms_collection = None
            self.umls_cache = None
            self.versions_collection = None
        except PyMongoError as e:
            logger.error(f"Error initializing MongoDB: {e}")
            raise RuntimeError(f"Failed to initialize MongoDB: {e}") from e

        # Initialize HTTP session (for potential future UTS API use)
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount('https://', HTTPAdapter(max_retries=retries))

        # Cache category embeddings
        self.category_embeddings = self._load_category_embeddings()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _connect_mongodb(self) -> MongoClient:
        """Connect to MongoDB with retry logic."""
        try:
            client = MongoClient(self.mongo_uri)
            client.admin.command('ping')
            return client
        except ConnectionFailure as e:
            logger.error(f"MongoDB connection attempt failed: {e}")
            raise

    def _create_indexes(self):
        """Create MongoDB indexes for performance."""
        try:
            self.umls_cache.create_index('symptom', unique=True)
            self.symptoms_collection.create_index([('category', 1), ('symptom', 1)], unique=True)
            self.medical_terms_collection.create_index('term', unique=True)
            self.synonyms_collection.create_index('term', unique=True)
            self.pathways_collection.create_index([('category', 1), ('key', 1)], unique=True)
            self.versions_collection.create_index('version', unique=True)
            logger.debug("MongoDB indexes created.")
        except PyMongoError as e:
            logger.error(f"Failed to create MongoDB indexes: {e}")

    def _load_category_embeddings(self) -> Dict[str, torch.Tensor]:
        """Load or compute category embeddings."""
        categories = [c.value for c in Category]
        embeddings = {}
        cache_file = Path('category_embeddings.json')
        if cache_file.exists():
            try:
                with cache_file.open('r') as f:
                    cached = json.load(f)
                embeddings = {k: torch.tensor(v) for k, v in cached.items()}
                logger.debug("Loaded category embeddings from cache.")
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to load category embeddings cache: {e}")
        if not embeddings:
            for cat in categories:
                embeddings[cat] = embed_text(cat)
            try:
                cache_file.parent.mkdir(exist_ok=True)
                with cache_file.open('w') as f:
                    json.dump({k: v.tolist() for k, v in embeddings.items()}, f)
                logger.debug("Saved category embeddings to cache.")
            except OSError as e:
                logger.warning(f"Failed to save category embeddings: {e}")
        return embeddings

    def search_local_umls_cui(self, term: str, max_attempts=3) -> Optional[Dict]:
        """Query local PostgreSQL UMLS database for CUI and semantic type."""
        cleaned_term = term.strip().lower()
        logger.debug(f"Searching PostgreSQL for CUI of '{cleaned_term}'")

        conn = get_postgres_connection()
        if not conn:
            logger.error(f"No PostgreSQL connection for '{cleaned_term}'")
            return None

        try:
            cursor = conn.cursor()
            for attempt in range(max_attempts):
                try:
                    # Exact match
                    start_time = time.time()
                    query = """
                        SELECT DISTINCT c.CUI
                        FROM umls.MRCONSO c
                        WHERE c.SAB = 'SNOMEDCT_US'
                        AND LOWER(c.STR) = %s
                        AND c.SUPPRESS = 'N'
                        LIMIT 1
                    """
                    cursor.execute(query, (cleaned_term,))
                    result = cursor.fetchone()
                    logger.debug(f"Exact match result: {result}, took {time.time() - start_time:.3f} seconds")
                    cui = result['cui'] if result else None

                    if not cui:
                        # Full-text search
                        start_time = time.time()
                        tsquery = ' & '.join(cleaned_term.split()) + ':*'
                        query = """
                            SELECT DISTINCT c.CUI
                            FROM umls.MRCONSO c
                            WHERE c.SAB = 'SNOMEDCT_US'
                            AND to_tsvector('english', c.STR) @@ to_tsquery('english', %s)
                            AND c.SUPPRESS = 'N'
                            LIMIT 1
                        """
                        cursor.execute(query, (tsquery,))
                        result = cursor.fetchone()
                        logger.debug(f"Full-text search result: {result}, took {time.time() - start_time:.3f} seconds")
                        cui = result['cui'] if result else None

                    if cui:
                        # Fetch semantic type
                        start_time = time.time()
                        query = """
                            SELECT DISTINCT sty.TUI, sty.STY
                            FROM umls.MRSTY sty
                            WHERE sty.CUI = %s
                            LIMIT 1
                        """
                        cursor.execute(query, (cui,))
                        result = cursor.fetchone()
                        logger.debug(f"Semantic type query for CUI {cui} took {time.time() - start_time:.3f} seconds")
                        semantic_type = result['sty'] if result and result['sty'] else SemanticType.UNKNOWN.value
                        return {'cui': cui, 'semantic_type': semantic_type}

                    logger.warning(f"No CUI found for term '{cleaned_term}'")
                    return None

                except Exception as e:
                    logger.error(f"Attempt {attempt + 1}/{max_attempts} failed for term '{cleaned_term}': {e}")
                    if attempt == max_attempts - 1:
                        logger.error(f"Max retries reached for term '{cleaned_term}'")
                        return None
                    continue
        finally:
            if cursor:
                cursor.close()
            if conn:
                pool.putconn(conn)
            logger.debug(f"Returned connection for '{cleaned_term}' to pool")

    def update_symptom(self, symptom: str, category: Category = Category.GENERAL, description: Optional[str] = None):
        """Update or add a symptom with UMLS metadata."""
        try:
            symptom_data = SymptomData(
                term=symptom,
                category=category,
                description=description or f"UMLS-derived: {symptom}"
            )

            # Use PostgreSQL for CUI mapping
            umls_data = self.search_local_umls_cui(symptom)
            if umls_data:
                symptom_data.umls_metadata = UmlsMetadata(
                    cui=umls_data['cui'],
                    semantic_type=umls_data['semantic_type']
                )
            else:
                logger.warning(f"No CUI found for '{symptom}' in PostgreSQL")

            # Update MongoDB
            if self.symptoms_collection:
                with self.client.start_session() as session:
                    with session.start_transaction():
                        self.symptoms_collection.update_one(
                            {'category': category.value, 'symptom': symptom},
                            {
                                '$set': {
                                    'description': symptom_data.description,
                                    'umls_cui': symptom_data.umls_metadata.cui,
                                    'semantic_type': symptom_data.umls_metadata.semantic_type
                                }
                            },
                            upsert=True,
                            session=session
                        )
                        self.versions_collection.update_one(
                            {'version': self.version},
                            {
                                '$set': {
                                    'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    'updated_collections': {'symptoms': True}
                                }
                            },
                            upsert=True,
                            session=session
                        )
                logger.info(f"Updated symptom '{symptom}' in MongoDB")

            # Update knowledge base
            self.knowledge_base['symptoms'].setdefault(category.value, {})[symptom] = {
                'description': symptom_data.description,
                'umls_cui': symptom_data.umls_metadata.cui,
                'semantic_type': symptom_data.umls_metadata.semantic_type
            }
            save_knowledge_base(self.knowledge_base)
            logger.debug(f"Updated knowledge base with symptom '{symptom}'")

        except (ValidationError, PyMongoError) as e:
            logger.error(f"Failed to update symptom '{symptom}': {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            self.client.close()
        self.session.close()
        if pool:
            pool.closeall()