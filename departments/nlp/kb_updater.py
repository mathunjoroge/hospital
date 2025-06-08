from datetime import datetime
from typing import Dict, Optional, List
from enum import Enum
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, PyMongoError, DuplicateKeyError
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
from tenacity import retry, stop_after_attempt
from pydantic import BaseModel, Field, ValidationError
from departments.nlp.logging_setup import get_logger
from departments.nlp.knowledge_base_io import load_knowledge_base, save_knowledge_base
from departments.nlp.nlp_utils import embed_text
from departments.nlp.nlp_common import FALLBACK_CUI_MAP  # Import from nlp_common.py
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from departments.nlp.config import (
    MONGO_URI, DB_NAME, POSTGRES_HOST, POSTGRES_PORT,
    POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
)

logger = get_logger(__name__)

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
    HEPATIC = "hepatic"

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
    category: Category
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
    metadata: Dict = Field(default_factory=lambda: {
        "source": ["Automated update"],
        "last_updated": datetime.now()  # Store as datetime object
    })

# Initialize PostgreSQL connection pool
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
    if pool is None:
        logger.error("Connection pool not initialized")
        return None
    try:
        conn = pool.getconn()
        logger.debug("Connected to PostgreSQL database from pool")
        return conn
    except Exception as e:
        logger.error(f"Failed to get connection from pool: {str(e)}")
        return None

class KnowledgeBaseUpdater:
    def __init__(self, mongo_uri: str = None, db_name: str = None, kb_prefix: str = 'kb_'):
        self.mongo_uri = mongo_uri or MONGO_URI or 'mongodb://localhost:27017'
        self.db_name = db_name or DB_NAME or 'clinical_db'
        self.kb_prefix = kb_prefix
        self.knowledge_base = load_knowledge_base()
        self.version = self.knowledge_base.get('version', '1.0.0')

        # Initialize transformer model
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
            self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
            logger.info("Initialized T5 model for synonym generation")
        except Exception as e:
            logger.warning(f"Failed to initialize T5 model: {e}")
            self.model = None
            self.tokenizer = None

        try:
            self.client = self._connect_mongodb()
            self.db = self.client[self.db_name]
            self.synonyms_collection = self.db[f'{self.kb_prefix}synonyms']
            self.pathways_collection = self.db[f'{self.kb_prefix}clinical_paths']
            self.diagnosis_relevance_collection = self.db[f'{self.kb_prefix}diagnosis_relevance']
            self.history_diagnoses_collection = self.db[f'{self.kb_prefix}history_diagnoses']
            self.management_config_collection = self.db[f'{self.kb_prefix}management_config']
            self.versions_collection = self.db[f'{self.kb_prefix}versions']
            self._create_indexes()
            logger.info("Connected to MongoDB for knowledge base updates")
        except ConnectionFailure as e:
            logger.error(f"MongoDB connection failed: {str(e)}")
            self.client = None
            self.db = None
            self.synonyms_collection = None
            self.pathways_collection = None
            self.diagnosis_relevance_collection = None
            self.history_diagnoses_collection = None
            self.management_config_collection = None
            self.versions_collection = None
            raise RuntimeError(f"Failed to initialize MongoDB: {str(e)}") from e
        except PyMongoError as e:
            logger.error(f"Error initializing MongoDB: {str(e)}")
            raise RuntimeError(f"Failed to initialize MongoDB: {str(e)}") from e

        self.category_embeddings = self._load_category_embeddings()

    @retry(stop=stop_after_attempt(3))
    def _connect_mongodb(self) -> MongoClient:
        try:
            client = MongoClient(self.mongo_uri)
            client.admin.command('ping')
            return client
        except ConnectionFailure as e:
            logger.error(f"MongoDB connection attempt failed: {str(e)}")
            raise

    def _create_indexes(self):
        try:
            index_name = 'category_1_key_1'
            existing_indexes = self.pathways_collection.index_information()
            if index_name in existing_indexes and not existing_indexes[index_name].get('unique'):
                logger.warning(f"Index {index_name} is not unique, dropping and recreating...")
                self.pathways_collection.drop_index(index_name)
                self.pathways_collection.create_index(
                    [('category', 1), ('key', 1)],
                    unique=True,
                    name=index_name
                )
                logger.info(f"Recreated unique index '{index_name}'")
            elif index_name not in existing_indexes:
                self.pathways_collection.create_index(
                    [('category', 1), ('key', 1)],
                    unique=True,
                    name=index_name
                )
                logger.info(f"Created unique index '{index_name}'")

            diagnosis_index_name = 'diagnosis_1'
            existing_indexes = self.diagnosis_relevance_collection.index_information()
            if diagnosis_index_name not in existing_indexes:
                null_count = self.diagnosis_relevance_collection.count_documents({'diagnosis': None})
                if null_count > 1:
                    logger.warning(f"Found {null_count} documents with null diagnosis. Keeping one, removing others.")
                    null_doc = self.diagnosis_relevance_collection.find_one({'diagnosis': None})
                    if null_doc:
                        self.diagnosis_relevance_collection.delete_many({
                            'diagnosis': None,
                            '_id': {'$ne': null_doc['_id']}
                        })
                        logger.info(f"Removed {null_count - 1} duplicate null diagnosis documents")
                    else:
                        self.diagnosis_relevance_collection.delete_many({'diagnosis': None})
                        logger.info(f"Removed all {null_count} null diagnosis documents")
                self.diagnosis_relevance_collection.create_index(
                    'diagnosis',
                    unique=True,
                    name=diagnosis_index_name
                )
                logger.info(f"Created unique index '{diagnosis_index_name}'")
            elif not existing_indexes[diagnosis_index_name].get('unique'):
                logger.warning(f"Index {diagnosis_index_name} is not unique, dropping and recreating...")
                self.diagnosis_relevance_collection.drop_index(diagnosis_index_name)
                null_count = self.diagnosis_relevance_collection.count_documents({'diagnosis': None})
                if null_count > 1:
                    logger.warning(f"Found {null_count} documents with null diagnosis. Keeping one, removing others.")
                    null_doc = self.diagnosis_relevance_collection.find_one({'diagnosis': None})
                    if null_doc:
                        self.diagnosis_relevance_collection.delete_many({
                            'diagnosis': None,
                            '_id': {'$ne': null_doc['_id']}
                        })
                        logger.info(f"Removed {null_count - 1} duplicate null diagnosis documents")
                self.diagnosis_relevance_collection.create_index(
                    'diagnosis',
                    unique=True,
                    name=diagnosis_index_name
                )
                logger.info(f"Recreated unique index '{diagnosis_index_name}'")

            history_index_name = 'key_1'
            existing_indexes = self.history_diagnoses_collection.index_information()
            if history_index_name not in existing_indexes:
                null_count = self.history_diagnoses_collection.count_documents({'key': None})
                if null_count > 1:
                    logger.warning(f"Found {null_count} documents with null key. Keeping one, removing others.")
                    null_doc = self.history_diagnoses_collection.find_one({'key': None})
                    if null_doc:
                        self.history_diagnoses_collection.delete_many({
                            'key': None,
                            '_id': {'$ne': null_doc['_id']}
                        })
                        logger.info(f"Removed {null_count - 1} duplicate null key documents")
                    else:
                        self.history_diagnoses_collection.delete_many({'key': None})
                        logger.info(f"Removed all {null_count} null key documents")
                self.history_diagnoses_collection.create_index(
                    'key',
                    unique=True,
                    name=history_index_name
                )
                logger.info(f"Created unique index '{history_index_name}'")
            elif not existing_indexes[history_index_name].get('unique'):
                logger.warning(f"Index {history_index_name} is not unique, dropping and recreating...")
                self.history_diagnoses_collection.drop_index(history_index_name)
                null_count = self.history_diagnoses_collection.count_documents({'key': None})
                if null_count > 1:
                    logger.warning(f"Found {null_count} documents with null key. Keeping one, removing others.")
                    null_doc = self.history_diagnoses_collection.find_one({'key': None})
                    if null_doc:
                        self.history_diagnoses_collection.delete_many({
                            'key': None,
                            '_id': {'$ne': null_doc['_id']}
                        })
                        logger.info(f"Removed {null_count - 1} duplicate null key documents")
                self.history_diagnoses_collection.create_index(
                    'key',
                    unique=True,
                    name=history_index_name
                )
                logger.info(f"Recreated unique index '{history_index_name}'")

            self.synonyms_collection.create_index('term', unique=True)
            self.management_config_collection.create_index('key', unique=True)
            self.versions_collection.create_index('version', unique=True)
            logger.debug("MongoDB indexes created")
        except DuplicateKeyError as e:
            logger.error(f"Duplicate key error during index creation: {str(e)}")
            raise
        except PyMongoError as e:
            logger.error(f"Failed to create MongoDB indexes: {str(e)}")
            raise

    def _load_category_embeddings(self) -> Dict[str, torch.Tensor]:
        categories = [c.value for c in Category]
        embeddings = {}
        conn = get_postgres_connection()
        if not conn:
            logger.error("Cannot load category embeddings without PostgreSQL connection")
            return embeddings

        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS category_embeddings (
                    category TEXT PRIMARY KEY,
                    embedding FLOAT[] NOT NULL
                )
            """)
            conn.commit()

            cursor.execute("SELECT category, embedding FROM category_embeddings")
            for row in cursor.fetchall():
                embeddings[row['category']] = torch.tensor(row['embedding'])
                logger.debug(f"Loaded embedding for category '{row['category']}'")

            missing_cats = set(categories) - set(embeddings.keys())
            if missing_cats:
                for cat in missing_cats:
                    embeddings[cat] = embed_text(cat)
                    cursor.execute("""
                        INSERT INTO category_embeddings (category, embedding)
                        VALUES (%s, %s)
                        ON CONFLICT (category) DO UPDATE SET embedding = EXCLUDED.embedding
                    """, (cat, embeddings[cat].tolist()))
                conn.commit()
                logger.debug(f"Computed and saved embeddings for {len(missing_cats)} categories")
        except Exception as e:
            logger.error(f"Failed to load/save category embeddings: {str(e)}")
            conn.rollback()
        finally:
            cursor.close()
            pool.putconn(conn)
        return embeddings

    def search_local_umls_cui(self, term: str, max_attempts: int = 3) -> Optional[Dict]:
        """Search for UMLS CUI in the PostgreSQL database with fallback."""
        cleaned_term = term.strip().lower()
        logger.debug(f"Searching UMLS for term: {cleaned_term}")

        # Check fallback dictionary
        if cleaned_term in FALLBACK_CUI_MAP:
            logger.debug(f"Fallback hit for '{cleaned_term}': {FALLBACK_CUI_MAP[cleaned_term]}")
            return {
                'cui': FALLBACK_CUI_MAP[cleaned_term]['umls_cui'],
                'semantic_type': FALLBACK_CUI_MAP[cleaned_term]['semantic_type']
            }

        conn = get_postgres_connection()
        if not conn:
            logger.error(f"No PostgreSQL connection for '{cleaned_term}'")
            return None

        try:
            cursor = conn.cursor()
            for attempt in range(max_attempts):
                try:
                    # Check cache
                    cursor.execute("""
                        SELECT cui, semantic_type FROM public.umls_cache 
                        WHERE term = %s
                    """, (cleaned_term,))
                    result = cursor.fetchone()
                    if result:
                        logger.debug(f"Cache hit for '{cleaned_term}'")
                        return {
                            'cui': result['cui'],
                            'semantic_type': result.get('semantic_type', SemanticType.UNKNOWN.value)
                        }

                    # Exact match query
                    query = """
                        SELECT DISTINCT c.CUI 
                        FROM umls.MRCONSO c
                        WHERE c.SAB = 'SNOMEDCT_US'
                        AND LOWER(c.STR) = %s
                        LIMIT 1
                    """
                    cursor.execute(query, (cleaned_term,))
                    result = cursor.fetchone()
                    cui = result['CUI'] if result else None

                    if not cui:
                        # Fallback to LIKE search
                        query = """
                            SELECT DISTINCT c.CUI 
                            FROM umls.MRCONSO c
                            WHERE c.SAB = 'SNOMEDCT_US'
                            AND LOWER(c.STR) LIKE %s
                            LIMIT 1
                        """
                        cursor.execute(query, ('%' + cleaned_term + '%',))
                        result = cursor.fetchone()
                        cui = result['CUI'] if result else None

                    if cui:
                        # Get semantic type
                        query = """
                            SELECT DISTINCT sty.STY 
                            FROM umls.MRSTY sty 
                            WHERE sty.CUI = %s
                            LIMIT 1
                        """
                        cursor.execute(query, (cui,))
                        result = cursor.fetchone()
                        semantic_type = result['STY'] if result else SemanticType.UNKNOWN.value

                        # Cache result
                        cursor.execute("""
                            INSERT INTO public.umls_cache (term, cui, semantic_type) 
                            VALUES (%s, %s, %s)
                            ON CONFLICT (term) 
                            DO UPDATE SET cui = EXCLUDED.cui, semantic_type = EXCLUDED.semantic_type
                        """, (cleaned_term, cui, semantic_type))
                        conn.commit()
                        logger.debug(f"Cached UMLS for '{cleaned_term}': CUI={cui}, SemanticType={semantic_type}")
                        return {'cui': cui, 'semantic_type': semantic_type}

                    logger.warning(f"No CUI found for term '{cleaned_term}'")
                    return None

                except Exception as e:
                    logger.error(f"Attempt {attempt + 1}/{max_attempts} failed for '{cleaned_term}': {str(e)}")
                    if attempt == max_attempts - 1:
                        return None
                    continue
        except Exception as e:
            logger.error(f"Failed to execute query for '{cleaned_term}': {str(e)}")
            return None
        finally:
            cursor.close()
            pool.putconn(conn)

    def update_symptom(self, symptom: str, category: Category = Category.GENERAL, description: Optional[str] = None, synonyms: List[str] = None):
        """Update symptom in the knowledge base."""
        try:
            symptom_data = SymptomData(
                term=symptom,
                category=category,
                description=description or f"Description for {symptom}",
                synonyms=synonyms or []
            )

            # Fetch UMLS or use fallback
            umls_data = self.search_local_umls_cui(symptom)
            if umls_data:
                symptom_data.umls_metadata = UmlsMetadata(
                    cui=umls_data['cui'],
                    semantic_type=umls_data['semantic_type']
                )

            conn = get_postgres_connection()
            if not conn:
                logger.error("Cannot update symptom without PostgreSQL connection")
                return

            try:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO symptoms (symptom, category, description, umls_cui, semantic_type)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (symptom) DO UPDATE
                    SET category = EXCLUDED.category,
                        description = EXCLUDED.description,
                        umls_cui = EXCLUDED.umls_cui,
                        semantic_type = EXCLUDED.semantic_type
                """, (
                    symptom_data.term.lower(),
                    symptom_data.category.value,
                    symptom_data.description,
                    symptom_data.umls_metadata.cui,
                    symptom_data.umls_metadata.semantic_type.value
                ))

                cursor.execute("""
                    INSERT INTO knowledge_base_metadata (key, version, last_updated)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (key) DO UPDATE
                    SET version = EXCLUDED.version, last_updated = EXCLUDED.last_updated
                """, ('knowledge_base', self.version, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                conn.commit()
                logger.info(f"Updated symptom '{symptom}' in PostgreSQL")
            except Exception as e:
                logger.error(f"Failed to update symptom '{symptom}': {str(e)}")
                conn.rollback()
            finally:
                cursor.close()
                pool.putconn(conn)

            if self.synonyms_collection and symptom_data.synonyms:
                try:
                    self.synonyms_collection.update_one(
                        {'term': symptom.lower()},
                        {'$set': {'aliases': symptom_data.synonyms}},
                        upsert=True
                    )
                    self.versions_collection.update_one(
                        {'version': self.version},
                        {
                            '$set': {
                                'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'updated_collections': {'synonyms': True}
                            }
                        },
                        upsert=True
                    )
                    logger.info(f"Updated synonyms for '{symptom}' in MongoDB")
                except PyMongoError as e:
                    logger.error(f"Failed to update synonyms for '{symptom}' in MongoDB: {str(e)}")

            self.knowledge_base['symptoms'].setdefault(category.value, {})[symptom.lower()] = {
                'description': symptom_data.description,
                'umls_cui': symptom_data.umls_metadata.cui,
                'semantic_type': symptom_data.umls_metadata.semantic_type.value
            }
            if symptom_data.synonyms:
                self.knowledge_base['synonyms'][symptom.lower()] = symptom_data.synonyms
            save_knowledge_base(self.knowledge_base)
            logger.debug(f"Updated knowledge base with symptom '{symptom}'")

        except ValidationError as e:
            logger.error(f"Validation error for symptom '{symptom}': {str(e)}")

    def is_new_symptom(self, symptom: str) -> bool:
        """Check if a symptom is new or not in the knowledge base."""
        try:
            symptom_lower = symptom.lower().strip()
            conn = get_postgres_connection()
            if not conn:
                logger.error("Cannot check symptom without PostgreSQL connection")
                return True

            try:
                cursor = conn.cursor()
                cursor.execute("SELECT 1 FROM symptoms WHERE symptom = %s", (symptom_lower,))
                exists = cursor.fetchone()
                logger.debug(f"Symptom '{symptom_lower}' {'exists' if exists else 'is new'} in knowledge base")
                return not exists
            except Exception as e:
                logger.error(f"Error checking symptom '{symptom_lower}': {str(e)}")
                return True
            finally:
                cursor.close()
                pool.putconn(conn)
        except Exception as e:
            logger.error(f"Failed to check symptom '{symptom}': {str(e)}")
            return True

    def generate_synonyms(self, symptom: str) -> List[str]:
        """Generate synonyms for a symptom."""
        symptom_lower = symptom.lower().strip()
        try:
            # Check MongoDB for existing synonyms
            if self.synonyms_collection:
                doc = self.synonyms_collection.find_one({'term': symptom_lower}, {'aliases': 1})
                if doc and 'aliases' in doc:
                    logger.debug(f"Found synonyms for '{symptom_lower}': {doc['aliases']}")
                    return doc['aliases']

            # Fallback to transformer model
            if self.model and self.tokenizer:
                input_text = f"Generate synonyms for the medical symptom: {symptom_lower}"
                inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                outputs = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=100,
                    num_return_sequences=5,
                    no_repeat_ngram_size=2,
                    early_stopping=True
                )
                synonyms = [
                    self.tokenizer.decode(output, skip_special_tokens=True).strip().lower()
                    for output in outputs
                ]
                synonyms = list(set(s for s in synonyms if s and s != symptom_lower))
                if synonyms and self.synonyms_collection:
                    self.synonyms_collection.update_one(
                        {'term': symptom_lower},
                        {'$set': {'aliases': synonyms}},
                        upsert=True
                    )
                    logger.debug(f"Saved synonyms for '{symptom_lower}' to MongoDB")
                logger.info(f"Generated synonyms for '{symptom_lower}': {synonyms}")
                return synonyms

            logger.warning(f"No synonym generation available for '{symptom_lower}' (no model)")
            return []
        except Exception as e:
            logger.error(f"Failed to generate synonyms for '{symptom}': {str(e)}")
            return []

    def update_knowledge_base(self, symptom: str, category: str, synonyms: List[str], note_text: str):
        """Update knowledge base with new symptom."""
        try:
            # Validate category
            try:
                category_enum = Category(category.lower())
            except ValueError:
                logger.warning(f"Invalid category '{category}' for symptom '{symptom}'. Using 'general'.")
                category_enum = Category.GENERAL

            # Prepare symptom data
            symptom_data = SymptomData(
                term=symptom,
                category=category_enum,
                synonyms=synonyms or [],
                description=f"Derived from note: {note_text[:50]}..." if note_text else f"Description for {symptom}"
            )

            # Fetch UMLS or use fallback
            umls_data = self.search_local_umls_cui(symptom)
            if umls_data:
                symptom_data.umls_metadata = UmlsMetadata(
                    cui=umls_data['cui'],
                    semantic_type=umls_data['semantic_type']
                )

            # Update PostgreSQL
            conn = get_postgres_connection()
            if not conn:
                logger.error("Cannot update symptom without PostgreSQL connection")
                return

            try:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO symptoms (symptom, category, description, umls_cui, semantic_type)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (symptom) DO UPDATE
                    SET category = EXCLUDED.category,
                        description = EXCLUDED.description,
                        umls_cui = EXCLUDED.umls_cui,
                        semantic_type = EXCLUDED.semantic_type
                """, (
                    symptom_data.term.lower(),
                    symptom_data.category.value,
                    symptom_data.description,
                    symptom_data.umls_metadata.cui,
                    symptom_data.umls_metadata.semantic_type.value
                ))

                cursor.execute("""
                    INSERT INTO knowledge_base_metadata (key, version, last_updated)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (key) DO UPDATE
                    SET version = EXCLUDED.version, last_updated = EXCLUDED.last_updated
                """, ('knowledge_base', self.version, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                conn.commit()
                logger.info(f"Updated symptom '{symptom}' in PostgreSQL")
            except Exception as e:
                logger.error(f"Failed to update symptom '{symptom}' in database: {str(e)}")
                conn.rollback()
            finally:
                cursor.close()
                pool.putconn(conn)

            # Update MongoDB synonyms
            if self.synonyms_collection and symptom_data.synonyms:
                try:
                    self.synonyms_collection.update_one(
                        {'term': symptom_data.term.lower()},
                        {'$set': {'aliases': symptom_data.synonyms}},
                        upsert=True
                    )
                    logger.info(f"Updated synonyms for '{symptom}' in MongoDB")
                except PyMongoError as e:
                    logger.error(f"Failed to update synonyms for '{symptom}' in MongoDB: {str(e)}")

            # Update in-memory knowledge base
            self.knowledge_base['symptoms'].setdefault(category_enum.value, {})[symptom.lower()] = {
                'description': symptom_data.description,
                'umls_cui': symptom_data.umls_metadata.cui,
                'semantic_type': symptom_data.umls_metadata.semantic_type.value
            }
            if symptom_data.synonyms:
                self.knowledge_base['synonyms'][symptom.lower()] = symptom_data.synonyms
            save_knowledge_base(self.knowledge_base)
            logger.info(f"Updated knowledge base with symptom '{symptom}'")
        except Exception as e:
            logger.error(f"Failed to update knowledge for '{symptom}': {str(e)}")

    def infer_category(self, symptom: str, context: str) -> str:
        """Infer category for a symptom based on context."""
        try:
            symptom_lower = symptom.lower().strip()
            context_lower = context.lower().strip()

            # Check PostgreSQL
            conn = get_postgres_connection()
            if conn:
                try:
                    cursor = conn.cursor()
                    cursor.execute("SELECT category FROM symptoms WHERE symptom = %s", (symptom_lower,))
                    result = cursor.fetchone()
                    if result:
                        logger.debug(f"Found category '{result['category']}' for symptom '{symptom_lower}'")
                        return result['category']
                finally:
                    cursor.close()
                    pool.putconn(conn)

            # Specific rules for symptoms
            if symptom_lower in ['headache', 'migraine', 'head pain', 'cephalalgia']:
                return Category.NEUROLOGICAL.value
            elif symptom_lower in ['fever', 'fevers', 'pyrexia', 'chills', 'shivering']:
                return Category.GENERAL.value
            elif symptom_lower in ['nausea', 'queasiness', 'vomiting', 'loss of appetite', 'anorexia', 'decreased appetite']:
                return Category.GASTROINTESTINAL.value
            elif symptom_lower in ['jaundice', 'jaundice in eyes', 'icterus'] or 'yellowing' in context_lower:
                return Category.HEPATIC.value
            elif symptom_lower == 'cough' or 'wheezing' in context_lower:
                return Category.RESPIRATORY.value
            elif 'chest pain' in context_lower or 'palpitations' in context_lower:
                return Category.CARDIOVASCULAR.value

            # Use embeddings if available
            if self.category_embeddings:
                symptom_embedding = embed_text(symptom)
                max_similarity = -1
                best_category = Category.GENERAL.value
                for category, cat_embedding in self.category_embeddings.items():
                    similarity = torch.cosine_similarity(cat_embedding, symptom_embedding, dim=0).item()
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_category = category
                if max_similarity > 0.7:
                    logger.debug(f"Inferred category '{best_category}' for '{symptom_lower}'")
                    return best_category

            logger.debug(f"No category inferred for '{symptom_lower}'. Defaulting to 'general'")
            return Category.GENERAL.value
        except Exception as e:
            logger.error(f"Error inferring category for '{symptom}': {str(e)}")
            return Category.GENERAL.value

    def add_symptom(self, symptom: str, category: str, synonyms: List[str], note_text: str) -> None:
        """Add a symptom to the knowledge base."""
        self.update_knowledge_base(symptom, category, synonyms, note_text)

    def add_clinical_path(self, category: str, path_key: str, path_data: Dict):
        """Add or update a clinical path in the knowledge base."""
        try:
            # Validate category
            try:
                category_enum = Category(category.lower())
            except ValueError:
                logger.warning(f"Invalid category '{category}' for clinical path '{path_key}'. Using 'general'.")
                category_enum = Category.GENERAL

            # Validate clinical path data
            if 'metadata' in path_data and 'last_updated' in path_data['metadata']:
                if isinstance(path_data['metadata']['last_updated'], str):
                    try:
                        path_data['metadata']['last_updated'] = datetime.strptime(path_data['metadata']['last_updated'], "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        path_data['metadata']['last_updated'] = datetime.now()

            clinical_path = ClinicalPath(**path_data)

            # Update in-memory knowledge base
            self.knowledge_base['clinical_pathways'].setdefault(category_enum.value, {})[path_key.lower()] = clinical_path.dict()

            # Save to MongoDB
            if self.pathways_collection:
                try:
                    # Format last_updated as string for MongoDB
                    formatted_path = clinical_path.dict()
                    if 'metadata' in formatted_path and 'last_updated' in formatted_path['metadata']:
                        formatted_path['metadata']['last_updated'] = formatted_path['metadata']['last_updated'].strftime("%Y-%m-%d %H:%M:%S")
                    self.pathways_collection.update_one(
                        {'category': category_enum.value, 'key': path_key.lower()},
                        {'$set': {'paths': {path_key.lower(): formatted_path}}},
                        upsert=True
                    )
                    self.versions_collection.update_one(
                        {'version': self.version},
                        {
                            '$set': {
                                'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'updated_collections': {'clinical_pathways': True}
                            }
                        },
                        upsert=True
                    )
                    logger.info(f"Updated clinical path '{path_key}' for category '{category_enum.value}' in MongoDB")
                except PyMongoError as e:
                    logger.error(f"Failed to update clinical path '{path_key}' in MongoDB: {str(e)}")

            save_knowledge_base(self.knowledge_base)
            logger.info(f"Updated knowledge base with clinical path '{path_key}' in category '{category_enum.value}'")
        except ValidationError as e:
            logger.error(f"Validation error for clinical path '{path_key}': {str(e)}")
        except Exception as e:
            logger.error(f"Failed to update clinical path '{path_key}': {str(e)}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            self.client.close()
            logger.debug("Closed MongoDB client")