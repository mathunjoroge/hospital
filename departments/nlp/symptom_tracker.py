from pymongo import MongoClient
from pymongo.errors import PyMongoError as MongoError
import logging
import re
from typing import List, Dict, Set, Tuple, Optional
import os
from dotenv import load_dotenv
from spacy.language import Language
from cachetools import Cache as LRUCache
from psycopg2.pool import SimpleConnectionPool
from departments.nlp.nlp_utils import preprocess_text, deduplicate
from departments.nlp.config import FALLBACK_CFG, MONGO_URI, DB_NAME, KB_PREFIX, POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
from departments.nlp.logging_setup import get_logger
from departments.nlp.nlp_pipeline import clean_term, search_local_umls_cui
from departments.nlp.kb_updater import KnowledgeBaseUpdater
from departments.nlp.nlp_utils import get_umls_cui
from departments.nlp.helper_functions import extract_aggravating_alleviating

logger = get_logger()
load_dotenv()

class SymptomTracker:
    _instance = None
    _cui_cache = LRUCache(maxsize=10000)

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, mongo_uri: str = MONGO_URI, db_name: str = DB_NAME, symptom_collection: str = 'symptoms', nlp: Optional[Language] = None):
        if getattr(self, '_initialized', False):
            return
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = f'{KB_PREFIX}{symptom_collection}'
        self.common_symptoms: Dict[str, Dict[str, Dict]] = {}
        self.synonyms: Dict[str, List[str]] = {}
        self.nlp = nlp

        # Initialize PostgreSQL connection pool
        try:
            self.pool = SimpleConnectionPool(
                minconn=1, maxconn=10,
                host=POSTGRES_HOST, port=POSTGRES_PORT,
                dbname=POSTGRES_DB, user=POSTGRES_USER, password=POSTGRES_PASSWORD
            )
            logger.info("Initialized PostgreSQL connection pool")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL connection pool: {e}")
            self.pool = None

        # MongoDB setup
        try:
            client = MongoClient(mongo_uri)
            client.admin.command('ping')
            db = client[db_name]
            self.collection = db[self.collection_name]
            index_name = "category_1_symptom_1"
            existing_indexes = self.collection.index_information()
            if index_name not in existing_indexes:
                self.collection.create_index([('category', 1), ('symptom', 1)], unique=True, name=index_name)
                logger.info(f"Created unique index '{index_name}' on 'category' and 'symptom'.")
            elif not existing_indexes[index_name].get('unique', False):
                logger.warning(f"Non-unique index '{index_name}' found. Dropping and recreating.")
                self.collection.drop_index(index_name)
                self.collection.create_index([('category', 1), ('symptom', 1)], unique=True, name=index_name)
                logger.info(f"Recreated unique index '{index_name}'.")
            else:
                logger.debug(f"Unique index '{index_name}' already exists.")

            cursor = self.collection.find({}, {'category': 1, 'symptom': 1, 'description': 1, 'umls_cui': 1, 'semantic_type': 1}, batch_size=100)
            for doc in cursor:
                category = doc.get('category')
                symptom = doc.get('symptom')
                description = doc.get('description')
                umls_cui = doc.get('umls_cui')
                semantic_type = doc.get('semantic_type', 'Unknown')
                if not all([category, symptom, description]) or not isinstance(category, str) or not isinstance(symptom, str) or not isinstance(description, str):
                    logger.warning(f"Skipping invalid symptom document: {doc}")
                    continue
                if category not in self.common_symptoms:
                    self.common_symptoms[category] = {}
                if symptom not in self.common_symptoms[category]:
                    self.common_symptoms[category][symptom] = {
                        'description': description,
                        'umls_cui': umls_cui,
                        'semantic_type': semantic_type
                    }
                else:
                    logger.warning(f"Duplicate symptom '{symptom}' in category '{category}' skipped")
            if not self.common_symptoms:
                logger.warning("No symptoms loaded from MongoDB. Initializing empty symptom set.")
            else:
                logger.info(f"Loaded {sum(len(symptoms) for symptoms in self.common_symptoms.values())} symptoms across {len(self.common_symptoms)} categories.")

            synonym_collection = db[f'{KB_PREFIX}synonyms']
            for doc in synonym_collection.find({}, {'term': 1, 'aliases': 1}):
                if 'term' in doc and 'aliases' in doc:
                    self.synonyms[doc['term']] = doc['aliases']
            if not self.synonyms:
                logger.warning("No synonyms loaded from MongoDB.")
            else:
                logger.info(f"Loaded {len(self.synonyms)} synonym mappings.")

            client.close()
        except MongoError as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading symptoms from MongoDB: {e}")
            raise

        self.kb_updater = KnowledgeBaseUpdater(mongo_uri=mongo_uri, db_name=db_name, kb_prefix=KB_PREFIX)
        self._initialized = True

    def _get_umls_cui(self, symptom: str) -> Tuple[Optional[str], Optional[str]]:
        """Retrieve UMLS CUI and semantic type using UMLSLookup."""
        return get_umls_cui(symptom)

    def add_symptom(self, category: str, symptom: str, description: str, umls_cui: Optional[str] = None, semantic_type: Optional[str] = 'Unknown'):
        if not self.collection:
            logger.error("MongoDB collection not initialized.")
            return
        try:
            self.collection.update_one(
                {'category': category, 'symptom': symptom},
                {
                    '$set': {
                        'description': description,
                        'umls_cui': umls_cui,
                        'semantic_type': semantic_type
                    }
                },
                upsert=True
            )
            if category not in self.common_symptoms:
                self.common_symptoms[category] = {}
            self.common_symptoms[category][symptom] = {
                'description': description,
                'umls_cui': umls_cui,
                'semantic_type': semantic_type
            }
            logger.info(f"Added symptom '{symptom}' to category '{category}' (CUI: {umls_cui}).")
        except Exception as e:
            logger.error(f"Error adding symptom to MongoDB: {e}")

    def remove_symptom(self, category: str, symptom: str):
        if not self.collection:
            logger.error("MongoDB collection not initialized.")
            return
        try:
            self.collection.delete_one({'category': category, 'symptom': symptom})
            if category in self.common_symptoms and symptom in self.common_symptoms[category]:
                del self.common_symptoms[category][symptom]
                if not self.common_symptoms[category]:
                    del self.common_symptoms[category]
                logger.info(f"Removed symptom '{symptom}' from category '{category}'.")
        except Exception as e:
            logger.error(f"Error removing symptom from MongoDB: {e}")

    def update_knowledge_base(self, note_text: str, expected_symptoms: List[str], extracted_symptoms: List[Dict], chief_complaint: str) -> None:
        extracted_desc = {s['description'].lower() for s in extracted_symptoms}
        missing_symptoms = [s.lower() for s in expected_symptoms if s.lower() not in extracted_desc]
        if not missing_symptoms:
            logger.info("All expected symptoms matched.")
            return

        logger.warning(f"Missing symptoms: {missing_symptoms}.")
        for symptom in missing_symptoms:
            if self.kb_updater.is_new_symptom(symptom):
                category = self._infer_category(symptom, chief_complaint)
                cui, semantic_type = self._get_umls_cui(symptom)
                description = f"UMLS-derived: {symptom}" if cui else f"Pending UMLS review: {symptom}"
                synonyms = self.synonyms.get(symptom.lower(), self.kb_updater.generate_synonyms(symptom))
                self.kb_updater.update_knowledge_base(symptom, category, synonyms, note_text)
                self.add_symptom(category, symptom, description, cui, semantic_type)
                logger.info(f"Updated knowledge base with symptom '{symptom}' (category: {category}, CUI: {cui})")
            else:
                logger.debug(f"Symptom '{symptom}' exists in knowledge base.")

    def _infer_category(self, symptom: str, chief_complaint: str) -> str:
        symptom_clean = preprocess_text(symptom)
        symptom_lower = symptom_clean.lower()
        try:
            client = MongoClient(self.mongo_uri)
            db = client[self.db_name]
            category_mapping = db[f'{KB_PREFIX}category_mappings']
            mapping = category_mapping.find_one({'symptom': symptom_lower}, {'category': 1})
            if mapping and 'category' in mapping:
                logger.debug(f"Inferred category '{mapping['category']}' for '{symptom_lower}'")
                return mapping['category']
        except Exception as e:
            logger.error(f"Error accessing category mappings: {e}")
        finally:
            if 'client' in locals():
                client.close()

        chief_clean = preprocess_text(chief_complaint)
        chief_lower = chief_clean.lower()
        if 'headache' in chief_lower or 'neurological' in chief_lower:
            return "neurological"
        if 'sinus' in chief_lower or 'respiratory' in chief_lower:
            return "respiratory"
        logger.warning(f"No category mapping for '{symptom_lower}'. Defaulting to 'general'.")
        return "general"

    def get_symptoms_by_category(self, category: str) -> Dict[str, Dict]:
        return self.common_symptoms.get(category, {})

    def search_symptom(self, symptom: str) -> Tuple[Optional[str], Optional[Dict]]:
        symptom_clean = preprocess_text(symptom)
        for category, symptoms in self.common_symptoms.items():
            for s, info in symptoms.items():
                if preprocess_text(s).lower() == symptom_clean.lower():
                    return category, info
        return None, None

    def get_all_symptoms(self) -> Set[str]:
        return {symptom for category in self.common_symptoms.values() for symptom in category.keys()}

    def get_categories(self) -> List[str]:
        return list(self.common_symptoms.keys())

    def extract_duration(self, text: str) -> str:
        if '3 days' in text.lower() or 'three days' in text.lower():
            return '3 days'
        return 'Unknown'

    def classify_severity(self, text: str) -> str:
        if 'severe' in text.lower() or 'worse' in text.lower():
            return 'Severe'
        return 'Unknown'

    def extract_location(self, text: str, symptom: str) -> str:
        return 'Unknown'

    def process_note(self, note, chief_complaint: str, expected_symptoms: List[str] = None) -> List[Dict]:
        logger.debug(f"Processing note ID {getattr(note, 'id', 'unknown')}")
        text = f"{getattr(note, 'situation', '') or ''} {getattr(note, 'hpi', '') or ''} {getattr(note, 'assessment', '') or ''} {getattr(note, 'aggravating_factors', '') or ''} {getattr(note, 'alleviating_factors', '') or ''} {getattr(note, 'Symptoms', '') or ''}".strip()
        if not text:
            logger.warning("No text available for symptom extraction")
            return []

        text_clean = preprocess_text(text)
        text_clean = re.sub(r'[;:]', ' ', re.sub(r'\s+', ' ', text_clean)).lower()
        symptoms = []
        matched_terms = set()
        chief_complaint_lower = preprocess_text(chief_complaint).lower()
        hpi_lower = preprocess_text(getattr(note, 'hpi', '') or '').lower()

        # Extract aggravating and alleviating factors
        try:
            aggravating_text = getattr(note, 'aggravating_factors', '') or ''
            alleviating_text = getattr(note, 'alleviating_factors', '') or ''
            aggravating_result = extract_aggravating_alleviating(aggravating_text, 'aggravating')
            alleviating_result = extract_aggravating_alleviating(alleviating_text, 'alleviating')
            aggravating = aggravating_result.lower().strip() if isinstance(aggravating_result, str) else aggravating_text.lower().strip()
            alleviating = alleviating_result.lower().strip() if isinstance(alleviating_result, str) else alleviating_text.lower().strip()
            logger.debug(f"Extracted aggravating: {aggravating[:50]}..., alleviating: {alleviating[:50]}...")
        except Exception as e:
            logger.error(f"Error extracting aggravating/alleviating factors: {e}")
            aggravating = 'Unknown'
            alleviating = 'Unknown'

        for category, symptom_dict in self.common_symptoms.items():
            for symptom, info in symptom_dict.items():
                symptom_clean = preprocess_text(symptom).lower()
                umls_cui = info.get('umls_cui')
                semantic_type = info.get('semantic_type', 'Unknown')
                synonym_list = self.synonyms.get(symptom_clean, [])
                matched = False
                matched_term = None
                patterns = [symptom_clean] + [preprocess_text(s).lower() for s in synonym_list]
                for pattern in patterns:
                    if re.search(r'\b' + re.escape(pattern) + r'\b', text_clean):
                        matched = True
                        matched_term = pattern
                        break
                if matched:
                    clean_symptom = re.sub(r'^(patient complains of |complains of |reports )\b', '', symptom_clean, flags=re.IGNORECASE)
                    symptoms.append({
                        'description': clean_symptom,
                        'category': category,
                        'definition': info['description'],
                        'duration': self.extract_duration(text_clean),
                        'severity': self.classify_severity(text_clean),
                        'location': self.extract_location(text_clean, clean_symptom),
                        'aggravating': aggravating,
                        'alleviating': alleviating,
                        'umls_cui': umls_cui,
                        'semantic_type': semantic_type
                    })
                    matched_terms.add(matched_term)
                    logger.debug(f"Matched symptom: {clean_symptom} (category: {category}, term: {matched_term})")

        if self.nlp:
            try:
                doc = self.nlp(text_clean)
                for ent in doc.ents:
                    ent_text = preprocess_text(ent.text).lower()
                    if ent_text in matched_terms:
                        continue
                    category, info = self.search_symptom(ent_text)
                    if not category:
                        category = self._infer_category(ent_text, chief_complaint)
                        cui, semantic_type = self._get_umls_cui(ent_text)
                        description = f"UMLS-derived: {ent_text}" if cui else f"Pending UMLS review: {ent_text}"
                        if ent_text not in self.get_all_symptoms():
                            self.add_symptom(category, ent_text, description, cui, semantic_type)
                            logger.info(f"Added new symptom '{ent_text}' to category '{category}'")
                    else:
                        cui = info.get('umls_cui')
                        semantic_type = info.get('semantic_type', 'Unknown')
                        description = info['description']
                    symptoms.append({
                        'description': ent_text,
                        'category': category,
                        'definition': description,
                        'duration': self.extract_duration(text_clean),
                        'severity': self.classify_severity(text_clean),
                        'location': self.extract_location(text_clean, ent_text),
                        'aggravating': aggravating,
                        'alleviating': alleviating,
                        'umls_cui': cui,
                        'semantic_type': semantic_type
                    })
                    matched_terms.add(ent_text)
            except Exception as e:
                logger.error(f"Error processing text with NLP: {e}")

        negated_symptoms = []
        negation_pattern = r"\b(no|without|denies|not)\b\s+([\w\s-]+?)(?=\.|,\s*|\s+and\b|\s+or\b|$)"
        matches = re.findall(negation_pattern, text_clean, re.IGNORECASE)
        for _, negated in matches:
            negated_clean = preprocess_text(negated.strip())
            if negated_clean and negated_clean not in chief_complaint_lower and negated_clean not in hpi_lower:
                negated_symptoms.append({
                    'description': negated_clean,
                    'category': self._infer_category(negated_clean, chief_complaint),
                    'definition': f"Negated: {negated_clean}",
                    'umls_cui': None,
                    'semantic_type': 'Unknown'
                })
                logger.debug(f"Detected negated symptom: {negated_clean}")

        valid_symptoms = []
        for s in symptoms:
            s_desc_lower = preprocess_text(s['description']).lower()
            is_negated = False
            for ns in negated_symptoms:
                ns_lower = preprocess_text(ns['description']).lower()
                if re.search(r'\b' + re.escape(ns_lower) + r'\b', s_desc_lower) or re.search(r'\b' + re.escape(s_desc_lower) + r'\b', ns_lower):
                    is_negated = True
                    logger.debug(f"Excluding negated symptom: {s['description']}")
                    break
            if not is_negated:
                valid_symptoms.append(s)

        symptom_descriptions = tuple(s['description'] for s in valid_symptoms)
        deduped_descriptions = deduplicate(symptom_descriptions, self.synonyms)
        unique_symptoms = [s for s in valid_symptoms if s['description'] in deduped_descriptions]

        logger.debug(f"Extracted symptoms: {[s['description'] for s in unique_symptoms]}")
        if expected_symptoms:
            self.update_knowledge_base(text_clean, expected_symptoms, unique_symptoms, chief_complaint)
        return unique_symptoms

    def get_negated_symptoms(self, note, chief_complaint: str) -> Set[str]:
        """Extract negated symptoms from a clinical note."""
        logger.debug(f"Extracting negated symptoms for note ID {getattr(note, 'id', 'unknown')}")
        text = f"{getattr(note, 'situation', '') or ''} {getattr(note, 'hpi', '') or ''} {getattr(note, 'assessment', '') or ''} {getattr(note, 'aggravating_factors', '') or ''} {getattr(note, 'alleviating_factors', '') or ''} {getattr(note, 'Symptoms', '') or ''}".strip()
        if not text:
            logger.warning("No text available for negated symptom extraction")
            return set()

        text_clean = preprocess_text(text).lower()
        text_clean = re.sub(r'[;:]', ' ', re.sub(r'\s+', ' ', text_clean))
        chief_complaint_lower = preprocess_text(chief_complaint).lower()
        hpi_lower = preprocess_text(getattr(note, 'hpi', '') or '').lower()

        negated_symptoms = set()
        negation_pattern = r"\b(no|without|denies|not)\b\s+([\w\s-]+?)(?=\.|,\s*|\s+and\b|\s+or\b|$)"
        matches = re.findall(negation_pattern, text_clean, re.IGNORECASE)
        for _, negated in matches:
            negated_clean = preprocess_text(negated.strip()).lower()
            if negated_clean and negated_clean not in chief_complaint_lower and negated_clean not in hpi_lower:
                negated_symptoms.add(negated_clean)
                logger.debug(f"Detected negated symptom: {negated_clean}")

        return negated_symptoms

    def __del__(self):
        if hasattr(self, 'pool') and self.pool:
            self.pool.closeall()
            logger.debug("Closed PostgreSQL connection pool")