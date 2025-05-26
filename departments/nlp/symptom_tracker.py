from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import logging
import re
import os
import json
from typing import List, Dict, Set, Tuple, Optional
from departments.nlp.logging_setup import get_logger
from departments.nlp.knowledge_base_io import load_knowledge_base
from departments.nlp.kb_updater import KnowledgeBaseUpdater
from departments.nlp.config import UTS_API_KEY, UTS_BASE_URL
from departments.nlp.nlp_pipeline import get_nlp
from departments.nlp.nlp_utils import preprocess_text, deduplicate
import requests
import time

logger = get_logger()

class SymptomTracker:
    def __init__(self, mongo_uri: str = 'mongodb://localhost:27017', db_name: str = 'clinical_db', collection_name: str = 'symptoms'):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.uts_api_key = UTS_API_KEY or 'mock_api_key'
        self.uts_base_url = UTS_BASE_URL
        self.common_symptoms: Dict[str, Dict[str, Dict]] = {}
        self.synonyms: Dict[str, List[str]] = {}

        # Initialize MongoDB with indexes
        try:
            client = MongoClient(mongo_uri)
            client.admin.command('ping')
            db = client[db_name]
            self.collection = db[collection_name]
            index_name = "category_1_symptom_1"
            existing_indexes = self.collection.index_information()
            if index_name in existing_indexes:
                if existing_indexes[index_name].get('unique', False):
                    logger.debug(f"Unique index '{index_name}' already exists on 'category' and 'symptom'.")
                else:
                    logger.warning(f"Index '{index_name}' exists but is not unique. Updating to unique index.")
                    duplicates = self.collection.aggregate([
                        {'$group': {
                            '_id': {'category': '$category', 'symptom': '$symptom'},
                            'count': {'$sum': 1},
                            'docs': {'$push': '$_id'}
                        }},
                        {'$match': {'count': {'$gt': 1}}}
                    ])
                    for dup in duplicates:
                        logger.warning(f"Duplicate found: {dup['_id']} with {dup['count']} entries")
                        doc_ids = dup['docs'][1:]
                        self.collection.delete_many({'_id': {'$in': doc_ids}})
                        logger.info(f"Removed {len(doc_ids)} duplicate documents for {dup['_id']}")
                    self.collection.drop_index(index_name)
                    self.collection.create_index([('category', 1), ('symptom', 1)], unique=True, name=index_name)
                    logger.info(f"Updated index '{index_name}' to unique on 'category' and 'symptom'.")
            else:
                self.collection.create_index([('category', 1), ('symptom', 1)], unique=True, name=index_name)
                logger.info(f"Created unique index '{index_name}' on 'category' and 'symptom'.")
            db['umls_cache'].create_index([('symptom', 1)], unique=True, name="symptom_1")
            cursor = self.collection.find(batch_size=100)
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
                logger.info(f"Loaded {sum(len(symptoms) for symptoms in self.common_symptoms.values())} symptoms from MongoDB across {len(self.common_symptoms)} categories.")
            client.close()
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}. Falling back to JSON.")
            self._load_json_fallback()
            self.collection = None
        except Exception as e:
            logger.error(f"Error loading symptoms from MongoDB: {str(e)}. Falling back to JSON.")
            self._load_json_fallback()
            self.collection = None

        # Load synonyms
        try:
            kb = load_knowledge_base()
            self.synonyms = kb.get('synonyms', {})
            if not self.synonyms:
                logger.warning("No synonyms loaded from knowledge base. Symptom matching may be limited.")
            else:
                logger.info(f"Loaded {len(self.synonyms)} synonym mappings from knowledge base.")
        except Exception as e:
            logger.error(f"Error loading synonyms from knowledge base: {str(e)}")
            self.synonyms = {}

        # Initialize KnowledgeBaseUpdater
        self.kb_updater = KnowledgeBaseUpdater(mongo_uri=mongo_uri, db_name=db_name, kb_prefix='kb_')

        # Initialize NLP pipeline
        self.nlp = None
        try:
            self.nlp = get_nlp()
            logger.info("Using shared medspacy pipeline from nlp_pipeline")
        except Exception as e:
            logger.error(f"Failed to initialize NLP pipeline: {str(e)}. Falling back to regex negation.")
            self.nlp = None
        logger.debug(f"NLP attribute set to: {self.nlp}")

    def _load_json_fallback(self):
        """Load symptoms from JSON fallback if MongoDB fails."""
        json_path = os.path.join(os.path.dirname(__file__), "knowledge_base", "symptoms.json")
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    logger.error(f"Expected dict in {json_path}, got {type(data)}. Using empty symptoms.")
                    self.common_symptoms = {}
                else:
                    valid_symptoms = {}
                    for category, symptoms in data.items():
                        if not isinstance(symptoms, dict):
                            logger.warning(f"Skipping invalid category {category}: {type(symptoms)}")
                            continue
                        valid_symptoms[category] = {}
                        for symptom, info in symptoms.items():
                            if not isinstance(info, dict) or 'description' not in info or not symptom or not info['description']:
                                logger.warning(f"Skipping invalid symptom {symptom} in {category}: {info}")
                                continue
                            valid_symptoms[category][symptom] = {
                                'description': info['description'],
                                'umls_cui': info.get('umls_cui'),
                                'semantic_type': info.get('semantic_type', 'Unknown')
                            }
                    self.common_symptoms = valid_symptoms
                    logger.info(f"Loaded {sum(len(symptoms) for symptoms in self.common_symptoms.values())} symptoms from {json_path}.")
        except FileNotFoundError:
            logger.error(f"Symptom JSON not found at {json_path}. Using empty symptom set.")
            self.common_symptoms = {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {json_path}: {str(e)}. Using empty symptom set.")
            self.common_symptoms = {}

    def _save_json_fallback(self, json_path: str, data: dict):
        """Save symptoms to JSON fallback."""
        try:
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Created symptom JSON at {json_path}.")
        except Exception as e:
            logger.error(f"Failed to create {json_path}: {str(e)}")

    def _get_uts_ticket(self, retries: int = 3, delay: float = 1.0) -> str:
        """Retrieve a single-use ticket for UTS API."""
        if self.uts_api_key == 'mock_api_key':
            logger.error("UTS_API_KEY is not set or is using mock key. Please configure a valid API key.")
            return ''
        for attempt in range(retries):
            try:
                ticket_url = f"{self.uts_base_url}/tickets"
                response = requests.post(ticket_url, data={'apiKey': self.uts_api_key}, timeout=5)
                response.raise_for_status()
                return response.text
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{retries} failed to get UTS ticket: {str(e)}")
                if attempt < retries - 1:
                    time.sleep(delay)
        logger.error(f"Failed to get UTS ticket after {retries} attempts")
        return ''

    def _get_umls_cui(self, symptom: str) -> Tuple[Optional[str], Optional[str]]:
        """Retrieve UMLS CUI and semantic type for a symptom with caching."""
        symptom_clean = preprocess_text(symptom)
        symptom_lower = symptom_clean.lower()
        try:
            client = MongoClient(self.mongo_uri)
            db = client[self.db_name]
            cache = db['umls_cache']
            cached = cache.find_one({'symptom': symptom_lower})
            if cached:
                logger.debug(f"Retrieved cached UMLS data for '{symptom_lower}'")
                client.close()
                return cached['cui'], cached['semantic_type']
        except Exception as e:
            logger.error(f"Error accessing UMLS cache: {str(e)}")

        try:
            ticket = self._get_uts_ticket()
            if not ticket:
                return None, 'Unknown'
            search_url = f"{self.uts_base_url}/search/current"
            params = {'string': symptom_lower, 'ticket': ticket, 'searchType': 'exact', 'sabs': 'SNOMEDCT_US'}
            response = requests.get(search_url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            if data.get('result', {}).get('results') and data['result']['results'][0]['ui'] != 'NONE':
                cui = data['result']['results'][0]['ui']
                concept_url = f"{self.uts_base_url}/content/current/CUI/{cui}"
                concept_response = requests.get(concept_url, params={'ticket': ticket}, timeout=5)
                concept_response.raise_for_status()
                concept_data = concept_response.json()
                semantic_type = concept_data['result'].get('semanticTypes', [{}])[0].get('name', 'Unknown')
                try:
                    cache.update_one(
                        {'symptom': symptom_lower},
                        {'$set': {
                            'cui': cui,
                            'semantic_type': semantic_type
                        }},
                        upsert=True
                    )
                    logger.debug(f"Cached UMLS data for '{symptom_lower}'")
                except Exception as e:
                    logger.error(f"Failed to cache UMLS data: {str(e)}")
                finally:
                    client.close()
                return cui, semantic_type
            logger.warning(f"No UMLS CUI found for '{symptom_lower}'. Marking for review.")
            return None, 'Unknown'
        except Exception as e:
            logger.error(f"UMLS CUI retrieval failed for '{symptom_lower}': {str(e)}")
            return None, 'Unknown'

    def add_symptom(self, category: str, symptom: str, description: str, umls_cui: Optional[str] = None, semantic_type: Optional[str] = 'Unknown'):
        """Add a symptom to MongoDB and in-memory store."""
        if not self.collection:
            logger.error("MongoDB collection not initialized. Cannot add symptom.")
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
            self._update_symptoms_json()
        except Exception as e:
            logger.error(f"Error adding symptom to MongoDB: {str(e)}")

    def remove_symptom(self, category: str, symptom: str):
        """Remove a symptom from MongoDB and in-memory store."""
        if not self.collection:
            logger.error("MongoDB collection not initialized. Cannot remove symptom.")
            return
        try:
            self.collection.delete_one({'category': category, 'symptom': symptom})
            if category in self.common_symptoms and symptom in self.common_symptoms[category]:
                del self.common_symptoms[category][symptom]
                if not self.common_symptoms[category]:
                    del self.common_symptoms[category]
                logger.info(f"Removed symptom '{symptom}' from category '{category}'.")
            self._update_symptoms_json()
        except Exception as e:
            logger.error(f"Error removing symptom from MongoDB: {str(e)}")

    def _update_symptoms_json(self):
        """Update JSON fallback with current symptoms."""
        json_path = os.path.join(os.path.dirname(__file__), "knowledge_base", "symptoms.json")
        try:
            self._save_json_fallback(json_path, self.common_symptoms)
        except Exception as e:
            logger.error(f"Failed to update {json_path}: {str(e)}")

    def update_knowledge_base(self, note_text: str, expected_symptoms: List[str], extracted_symptoms: List[Dict], chief_complaint: str) -> None:
        """Update knowledge base with missing symptoms."""
        extracted_desc = {s['description'].lower() for s in extracted_symptoms}
        missing_symptoms = [s.lower() for s in expected_symptoms if s.lower() not in extracted_desc]
        if not missing_symptoms:
            logger.info("All expected symptoms matched. No knowledge base update needed.")
            return

        logger.warning(f"Missing symptoms: {missing_symptoms}. Initiating knowledge base update.")
        for symptom in missing_symptoms:
            if self.kb_updater.is_new_symptom(symptom):
                category = self._infer_category(symptom, chief_complaint)
                cui, semantic_type = self._get_umls_cui(symptom)
                description = f"UMLS-derived: {symptom}" if cui else f"Pending UMLS review: {symptom}"
                synonyms = self.synonyms.get(symptom.lower(), self.kb_updater.generate_synonyms(symptom))
                self.kb_updater.update_knowledge_base(symptom, category, synonyms, note_text)
                self.add_symptom(category, symptom, description, cui, semantic_type)
                logger.info(f"Updated knowledge base with new symptom '{symptom}' (category: {category}, synonyms: {synonyms}, CUI: {cui})")
            else:
                logger.debug(f"Symptom '{symptom}' already exists in knowledge base. Skipping update.")

    def _infer_category(self, symptom: str, chief_complaint: str) -> str:
        """Infer symptom category from MongoDB mapping or chief complaint."""
        symptom_clean = preprocess_text(symptom)
        symptom_lower = symptom_clean.lower()
        try:
            client = MongoClient(self.mongo_uri)
            db = client[self.db_name]
            category_mapping = db.get_collection('category_mappings')
            mapping = category_mapping.find_one({'symptom': symptom_lower})
            if mapping and 'category' in mapping:
                logger.debug(f"Inferred category '{mapping['category']}' for symptom '{symptom_lower}' from MongoDB")
                client.close()
                return mapping['category']
        except Exception as e:
            logger.error(f"Error accessing category mappings: {str(e)}")

        # Fallback to chief complaint-based inference
        chief_clean = preprocess_text(chief_complaint)
        chief_lower = chief_clean.lower()
        if 'headache' in chief_lower or 'neurological' in chief_lower:
            return "neurological"
        if 'sinus' in chief_lower or 'respiratory' in chief_lower:
            return "respiratory"
        logger.warning(f"No category mapping found for '{symptom_lower}'. Defaulting to 'general'.")
        return "general"

    def get_symptoms_by_category(self, category: str) -> Dict[str, Dict]:
        """Retrieve symptoms for a given category."""
        return self.common_symptoms.get(category, {})

    def search_symptom(self, symptom: str) -> Tuple[Optional[str], Optional[Dict]]:
        """Search for a symptom across categories."""
        symptom_clean = preprocess_text(symptom)
        for category, symptoms in self.common_symptoms.items():
            for s, info in symptoms.items():
                if preprocess_text(s).lower() == symptom_clean.lower():
                    return category, info
        return None, None

    def get_all_symptoms(self) -> Set[str]:
        """Retrieve all symptom names."""
        return {symptom for category in self.common_symptoms.values() for symptom in category.keys()}

    def get_categories(self) -> List[str]:
        """Retrieve all symptom categories."""
        return list(self.common_symptoms.keys())

    def extract_duration(self, text: str) -> str:
        """Placeholder for duration extraction."""
        if '3 days' in text.lower() or 'three days' in text.lower():
            return '3 days'
        return 'Unknown'

    def classify_severity(self, text: str) -> str:
        """Placeholder for severity classification."""
        if 'severe' in text.lower() or 'worse' in text.lower():
            return 'Severe'
        return 'Unknown'

    def extract_location(self, text: str, symptom: str) -> str:
        """Placeholder for location extraction."""
        return 'Unknown'

    def process_note(self, note, chief_complaint: str, expected_symptoms: List[str] = None) -> List[Dict]:
        """Extract symptoms from a SOAP note."""
        logger.debug(f"Processing note ID {getattr(note, 'id', 'unknown')} for symptoms with chief complaint: {chief_complaint}")
        text = f"{note.situation or ''} {note.hpi or ''} {note.assessment or ''} {note.aggravating_factors or ''} {note.alleviating_factors or ''} {getattr(note, 'symptoms', '') or ''}".strip()
        if not text:
            logger.warning("No text available for symptom extraction")
            return []

        text_clean = preprocess_text(text)
        text_clean = re.sub(r'[;:]', ' ', re.sub(r'\s+', ' ', text_clean)).lower()
        symptoms = []
        matched_terms = set()
        chief_complaint_lower = preprocess_text(chief_complaint).lower()
        hpi_lower = preprocess_text(note.hpi or '').lower()

        # Match symptoms using common_symptoms and synonyms
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
                        'aggravating': 'Unknown',
                        'alleviating': 'Unknown',
                        'umls_cui': umls_cui,
                        'semantic_type': semantic_type
                    })
                    matched_terms.add(matched_term)
                    logger.debug(f"Matched symptom: {clean_symptom} (category: {category}, term: {matched_term}, CUI: {umls_cui})")

        # NLP-based symptom extraction
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
                            logger.info(f"Added new symptom '{ent_text}' from NLP to category '{category}' (CUI: {cui})")
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
                        'aggravating': 'Unknown',
                        'alleviating': 'Unknown',
                        'umls_cui': cui,
                        'semantic_type': semantic_type
                    })
                    matched_terms.add(ent_text)
            except Exception as e:
                logger.error(f"Error processing text with NLP: {str(e)}")

        # Negation detection
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
                logger.debug(f"Detected negated symptom (regex): {negated_clean}")

        # Filter out negated symptoms
        valid_symptoms = []
        for s in symptoms:
            s_desc_lower = preprocess_text(s['description']).lower()
            is_negated = False
            for ns in negated_symptoms:
                ns_lower = preprocess_text(ns['description']).lower()
                if re.search(r'\b' + re.escape(ns_lower) + r'\b', s_desc_lower) or re.search(r'\b' + re.escape(s_desc_lower) + r'\b', ns_lower):
                    is_negated = True
                    logger.debug(f"Excluding negated symptom: {s['description']} (negated by: {ns['description']})")
                    break
            if not is_negated:
                valid_symptoms.append(s)

        # Deduplicate symptoms
        symptom_descriptions = tuple(s['description'] for s in valid_symptoms)
        deduped_descriptions = deduplicate(symptom_descriptions, self.synonyms)
        unique_symptoms = [s for s in valid_symptoms if s['description'] in deduped_descriptions]

        logger.debug(f"Extracted symptoms: {[s['description'] for s in unique_symptoms]}, Negated: {[ns['description'] for ns in negated_symptoms]}")
        if expected_symptoms:
            self.update_knowledge_base(text_clean, expected_symptoms, unique_symptoms, chief_complaint)
        return unique_symptoms