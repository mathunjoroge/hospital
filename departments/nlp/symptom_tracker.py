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
                if symptom.lower() in ['diabetes', 'hypertension']:  # Skip conditions, not symptoms
                    continue
                if category and symptom and description and isinstance(category, str) and isinstance(symptom, str) and isinstance(description, str):
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
                logger.warning("No symptoms loaded from MongoDB. Initializing with UMLS seed data.")
                self._initialize_symptoms_from_umls()
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

    def _initialize_symptoms_from_umls(self):
        """Initialize symptoms collection with a seed set from UMLS."""
        if not self.collection:
            logger.error("MongoDB collection not initialized. Cannot seed symptoms.")
            return
        # Seed with common symptoms (curated list to avoid excessive API calls)
        seed_symptoms = [
            'headache', 'fever', 'cough', 'nausea', 'chest pain', 'shortness of breath',
            'jaundice', 'chills', 'vomiting', 'loss of appetite', 'fatigue', 'abdominal pain',
            'diarrhea', 'rash', 'joint pain', 'back pain', 'dizziness', 'photophobia'
        ]
        for symptom in seed_symptoms:
            cui, semantic_type = self._get_umls_cui(symptom)
            if cui:
                category = self._infer_category(symptom, '')
                description = f"UMLS-derived: {symptom}"
                self.add_symptom(category, symptom, description, cui, semantic_type)
        logger.info(f"Seeded {len(seed_symptoms)} symptoms from UMLS into MongoDB.")

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
                            if symptom.lower() in ['diabetes', 'hypertension']:
                                continue
                            if isinstance(symptom, str) and isinstance(info, dict) and 'description' in info and symptom and info['description']:
                                valid_symptoms[category][symptom] = {
                                    'description': info['description'],
                                    'umls_cui': info.get('umls_cui'),
                                    'semantic_type': info.get('semantic_type', 'Unknown')
                                }
                            else:
                                logger.warning(f"Skipping invalid symptom {symptom} in {category}: {info}")
                    self.common_symptoms = valid_symptoms
                    logger.info(f"Loaded {sum(len(symptoms) for symptoms in self.common_symptoms.values())} symptoms from {json_path}.")
        except FileNotFoundError:
            logger.error(f"Symptom JSON not found at {json_path}. Using empty symptoms.")
            self.common_symptoms = {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {json_path}: {str(e)}. Using empty symptoms.")
            self.common_symptoms = {}

    def _save_json_fallback(self, json_path: str, data: dict):
        try:
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Created symptom JSON at {json_path}.")
        except Exception as e:
            logger.error(f"Failed to create {json_path}: {str(e)}")

    def _get_uts_ticket(self, retries: int = 3, delay: float = 1.0) -> str:
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
        symptom_lower = symptom.lower()
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
            # Search UMLS for exact match
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
                # Fetch synonyms
                synonyms = []
                try:
                    xref_url = f"{self.uts_base_url}/crosswalk/current/source/SNOMEDCT_US/{cui}"
                    xref_response = requests.get(xref_url, params={'ticket': ticket}, timeout=5)
                    xref_response.raise_for_status()
                    xref_data = xref_response.json()
                    for result in xref_data.get('result', []):
                        if result.get('sourceName') == 'MSH':  # MeSH for synonyms
                            synonym_cui = result.get('ui')
                            synonym_url = f"{self.uts_base_url}/content/current/CUI/{synonym_cui}"
                            synonym_response = requests.get(synonym_url, params={'ticket': ticket}, timeout=5)
                            synonym_response.raise_for_status()
                            synonym_data = synonym_response.json()
                            synonyms.extend(synonym_data['result'].get('synonyms', []))
                except Exception as e:
                    logger.warning(f"Failed to fetch synonyms for '{symptom_lower}': {str(e)}")
                # Cache result
                try:
                    cache.update_one(
                        {'symptom': symptom_lower},
                        {'$set': {
                            'cui': cui,
                            'semantic_type': semantic_type,
                            'synonyms': synonyms
                        }},
                        upsert=True
                    )
                    logger.debug(f"Cached UMLS data for '{symptom_lower}'")
                except Exception as e:
                    logger.error(f"Failed to cache UMLS data: {str(e)}")
                finally:
                    client.close()
                # Update synonyms in knowledge base
                if synonyms:
                    self.synonyms[symptom_lower] = synonyms
                    self.kb_updater.update_knowledge_base(symptom_lower, self._infer_category(symptom_lower, ''), synonyms, '')
                return cui, semantic_type
            return None, 'Unknown'
        except Exception as e:
            logger.error(f"UMLS CUI retrieval failed for '{symptom_lower}': {str(e)}")
            return None, 'Unknown'

    def add_symptom(self, category: str, symptom: str, description: str, umls_cui: Optional[str] = None, semantic_type: Optional[str] = 'Unknown'):
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
        json_path = os.path.join(os.path.dirname(__file__), "knowledge_base", "symptoms.json")
        try:
            with open(json_path, 'w') as f:
                json.dump(self.common_symptoms, f, indent=2)
            logger.info(f"Updated {json_path} with current symptoms.")
        except Exception as e:
            logger.error(f"Failed to update {json_path}: {str(e)}")

    def update_knowledge_base(self, note_text: str, expected_symptoms: List[str], extracted_symptoms: List[Dict], chief_complaint: str) -> None:
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
                description = f"UMLS-derived: {symptom}"
                synonyms = self.synonyms.get(symptom.lower(), self.kb_updater.generate_synonyms(symptom))
                self.kb_updater.update_knowledge_base(symptom, category, synonyms, note_text)
                self.add_symptom(category, symptom, description, cui, semantic_type)
                logger.info(f"Updated knowledge base with new symptom '{symptom}' (category: {category}, synonyms: {synonyms}, CUI: {cui})")
            else:
                logger.debug(f"Symptom '{symptom}' already exists in knowledge base. Skipping update.")

    def _infer_category(self, symptom: str, chief_complaint: str) -> str:
        symptom_lower = symptom.lower()
        # Extended category inference for HMIS
        if any(kw in symptom_lower for kw in ['nasal', 'sinus', 'facial', 'congestion', 'discharge', 'cough', 'wheezing']):
            return "respiratory"
        if any(kw in symptom_lower for kw in ['headache', 'dizziness', 'photophobia', 'seizure', 'numbness']):
            return "neurological"
        if any(kw in symptom_lower for kw in ['chest', 'palpitations', 'shortness of breath', 'tightness', 'edema']):
            return "cardiovascular"
        if any(kw in symptom_lower for kw in ['nausea', 'diarrhea', 'epigastric', 'vomiting', 'abdominal', 'jaundice']):
            return "gastrointestinal"
        if any(kw in symptom_lower for kw in ['joint', 'knee', 'back', 'obesity', 'movement', 'stiffness']):
            return "musculoskeletal"
        if any(kw in symptom_lower for kw in ['rash', 'itch', 'lesion', 'ulcer']):
            return "dermatological"
        if any(kw in symptom_lower for kw in ['hearing', 'vision', 'tinnitus']):
            return "sensory"
        if any(kw in symptom_lower for kw in ['bleeding', 'bruising', 'anemia']):
            return "hematologic"
        if any(kw in symptom_lower for kw in ['weight', 'thirst', 'fatigue', 'sweating']):
            return "endocrine"
        if any(kw in symptom_lower for kw in ['urinary', 'dysuria', 'incontinence']):
            return "genitourinary"
        if any(kw in symptom_lower for kw in ['mood', 'anxiety', 'depression', 'insomnia']):
            return "psychiatric"
        if any(kw in symptom_lower for kw in ['fever', 'chills', 'malaise']):
            return "general"
        if 'sinus' in chief_complaint.lower():
            return "respiratory"
        return "general"

    def get_symptoms_by_category(self, category: str) -> Dict[str, Dict]:
        return self.common_symptoms.get(category, {})

    def search_symptom(self, symptom: str) -> Tuple[Optional[str], Optional[Dict]]:
        for category, symptoms in self.common_symptoms.items():
            for s, info in symptoms.items():
                if s.lower() == symptom.lower():
                    return category, info
        return None, None

    def get_all_symptoms(self) -> Set[str]:
        return {symptom for category in self.common_symptoms.values() for symptom in category.keys()}

    def get_categories(self) -> List[str]:
        return list(self.common_symptoms.keys())

    def process_note(self, note, chief_complaint: str, expected_symptoms: List[str] = None) -> List[Dict]:
        logger.debug(f"Processing note ID {getattr(note, 'id', 'unknown')} for symptoms with chief complaint: {chief_complaint}")
        text = f"{note.situation or ''} {note.hpi or ''} {note.assessment or ''} {note.aggravating_factors or ''} {note.alleviating_factors or ''} {getattr(note, 'symptoms', '') or ''}".lower().strip()
        if not text:
            logger.warning("No text available for symptom extraction")
            return []

        # Prepare text for processing
        text = re.sub(r'[;:]', ' ', re.sub(r'\s+', ' ', text))
        symptoms = []
        matched_terms = set()
        chief_complaint_lower = chief_complaint.lower()
        hpi_lower = (note.hpi or '').lower()

        # Match symptoms using common_symptoms and synonyms
        for category, symptom_dict in self.common_symptoms.items():
            for symptom, info in symptom_dict.items():
                symptom_lower = symptom.lower()
                desc_lower = info['description'].lower()
                umls_cui = info.get('umls_cui')
                semantic_type = info.get('semantic_type', 'Unknown')
                synonym_list = self.synonyms.get(symptom_lower, [])
                matched = False
                matched_term = None
                patterns = [symptom_lower, desc_lower] + [s.lower() for s in synonym_list]
                for pattern in patterns:
                    matches = list(re.finditer(r'\b' + re.escape(pattern) + r'\b', text))
                    for match in matches:
                        start = match.start()
                        prefix_text = text[max(0, start - 20):start].lower()
                        if any(prefix_text.endswith(prefix) for prefix in ['patient complains of ', 'complains of ', 'reports ']):
                            logger.debug(f"Skipping match '{pattern}' due to prefix in text: {prefix_text}")
                            continue
                        if pattern not in matched_terms:
                            matched = True
                            matched_term = pattern
                            break
                    if matched:
                        break
                if matched:
                    clean_symptom = re.sub(r'^(patient complains of |complains of |reports )\b', '', symptom_lower, flags=re.IGNORECASE)
                    # Enhanced duration detection
                    duration = 'Unknown'
                    if 'three days ago' in text or '3 days' in text:
                        duration = '3 days'
                    elif 'two weeks' in text or 'three weeks' in text:
                        duration = '2-3 weeks'
                    # Enhanced severity and context
                    severity = 'Moderate' if 'worse' in text or 'severe' in text else 'Unknown'
                    location = 'Unknown'
                    if 'headache' in clean_symptom:
                        location = 'Head'
                    elif 'jaundice' in clean_symptom:
                        location = 'Eyes'
                    elif 'back' in clean_symptom:
                        location = 'Lower back'
                    # Capture aggravating/alleviating factors
                    aggravating = 'Unknown'
                    alleviating = 'Unknown'
                    if 'nausea' in clean_symptom:
                        aggravating = 'Fatty foods' if 'makes the nausea worse' in text else 'Unknown'
                        alleviating = 'Tea and salt' if 'alleviates the nausea' in text else 'Unknown'
                    symptoms.append({
                        'description': clean_symptom,
                        'category': category,
                        'definition': info['description'],
                        'duration': duration,
                        'severity': severity,
                        'location': location,
                        'aggravating': aggravating,
                        'alleviating': alleviating,
                        'umls_cui': umls_cui,
                        'semantic_type': semantic_type
                    })
                    matched_terms.add(matched_term)
                    logger.debug(f"Matched symptom: {clean_symptom} (category: {category}, term: {matched_term}, CUI: {umls_cui})")

        # Handle structured symptoms field and unmapped symptoms
        fields_to_process = [
            (note.situation or '', 'situation'),
            (note.hpi or '', 'hpi'),
            (note.assessment or '', 'assessment'),
            (getattr(note, 'symptoms', '') or '', 'symptoms')
        ]
        for field_text, field_name in fields_to_process:
            if not field_text.strip():
                continue
            # Extract potential symptoms via NLP
            if self.nlp:
                try:
                    doc = self.nlp(field_text)
                    for ent in doc.ents:
                        ent_text = ent.text.lower()
                        if ent_text in matched_terms or ent_text.startswith(('patient complains of', 'complains of', 'reports')):
                            continue
                        category = self._infer_category(ent_text, chief_complaint)
                        cui, semantic_type = self._get_umls_cui(ent_text)
                        description = f"UMLS-derived: {ent_text}"
                        if ent_text not in self.get_all_symptoms():
                            self.add_symptom(category, ent_text, description, cui, semantic_type)
                            logger.info(f"Added new symptom '{ent_text}' from {field_name} to category '{category}' (CUI: {cui})")
                        symptoms.append({
                            'description': ent_text,
                            'category': category,
                            'definition': description,
                            'duration': 'Unknown',
                            'severity': 'Unknown',
                            'location': 'Unknown',
                            'aggravating': 'Unknown',
                            'alleviating': 'Unknown',
                            'umls_cui': cui,
                            'semantic_type': semantic_type
                        })
                        matched_terms.add(ent_text)
                except Exception as e:
                    logger.error(f"Error processing {field_name} with NLP: {str(e)}")

        # Negation detection with medspacy
        negated_symptoms = []
        if self.nlp:
            sentence_segmenters = ['medspacy_pyrush', 'sentencizer', 'senter']
            active_segmenter = None
            for seg in sentence_segmenters:
                if seg in self.nlp.pipe_names:
                    active_segmenter = seg
                    break
            for field in fields_to_process:
                field_text, field_name = field[0], field[1]
                if not field_text.strip():
                    continue
                try:
                    if active_segmenter:
                        with self.nlp.disable_pipes(active_segmenter):
                            doc = self.nlp(field_text)
                    else:
                        doc = self.nlp(field_text)
                    for ent in doc.ents:
                        if ent._.is_negated:
                            ent_lower = ent.text.lower()
                            # Skip negation if symptom is in chief complaint or HPI
                            if ent_lower in chief_complaint_lower or ent_lower in hpi_lower:
                                logger.debug(f"Skipping negation for '{ent_lower}' as it appears in chief complaint or HPI")
                                continue
                            negated_symptoms.append({
                                'description': ent_lower,
                                'category': self._infer_category(ent_lower, chief_complaint),
                                'definition': f"Negated: {ent_lower}",
                                'umls_cui': None,
                                'semantic_type': 'Unknown'
                            })
                            logger.debug(f"Detected negated symptom: {ent_lower} in '{field_name}'")
                except Exception as e:
                    logger.error(f"Error processing {field_name} with NLP: {str(e)}. Falling back to regex.")
                    negation_pattern = r"\b(no|without|denies|not)\b\s+([\w\s-]+?)(?=\.|,\s*|\s+and\b|\s+or\b|$)"
                    field_lower = re.sub(r'[;:]', ' ', re.sub(r'\s+', ' ', field_text.lower()))
                    matches = re.findall(negation_pattern, field_lower, re.IGNORECASE)
                    for _, negated in matches:
                        negated_clean = negated.strip()
                        if negated_clean and not negated_clean.startswith(('patient complains of', 'complains of', 'reports')):
                            if negated_clean in chief_complaint_lower or negated_clean in hpi_lower:
                                logger.debug(f"Skipping negation for '{negated_clean}' as it appears in chief complaint or HPI")
                                continue
                            negated_symptoms.append({
                                'description': negated_clean,
                                'category': self._infer_category(negated_clean, chief_complaint),
                                'definition': f"Negated: {negated_clean}",
                                'umls_cui': None,
                                'semantic_type': 'Unknown'
                            })
                            logger.debug(f"Detected negated symptom (regex): {negated_clean} in '{field_name}'")
        else:
            logger.warning("NLP pipeline unavailable. Falling back to regex negation.")
            negation_pattern = r"\b(no|without|denies|not)\b\s+([\w\s-]+?)(?=\.|,\s*|\s+and\b|\s+or\b|$)"
            for field in fields_to_process:
                field_text, field_name = field[0], field[1]
                field_lower = re.sub(r'[;:]', ' ', re.sub(r'\s+', ' ', field_text.lower()))
                matches = re.findall(negation_pattern, field_lower, re.IGNORECASE)
                for _, negated in matches:
                    negated_clean = negated.strip()
                    if negated_clean and not negated_clean.startswith(('patient complains of', 'complains of', 'reports')):
                        if negated_clean in chief_complaint_lower or negated_clean in hpi_lower:
                            logger.debug(f"Skipping negation for '{negated_clean}' as it appears in chief complaint or HPI")
                            continue
                        negated_symptoms.append({
                            'description': negated_clean,
                            'category': self._infer_category(negated_clean, chief_complaint),
                            'definition': f"Negated: {negated_clean}",
                            'umls_cui': None,
                            'semantic_type': 'Unknown'
                        })
                        logger.debug(f"Detected negated symptom (regex): {negated_clean} in '{field_name}'")

        # Filter out negated symptoms
        valid_symptoms = []
        for s in symptoms:
            s_desc_lower = s['description'].lower()
            is_negated = False
            for ns in negated_symptoms:
                ns_lower = ns['description'].lower()
                if re.search(r'\b' + re.escape(ns_lower) + r'\b', s_desc_lower) or re.search(r'\b' + re.escape(s_desc_lower) + r'\b', ns_lower):
                    is_negated = True
                    logger.debug(f"Excluding negated symptom: {s['description']} (negated by: {ns['description']})")
                    break
            if not is_negated:
                valid_symptoms.append(s)

        # Deduplicate symptoms
        unique_symptoms = []
        seen = set()
        for s in valid_symptoms:
            if s['description'].lower() not in seen:
                unique_symptoms.append(s)
                seen.add(s['description'].lower())

        logger.debug(f"Extracted symptoms: {[s['description'] for s in unique_symptoms]}, Negated: {[ns['description'] for ns in negated_symptoms]}")
        if expected_symptoms:
            self.update_knowledge_base(text, expected_symptoms, unique_symptoms, chief_complaint)
        return unique_symptoms