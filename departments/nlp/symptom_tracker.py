# departments/nlp/symptom_tracker.py
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import logging
import re
import os
import json
from typing import List, Dict, Set, Tuple, Optional

from sqlalchemy import null
from departments.nlp.logging_setup import logger
from departments.nlp.knowledge_base import load_knowledge_base
from departments.nlp.kb_updater import KnowledgeBaseUpdater
from departments.nlp.config import UTS_API_KEY
import requests

class SymptomTracker:
    def __init__(self, mongo_uri: str = 'mongodb://localhost:27017', db_name: str = 'clinical_db', collection_name: str = 'symptoms'):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.uts_api_key = UTS_API_KEY or 'mock_api_key'
        self.common_symptoms: Dict[str, Dict[str, Dict]] = {}  # Updated to store UMLS metadata
        self.synonyms: Dict[str, List[str]] = {}
        try:
            client = MongoClient(mongo_uri)
            client.admin.command('ping')
            db = client[db_name]
            collection = db[collection_name]
            symptoms = collection.find()
            for doc in symptoms:
                category = doc.get('category')
                symptom = doc.get('symptom')
                description = doc.get('description')
                umls_cui = doc.get('umls_cui')
                semantic_type = doc.get('semantic_type', 'Unknown')
                if symptom.lower() in ['diabetes', 'hypertension']:
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
            client.close()
            if not self.common_symptoms:
                logger.warning("No symptoms loaded from MongoDB. Collection may be empty.")
            else:
                logger.info(f"Loaded {sum(len(symptoms) for symptoms in self.common_symptoms.values())} symptoms from MongoDB across {len(self.common_symptoms)} categories.")
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}. Falling back to JSON.")
            self._load_json_fallback()
        except Exception as e:
            logger.error(f"Error loading symptoms from MongoDB: {str(e)}. Falling back to JSON.")
            self._load_json_fallback()
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

    def _load_json_fallback(self):
        default_symptoms = {
            "respiratory": {
                "facial pain": {"description": "Pain in the facial region, often over sinuses", "umls_cui": null, "semantic_type": "Unknown"},
                "nasal congestion": {"description": "Blocked or stuffy nose", "umls_cui": null, "semantic_type": "Unknown"},
                "purulent nasal discharge": {"description": "Yellow or green nasal discharge", "umls_cui": null, "semantic_type": "Unknown"},
                "fever": {"description": "Elevated body temperature", "umls_cui": null, "semantic_type": "Unknown"},
                "cough": {"description": "Persistent or intermittent coughing", "umls_cui": null, "semantic_type": "Unknown"}
            },
            "neurological": {
                "headache": {"description": "Pain in the head or neck", "umls_cui": null, "semantic_type": "Unknown"},
                "photophobia": {"description": "Sensitivity to light", "umls_cui": null, "semantic_type": "Unknown"}
            },
            "cardiovascular": {
                "chest pain": {"description": "Pain or discomfort in the chest", "umls_cui": null, "semantic_type": "Unknown"},
                "shortness of breath": {"description": "Difficulty breathing or feeling out of breath", "umls_cui": null, "semantic_type": "Unknown"},
                "palpitations": {"description": "Irregular or rapid heartbeat", "umls_cui": null, "semantic_type": "Unknown"},
                "chest tightness": {"description": "Pressure or tightness in the chest", "umls_cui": "C0242209", "semantic_type": "Sign or Symptom"}
            },
            "gastrointestinal": {
                "epigastric pain": {"description": "Pain in the upper abdomen", "umls_cui": null, "semantic_type": "Unknown"},
                "nausea": {"description": "Feeling of sickness with an inclination to vomit", "umls_cui": null, "semantic_type": "Unknown"},
                "diarrhea": {"description": "Frequent loose or watery stools", "umls_cui": null, "semantic_type": "Unknown"}
            },
            "musculoskeletal": {
                "knee pain": {"description": "Pain in or around the knee joint", "umls_cui": null, "semantic_type": "Unknown"},
                "back pain": {"description": "Pain in the lower or upper back", "umls_cui": "C0004604", "semantic_type": "Sign or Symptom"},
                "joint pain": {"description": "Pain in joints", "umls_cui": null, "semantic_type": "Unknown"},
                "pain on movement": {"description": "Pain exacerbated by movement", "umls_cui": null, "semantic_type": "Unknown"},
                "obesity": {"description": "Excess body weight contributing to musculoskeletal strain", "umls_cui": "C0028754", "semantic_type": "Disease or Syndrome"}
            },
            "dermatological": {
                "rash": {"description": "Skin eruption or redness", "umls_cui": null, "semantic_type": "Unknown"}
            },
            "sensory": {
                "hearing loss": {"description": "Reduced ability to hear", "umls_cui": null, "semantic_type": "Unknown"},
                "vision changes": {"description": "Altered visual perception", "umls_cui": null, "semantic_type": "Unknown"}
            },
            "hematologic": {
                "bleeding": {"description": "Abnormal bleeding or bruising", "umls_cui": null, "semantic_type": "Unknown"}
            },
            "endocrine": {
                "weight changes": {"description": "Unexplained weight gain or loss", "umls_cui": null, "semantic_type": "Unknown"},
                "thirst": {"description": "Excessive thirst", "umls_cui": null, "semantic_type": "Unknown"}
            },
            "genitourinary": {
                "urinary changes": {"description": "Changes in urination frequency or quality", "umls_cui": null, "semantic_type": "Unknown"}
            },
            "psychiatric": {
                "mood changes": {"description": "Altered mood or emotional state", "umls_cui": null, "semantic_type": "Unknown"}
            }
        }
        json_path = os.path.join(os.path.dirname(__file__), "knowledge_base", "symptoms.json")
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    logger.error(f"Expected dict in {json_path}, got {type(data)}. Using default symptoms.")
                    self.common_symptoms = default_symptoms
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
                    self.common_symptoms = valid_symptoms or default_symptoms
                    logger.info(f"Loaded {sum(len(symptoms) for symptoms in self.common_symptoms.values())} symptoms from {json_path}.")
        except FileNotFoundError:
            logger.error(f"Symptom JSON not found at {json_path}. Using default symptoms.")
            self.common_symptoms = default_symptoms
            self._save_json_fallback(json_path, default_symptoms)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {json_path}: {str(e)}. Using default symptoms.")
            self.common_symptoms = default_symptoms
            self._save_json_fallback(json_path, default_symptoms)

    def _save_json_fallback(self, json_path: str, data: dict):
        try:
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Created default symptom JSON at {json_path}.")
        except Exception as e:
            logger.error(f"Failed to create {json_path}: {str(e)}")

    def _get_umls_cui(self, symptom: str) -> Tuple[Optional[str], Optional[str]]:
        """Retrieve UMLS CUI and semantic type for a symptom."""
        if self.uts_api_key == 'mock_api_key':
            mock_data = {
                'chest tightness': {'cui': 'C0242209', 'semantic_type': 'Sign or Symptom'},
                'back pain': {'cui': 'C0004604', 'semantic_type': 'Sign or Symptom'},
                'obesity': {'cui': 'C0028754', 'semantic_type': 'Disease or Syndrome'}
            }
            return mock_data.get(symptom.lower(), {'cui': None, 'semantic_type': 'Unknown'}).values()
        try:
            ticket_url = "https://uts-ws.nlm.nih.gov/rest/tickets"
            ticket_response = requests.post(ticket_url, data={'apiKey': self.uts_api_key})
            ticket_response.raise_for_status()
            ticket = ticket_response.text
            search_url = "https://uts-ws.nlm.nih.gov/rest/search/current"
            params = {'string': symptom, 'ticket': ticket, 'searchType': 'exact', 'sabs': 'SNOMEDCT_US'}
            response = requests.get(search_url, params=params)
            response.raise_for_status()
            data = response.json()
            if data.get('result', {}).get('results'):
                cui = data['result']['results'][0]['ui']
                concept_url = f"https://uts-ws.nlm.nih.gov/rest/content/current/CUI/{cui}"
                concept_response = requests.get(concept_url, params={'ticket': ticket})
                concept_response.raise_for_status()
                concept_data = concept_response.json()
                semantic_type = concept_data['result'].get('semanticTypes', [{}])[0].get('name', 'Unknown')
                return cui, semantic_type
            return None, 'Unknown'
        except Exception as e:
            logger.error(f"UMLS CUI retrieval failed for '{symptom}': {str(e)}")
            return None, 'Unknown'

    def add_symptom(self, category: str, symptom: str, description: str, umls_cui: Optional[str] = None, semantic_type: Optional[str] = 'Unknown'):
        try:
            client = MongoClient(self.mongo_uri)
            db = client[self.db_name]
            collection = db[self.collection_name]
            collection.update_one(
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
            client.close()
            self._update_symptoms_json()
        except Exception as e:
            logger.error(f"Error adding symptom to MongoDB: {str(e)}")

    def remove_symptom(self, category: str, symptom: str):
        try:
            client = MongoClient(self.mongo_uri)
            db = client[self.db_name]
            collection = db[self.collection_name]
            collection.delete_one({'category': category, 'symptom': symptom})
            if category in self.common_symptoms and symptom in self.common_symptoms[category]:
                del self.common_symptoms[category][symptom]
                if not self.common_symptoms[category]:
                    del self.common_symptoms[category]
                logger.info(f"Removed symptom '{symptom}' from category '{category}'.")
            client.close()
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
        extracted_cuis = {s.get('umls_cui') for s in extracted_symptoms if s.get('umls_cui')}
        missing_symptoms = [s.lower() for s in expected_symptoms if s.lower() not in extracted_desc]
        if not missing_symptoms:
            logger.info("All expected symptoms matched. No knowledge base update needed.")
            return

        logger.warning(f"Missing symptoms: {missing_symptoms}. Initiating knowledge base update.")
        for symptom in missing_symptoms:
            if self.kb_updater.is_new_symptom(symptom):
                category = self._infer_category(symptom, chief_complaint)
                synonyms = self.kb_updater.generate_synonyms(symptom)
                cui, semantic_type = self._get_umls_cui(symptom)
                description = f"Automatically added: {symptom}"
                self.kb_updater.update_knowledge_base(symptom, category, synonyms, note_text)
                self.add_symptom(category, symptom, description, cui, semantic_type)
                logger.info(f"Updated knowledge base with new symptom '{symptom}' (category: {category}, synonyms: {synonyms}, CUI: {cui})")
            else:
                logger.debug(f"Symptom '{symptom}' already exists in knowledge base. Skipping update.")

    def _infer_category(self, symptom: str, chief_complaint: str) -> str:
        symptom_lower = symptom.lower()
        if any(kw in symptom_lower for kw in ['nasal', 'sinus', 'facial', 'congestion', 'discharge', 'cough']):
            return "respiratory"
        if any(kw in symptom_lower for kw in ['headache', 'dizziness', 'photophobia']):
            return "neurological"
        if any(kw in symptom_lower for kw in ['chest', 'palpitations', 'shortness of breath', 'tightness']):
            return "cardiovascular"
        if any(kw in symptom_lower for kw in ['nausea', 'diarrhea', 'epigastric']):
            return "gastrointestinal"
        if any(kw in symptom_lower for kw in ['joint', 'knee', 'back', 'obesity', 'movement']):
            return "musculoskeletal"
        if 'rash' in symptom_lower:
            return "dermatological"
        if any(kw in symptom_lower for kw in ['hearing', 'vision']):
            return "sensory"
        if 'bleeding' in symptom_lower:
            return "hematologic"
        if any(kw in symptom_lower for kw in ['weight', 'thirst']):
            return "endocrine"
        if 'urinary' in symptom_lower:
            return "genitourinary"
        if 'mood' in symptom_lower:
            return "psychiatric"
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
        text = re.sub(r'[;:]', ' ', re.sub(r'\s+', ' ', text))
        symptoms = []
        matched_terms = set()
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
                    if re.search(r'(?:\b|\s)' + re.escape(pattern) + r'(?:\b|\s)', text) and pattern not in matched_terms:
                        matched = True
                        matched_term = pattern
                        break
                if matched:
                    duration = '2-3 weeks' if 'two weeks' in text or 'three weeks' in text else 'Unknown'
                    severity = 'Moderate' if 'worse' in text else 'Unknown'
                    location = 'Lower back' if 'back' in text else 'Unknown'
                    symptoms.append({
                        'description': symptom,
                        'category': category,
                        'definition': info['description'],
                        'duration': duration,
                        'severity': severity,
                        'location': location,
                        'umls_cui': umls_cui,
                        'semantic_type': semantic_type
                    })
                    matched_terms.add(matched_term)
                    logger.debug(f"Matched symptom: {symptom} (category: {category}, term: {matched_term}, CUI: {umls_cui})")
        # Handle structured symptoms field
        structured_symptoms = getattr(note, 'symptoms', None)
        if structured_symptoms:
            if isinstance(structured_symptoms, str):
                structured_symptoms = [s.strip() for s in structured_symptoms.split(',')]
            for symptom in structured_symptoms:
                if isinstance(symptom, str) and symptom.lower() not in matched_terms:
                    category = self._infer_category(symptom, chief_complaint)
                    cui, semantic_type = self._get_umls_cui(symptom)
                    description = self.common_symptoms.get(category, {}).get(symptom, {}).get('description', f"Automatically detected: {symptom}")
                    if symptom.lower() not in self.get_all_symptoms():
                        self.add_symptom(category, symptom, description, cui, semantic_type)
                        self.kb_updater.update_knowledge_base(symptom, category, self.kb_updater.generate_synonyms(symptom), text)
                        logger.info(f"Added structured symptom '{symptom}' to category '{category}' (CUI: {cui})")
                    symptoms.append({
                        'description': symptom,
                        'category': category,
                        'definition': description,
                        'duration': 'Unknown',
                        'severity': 'Unknown',
                        'location': 'Unknown',
                        'umls_cui': cui,
                        'semantic_type': semantic_type
                    })
                    matched_terms.add(symptom.lower())
        negation_pattern = r"\b(no|without|denies|not)\b\s+([\w\s-]+?)(?=\.|,\s*|\s+and\b|\s+or\b|\s+no\b|\s+without\b|\s+denies\b|\s+not\b|$)"
        negated_symptoms = []
        for field in [note.situation or '', note.hpi or '', note.assessment or '', note.aggravating_factors or '', note.alleviating_factors or '', getattr(note, 'symptoms', '') or '']:
            field_lower = re.sub(r'[;:]', ' ', re.sub(r'\s+', ' ', field.lower()))
            matches = re.findall(negation_pattern, field_lower, re.IGNORECASE)
            for _, negated in matches:
                negated_clean = negated.strip()
                if negated_clean:
                    negated_symptoms.append(negated_clean)
        valid_symptoms = []
        for s in symptoms:
            s_desc_lower = s['description'].lower()
            is_negated = False
            for ns in negated_symptoms:
                ns_lower = ns.lower()
                if re.search(r'(?:\b|\s)' + re.escape(ns_lower) + r'(?:\b|\s)', s_desc_lower) or re.search(r'(?:\b|\s)' + re.escape(s_desc_lower) + r'(?:\b|\s)', ns_lower):
                    is_negated = True
                    logger.debug(f"Excluding negated symptom: {s['description']} (negated by: {ns})")
                    break
            if not is_negated:
                valid_symptoms.append(s)
        unique_symptoms = []
        seen = set()
        for s in valid_symptoms:
            if s['description'].lower() not in seen:
                unique_symptoms.append(s)
                seen.add(s['description'].lower())
        logger.debug(f"Extracted symptoms: {[s['description'] for s in unique_symptoms]}, Negated: {negated_symptoms}")
        if expected_symptoms:
            self.update_knowledge_base(text, expected_symptoms, unique_symptoms, chief_complaint)
        return unique_symptoms