# departments/nlp/symptom_tracker.py
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import logging
import re
import os
import json
from typing import List, Dict, Set, Tuple, Optional
from departments.nlp.logging_setup import logger
from departments.nlp.knowledge_base import load_knowledge_base

class SymptomTracker:
    def __init__(self, mongo_uri: str = 'mongodb://localhost:27017', db_name: str = 'clinical_db', collection_name: str = 'symptoms'):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.common_symptoms: Dict[str, Dict[str, str]] = {}
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
                if symptom.lower() in ['diabetes', 'hypertension']:
                    continue
                if category and symptom and description and isinstance(category, str) and isinstance(symptom, str) and isinstance(description, str):
                    if category not in self.common_symptoms:
                        self.common_symptoms[category] = {}
                    if symptom not in self.common_symptoms[category]:
                        self.common_symptoms[category][symptom] = description
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

    def _load_json_fallback(self):
        default_symptoms = {
            "respiratory": {
                "facial pain": "Pain in the facial region, often over sinuses",
                "nasal congestion": "Blocked or stuffy nose",
                "purulent nasal discharge": "Yellow or green nasal discharge",
                "fever": "Elevated body temperature",
                "cough": "Persistent or intermittent coughing"
            },
            "neurological": {
                "headache": "Pain in the head or neck",
                "photophobia": "Sensitivity to light"
            },
            "cardiovascular": {
                "chest pain": "Pain or discomfort in the chest",
                "shortness of breath": "Difficulty breathing or feeling out of breath",
                "palpitations": "Irregular or rapid heartbeat"
            },
            "gastrointestinal": {
                "epigastric pain": "Pain in the upper abdomen",
                "nausea": "Feeling of sickness with an inclination to vomit",
                "diarrhea": "Frequent loose or watery stools"
            },
            "musculoskeletal": {
                "knee pain": "Pain in or around the knee joint",
                "back pain": "Pain in the lower or upper back",
                "joint pain": "Pain in joints",
                "pain on movement": "Pain exacerbated by movement",
                "obesity": "Excess body weight contributing to musculoskeletal strain"
            },
            "dermatological": {
                "rash": "Skin eruption or redness"
            },
            "sensory": {
                "hearing loss": "Reduced ability to hear",
                "vision changes": "Altered visual perception"
            },
            "hematologic": {
                "bleeding": "Abnormal bleeding or bruising"
            },
            "endocrine": {
                "weight changes": "Unexplained weight gain or loss",
                "thirst": "Excessive thirst"
            },
            "genitourinary": {
                "urinary changes": "Changes in urination frequency or quality"
            },
            "psychiatric": {
                "mood changes": "Altered mood or emotional state"
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
                        for symptom, desc in symptoms.items():
                            if symptom.lower() in ['diabetes', 'hypertension']:
                                continue
                            if isinstance(symptom, str) and isinstance(desc, str) and symptom and desc:
                                valid_symptoms[category][symptom] = desc
                            else:
                                logger.warning(f"Skipping invalid symptom {symptom} in {category}: {desc}")
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

    def add_symptom(self, category: str, symptom: str, description: str):
        try:
            client = MongoClient(self.mongo_uri)
            db = client[self.db_name]
            collection = db[self.collection_name]
            collection.update_one(
                {'category': category, 'symptom': symptom},
                {'$set': {'description': description}},
                upsert=True
            )
            if category not in self.common_symptoms:
                self.common_symptoms[category] = {}
            self.common_symptoms[category][symptom] = description
            logger.info(f"Added symptom '{symptom}' to category '{category}'.")
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
        missing_symptoms = [s.lower() for s in expected_symptoms if s.lower() not in extracted_desc]
        if not missing_symptoms:
            logger.info("All expected symptoms matched. No knowledge base update needed.")
            return

        logger.warning(f"Missing symptoms: {missing_symptoms}. Initiating knowledge base update.")
        synonyms_json_path = os.path.join(os.path.dirname(__file__), "knowledge_base", "synonyms.json")
        symptoms_json_path = os.path.join(os.path.dirname(__file__), "knowledge_base", "symptoms.json")
        
        try:
            with open(synonyms_json_path, 'r') as f:
                synonyms = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            synonyms = {}

        note_words = set(re.findall(r'\b[\w\s-]+\b', note_text.lower()))
        for symptom in missing_symptoms:
            possible_synonyms = [
                w for w in note_words
                if w not in extracted_desc and any(kw in w for kw in symptom.split()) and len(w) > 3
            ]
            if possible_synonyms:
                if symptom not in synonyms:
                    synonyms[symptom] = []
                synonyms[symptom].extend([s for s in possible_synonyms if s not in synonyms[symptom]])
                logger.info(f"Added synonyms for '{symptom}': {possible_synonyms}")

                category = self._infer_category(symptom, chief_complaint)
                description = f"Automatically added: {symptom}"
                if category not in self.common_symptoms:
                    self.common_symptoms[category] = {}
                if symptom not in self.common_symptoms[category]:
                    self.common_symptoms[category][symptom] = description
                    self.add_symptom(category, symptom, description)
                    logger.info(f"Added symptom '{symptom}' to category '{category}' in common_symptoms.")

        try:
            with open(synonyms_json_path, 'w') as f:
                json.dump(synonyms, f, indent=2)
            logger.info(f"Updated {synonyms_json_path} with new synonyms.")
        except Exception as e:
            logger.error(f"Failed to update {synonyms_json_path}: {str(e)}")

        self._update_symptoms_json()

    def _infer_category(self, symptom: str, chief_complaint: str) -> str:
        symptom_lower = symptom.lower()
        if any(kw in symptom_lower for kw in ['nasal', 'sinus', 'facial', 'congestion', 'discharge']):
            return "respiratory"
        if any(kw in symptom_lower for kw in ['headache', 'dizziness', 'photophobia']):
            return "neurological"
        if any(kw in symptom_lower for kw in ['chest', 'palpitations', 'shortness of breath']):
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

    def get_symptoms_by_category(self, category: str) -> Dict[str, str]:
        return self.common_symptoms.get(category, {})

    def search_symptom(self, symptom: str) -> Tuple[Optional[str], Optional[str]]:
        for category, symptoms in self.common_symptoms.items():
            for s, desc in symptoms.items():
                if s.lower() == symptom.lower():
                    return category, desc
        return None, None

    def get_all_symptoms(self) -> Set[str]:
        return {symptom for category in self.common_symptoms.values() for symptom in category.keys()}

    def get_categories(self) -> List[str]:
        return list(self.common_symptoms.keys())

    def process_note(self, note, chief_complaint: str, expected_symptoms: List[str] = None) -> List[Dict]:
        logger.debug(f"Processing note ID {getattr(note, 'id', 'unknown')} for symptoms with chief complaint: {chief_complaint}")
        text = f"{note.situation or ''} {note.hpi or ''} {note.assessment or ''} {note.aggravating_factors or ''} {note.alleviating_factors or ''}".lower().strip()
        if not text:
            logger.warning("No text available for symptom extraction")
            return []
        text = re.sub(r'[;:]', ' ', re.sub(r'\s+', ' ', text))
        symptoms = []
        matched_terms = set()
        for category, symptom_dict in self.common_symptoms.items():
            for symptom, description in symptom_dict.items():
                symptom_lower = symptom.lower()
                desc_lower = description.lower()
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
                        'definition': description,
                        'duration': duration,
                        'severity': severity,
                        'location': location
                    })
                    matched_terms.add(matched_term)
                    logger.debug(f"Matched symptom: {symptom} (category: {category}, term: {matched_term})")
        negation_pattern = r"\b(no|without|denies|not)\b\s+([\w\s-]+?)(?=\.|,\s*|\s+and\b|\s+or\b|\s+no\b|\s+without\b|\s+denies\b|\s+not\b|$)"
        negated_symptoms = []
        for field in [note.situation or '', note.hpi or '', note.assessment or '', note.aggravating_factors or '', note.alleviating_factors or '']:
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