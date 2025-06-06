import re
from typing import List, Dict, Tuple, Optional, Set
import medspacy
from pymongo import MongoClient
import torch
from datetime import datetime

from departments.nlp.nlp_utils import preprocess_text, deduplicate, get_umls_cui
from departments.nlp.nlp_pipeline import clean_term, extract_aggravating_alleviating, FALLBACK_CUI_MAP
from departments.nlp.knowledge_base_io import load_knowledge_base
from departments.nlp.logging_setup import get_logger
from departments.nlp.config import MONGO_URI, DB_NAME, SYMPTOMS_COLLECTION

logger = get_logger(__name__)

class SymptomTracker:
    """Track and analyze symptoms from clinical notes using MongoDB and NLP."""
    def __init__(self, mongo_uri: str = MONGO_URI, db_name: str = DB_NAME, symptom_collection: str = SYMPTOMS_COLLECTION):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = symptom_collection
        self.client = None
        self.collection = None
        self.nlp = medspacy.load()
        self.knowledge_base = load_knowledge_base()
        self.synonyms = self.knowledge_base.get('synonyms', {})
        self.medical_terms = self.knowledge_base.get('medical_terms', [])
        try:
            self.client = MongoClient(mongo_uri)
            self.collection = self.client[db_name][symptom_collection]
            logger.info(f"Connected to MongoDB: {db_name}.{symptom_collection}")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}", exc_info=True)
            self.collection = None

    def search_symptom(self, symptom: str) -> Tuple[Optional[str], Optional[Dict]]:
        """Search for symptom in the knowledge base or MongoDB."""
        symptom_clean = clean_term(preprocess_text(symptom)).lower()
        if not symptom_clean:
            logger.debug(f"Empty symptom after cleaning: {symptom}")
            return None, None
        try:
            # Search knowledge base
            for category, terms in self.knowledge_base.get('symptoms', {}).items():
                for term, data in terms.items():
                    if symptom_clean == term.lower():
                        logger.debug(f"Found symptom '{symptom_clean}' in knowledge base, category: {category}")
                        return category, data
            # Search MongoDB
            if self.collection is not None:
                # Try both 'term' and 'symptom' for backward compatibility
                result = self.collection.find_one({"$or": [{"term": symptom_clean}, {"symptom": symptom_clean}]})
                if result:
                    logger.debug(f"Found symptom '{symptom}' in MongoDB, category: {result.get('category')}")
                    return result.get('category', 'Uncategorized'), {
                        'description': result.get('description', symptom_clean),
                        'umls_cui': result.get('cui'),
                        'semantic_type': result.get('semantic_type', 'Unknown')
                    }
            logger.debug(f"No match found for symptom '{symptom_clean}'")
            return None, None
        except Exception as e:
            logger.error(f"Search failed for symptom '{symptom}': {e}", exc_info=True)
            return None, None

    def add_symptom(self, category: str, symptom: str, description: str, cui: Optional[str], semantic_type: Optional[str]) -> None:
        """Add a new symptom to MongoDB."""
        if not symptom or not isinstance(symptom, str):
            logger.warning(f"Invalid symptom for adding: {symptom}")
            return
        symptom_clean = clean_term(preprocess_text(symptom)).lower()
        if not symptom_clean:
            logger.debug(f"Empty symptom after cleaning: {symptom}")
            return
        try:
            if self.collection is not None:
                doc = {
                    'term': symptom_clean,  # Use 'term' for unique index compatibility
                    'symptom': symptom_clean,  # For backward compatibility
                    'category': category or 'Uncategorized',
                    'description': description or symptom_clean,
                    'cui': cui,
                    'semantic_type': semantic_type or 'Unknown',
                    'timestamp': datetime.utcnow()
                }
                self.collection.update_one(
                    {'term': symptom_clean},
                    {'$set': doc},
                    upsert=True
                )
                logger.debug(f"Added/updated symptom '{symptom_clean}' to MongoDB")
        except Exception as e:
            logger.error(f"Failed to add symptom '{symptom_clean}' to MongoDB: {e}", exc_info=True)

    def _get_umls_cui(self, symptom: str) -> Tuple[Optional[str], Optional[str]]:
        """Retrieve UMLS CUI and semantic type for a symptom."""
        return get_umls_cui(symptom)

    def _infer_category(self, symptom: str, context: str) -> str:
        """Infer symptom category based on context."""
        symptom_clean = clean_term(preprocess_text(symptom)).lower()
        context_clean = clean_term(preprocess_text(context)).lower()
        if not symptom_clean:
            return 'Uncategorized'
        for category, terms in self.knowledge_base.get('symptoms', {}).items():
            for term in terms:
                if symptom_clean == term.lower():
                    return category
        # Fallback to context-based inference
        if 'head' in symptom_clean or 'head' in context_clean:
            return 'Neurological'
        if 'fever' in symptom_clean or 'chills' in context_clean:
            return 'Systemic'
        if 'nausea' in symptom_clean or 'vomiting' in context_clean:
            return 'Gastrointestinal'
        logger.debug(f"Could not infer category for symptom '{symptom_clean}', context: {context_clean[:50]}...")
        return 'Uncategorized'

    def get_all_symptoms(self) -> Set[str]:
        """Retrieve all symptoms from knowledge base and MongoDB."""
        symptoms = set()
        # Knowledge base symptoms
        for category, terms in self.knowledge_base.get('symptoms', {}).items():
            symptoms.update(term.lower() for term in terms.keys())
        # MongoDB symptoms
        try:
            if self.collection is not None:
                cursor = self.collection.find({}, {'term': 1})
                symptoms.update(doc['term'].lower() for doc in cursor if 'term' in doc and doc['term'])
        except Exception as e:
            logger.error(f"Failed to retrieve symptoms from MongoDB: {e}", exc_info=True)
        logger.debug(f"Retrieved {len(symptoms)} unique symptoms")
        return symptoms

    def extract_duration(self, text: str) -> str:
        """Extract duration from text."""
        if not text:
            return 'Unknown'
        patterns = [
            r'(\d+\s*(?:day|week|month|year)s?\s*(?:ago)?)',
            r'(since\s*\w+\s*\d{4})',
            r'(for\s*\d+\s*(?:day|week|month|year)s?)'
        ]
        for pattern in patterns:
            match = re.search(pattern, text.lower(), re.IGNORECASE)
            if match:
                logger.debug(f"Extracted duration: {match.group(0)}")
                return match.group(0).capitalize()
        return 'Unknown'

    def classify_severity(self, text: str) -> str:
        """Classify symptom severity based on text."""
        if not text:
            return 'Unknown'
        severe_keywords = ['severe', 'intense', 'debilitating', 'unbearable']
        moderate_keywords = ['moderate', 'persistent', 'recurrent']
        for keyword in severe_keywords:
            if keyword in text.lower():
                return 'Severe'
        for keyword in moderate_keywords:
            if keyword in text.lower():
                return 'Moderate'
        logger.debug("No severity keywords found")
        return 'Mild'

    def extract_location(self, text: str, symptom: str) -> str:
        """Extract symptom location from text."""
        if not text or not symptom:
            return 'Unknown'
        location_pattern = r'\b(in|on|of|at)\s+(\w+\s*(?:\w+\s*){0,2})\b'
        symptom_clean = clean_term(preprocess_text(symptom)).lower()
        text_clean = clean_term(preprocess_text(text)).lower()
        matches = re.findall(location_pattern, text_clean, re.IGNORECASE)
        for prep, location in matches:
            if symptom_clean in text_clean and location in text_clean:
                logger.debug(f"Extracted location for '{symptom_clean}': {location}")
                return location.capitalize()
        return 'Unknown'

    def validate_symptom_string(self, symptom: str) -> List[str]:
        """Validate and clean symptom strings."""
        if not symptom or not isinstance(symptom, str) or len(symptom.strip()) < 2:
            logger.debug(f"Invalid symptom string: {symptom}")
            return []
        cleaned = clean_term(preprocess_text(symptom)).lower()
        if not cleaned:
            logger.warning(f"Empty symptom string after cleaning: {symptom}")
            return []
        # Split and deduplicate
        terms = cleaned.split()
        terms = list(dict.fromkeys(terms))  # Remove duplicates
        # Filter out stop terms
        stop_terms = {'of', 'and', 'the', 'with', 'patient', 'complains', 'reports', 'has', 'taking', 'makes', 'worse', 'alleviates'}
        valid_terms = [t for t in terms if t not in stop_terms]
        # Group multi-word symptoms
        result = []
        i = 0
        while i < len(valid_terms):
            matched = False
            for j in range(3, 0, -1):  # Try 3, 2, 1 word combinations
                if i + j <= len(valid_terms):
                    phrase = ' '.join(valid_terms[i:i+j]).lower()
                    if phrase in self.get_all_symptoms() or phrase in FALLBACK_CUI_MAP:
                        result.append(phrase)
                        i += j
                        matched = True
                        logger.debug(f"Matched multi-word symptom: {phrase}")
                        break
            if not matched:
                result.append(valid_terms[i])
                i += 1
        logger.debug(f"Validated symptom string '{symptom}' -> {result}")
        return result

    def update_knowledge_base(self, symptoms_output: List[Dict], expected: List[str], extracted: List[Dict], chief_complaint: str) -> None:
        """Update knowledge base with new symptoms."""
        if not extracted:
            logger.debug("No extracted symptoms to update knowledge base")
            return
        expected_clean = {clean_term(preprocess_text(s)).lower() for s in expected}
        extracted_clean = {clean_term(preprocess_text(s['description'])).lower() for s in extracted}
        missing = expected_clean - extracted_clean
        if missing:
            logger.warning(f"Missing expected symptoms: {missing}")
            for symptom in missing:
                category = self._infer_category(symptom, chief_complaint)
                cui, semantic_type = self._get_umls_cui(symptom)
                description = f"UMLS-derived: {symptom}" if cui else f"Pending UMLS review: {symptom}"
                self.add_symptom(category, symptom, description, cui, semantic_type)
                logger.debug(f"Added missing symptom '{symptom}' to knowledge base")

    def get_negated_symptoms(self, note, chief_complaint: str) -> Set[str]:
        """Extract negated symptoms from note."""
        logger.debug(f"Extracting negated symptoms for note ID {getattr(note, 'id', 'unknown')}")
        # Construct text from note attributes, including medical_history
        attributes = [
            str(getattr(note, 'situation', '') or ''),
            str(getattr(note, 'hpi', '') or ''),
            str(getattr(note, 'assessment', '') or ''),
            str(getattr(note, 'aggravating_factors', '') or ''),
            str(getattr(note, 'alleviating_factors', '') or ''),
            str(getattr(note, 'Symptoms', '') or getattr(note, 'symptoms', '') or ''),
            str(getattr(note, 'medical_history', '') or '')
        ]
        text = ' '.join(attr for attr in attributes if attr).strip()
        if not text:
            logger.warning("No text available for negated symptom extraction")
            return set()

        text_clean = preprocess_text(text).lower()
        text_clean = re.sub(r'[;:]', ' ', re.sub(r'\s+', ' ', text_clean))
        chief_complaint_lower = preprocess_text(chief_complaint).lower()
        hpi_lower = preprocess_text(getattr(note, 'hpi', '') or '').lower()

        negated_symptoms = set()
        negation_pattern = r'\b(no|without|denies|not)\b\s+([\w\s-]+?)(?=\s*\.|,\s*|\s+and\b|\s+or\b|$)'
        matches = re.findall(negation_pattern, text_clean, re.IGNORECASE)
        for _, negated in matches:
            negated_clean = preprocess_text(negated.strip()).lower()
            if negated_clean and negated_clean not in chief_complaint_lower and negated_clean not in hpi_lower:
                negated_symptoms.add(negated_clean)
                logger.debug(f"Detected negated symptom: '{negated_clean}'")
        return negated_symptoms

    def process_note(self, note, chief_complaint: str, expected_symptoms: List[str] = None) -> List[Dict]:
        """Process a clinical note to extract symptoms."""
        logger.debug(f"Processing note ID {getattr(note, 'id', 'unknown')}")
        # Safely construct text from note attributes
        attributes = [
            str(getattr(note, 'situation', '') or ''),
            str(getattr(note, 'hpi', '') or ''),
            str(getattr(note, 'assessment', '') or ''),
            str(getattr(note, 'aggravating_factors', '') or ''),
            str(getattr(note, 'alleviating_factors', '') or ''),
            str(getattr(note, 'Symptoms', '') or getattr(note, 'symptoms', '') or '')
        ]
        text = ' '.join(attr for attr in attributes if attr).strip()
        if not text:
            logger.warning("No text available for symptom extraction")
            return []

        # Enhanced preprocessing
        text_clean = preprocess_text(text)
        if not text_clean:
            logger.warning("Text is empty after preprocessing")
            return []
        # Validate and clean symptom strings
        symptom_terms = self.validate_symptom_string(text_clean)
        text_clean = ' '.join(symptom_terms).lower()
        text_clean = re.sub(r'[;:]', ' ', re.sub(r'\s+', ' ', text_clean)).lower()
        logger.debug(f"Cleaned text: {text_clean[:100]}...")

        symptoms = []
        matched_terms = set()
        chief_complaint_lower = preprocess_text(chief_complaint).lower()
        hpi_lower = preprocess_text(getattr(note, 'hpi', '') or '').lower()

        try:
            aggravating_text = getattr(note, 'aggravating_factors', '') or ''
            alleviating_text = getattr(note, 'alleviating_factors', '') or ''
            aggravating_result = extract_aggravating_alleviating(aggravating_text, 'aggravating')
            alleviating_result = extract_aggravating_alleviating(alleviating_text, 'alleviating')
            aggravating = aggravating_result.lower().strip() if isinstance(aggravating_result, str) else aggravating_text.lower().strip()
            alleviating = alleviating_result.lower().strip() if isinstance(alleviating_result, str) else alleviating_text.lower().strip()
            logger.debug(f"Extracted aggravating: {aggravating[:50]}..., alleviating: {alleviating[:50]}...")
        except Exception as e:
            logger.error(f"Error extracting aggravating/alleviating factors: {e}", exc_info=True)
            aggravating = 'Unknown'
            alleviating = 'Unknown'

        # Process each validated symptom term
        for symptom_term in symptom_terms:
            symptom_clean = preprocess_text(symptom_term).lower()
            if not symptom_clean:
                continue
            category, info = self.search_symptom(symptom_clean)
            if not category:
                category = self._infer_category(symptom_clean, chief_complaint)
                cui, semantic_type = self._get_umls_cui(symptom_clean)
                description = f"UMLS-derived: {symptom_clean}" if cui else f"Pending UMLS review: {symptom_clean}"
                self.add_symptom(category, symptom_clean, description, cui, semantic_type)
            else:
                cui = info.get('umls_cui')
                semantic_type = info.get('semantic_type', 'Unknown')
                description = info['description']
            symptoms.append({
                'description': symptom_clean,
                'category': category,
                'definition': description,
                'duration': self.extract_duration(text_clean),
                'severity': self.classify_severity(text_clean),
                'location': self.extract_location(text_clean, symptom_clean),
                'aggravating': aggravating,
                'alleviating': alleviating,
                'umls_cui': cui,
                'semantic_type': semantic_type
            })
            matched_terms.add(symptom_clean)
            # Handle synonyms
            synonym_list = self.synonyms.get(symptom_clean, [])
            for synonym in synonym_list:
                if re.search(r'\b' + re.escape(preprocess_text(synonym).lower()) + r'\b', text_clean):
                    matched_terms.add(synonym.lower())

        # NLP-based extraction
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
                        self.add_symptom(category, ent_text, description, cui, semantic_type)
                    else:
                        cui, semantic_type = info.get('umls_cui', ''), info.get('semantic_type', 'Unknown')
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
                    logger.debug(f"NLP extracted symptom: {ent_text} (CUI: {cui})")
            except Exception as e:
                logger.error(f"Error processing text with NLP: {e}", exc_info=True)

        # Negated symptoms
        negated_symptoms = self.get_negated_symptoms(note, chief_complaint)
        valid_symptoms = []
        for s in symptoms:
            s_desc_lower = preprocess_text(s['description']).lower()
            is_negated = any(preprocess_text(negated).lower() in s_desc_lower for negated in negated_symptoms)
            if not is_negated:
                valid_symptoms.append(s)
            else:
                logger.debug(f"Excluding negated symptom: '{s['description']}'")

        # Deduplicate symptoms
        symptom_descriptions = tuple(s['description'] for s in valid_symptoms)
        deduped_descriptions = deduplicate(symptom_descriptions, self.synonyms)
        unique_symptoms = [s for s in valid_symptoms if s['description'] in deduped_descriptions]

        logger.debug(f"Extracted symptoms: {[s['description'] for s in unique_symptoms]}")
        if expected_symptoms:
            self.update_knowledge_base(unique_symptoms, expected_symptoms, unique_symptoms, chief_complaint)
        return unique_symptoms