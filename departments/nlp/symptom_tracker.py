import re
from typing import List, Dict, Tuple, Optional, Set
import medspacy
from pymongo import MongoClient
import torch
import numpy as np
from datetime import datetime

from departments.nlp.nlp_utils import preprocess_text, deduplicate, get_umls_cui, get_negated_symptoms
from departments.nlp.nlp_pipeline import extract_aggravating_alleviating, extract_clinical_phrases, get_semantic_types, get_nlp, search_local_umls_cui
from departments.nlp.knowledge_base_io import load_knowledge_base
from departments.nlp.nlp_common import clean_term, FALLBACK_CUI_MAP
from departments.nlp.logging_setup import get_logger
from departments.nlp.nlp_utils import search_local_umls_cui
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
        self.nlp = get_nlp()  # Use get_nlp from nlp_pipeline.py
        self.knowledge_base = load_knowledge_base()
        self.synonyms = self.knowledge_base.get('synonyms', {})
        self.medical_terms = self.knowledge_base.get('medical_terms', [])
        self.stop_terms = self.knowledge_base.get('medical_stop_words', set()).union({
            'the', 'and', 'or', 'is', 'started', 'days', 'weeks', 'months', 'years', 'ago',
            'taking', 'makes', 'alleviates', 'foods', 'fats', 'strong', 'tea', 'licking', 'salt', 'worse'
        })
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
                    'term': symptom_clean,
                    'symptom': symptom_clean,
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

    from departments.nlp.nlp_utils import get_umls_cui
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
        # Fallback to context-based inference (without embeddings)
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
        for category, terms in self.knowledge_base.get('symptoms', {}).items():
            symptoms.update(term.lower() for term in terms.keys())
        try:
            if self.collection is not None:
                cursor = self.collection.find({}, {'term': 1})
                symptoms.update(doc['term'].lower() for doc in cursor if 'term' in doc and doc['term'])
        except Exception as e:
            logger.error(f"Failed to retrieve symptoms from MongoDB: {e}", exc_info=True)
        logger.debug(f"Retrieved {len(symptoms)} unique symptoms")
        return symptoms

    def extract_duration(self, text: str, symptom: str) -> str:
        """Extract duration of a symptom from text."""
        if not text or not symptom:
            return 'Unknown'
        symptom_clean = clean_term(preprocess_text(symptom)).lower()
        # Compile patterns with re.escape to handle special characters and support hyphenated durations
        patterns = [
            re.compile(rf'\b{re.escape(symptom_clean)}\s*(?:started|for)\s*(\d+\s*-?(?:day|week|month|year)s?\s*(?:ago)?)', re.IGNORECASE),
            re.compile(rf'(\d+\s*-?(?:day|week|month|year)s?)\s*(?:of|with)\s*{re.escape(symptom_clean)}', re.IGNORECASE)
        ]
        for pattern in patterns:
            match = pattern.search(text.lower())
            if match:
                logger.debug(f"Extracted duration for '{symptom_clean}': {match.group(1)}")
                return match.group(1).capitalize()
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
        # Compile pattern to support hyphenated locations and improve performance
        location_pattern = re.compile(r'\b(in|on|of|at)\s+([\w-]+\s*(?:[\w-]+\s*){0,2})\b', re.IGNORECASE)
        symptom_clean = clean_term(preprocess_text(symptom)).lower()
        text_clean = clean_term(preprocess_text(text)).lower()
        matches = location_pattern.findall(text_clean)
        for prep, location in matches:
            # Use regex to ensure exact word match for location
            if symptom_clean in text_clean and re.search(r'\b' + re.escape(location) + r'\b', text_clean):
                logger.debug(f"Extracted location for '{symptom_clean}': {location}")
                return location.capitalize()
        return 'Unknown'

    def validate_symptom_string(self, symptom: str) -> List[str]:
        if not symptom or not isinstance(symptom, str) or len(symptom.strip()) < 2:
            logger.debug(f"Invalid symptom string: {symptom}")
            return []
        cleaned = clean_term(preprocess_text(symptom)).lower()
        if not cleaned:
            logger.warning(f"Empty symptom string after cleaning: {symptom}")
            return []

        # Use extract_clinical_phrases instead of noun_chunks
        phrases = extract_clinical_phrases(cleaned)
        result = []
        stop_terms = {'of', 'and', 'the', 'with', 'patient', 'complains', 'reports', 'has', 'taking', 'makes', 'worse', 'alleviates'}

        # Filter phrases
        for phrase in phrases:
            if phrase in self.get_all_symptoms() or phrase in FALLBACK_CUI_MAP:
                result.append(phrase)
                logger.debug(f"Matched phrase: {phrase}")
            elif not any(term in stop_terms for term in phrase.split()):
                result.append(phrase)

        # Fallback to single tokens for unmatched terms
        doc = self.nlp(cleaned)
        for token in doc:
            token_text = token.text.lower()
            if token_text not in stop_terms and token_text not in ' '.join(result):
                result.append(token_text)

        # Deduplicate while preserving order
        seen = set()
        result = [x for x in result if not (x in seen or seen.add(x))]
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
        
        doc = self.nlp(text)
        negated_symptoms = set()
        for ent in doc.ents:
            if ent._.is_negated:
                negated_clean = clean_term(preprocess_text(ent.text)).lower()
                if negated_clean and negated_clean not in chief_complaint.lower():
                    negated_symptoms.add(negated_clean)
                    logger.debug(f"Detected negated symptom: '{negated_clean}'")
        return negated_symptoms

    def process_note(self, note, chief_complaint: str, expected_symptoms: List[str] = None) -> List[Dict]:
        """Process a clinical note to extract symptoms."""
        logger.debug(f"Processing note ID {getattr(note, 'id', 'unknown')}")
        start_time = datetime.utcnow()

        # Input validation
        if expected_symptoms is None:
            expected_symptoms = []
        if not isinstance(chief_complaint, str):
            chief_complaint = ""
            logger.warning("Invalid chief_complaint: setting to empty string")
        if not isinstance(expected_symptoms, list):
            expected_symptoms = []
            logger.warning("Invalid expected_symptoms: setting to empty list")

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

        try:
            # Initialize symptom_terms
            symptom_terms = []

            # Extract phrases using validate_symptom_string and extract_clinical_phrases
            text_clean = preprocess_text(text)
            if not text_clean:
                logger.warning("Text is empty after preprocessing")
                return []

            # Use validate_symptom_string for initial extraction
            symptom_terms = self.validate_symptom_string(text_clean)

            # Enhance with extract_clinical_phrases for multi-word phrases
            extracted_phrases = extract_clinical_phrases([text, chief_complaint] if chief_complaint.strip() else [text])
            note_phrases = extracted_phrases[0] if extracted_phrases else []
            cc_phrases = extracted_phrases[1] if len(extracted_phrases) > 1 and chief_complaint.strip() else []
            symptom_terms.extend(note_phrases + cc_phrases)

            # Add expected symptoms
            symptom_terms.extend(clean_term(preprocess_text(s)) for s in expected_symptoms if isinstance(s, str) and clean_term(preprocess_text(s)))

            # Remove duplicates
            symptom_terms = list(set(t for t in symptom_terms if t))

            # Filter out non-symptom terms (negated and stop terms)
            non_symptom_terms = get_negated_symptoms(text, self.nlp)
            non_symptom_terms = {clean_term(t) for t in non_symptom_terms if clean_term(t)}
            non_symptom_terms.update(self.stop_terms)
            symptom_terms = [t for t in symptom_terms if t and t not in non_symptom_terms]
            logger.debug(f"Extracted {len(symptom_terms)} symptom terms after filtering: {symptom_terms[:50]}...")

            if not symptom_terms:
                logger.info(f"No valid symptoms found in note, took {(datetime.utcnow() - start_time).total_seconds():.3f} seconds")
                return []

            # Map symptoms to UMLS CUIs
            cui_results = search_local_umls_cui(symptom_terms)
            valid_cuis = [cui for cui in cui_results.values() if cui and isinstance(cui, str) and cui.startswith('C')]
            semantic_types = get_semantic_types(valid_cuis) if valid_cuis else {}

            symptoms = []
            matched_terms = set()
            for symptom_term in symptom_terms:
                symptom_clean = preprocess_text(symptom_term).lower()
                if not symptom_clean or symptom_clean in matched_terms:
                    continue

                # Search for symptom in knowledge base or MongoDB
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

                # Extract additional attributes
                aggravating = extract_aggravating_alleviating(getattr(note, 'aggravating_factors', '') or '', 'aggravating')
                alleviating = extract_aggravating_alleviating(getattr(note, 'alleviating_factors', '') or '', 'alleviating')

                symptoms.append({
                    'description': symptom_clean,
                    'category': category,
                    'definition': description,
                    'duration': self.extract_duration(text_clean, symptom_clean),
                    'severity': self.classify_severity(text_clean),
                    'location': self.extract_location(text_clean, symptom_clean),
                    'aggravating': aggravating.lower().strip() if isinstance(aggravating, str) else 'Unknown',
                    'alleviating': alleviating.lower().strip() if isinstance(alleviating, str) else 'Unknown',
                    'umls_cui': cui,
                    'semantic_type': semantic_type
                })
                matched_terms.add(symptom_clean)

                # Handle synonyms
                synonym_list = self.synonyms.get(symptom_clean, [])
                for synonym in synonym_list:
                    if re.search(r'\b' + re.escape(preprocess_text(synonym).lower()) + r'\b', text_clean):
                        matched_terms.add(synonym.lower())

            # NLP-based extraction for additional entities
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
                            cui = info.get('umls_cui', '')
                            semantic_type = info.get('semantic_type', 'Unknown')
                            description = info['description']
                        symptoms.append({
                            'description': ent_text,
                            'category': category,
                            'definition': description,
                            'duration': self.extract_duration(text_clean, ent_text),
                            'severity': self.classify_severity(text_clean),
                            'location': self.extract_location(text_clean, ent_text),
                            'aggravating': aggravating.lower().strip() if isinstance(aggravating, str) else 'Unknown',
                            'alleviating': alleviating.lower().strip() if isinstance(alleviating, str) else 'Unknown',
                            'umls_cui': cui,
                            'semantic_type': semantic_type
                        })
                        matched_terms.add(ent_text)
                        logger.debug(f"NLP extracted symptom: {ent_text} (CUI: {cui})")
                except Exception as e:
                    logger.error(f"Error processing text with NLP: {e}", exc_info=True)

            # Filter out negated symptoms
            negated_symptoms = self.get_negated_symptoms(note, chief_complaint)
            valid_symptoms = [
                s for s in symptoms
                if not any(preprocess_text(negated).lower() in preprocess_text(s['description']).lower() for negated in negated_symptoms)
            ]

            # Deduplicate symptoms
            symptom_descriptions = tuple(s['description'] for s in valid_symptoms)
            deduped_descriptions = deduplicate(symptom_descriptions, self.synonyms)
            unique_symptoms = [s for s in valid_symptoms if s['description'] in deduped_descriptions]

            # Update knowledge base with expected symptoms
            if expected_symptoms:
                self.update_knowledge_base(unique_symptoms, expected_symptoms, unique_symptoms, chief_complaint)

            logger.info(f"Processed note, extracted {len(unique_symptoms)} symptoms, took {(datetime.utcnow() - start_time).total_seconds():.3f} seconds")
            return unique_symptoms

        except Exception as e:
            logger.error(f"Error processing note: {e}", exc_info=True)
            return []