from typing import List, Dict, Set, Tuple, Optional
import torch
import re
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import spacy
import medspacy
from medspacy.target_matcher import TargetRule
from scispacy.linking import EntityLinker
import requests
from tenacity import retry, stop_after_attempt, wait_fixed
from departments.models.medicine import SOAPNote
from departments.models.records import Patient
from departments.nlp.logging_setup import get_logger
from departments.nlp.nlp_pipeline import get_nlp
from departments.nlp.config import SIMILARITY_THRESHOLD, CONFIDENCE_THRESHOLD, EMBEDDING_DIM, UTS_API_KEY, UTS_BASE_URL
from departments.nlp.nlp_utils import embed_text, preprocess_text, deduplicate, get_patient_info
from departments.nlp.helper_functions import extract_duration, classify_severity, extract_location, extract_aggravating_alleviating
from departments.nlp.models.transformer_model import model, tokenizer
from departments.nlp.symptom_tracker import SymptomTracker
from departments.nlp.knowledge_base import load_knowledge_base
from departments.nlp.kb_updater import KnowledgeBaseUpdater

logger = get_logger()

class ClinicalAnalyzer:
    def __init__(self):
        self.model = model
        self.tokenizer = tokenizer
        self.uts_api_key = UTS_API_KEY or 'mock_api_key'
        self.uts_base_url = UTS_BASE_URL
        mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
        db_name = os.getenv('DB_NAME', 'clinical_db')
        kb_prefix = os.getenv('KB_PREFIX', 'kb_')
        self.medical_stop_words: Set[str] = set()
        self.medical_terms: List[Dict] = []
        self.synonyms: Dict[str, List[str]] = {}
        self.clinical_pathways: Dict[str, Dict[str, Dict]] = {}
        self.history_diagnoses: Dict[str, List[str]] = {}
        self.diagnosis_relevance: Dict[str, List[Dict]] = {}
        self.management_config: Dict[str, str] = {}
        self.diagnosis_treatments: Dict[str, Dict] = {}

        # Connect to MongoDB with retry
        @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
        def connect_to_mongo():
            client = MongoClient(mongo_uri)
            client.admin.command('ping')
            return client

        try:
            client = connect_to_mongo()
            db = client[db_name]
            collections = {
                'medical_stop_words': f'{kb_prefix}medical_stop_words',
                'medical_terms': f'{kb_prefix}medical_terms',
                'synonyms': f'{kb_prefix}synonyms',
                'clinical_pathways': f'{kb_prefix}clinical_pathways',
                'history_diagnoses': f'{kb_prefix}history_diagnoses',
                'diagnosis_relevance': f'{kb_prefix}diagnosis_relevance',
                'management_config': f'{kb_prefix}management_config',
                'diagnosis_treatments': f'{kb_prefix}diagnosis_treatments'
            }
            self.medical_stop_words = {doc['word'] for doc in db[collections['medical_stop_words']].find() if 'word' in doc}
            self.medical_terms = [
                {
                    'term': doc['term'],
                    'category': doc.get('category', 'unknown'),
                    'umls_cui': doc.get('umls_cui'),
                    'semantic_type': doc.get('semantic_type', 'Unknown')
                } for doc in db[collections['medical_terms']].find() if 'term' in doc
            ]
            for doc in db[collections['synonyms']].find():
                if 'key' in doc and 'aliases' in doc:
                    self.synonyms[doc['key']] = doc['aliases']
            for doc in db[collections['clinical_pathways']].find():
                if 'category' in doc and 'key' in doc and 'path' in doc:
                    if doc['category'] not in self.clinical_pathways:
                        self.clinical_pathways[doc['category']] = {}
                    self.clinical_pathways[doc['category']][doc['key']] = doc['path']
            for doc in db[collections['history_diagnoses']].find():
                if 'condition' in doc and 'aliases' in doc:
                    self.history_diagnoses[doc['condition']] = doc['aliases']
            for doc in db[collections['diagnosis_relevance']].find():
                if 'condition' in doc and 'required' in doc:
                    self.diagnosis_relevance[doc['condition']] = doc['required']
            for doc in db[collections['management_config']].find():
                if 'key' in doc and 'value' in doc:
                    self.management_config[doc['key']] = doc['value']
            for doc in db[collections['diagnosis_treatments']].find():
                if 'diagnosis' in doc and 'mappings' in doc:
                    self.diagnosis_treatments[doc['diagnosis']] = doc['mappings']
            client.close()
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}. Falling back to JSON.")
            self._load_knowledge_fallback()
        except Exception as e:
            logger.error(f"Error loading from MongoDB: {str(e)}. Falling back to JSON.")
            self._load_knowledge_fallback()

        # Initialize KnowledgeBaseUpdater
        self.kb_updater = KnowledgeBaseUpdater(mongo_uri=mongo_uri, db_name=db_name, kb_prefix=kb_prefix)

        # Validate loaded data
        for name, data in [
            ('medical_stop_words', self.medical_stop_words),
            ('medical_terms', self.medical_terms),
            ('synonyms', self.synonyms),
            ('clinical_pathways', self.clinical_pathways),
            ('history_diagnoses', self.history_diagnoses),
            ('diagnosis_relevance', self.diagnosis_relevance),
            ('management_config', self.management_config),
            ('diagnosis_treatments', self.diagnosis_treatments)
        ]:
            if not data:
                logger.warning(f"No data loaded for {name}. Functionality may be limited.")

        self.diagnoses_list: Set[str] = set()
        for category, pathways in self.clinical_pathways.items():
            for key, path in pathways.items():
                differentials = path.get('differentials', [])
                self.diagnoses_list.update(d.lower() for d in differentials if isinstance(d, str))

        # Initialize SymptomTracker
        self.common_symptoms = SymptomTracker(
            mongo_uri=mongo_uri,
            db_name=db_name,
            collection_name=os.getenv('SYMPTOMS_COLLECTION', 'symptoms')
        )
        if not self.common_symptoms.get_all_symptoms():
            logger.warning("No symptoms loaded into SymptomTracker.")

        # Initialize spaCy with medspacy
        self.nlp = get_nlp()
        self._add_symptom_rules()

    def _add_symptom_rules(self):
        """Add dynamic symptom rules to medspacy_target_matcher from MongoDB."""
        try:
            target_matcher = self.nlp.get_pipe("medspacy_target_matcher")
            client = MongoClient(os.getenv('MONGO_URI', 'mongodb://localhost:27017'))
            db = client[os.getenv('DB_NAME', 'clinical_db')]
            symptoms_collection = db[os.getenv('SYMPTOMS_COLLECTION', 'symptoms')]
            rules = []
            for doc in symptoms_collection.find():
                if 'symptom' in doc and 'umls_cui' in doc:
                    rules.append(
                        TargetRule(
                            literal=doc['symptom'].lower(),
                            category="SYMPTOM",
                            attributes={"cui": doc['umls_cui']}
                        )
                    )
            if rules:
                target_matcher.add(rules)
                logger.debug(f"Added {len(rules)} dynamic symptom rules from MongoDB")
            else:
                logger.warning("No symptom rules found in MongoDB. Relying on spaCy entities.")
            client.close()
        except Exception as e:
            logger.error(f"Failed to add symptom rules: {str(e)}")

    def _load_knowledge_fallback(self):
        """Load knowledge base from JSON if MongoDB fails."""
        kb = load_knowledge_base()
        self.medical_stop_words = kb.get('medical_stop_words', set())
        self.medical_terms = kb.get('medical_terms', [])
        self.synonyms = kb.get('synonyms', {})
        self.clinical_pathways = kb.get('clinical_pathways', {})
        self.history_diagnoses = kb.get('history_diagnoses', {})
        self.diagnosis_relevance = kb.get('diagnosis_relevance', {})
        self.management_config = kb.get('management_config', {})
        self.diagnosis_treatments = kb.get('diagnosis_treatments', {})

    def _get_uts_ticket(self) -> str:
        """Retrieve a single-use ticket for UTS API."""
        if not self.uts_api_key or self.uts_api_key == 'mock_api_key':
            logger.error("UTS_API_KEY is not set or is using mock key. Please configure a valid API key.")
            return ''
        try:
            ticket_url = f"{self.uts_base_url}/tickets"
            response = requests.post(ticket_url, data={'apiKey': self.uts_api_key})
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Failed to get UTS ticket: {str(e)}")
            return ''

    def _get_umls_cui(self, symptom: str) -> Tuple[Optional[str], Optional[str]]:
        """Retrieve UMLS CUI and semantic type for a symptom with caching."""
        symptom_lower = symptom.lower()
        try:
            client = MongoClient(os.getenv('MONGO_URI', 'mongodb://localhost:27017'))
            db = client[os.getenv('DB_NAME', 'clinical_db')]
            cache = db['umls_cache']
            cached = cache.find_one({'symptom': symptom_lower})
            if cached:
                logger.debug(f"Retrieved cached UMLS data for '{symptom_lower}'")
                client.close()
                return cached['cui'], cached['semantic_type']
        except Exception as e:
            logger.error(f"Error accessing UMLS cache: {str(e)}")

        if self.uts_api_key == 'mock_api_key':
            logger.warning("Using mock UMLS data due to missing API key")
            return None, 'Unknown'

        try:
            ticket = self._get_uts_ticket()
            if not ticket:
                return None, 'Unknown'
            search_url = f"{self.uts_base_url}/search/current"
            params = {'string': symptom_lower, 'ticket': ticket, 'searchType': 'exact', 'sabs': 'SNOMEDCT_US'}
            response = requests.get(search_url, params=params)
            response.raise_for_status()
            data = response.json()
            if data.get('result', {}).get('results'):
                cui = data['result']['results'][0]['ui']
                concept_url = f"{self.uts_base_url}/content/current/CUI/{cui}"
                concept_response = requests.get(concept_url, params={'ticket': ticket})
                concept_response.raise_for_status()
                concept_data = concept_response.json()
                semantic_type = concept_data['result'].get('semanticTypes', [{}])[0].get('name', 'Unknown')
                try:
                    cache.insert_one({
                        'symptom': symptom_lower,
                        'cui': cui,
                        'semantic_type': semantic_type
                    })
                    logger.debug(f"Cached UMLS data for '{symptom_lower}'")
                except Exception as e:
                    logger.error(f"Failed to cache UMLS data: {str(e)}")
                finally:
                    client.close()
                return cui, semantic_type
            return None, 'Unknown'
        except Exception as e:
            logger.error(f"UMLS CUI retrieval failed for '{symptom_lower}': {str(e)}")
            return None, 'Unknown'

    def extract_clinical_features(self, note: SOAPNote, expected_symptoms: Optional[List[str]] = None) -> Dict:
        """Extract clinical features from a SOAP note, integrating spaCy and UMLS."""
        if not isinstance(note, SOAPNote):
            logger.error(f"Invalid note type: {type(note)}")
            return {
                'chief_complaint': '',
                'symptoms': [],
                'assessment': '',
                'context': {},
                'hpi': '',
                'history': '',
                'medications': '',
                'recommendation': '',
                'additional_notes': '',
                'aggravating_factors': '',
                'alleviating_factors': ''
            }

        try:
            features = {
                'chief_complaint': (note.situation or '').lower().strip() or 'Unknown',
                'hpi': (note.hpi or '').lower().strip(),
                'history': (note.medical_history or '').lower().strip(),
                'medications': (note.medication_history or '').lower().strip(),
                'assessment': (note.assessment or '').lower().strip(),
                'recommendation': (note.recommendation or '').lower().strip(),
                'additional_notes': (note.additional_notes or '').lower().strip(),
                'aggravating_factors': (note.aggravating_factors or '').lower().strip(),
                'alleviating_factors': (note.alleviating_factors or '').lower().strip(),
                'symptoms': []
            }

            # Combine text for processing
            text = f"{features['chief_complaint']} {features['hpi']} {features['assessment']} {features['aggravating_factors']} {features['alleviating_factors']}".strip()
            if not text:
                logger.warning(f"No valid text for note ID {note.id}, skipping NLP processing")
                text = features['chief_complaint'] or 'Unknown'

            # spaCy-based symptom extraction
            spacy_symptoms = []
            try:
                doc = self.nlp(text)
                for ent in doc.ents:
                    for umls_ent in getattr(ent._, 'kb_ents', []):
                        cui, score = umls_ent[0], umls_ent[1]
                        if score < 0.7:  # Confidence threshold
                            continue
                        linker = self.nlp.get_pipe("scispacy_linker")
                        concept = linker.kb.cui_to_entity.get(cui, None)
                        if concept:
                            spacy_symptoms.append({
                                'description': ent.text.lower(),
                                'category': self.common_symptoms._infer_category(ent.text.lower(), features['chief_complaint']),
                                'definition': concept.canonical_name,
                                'duration': extract_duration(text) or 'Unknown',
                                'severity': classify_severity(text) or 'Unknown',
                                'location': extract_location(text, ent.text.lower()) or 'Unknown',
                                'aggravating': features['aggravating_factors'],
                                'alleviating': features['alleviating_factors'],
                                'umls_cui': cui,
                                'semantic_type': concept.types[0] if concept.types else 'Unknown'
                            })
            except Exception as e:
                logger.error(f"NLP processing error for note ID {note.id}: {str(e)}")

            # Merge spaCy symptoms with SymptomTracker, prioritizing tracker metadata
            tracker_symptoms = self.common_symptoms.process_note(note, features['chief_complaint'], expected_symptoms)
            spacy_descriptions = {s['description'] for s in spacy_symptoms}
            features['symptoms'] = tracker_symptoms + [s for s in spacy_symptoms if s['description'] not in {t['description'] for t in tracker_symptoms}]

            # Detect new symptoms dynamically
            potential_symptoms = getattr(note, 'symptoms', None) or [s.strip() for s in text.split(',') if s.strip() and len(s.strip()) > 2]
            for symptom in potential_symptoms:
                symptom_lower = symptom.lower()
                if self.kb_updater.is_new_symptom(symptom_lower):
                    category = self.kb_updater.infer_category(symptom_lower, text)
                    synonyms = self.kb_updater.generate_synonyms(symptom_lower)
                    cui, semantic_type = self._get_umls_cui(symptom_lower)
                    self.kb_updater.update_knowledge_base(symptom_lower, category, synonyms, text)
                    features['symptoms'].append({
                        'description': symptom_lower,
                        'category': category,
                        'definition': f"Newly detected: {symptom_lower}",
                        'duration': extract_duration(text) or 'Unknown',
                        'severity': classify_severity(text) or 'Unknown',
                        'location': extract_location(text, symptom_lower) or 'Unknown',
                        'aggravating': features['aggravating_factors'],
                        'alleviating': features['alleviating_factors'],
                        'umls_cui': cui,
                        'semantic_type': semantic_type
                    })
                    logger.info(f"Added new symptom {symptom_lower} to knowledge base (category: {category}, CUI: {cui})")

            # Fallback symptom extraction
            if not features['symptoms'] and expected_symptoms:
                logger.warning(f"No symptoms extracted for note ID {note.id}. Using fallback.")
                for symptom in expected_symptoms:
                    symptom_lower = symptom.lower()
                    synonyms = self.synonyms.get(symptom_lower, [])
                    patterns = [symptom_lower] + [s.lower() for s in synonyms]
                    cui, semantic_type = self._get_umls_cui(symptom_lower)
                    for pattern in patterns:
                        if pattern in text:
                            duration = extract_duration(text) or 'Unknown'
                            severity = classify_severity(text) or 'Unknown'
                            location = extract_location(text, symptom_lower) or 'Unknown'
                            features['symptoms'].append({
                                'description': symptom_lower,
                                'category': self.common_symptoms._infer_category(symptom_lower, features['chief_complaint']),
                                'definition': f"Automatically extracted: {symptom_lower}",
                                'duration': duration,
                                'severity': severity,
                                'location': location,
                                'aggravating': features['aggravating_factors'],
                                'alleviating': features['alleviating_factors'],
                                'umls_cui': cui,
                                'semantic_type': semantic_type
                            })
                            break

            # Context extraction
            context = {}
            for factor in ['aggravating', 'alleviating']:
                field = f"{factor}_factors"
                if getattr(note, field, None):
                    try:
                        context.update(extract_aggravating_alleviating(note, factor))
                    except TypeError as e:
                        logger.error(f"Error calling extract_aggravating_alleviating for {factor}: {str(e)}")
                        context[factor] = getattr(note, field).lower().strip()
            features['context'] = {
                'aggravating': context.get('aggravating', ''),
                'alleviating': context.get('alleviating', ''),
                'sedentary': 'sedentary' in features['hpi'] or 'sitting' in features['aggravating_factors'],
                'medication': features['medications']
            }

            features['symptoms'] = deduplicate(features['symptoms'])
            logger.debug(f"Extracted features for note ID {note.id}: {features}")
            return features
        except Exception as e:
            logger.error(f"Error extracting features for note ID {note.id}: {str(e)}")
            return {
                'chief_complaint': '',
                'symptoms': [],
                'assessment': '',
                'context': {},
                'hpi': '',
                'history': '',
                'medications': '',
                'recommendation': '',
                'additional_notes': '',
                'aggravating_factors': '',
                'alleviating_factors': ''
            }

    def is_relevant_dx(self, dx: str, age: Optional[int], sex: Optional[str], symptom_type: str, symptom_category: str, features: Dict) -> bool:
        """Check if a diagnosis is relevant based on patient and symptom data."""
        dx_lower = dx.lower()
        symptom_words = {s.get('description', '').lower() for s in features.get('symptoms', [])}
        symptom_cuis = {s.get('umls_cui') for s in features.get('symptoms', []) if s.get('umls_cui')}
        semantic_types = {s.get('semantic_type') for s in features.get('symptoms', []) if s.get('semantic_type')}
        history = features.get('history', '').lower()
        chief_complaint = features.get('chief_complaint', '').lower()
        assessment = features.get('assessment', '').lower()

        if dx_lower in assessment:
            return True
        if age is not None and 'pediatric' in dx_lower and age > 18:
            return False
        if sex and 'prostate' in dx_lower and sex.lower() == 'female':
            return False
        if sex and 'ovarian' in dx_lower and sex.lower() == 'male':
            return False

        required_symptoms = [r['symptom'] for r in self.diagnosis_relevance.get(dx_lower, [])]
        matches = sum(1 for req in required_symptoms if req in symptom_words or req in history or req in chief_complaint)
        if 'Sign or Symptom' in semantic_types:
            matches += 0.5
        critical_conditions = {'myocardial infarction', 'pulmonary embolism', 'aortic dissection'}
        min_matches = 2 if dx_lower in critical_conditions else 1
        relevance = matches >= min_matches or any(req in chief_complaint for req in required_symptoms)
        if not relevance:
            logger.debug(f"Excluded dx {dx_lower}: insufficient matches ({matches}/{min_matches})")
        return relevance

    def generate_differential_dx(self, features: Dict, patient: Optional[Patient] = None) -> List[Tuple[str, float, str]]:
        """Generate differential diagnoses using a tiered system."""
        try:
            dx_scores: Dict[str, Tuple[float, str]] = {}
            tiered_dx = {'tier1': [], 'tier2': [], 'tier3': []}
            symptoms = features.get('symptoms', [])
            symptom_descriptions = {s.get('description', '').lower() for s in symptoms}
            symptom_cuis = {s.get('umls_cui') for s in symptoms if s.get('umls_cui')}
            history = features.get('history', '').lower()
            additional_notes = features.get('additional_notes', '').lower()
            chief_complaint = features.get('chief_complaint', '').lower()
            assessment = features.get('assessment', '').lower()
            text = f"{chief_complaint} {features.get('hpi', '')} {additional_notes}".strip()

            # Check for empty text
            if not text:
                logger.warning("Empty text for embedding, skipping embedding-based scoring")
                text_embedding = None
            else:
                text_embedding = embed_text(text)

            # Patient info
            patient_id = getattr(patient, 'patient_id', None) if patient else None
            if patient_id:
                patient_info = get_patient_info(patient_id)
                sex = patient_info['sex']
                age = patient_info['age']
            else:
                sex = None
                age = None
                logger.warning("No patient_id for differential diagnosis, using default values")

            # High-risk and rare conditions
            high_risk_conditions = {'pulmonary embolism', 'myocardial infarction', 'meningitis', 'aortic dissection'}
            rare_conditions = {'malaria', 'leptospirosis', 'dengue'}

            # Primary diagnosis from assessment
            if assessment and 'primary assessment:' in assessment:
                dx_name = re.search(r"primary assessment: (.*?)(?:\.|$)", assessment, re.DOTALL)
                if dx_name:
                    dx_name = dx_name.group(1).strip().lower()
                    if self.is_relevant_dx(dx_name, age, sex, '', '', features):
                        matches = sum(1 for req in [r['symptom'] for r in self.diagnosis_relevance.get(dx_name, [])] if req in symptom_descriptions)
                        if matches >= min(2, len(self.diagnosis_relevance.get(dx_name, []))):
                            dx_scores[dx_name.capitalize()] = (0.95, f"Primary diagnosis: {dx_name}")
                            tiered_dx['tier1'].append((dx_name.capitalize(), 0.95, f"Primary diagnosis: {dx_name}"))

            # Symptom-based differentials
            for symptom in symptoms:
                symptom_type = symptom.get('description', '').lower()
                symptom_category = symptom.get('category', '').lower()
                symptom_cui = symptom.get('umls_cui')
                location = symptom.get('location', '').lower()
                aggravating = symptom.get('aggravating', '').lower()
                alleviating = symptom.get('alleviating', '').lower()
                for category, pathways in self.clinical_pathways.items():
                    for key, path in pathways.items():
                        key_lower = key.lower()
                        synonyms = self.synonyms.get(symptom_type, [])
                        path_cui = path.get('metadata', {}).get('umls_cui')
                        if not (any(k.lower() in {symptom_type, location, chief_complaint} for k in key_lower.split('|')) or
                                symptom_type in synonyms or symptom_category == category.lower() or
                                (symptom_cui and path_cui and symptom_cui == path_cui)):
                            continue
                        differentials = path.get('differentials', [])
                        contextual_triggers = path.get('contextual_triggers', [])
                        for diff in differentials:
                            if diff.lower() == assessment:
                                continue
                            if not self.is_relevant_dx(diff, age, sex, symptom_type, symptom_category, features):
                                continue
                            required_symptoms = [r['symptom'] for r in self.diagnosis_relevance.get(diff.lower(), [])]
                            matches = sum(1 for req in required_symptoms if req in symptom_descriptions or req in text.lower())
                            score = 0.5
                            if symptom_type in chief_complaint:
                                score += 0.2
                            if symptom_category == category.lower():
                                score += 0.15
                            if symptom_cui and path_cui and symptom_cui == path_cui:
                                score += 0.1
                            score += matches / max(len(required_symptoms), 1) * 0.25
                            reasoning = f"Matches symptom: {symptom_type} (category: {symptom_category}, CUI: {symptom_cui}) in {location}"
                            if aggravating and alleviating:
                                reasoning += f"; influenced by {aggravating}/{alleviating}"

                            if diff.lower() in high_risk_conditions or diff.lower() in rare_conditions:
                                if matches >= 3 and (not contextual_triggers or any(t.lower() in text for t in contextual_triggers)):
                                    tiered_dx['tier3'].append((diff, min(score, 0.7), reasoning + "; high-risk/rare condition"))
                                continue
                            elif matches >= 2:
                                tiered_dx['tier1'].append((diff, min(score, 0.85), reasoning))
                            else:
                                tiered_dx['tier2'].append((diff, min(score, 0.65), reasoning))
                            dx_scores[diff] = (min(score, 0.95), reasoning)

            # History-based differentials
            for condition, aliases in self.history_diagnoses.items():
                if any(alias.lower() in history for alias in aliases):
                    if condition.lower() != assessment and self.is_relevant_dx(condition, age, sex, '', '', features):
                        matches = sum(1 for req in [r['symptom'] for r in self.diagnosis_relevance.get(condition.lower(), [])] if req in symptom_descriptions)
                        if matches >= 1:
                            tiered_dx['tier2'].append((condition, 0.7, f"Supported by medical history: {condition}"))
                            dx_scores[condition] = (0.7, f"Supported by medical history: {condition}")

            # Contextual differentials
            for symptom in symptoms:
                symptom_type = symptom.get('description', '').lower()
                symptom_category = symptom.get('category', '').lower()
                for category, pathways in self.clinical_pathways.items():
                    for key, path in pathways.items():
                        if symptom_type in key.lower() or symptom_category == category.lower():
                            for diff in path.get('differentials', []):
                                if self.is_relevant_dx(diff, age, sex, symptom_type, symptom_category, features):
                                    dx_scores[diff] = (0.75, f"Supported by symptom: {symptom_type}")
                                    tiered_dx['tier2'].append((diff, 0.75, f"Supported by symptom: {symptom_type}"))

            # Embedding-based scoring
            if text_embedding is not None:
                for dx in dx_scores:
                    try:
                        dx_embedding = embed_text(dx)
                        similarity = torch.cosine_similarity(text_embedding.unsqueeze(0), dx_embedding.unsqueeze(0)).item()
                        if similarity < SIMILARITY_THRESHOLD:
                            continue
                        old_score, reasoning = dx_scores[dx]
                        adjusted_score = min(old_score + similarity * 0.05, 0.95)
                        dx_scores[dx] = (adjusted_score, reasoning)
                        for tier in tiered_dx:
                            for i, (t_dx, t_score, t_reason) in enumerate(tiered_dx[tier]):
                                if t_dx == dx:
                                    tiered_dx[tier][i] = (dx, adjusted_score, t_reason)
                                    break
                    except Exception as e:
                        logger.warning(f"Similarity failed for dx {dx}: {str(e)}")

            # Normalize and rank
            ranked_dx = tiered_dx['tier1'] + tiered_dx['tier2'] + tiered_dx['tier3']
            if ranked_dx:
                total_score = sum(score for _, score, _ in ranked_dx)
                if total_score > 0:
                    ranked_dx = [(dx, score / total_score * 0.9, reason) for dx, score, reason in ranked_dx]
            ranked_dx = sorted(ranked_dx, key=lambda x: x[1], reverse=True)[:5]
            if not ranked_dx:
                ranked_dx = [("Undetermined", 0.1, "Insufficient data")]

            logger.debug(f"Differentials: {ranked_dx}")
            return ranked_dx
        except Exception as e:
            logger.error(f"Error generating differentials: {str(e)}")
            return [("Undetermined", 0.1, "Error in differential diagnosis")]

    def generate_management_plan(self, features: Dict, differentials: List[Tuple[str, float, str]]) -> Dict:
        """Generate a management plan based on features and differentials."""
        try:
            plan = {
                'workup': {'urgent': [], 'routine': []},
                'treatment': {'symptomatic': [], 'definitive': [], 'lifestyle': []},
                'follow_up': [],
                'references': []
            }
            symptoms = features.get('symptoms', [])
            symptom_descriptions = {s.get('description', '').lower() for s in symptoms}
            symptom_cuis = {s.get('umls_cui') for s in symptoms if s.get('umls_cui')}
            symptom_categories = {s.get('category', '').lower() for s in symptoms}
            primary_dx = features.get('assessment', '').lower()
            additional_notes = features.get('additional_notes', '').lower()
            high_risk_conditions = {'temporal arteritis', 'atrial fibrillation', 'subarachnoid hemorrhage', 'myocardial infarction', 'pulmonary embolism', 'aortic dissection'}
            high_risk = False
            validated_differentials = [(dx, score, reason) for dx, score, reason in differentials if isinstance(dx, str) and isinstance(score, (int, float)) and isinstance(reason, str)]
            filtered_dx = {dx.lower() for dx, _, _ in validated_differentials if dx.lower() in high_risk_conditions and (high_risk := True)}

            # Primary diagnosis pathway
            for category, pathways in self.clinical_pathways.items():
                for key, path in pathways.items():
                    differentials = path.get('differentials', [])
                    key_parts = key.lower().split('|')
                    path_cui = path.get('metadata', {}).get('umls_cui')
                    if any(d.lower() in primary_dx for d in differentials) or any(k in symptom_descriptions or k in primary_dx for k in key_parts) or \
                       any(symptom_cuis and path_cui and cui == path_cui for cui in symptom_cuis):
                        workup = path.get('workup', {})
                        for w in workup.get('urgent', []):
                            parsed = parse_conditional_workup(w, symptoms)
                            if parsed:
                                plan['workup']['urgent'].append(parsed)
                        for w in workup.get('routine', []):
                            parsed = parse_conditional_workup(w, symptoms)
                            if parsed:
                                plan['workup']['routine'].append(parsed)
                        management = path.get('management', {})
                        plan['treatment']['symptomatic'].extend(management.get('symptomatic', []))
                        plan['treatment']['definitive'].extend(management.get('definitive', []))
                        plan['treatment']['lifestyle'].extend(management.get('lifestyle', []))
                        plan['follow_up'].extend(path.get('follow_up', []))
                        plan['references'].extend(path.get('references', []))

            # Secondary differential plans
            for dx, score, _ in validated_differentials:
                if score < 0.8 and dx.lower() not in primary_dx:
                    continue
                for diag_key, mappings in self.diagnosis_treatments.items():
                    if diag_key.lower() in dx.lower():
                        workup = mappings.get('workup', {})
                        for w in workup.get('urgent', []) if score >= CONFIDENCE_THRESHOLD else workup.get('routine', []):
                            parsed = parse_conditional_workup(w, symptoms)
                            if parsed:
                                plan['workup']['urgent' if score >= CONFIDENCE_THRESHOLD else 'routine'].append(parsed)
                        treatment = mappings.get('treatment', {})
                        plan['treatment']['definitive'].extend(treatment.get('definitive', []) if score >= CONFIDENCE_THRESHOLD else treatment.get('symptomatic', []))
                        plan['treatment']['lifestyle'].extend(treatment.get('lifestyle', []))
                        plan['references'].extend(treatment.get('references', []))

            # Follow-up
            follow_up_match = re.search(r'Follow-Up:\s*([^\.]+)', features.get('recommendation', ''), re.IGNORECASE)
            plan['follow_up'] = [follow_up_match.group(1).strip()] if follow_up_match else \
                               ['Follow-up in 3-5 days or sooner if symptoms worsen'] if high_risk else \
                               ['Follow-up in 2 weeks']

            # Deduplicate
            for key in plan['workup']:
                plan['workup'][key] = deduplicate(plan['workup'][key])
                if key == 'routine':
                    plan['workup'][key] = [item for item in plan['workup'][key] if item not in plan['workup']['urgent']]
            for key in plan['treatment']:
                plan['treatment'][key] = deduplicate(plan['treatment'][key])
            plan['follow_up'] = deduplicate(plan['follow_up'])
            plan['references'] = deduplicate(plan['references'])

            if not any(plan['workup'].values()) and not any(plan['treatment'].values()):
                plan['treatment']['definitive'] = ['Pending diagnosis']

            logger.debug(f"Management plan: {plan}")
            return plan
        except Exception as e:
            logger.error(f"Error generating management plan: {str(e)}")
            return {
                'workup': {'urgent': [], 'routine': []},
                'treatment': {'symptomatic': [], 'definitive': ['Pending diagnosis'], 'lifestyle': []},
                'follow_up': ['Follow-up in 2 weeks'],
                'references': []
            }

def parse_conditional_workup(workup: str, symptoms: List[Dict]) -> str:
    """Parse conditional workup instructions."""
    if not isinstance(workup, str):
        logger.warning(f"Invalid workup format: {workup}")
        return ''
    if 'if' not in workup.lower():
        return workup
    condition = workup.lower().split('if')[1].strip()
    for symptom in symptoms:
        desc = symptom.get('description', '').lower()
        cui = symptom.get('umls_cui')
        if condition in desc or condition in symptom.get('category', '').lower() or \
           (cui and cui in condition):
            return workup.split('if')[0].strip()
    return ''