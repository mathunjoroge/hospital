from typing import List, Dict, Set, Tuple
import torch
import re
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from departments.models.medicine import SOAPNote
from departments.models.records import Patient
from departments.nlp.logging_setup import logger
from departments.nlp.config import SIMILARITY_THRESHOLD, CONFIDENCE_THRESHOLD, EMBEDDING_DIM
from departments.nlp.nlp_utils import embed_text, preprocess_text, deduplicate
from departments.nlp.helper_functions import extract_duration, classify_severity, extract_location, extract_aggravating_alleviating
from departments.nlp.models.transformer_model import model, tokenizer
from departments.nlp.symptom_tracker import SymptomTracker

# Load environment variables
load_dotenv()

class ClinicalAnalyzer:
    def __init__(self):
        self.model = model
        self.tokenizer = tokenizer
        
        # MongoDB connection parameters
        mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
        db_name = os.getenv('DB_NAME', 'clinical_db')
        kb_prefix = os.getenv('KB_PREFIX', 'kb_')
        
        # Initialize knowledge base components
        self.medical_stop_words: Set[str] = set()
        self.medical_terms: Set[str] = set()
        self.synonyms: Dict[str, List[str]] = {}
        self.clinical_pathways: Dict[str, Dict[str, Dict]] = {}
        self.history_diagnoses: Dict[str, List[str]] = {}
        self.diagnosis_relevance: Dict[str, List[str]] = {}
        self.management_config: Dict[str, str] = {}
        self.diagnosis_treatments: Dict[str, Dict] = {}
        
        try:
            client = MongoClient(mongo_uri)
            client.admin.command('ping')
            db = client[db_name]
            
            # Load medical_stop_words
            stop_words_coll = db[f'{kb_prefix}medical_stop_words']
            self.medical_stop_words = {doc['word'] for doc in stop_words_coll.find() if 'word' in doc and isinstance(doc['word'], str)}
            logger.info(f"Loaded {len(self.medical_stop_words)} medical stop words.")
            
            # Load medical_terms
            terms_coll = db[f'{kb_prefix}medical_terms']
            self.medical_terms = {doc['term'] for doc in terms_coll.find() if 'term' in doc and isinstance(doc['term'], str)}
            logger.info(f"Loaded {len(self.medical_terms)} medical terms.")
            
            # Load synonyms
            synonyms_coll = db[f'{kb_prefix}synonyms']
            for doc in synonyms_coll.find():
                if 'key' in doc and 'aliases' in doc and isinstance(doc['key'], str) and isinstance(doc['aliases'], list):
                    self.synonyms[doc['key']] = [a for a in doc['aliases'] if isinstance(a, str)]
            logger.info(f"Loaded {len(self.synonyms)} synonym mappings.")
            
            # Load clinical_pathways
            pathways_coll = db[f'{kb_prefix}clinical_pathways']
            for doc in pathways_coll.find():
                if 'category' in doc and 'key' in doc and 'path' in doc and isinstance(doc['category'], str) and isinstance(doc['path'], dict):
                    category = doc['category']
                    if category not in self.clinical_pathways:
                        self.clinical_pathways[category] = {}
                    self.clinical_pathways[category][doc['key']] = doc['path']
            logger.info(f"Loaded {sum(len(paths) for paths in self.clinical_pathways.values())} clinical pathways.")
            
            # Load history_diagnoses
            history_coll = db[f'{kb_prefix}history_diagnoses']
            for doc in history_coll.find():
                if 'condition' in doc and 'aliases' in doc and isinstance(doc['condition'], str) and isinstance(doc['aliases'], list):
                    self.history_diagnoses[doc['condition']] = [a for a in doc['aliases'] if isinstance(a, str)]
            logger.info(f"Loaded {len(self.history_diagnoses)} history diagnoses.")
            
            # Load diagnosis_relevance
            relevance_coll = db[f'{kb_prefix}diagnosis_relevance']
            for doc in relevance_coll.find():
                if 'condition' in doc and 'required' in doc and isinstance(doc['condition'], str) and isinstance(doc['required'], list):
                    self.diagnosis_relevance[doc['condition']] = [r for r in doc['required'] if isinstance(r, str)]
            logger.info(f"Loaded {len(self.diagnosis_relevance)} diagnosis relevance mappings.")
            
            # Load management_config
            config_coll = db[f'{kb_prefix}management_config']
            for doc in config_coll.find():
                if 'key' in doc and 'value' in doc and isinstance(doc['key'], str):
                    self.management_config[doc['key']] = doc['value']
            logger.info(f"Loaded {len(self.management_config)} management config entries.")
            
            # Load diagnosis_treatments
            treatments_coll = db[f'{kb_prefix}diagnosis_treatments']
            for doc in treatments_coll.find():
                if 'diagnosis' in doc and 'mappings' in doc and isinstance(doc['diagnosis'], str) and isinstance(doc['mappings'], dict):
                    self.diagnosis_treatments[doc['diagnosis']] = doc['mappings']
            logger.info(f"Loaded {len(self.diagnosis_treatments)} diagnosis treatments.")
            
            client.close()
            
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
        
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
        except Exception as e:
            logger.error(f"Error loading knowledge base from MongoDB: {str(e)}")
        
        # Cache diagnoses list
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
            logger.warning("No symptoms loaded into SymptomTracker. Symptom extraction may be limited.")

    def extract_clinical_features(self, note: SOAPNote) -> Dict:
        """Extract structured clinical features from SOAP note."""
        logger.debug(f"Extracting features for note {note.id}")
        features = {
            'chief_complaint': "",
            'hpi': note.hpi or "",
            'history': note.medical_history or "",
            'medications': note.medication_history or "",
            'assessment': note.assessment or "",
            'recommendation': note.recommendation or "",
            'additional_notes': note.additional_notes or "",
            'symptoms': [],
            'aggravating_factors': note.aggravating_factors or "",
            'alleviating_factors': note.alleviating_factors or ""
        }

        # Set chief complaint
        if hasattr(note, 'situation') and note.situation:
            # Extract clinical terms from situation
            situation = note.situation.lower()
            for prefix in ["patient presents with", "patient reports", "patient experiencing"]:
                situation = situation.replace(prefix, "").strip()
            # Use synonym mappings to identify key symptoms
            chief_complaint = situation
            for key, aliases in self.synonyms.items():
                if key in situation or any(alias in situation for alias in aliases):
                    chief_complaint = key
                    break
            features['chief_complaint'] = chief_complaint
            logger.debug(f"Chief complaint set: {features['chief_complaint']}")
        else:
            logger.warning(f"No situation for note {note.id}")

        # Extract symptoms
        text = f"{features['chief_complaint']} {features['hpi']} {features['additional_notes']}"
        negated_terms = set()
        for match in re.finditer(r'\b(?:no|denies|without)\s+([\w\s]+?)(?:\s|$)', text.lower()):
            term = match.group(1).strip()
            if term in self.common_symptoms.get_all_symptoms() or term in self.medical_terms:
                negated_terms.add(term)
        logger.debug(f"Negated terms: {negated_terms}")

        if features['chief_complaint']:
            chief_symptom = preprocess_text(features['chief_complaint'], self.medical_stop_words)
            if chief_symptom and chief_symptom not in negated_terms:
                category, description = self.common_symptoms.search_symptom(chief_symptom)
                symptom_dict = {
                    'description': chief_symptom,
                    'category': category or 'unknown',
                    'definition': description or 'No description available',
                    'duration': extract_duration(text),
                    'severity': classify_severity(text),
                    'location': extract_location(features['chief_complaint'] + " " + features['hpi']),
                    'aggravating': features['aggravating_factors'] or extract_aggravating_alleviating(text, "aggravating"),
                    'alleviating': features['alleviating_factors'] or extract_aggravating_alleviating(text, "alleviating")
                }
                features['symptoms'].append(symptom_dict)
                logger.debug(f"Added chief symptom: {symptom_dict}")
            # Split composite chief complaints
            if ' and ' in features['chief_complaint']:
                for term in features['chief_complaint'].split(' and '):
                    term = term.strip()
                    if not term or term in negated_terms:
                        continue
                    if any(word in self.common_symptoms.get_all_symptoms() or word in self.medical_terms for word in term.split()):
                        category, description = self.common_symptoms.search_symptom(term)
                        symptom_dict = {
                            'description': term,
                            'category': category or 'unknown',
                            'definition': description or 'No description available',
                            'duration': extract_duration(text),
                            'severity': classify_severity(text),
                            'location': extract_location(term + " " + text),
                            'aggravating': features['aggravating_factors'] or extract_aggravating_alleviating(text, "aggravating"),
                            'alleviating': features['alleviating_factors'] or extract_aggravating_alleviating(text, "alleviating")
                        }
                        features['symptoms'].append(symptom_dict)
                        logger.debug(f"Added split symptom: {symptom_dict}")

        # Rule-based symptom extraction
        symptom_candidates = set(preprocess_text(text, self.medical_stop_words).split())
        for term in symptom_candidates:
            if not isinstance(term, str):
                logger.warning(f"Non-string symptom candidate: {term}")
                continue
            if (term in self.common_symptoms.get_all_symptoms() or term in self.medical_terms) and term not in negated_terms:
                category, description = self.common_symptoms.search_symptom(term)
                symptom_dict = {
                    'description': term,
                    'category': category or 'unknown',
                    'definition': description or 'No description available',
                    'duration': extract_duration(text),
                    'severity': classify_severity(text),
                    'location': extract_location(term + " " + text),
                    'aggravating': features['aggravating_factors'] or extract_aggravating_alleviating(text, "aggravating"),
                    'alleviating': features['alleviating_factors'] or extract_aggravating_alleviating(text, "alleviating")
                }
                features['symptoms'].append(symptom_dict)
                logger.debug(f"Added rule-based symptom: {symptom_dict}")

        # Embedding-based symptom validation with category-specific embeddings
        for symptom in features['symptoms'][:]:  # Copy to allow modification
            term = symptom.get('description', '')
            if not isinstance(term, str):
                continue
            if term in negated_terms:
                continue
            category = symptom.get('category', 'unknown').lower()
            try:
                clinical_embedding = embed_text(f"{category} symptom")
                term_embedding = embed_text(term)
                location = symptom.get('location', 'Unspecified')
                context_term = f"{term} {location.lower()}" if location != "Unspecified" else term
                context_embedding = embed_text(context_term)
                similarity = torch.cosine_similarity(context_embedding.unsqueeze(0), clinical_embedding.unsqueeze(0)).item()
                if similarity <= 0.85:  # Increased SIMILARITY_THRESHOLD
                    features['symptoms'].remove(symptom)
                    logger.debug(f"Removed low-similarity symptom: {term}, similarity: {similarity}")
            except Exception as e:
                logger.warning(f"Embedding failed for term {term}: {str(e)}")

        # Deduplicate symptoms
        original_symptoms = features['symptoms'].copy()
        symptom_descriptions = [s.get('description', '') for s in original_symptoms if isinstance(s, dict)]
        deduped_descriptions = deduplicate(tuple(symptom_descriptions), self.synonyms)
        features['symptoms'] = []
        seen = set()
        for desc in deduped_descriptions:
            if not isinstance(desc, str):
                continue
            desc_lower = desc.lower()
            if desc_lower not in seen:
                seen.add(desc_lower)
                for symptom in original_symptoms:
                    if not isinstance(symptom, dict):
                        continue
                    if symptom.get('description', '').lower() == desc_lower:
                        features['symptoms'].append(symptom)
                        break
        logger.debug(f"Final symptoms: {features['symptoms']}")
        return features

    def is_relevant_dx(self, dx: str, age: int, sex: str, symptom_type: str, symptom_category: str, features: Dict) -> bool:
        """Check if a diagnosis is relevant based on demographics and clinical features."""
        dx_lower = dx.lower()
        symptom_words = {s.get('description', '').lower() for s in features.get('symptoms', [])}
        history = features.get('history', '').lower()
        chief_complaint = features.get('chief_complaint', '').lower()
        
        # Demographic filters
        if age and 'pediatric' in dx_lower and age > 18:
            return False
        if sex and 'prostate' in dx_lower and sex.lower() == 'female':
            return False
        if sex and 'ovarian' in dx_lower and sex.lower() == 'male':
            return False
        
        # Symptom and context relevance
        required_symptoms = self.diagnosis_relevance.get(dx_lower, [])
        matches = sum(1 for req in required_symptoms if req in symptom_words or req in history or req in chief_complaint)
        critical_conditions = {'myocardial infarction', 'pericarditis', 'pulmonary embolism', 'angina', 'aortic dissection'}
        relevance = (matches >= 2 or (matches >= 1 and dx_lower in critical_conditions) or
                     any(req in chief_complaint for req in required_symptoms))
        if not relevance:
            logger.debug(f"Excluded dx {dx_lower}: insufficient symptom matches ({matches}/{len(required_symptoms)} required)")
        return relevance

    def generate_differential_dx(self, features: Dict, patient: Patient = None) -> List[Tuple]:
        """Generate ranked differential diagnoses with demographic and context filtering.
        
        Returns:
            List of tuples (diagnosis: str, score: float, reasoning: str)
        """
        logger.debug(f"Generating differentials for chief complaint: {features.get('chief_complaint')}")
        dx_scores: Dict[str, Tuple[float, str]] = {}
        symptoms = features.get('symptoms', [])
        history = features.get('history', '').lower()
        additional_notes = features.get('additional_notes', '').lower()
        text = f"{features.get('chief_complaint', '')} {features.get('hpi', '')} {additional_notes}"
        text_embedding = embed_text(text)
        primary_dx = features.get('assessment', '').lower()
        chief_complaint = features.get('chief_complaint', '').lower()
        age = patient.age if patient and hasattr(patient, 'age') else None
        sex = patient.sex if patient and hasattr(patient, 'sex') else None

        # Symptom and location matching with category consideration
        for symptom in symptoms:
            if not isinstance(symptom, dict):
                continue
            symptom_type = symptom.get('description', '').lower()
            symptom_category = symptom.get('category', 'unknown').lower()
            location = symptom.get('location', '').lower()
            aggravating = symptom.get('aggravating', '').lower()
            alleviating = symptom.get('alleviating', '').lower()
            for category, pathways in self.clinical_pathways.items():
                for key, path in pathways.items():
                    if not isinstance(path, dict):
                        continue
                    key_lower = key.lower()
                    synonyms = self.synonyms.get(symptom_type, [])
                    if (symptom_type == key_lower or location == key_lower or symptom_type in synonyms or
                        symptom_category == category.lower()):
                        differentials = path.get('differentials', [])
                        for diff in differentials:
                            if not isinstance(diff, str):
                                continue
                            if diff.lower() == primary_dx:
                                continue
                            if not self.is_relevant_dx(diff, age, sex, symptom_type, symptom_category, features):
                                continue
                            score = 0.5
                            if symptom_type in chief_complaint:
                                score += 0.3
                            if symptom_category == category.lower():
                                score += 0.2
                            required_symptoms = self.diagnosis_relevance.get(diff.lower(), [])
                            matches = sum(1 for req in required_symptoms if req in symptom_type or req in text.lower())
                            score += matches / max(len(required_symptoms), 1) * 0.3
                            reasoning = f"Matches symptom: {symptom_type} (category: {symptom_category}) in {location}"
                            if aggravating and alleviating:
                                reasoning += f"; influenced by {aggravating}/{alleviating}"
                            dx_scores[diff] = (min(score, 0.95), reasoning)
                            logger.debug(f"Added symptom-based dx: {diff}, score: {score}")

        # History-based differentials
        for condition, aliases in self.history_diagnoses.items():
            if any(alias.lower() in history for alias in aliases):
                if condition.lower() != primary_dx and self.is_relevant_dx(condition, age, sex, '', '', features):
                    dx_scores[condition] = (0.75, f"Supported by medical history: {condition}")
                    logger.debug(f"Added history-based dx: {condition}")

        # Contextual adjustments
        for symptom in symptoms:
            symptom_type = symptom.get('description', '').lower()
            symptom_category = symptom.get('category', '').lower()
            if 'new pet' in additional_notes and 'cough' in symptom_type:
                dx_scores['Allergic cough'] = (0.75, f"Supported by new pet exposure and symptom: {symptom_type}")
            if 'new medication' in additional_notes and 'rash' in symptom_type:
                dx_scores['Drug reaction'] = (0.75, f"Suggested by new medication and symptom: {symptom_type}")
            if 'travel' in additional_notes and 'diarrhea' in symptom_type:
                dx_scores['Traveler’s diarrhea'] = (0.75, f"Suggested by recent travel and symptom: {symptom_type}")
            if 'sedentary job' in history and 'back pain' in symptom_type:
                dx_scores['Mechanical low back pain'] = (0.75, f"Supported by sedentary lifestyle and symptom: {symptom_type}")
            if 'palpitations' in symptom_type and 'no weight loss' in text.lower():
                dx_scores.pop('Hyperthyroidism', None)
            if 'vaginal discharge' in symptom_type and 'antibiotics' in features.get('medications', '').lower():
                dx_scores['Candidiasis'] = (0.85, f"Supported by recent antibiotic use and symptom: {symptom_type}")

        # Embedding-based scoring
        for dx in dx_scores:
            try:
                dx_embedding = embed_text(dx)
                similarity = torch.cosine_similarity(text_embedding.unsqueeze(0), dx_embedding.unsqueeze(0)).item()
                old_score, reasoning = dx_scores[dx]
                dx_scores[dx] = (min(old_score + similarity * 0.1, 0.95), reasoning)
            except Exception as e:
                logger.warning(f"Similarity failed for dx {dx}: {str(e)}")

        # Normalize scores
        if dx_scores:
            total_score = sum(score for score, _ in dx_scores.values())
            if total_score > 0:
                dx_scores = {dx: (score / total_score * 0.9, reason) for dx, (score, reason) in dx_scores.items()}

        ranked_dx = [(dx, score, reason) for dx, (score, reason) in sorted(dx_scores.items(), key=lambda x: x[1][0], reverse=True)[:5]]
        if not ranked_dx:
            ranked_dx = [("Undetermined", 0.1, "Insufficient data")]
            logger.warning("No differentials generated; knowledge base or relevance criteria may need review")
        logger.debug(f"Returning differentials: {ranked_dx}")
        return ranked_dx

    def generate_management_plan(self, features: Dict, differentials: List[Tuple]) -> Dict:
        """Generate tailored management plan with filtered workups and treatments.
        
        Args:
            features: Dictionary of extracted clinical features
            differentials: List of tuples (diagnosis: str, score: float, reasoning: str)
        
        Returns:
            Dictionary with workup, treatment, and follow-up plans
        """
        logger.debug(f"Generating management plan for {features.get('chief_complaint')}")
        plan = {
            'workup': {'urgent': [], 'routine': []},
            'treatment': {'symptomatic': [], 'definitive': []},
            'follow_up': []
        }
        symptoms = features.get('symptoms', [])
        symptom_descriptions = {s.get('description', '').lower() for s in symptoms if isinstance(s, dict)}
        symptom_categories = {s.get('category', '').lower() for s in symptoms if isinstance(s, dict)}
        primary_dx = features.get('assessment', '').lower()
        filtered_dx = set()
        high_risk_conditions = {'temporal arteritis', 'atrial fibrillation', 'subarachnoid hemorrhage', 'myocardial infarction', 'pulmonary embolism', 'aortic dissection'}
        high_risk = False

        # Validate differentials
        validated_differentials = []
        for diff in differentials:
            if not isinstance(diff, tuple) or len(diff) != 3:
                continue
            dx, score, reason = diff
            if not isinstance(dx, str) or not isinstance(score, (int, float)) or not isinstance(reason, str):
                continue
            validated_differentials.append(diff)
            filtered_dx.add(dx.lower())
            if dx.lower() in high_risk_conditions:
                high_risk = True

        # Primary diagnosis-based plan
        for category, pathways in self.clinical_pathways.items():
            for key, path in pathways.items():
                if not isinstance(path, dict):
                    continue
                differentials = path.get('differentials', [])
                if any(d.lower() in primary_dx for d in differentials):
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
                    logger.debug(f"Added primary dx-based plan for {key}")

        # Differential-based management
        for diff in validated_differentials:
            dx, score, _ = diff
            if score < 0.4:  # Skip low-confidence diagnoses
                continue
            for diag_key, mappings in self.diagnosis_treatments.items():
                if not isinstance(mappings, dict):
                    continue
                if diag_key.lower() in dx.lower():
                    workup = mappings.get('workup', {})
                    for w in workup.get('urgent', []) if score >= CONFIDENCE_THRESHOLD else workup.get('routine', []):
                        parsed = parse_conditional_workup(w, symptoms)
                        if parsed:
                            plan['workup']['urgent' if score >= CONFIDENCE_THRESHOLD else 'routine'].append(parsed)
                    treatment = mappings.get('treatment', {})
                    plan['treatment']['definitive'].extend(treatment.get('definitive', []) if score >= CONFIDENCE_THRESHOLD else treatment.get('symptomatic', []))
                    logger.debug(f"Added dx-based plan for {dx}")

        # Contextual adjustments
        additional_notes = features.get('additional_notes', '').lower()
        if 'new pet' in additional_notes and 'respiratory' in symptom_categories:
            plan['workup']['routine'].append("Allergy testing")
        if 'new medication' in additional_notes and 'dermatological' in symptom_categories:
            plan['workup']['routine'].append("Medication history review")
        if 'travel' in additional_notes and 'gastrointestinal' in symptom_categories:
            plan['workup']['routine'].append("Stool culture")
        if 'sedentary job' in additional_notes and 'musculoskeletal' in symptom_categories:
            plan['treatment']['definitive'].append("Ergonomic counseling")

        # Follow-up customization
        plan['follow_up'] = ["Follow-up in 3-5 days or sooner if symptoms worsen"] if high_risk else ["Follow-up in 1-2 weeks"]

        # Deduplicate and filter
        for key in plan['workup']:
            plan['workup'][key] = deduplicate(tuple(sorted(set(plan['workup'][key]))), self.synonyms)
            if key == 'routine':
                plan['workup'][key] = [item for item in plan['workup'][key] if item not in plan['workup']['urgent']]
        for key in plan['treatment']:
            plan['treatment'][key] = deduplicate(tuple(sorted(set(plan['treatment'][key]))), self.synonyms)
        logger.debug(f"Final plan: {plan}")
        return plan

def parse_conditional_workup(workup: str, symptoms: List[Dict]) -> str:
    """Parse conditional workup requirements with severity and duration checks."""
    if not isinstance(workup, str):
        logger.warning(f"Invalid workup format: {workup}")
        return ""
    if 'if' not in workup.lower():
        return workup
    condition = workup.lower().split('if')[1].strip()
    for symptom in symptoms:
        if not isinstance(symptom, dict):
            continue
        desc = symptom.get('description', '').lower()
        severity = symptom.get('severity', '').lower()
        if condition in desc and (severity in ['severe', 'moderate'] or 'urgent' in condition):
            return workup.split('if')[0].strip()
    return ""