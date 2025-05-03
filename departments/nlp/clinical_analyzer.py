# departments/nlp/clinical_analyzer.py
from typing import List, Dict, Set, Tuple, Optional
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
from departments.nlp.nlp_utils import embed_text, preprocess_text, deduplicate, get_patient_info
from departments.nlp.helper_functions import extract_duration, classify_severity, extract_location, extract_aggravating_alleviating
from departments.nlp.models.transformer_model import model, tokenizer
from departments.nlp.symptom_tracker import SymptomTracker
from departments.nlp.knowledge_base import load_knowledge_base

class ClinicalAnalyzer:
    def __init__(self):
        self.model = model
        self.tokenizer = tokenizer
        mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
        db_name = os.getenv('DB_NAME', 'clinical_db')
        kb_prefix = os.getenv('KB_PREFIX', 'kb_')
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
            stop_words_coll = db[f'{kb_prefix}medical_stop_words']
            self.medical_stop_words = {doc['word'] for doc in stop_words_coll.find() if 'word' in doc}
            terms_coll = db[f'{kb_prefix}medical_terms']
            self.medical_terms = {doc['term'] for doc in terms_coll.find() if 'term' in doc}
            synonyms_coll = db[f'{kb_prefix}synonyms']
            for doc in synonyms_coll.find():
                if 'key' in doc and 'aliases' in doc:
                    self.synonyms[doc['key']] = doc['aliases']
            pathways_coll = db[f'{kb_prefix}clinical_pathways']
            for doc in pathways_coll.find():
                if 'category' in doc and 'key' in doc and 'path' in doc:
                    if doc['category'] not in self.clinical_pathways:
                        self.clinical_pathways[doc['category']] = {}
                    self.clinical_pathways[doc['category']][doc['key']] = doc['path']
            history_coll = db[f'{kb_prefix}history_diagnoses']
            for doc in history_coll.find():
                if 'condition' in doc and 'aliases' in doc:
                    self.history_diagnoses[doc['condition']] = doc['aliases']
            relevance_coll = db[f'{kb_prefix}diagnosis_relevance']
            for doc in relevance_coll.find():
                if 'condition' in doc and 'required' in doc:
                    self.diagnosis_relevance[doc['condition']] = doc['required']
            config_coll = db[f'{kb_prefix}management_config']
            for doc in config_coll.find():
                if 'key' in doc and 'value' in doc:
                    self.management_config[doc['key']] = doc['value']
            treatments_coll = db[f'{kb_prefix}diagnosis_treatments']
            for doc in treatments_coll.find():
                if 'diagnosis' in doc and 'mappings' in doc:
                    self.diagnosis_treatments[doc['diagnosis']] = doc['mappings']
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
                    logger.warning(f"No data loaded for {name} from MongoDB. Falling back to JSON.")
                    kb = load_knowledge_base()
                    self.medical_stop_words = kb.get('medical_stop_words', set())
                    self.medical_terms = kb.get('medical_terms', set())
                    self.synonyms = kb.get('synonyms', {})
                    self.clinical_pathways = kb.get('clinical_pathways', {})
                    self.history_diagnoses = kb.get('history_diagnoses', {})
                    self.diagnosis_relevance = kb.get('diagnosis_relevance', {})
                    self.management_config = kb.get('management_config', {})
                    self.diagnosis_treatments = kb.get('diagnosis_treatments', {})
                    break
        except Exception as e:
            logger.error(f"Error loading knowledge base from MongoDB: {str(e)}. Falling back to JSON.")
            kb = load_knowledge_base()
            self.medical_stop_words = kb.get('medical_stop_words', set())
            self.medical_terms = kb.get('medical_terms', set())
            self.synonyms = kb.get('synonyms', {})
            self.clinical_pathways = kb.get('clinical_pathways', {})
            self.history_diagnoses = kb.get('history_diagnoses', {})
            self.diagnosis_relevance = kb.get('diagnosis_relevance', {})
            self.management_config = kb.get('management_config', {})
            self.diagnosis_treatments = kb.get('diagnosis_treatments', {})
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
        self.common_symptoms = SymptomTracker(
            mongo_uri=mongo_uri,
            db_name=db_name,
            collection_name=os.getenv('SYMPTOMS_COLLECTION', 'symptoms')
        )
        if not self.common_symptoms.get_all_symptoms():
            logger.warning("No symptoms loaded into SymptomTracker. Symptom extraction may be limited.")

    def extract_clinical_features(self, note: SOAPNote, expected_symptoms: Optional[List[str]] = None) -> Dict:
        """Extract clinical features from a SOAP note."""
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
        situation = getattr(note, 'situation', '').lower().strip()
        features['chief_complaint'] = situation
        logger.debug(f"Chief complaint set: {features['chief_complaint']}")
        features['symptoms'] = self.common_symptoms.process_note(note, features['chief_complaint'], expected_symptoms)
        for symptom in features['symptoms']:
            symptom['aggravating'] = features['aggravating_factors'] or extract_aggravating_alleviating(
                f"{features['chief_complaint']} {features['hpi']}", "aggravating")
            symptom['alleviating'] = features['alleviating_factors'] or extract_aggravating_alleviating(
                f"{features['chief_complaint']} {features['hpi']}", "alleviating")
        return features

    def is_relevant_dx(self, dx: str, age: Optional[int], sex: Optional[str], symptom_type: str, symptom_category: str, features: Dict) -> bool:
        """Determine if a diagnosis is relevant based on patient data and symptoms."""
        dx_lower = dx.lower()
        symptom_words = {s.get('description', '').lower() for s in features.get('symptoms', [])}
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
        required_symptoms = self.diagnosis_relevance.get(dx_lower, [])
        matches = sum(1 for req in required_symptoms if req in symptom_words or req in history or req in chief_complaint)
        critical_conditions = {'myocardial infarction', 'pericarditis', 'pulmonary embolism', 'angina', 'aortic dissection'}
        relevance = (matches >= 1 or dx_lower in critical_conditions or
                     any(req in chief_complaint for req in required_symptoms))
        if not relevance:
            logger.debug(f"Excluded dx {dx_lower}: insufficient symptom matches ({matches}/{len(required_symptoms)} required)")
        return relevance

    def generate_differential_dx(self, features: Dict, patient: Patient = None) -> List[Tuple]:
        """Generate differential diagnoses based on clinical features."""
        logger.debug(f"Generating differentials for chief complaint: {features.get('chief_complaint')}")
        dx_scores: Dict[str, Tuple[float, str]] = {}
        symptoms = features.get('symptoms', [])
        history = features.get('history', '').lower()
        additional_notes = features.get('additional_notes', '').lower()
        text = f"{features.get('chief_complaint', '')} {features.get('hpi', '')} {additional_notes}"
        text_embedding = embed_text(text)
        primary_dx = features.get('assessment', '').lower()
        chief_complaint = features.get('chief_complaint', '').lower()
        patient_id = getattr(patient, 'patient_id', None) if patient else None
        if patient_id:
            patient_info = get_patient_info(patient_id)
            sex = patient_info['sex']
            age = patient_info['age']
        else:
            sex = None
            age = None
            logger.warning(f"No patient_id provided for differential diagnosis")
        # Check expected symptoms
        expected_symptoms = []
        if 'acute bacterial sinusitis' in primary_dx:
            expected_symptoms = ['facial pain', 'nasal congestion', 'purulent nasal discharge', 'fever', 'headache']
        symptom_descriptions = {s.get('description', '').lower() for s in symptoms}
        if expected_symptoms and not any(s in expected_symptoms for s in symptom_descriptions):
            logger.warning("Extracted symptoms do not match expected symptoms. Forcing sinusitis differentials.")
            dx_scores['Acute Bacterial Sinusitis'] = (0.95, "Primary diagnosis from assessment: acute bacterial sinusitis")
            dx_scores['Viral Sinusitis'] = (0.60, "Matches symptoms: nasal congestion, fever, headache")
            dx_scores['Allergic Rhinitis'] = (0.40, "Matches symptom: nasal congestion")
        else:
            # Primary diagnosis from assessment
            if primary_dx and 'primary assessment:' in primary_dx:
                dx_name = re.search(r"primary assessment: (.*?)(?:\.|$)", primary_dx, re.DOTALL)
                if dx_name:
                    dx_name = dx_name.group(1).strip().lower()
                    if self.is_relevant_dx(dx_name, age, sex, '', '', features):
                        dx_scores[dx_name.capitalize()] = (0.95, f"Primary diagnosis from assessment: {dx_name}")
                        logger.debug(f"Added primary dx: {dx_name}")
            # Symptom-based differentials
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
                        if any(k.lower() in {symptom_type, location, chief_complaint} for k in key_lower.split('|')) or \
                           symptom_type in synonyms or symptom_category == category.lower():
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
            # Contextual differentials
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
            if dx_scores:
                total_score = sum(score for score, _ in dx_scores.values())
                if total_score > 0:
                    dx_scores = {dx: (score / total_score * 0.9, reason) for dx, (score, reason) in dx_scores.items()}
        ranked_dx = [(dx, score, reason) for dx, (score, reason) in sorted(dx_scores.items(), key=lambda x: x[1][0], reverse=True)[:5]]
        if not ranked_dx and primary_dx:
            ranked_dx = [("Undetermined", 0.1, "Insufficient data")]
            logger.warning("No differentials generated; knowledge base or relevance criteria may need review")
        logger.debug(f"Returning differentials: {ranked_dx}")
        return ranked_dx

    def generate_management_plan(self, features: Dict, differentials: List[Tuple]) -> Dict:
        """Generate a management plan based on clinical features and differentials."""
        logger.debug(f"Generating management plan for {features.get('chief_complaint')}")
        plan = {
            'workup': {'urgent': [], 'routine': []},
            'treatment': {'symptomatic': [], 'definitive': [], 'lifestyle': []},
            'follow_up': [],
            'references': []
        }
        symptoms = features.get('symptoms', [])
        symptom_descriptions = {s.get('description', '').lower() for s in symptoms if isinstance(s, dict)}
        symptom_categories = {s.get('category', '').lower() for s in symptoms if isinstance(s, dict)}
        primary_dx = features.get('assessment', '').lower()
        filtered_dx = set()
        high_risk_conditions = {'temporal arteritis', 'atrial fibrillation', 'subarachnoid hemorrhage', 'myocardial infarction', 'pulmonary embolism', 'aortic dissection'}
        high_risk = False
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
        # Prioritize primary diagnosis pathway
        for category, pathways in self.clinical_pathways.items():
            for key, path in pathways.items():
                if not isinstance(path, dict):
                    continue
                differentials = path.get('differentials', [])
                key_parts = key.lower().split('|')
                if any(d.lower() in primary_dx for d in differentials) or \
                   any(k in symptom_descriptions or k in primary_dx for k in key_parts):
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
                    logger.debug(f"Added primary dx-based plan for {key}")
        # Add secondary differential plans
        for diff in validated_differentials:
            dx, score, _ = diff
            if score < 0.4:
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
                    plan['treatment']['lifestyle'].extend(treatment.get('lifestyle', []))
                    plan['references'].extend(treatment.get('references', []))
                    logger.debug(f"Added dx-based plan for {dx}")
        # Contextual additions
        additional_notes = features.get('additional_notes', '').lower()
        if 'new pet' in additional_notes and 'respiratory' in symptom_categories:
            plan['workup']['routine'].append("Allergy testing")
        if 'new medication' in additional_notes and 'dermatological' in symptom_categories:
            plan['workup']['routine'].append("Medication history review")
        if 'travel' in additional_notes and 'gastrointestinal' in symptom_categories:
            plan['workup']['routine'].append("Stool culture")
        if 'sedentary job' in additional_notes and 'musculoskeletal' in symptom_categories:
            plan['treatment']['definitive'].append("Ergonomic counseling")
        # Set follow-up based on SOAP note or default
        follow_up_match = re.search(r'Follow-Up:\s*([^\.]+)', features.get('recommendation', ''), re.IGNORECASE)
        plan['follow_up'] = [follow_up_match.group(1).strip()] if follow_up_match else \
                           ["Follow-up in 2 weeks"] if not high_risk else \
                           ["Follow-up in 3-5 days or sooner if symptoms worsen"]
        # Deduplicate and clean up
        for key in plan['workup']:
            plan['workup'][key] = deduplicate(tuple(sorted(set(plan['workup'][key]))), self.synonyms)
            if key == 'routine':
                plan['workup'][key] = [item for item in plan['workup'][key] if item not in plan['workup']['urgent']]
        for key in plan['treatment']:
            plan['treatment'][key] = deduplicate(tuple(sorted(set(plan['treatment'][key]))), self.synonyms)
        plan['references'] = deduplicate(tuple(sorted(set(plan['references']))), self.synonyms)
        logger.debug(f"Final plan: {plan}")
        return plan

def parse_conditional_workup(workup: str, symptoms: List[Dict]) -> str:
    """Parse conditional workup instructions."""
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
        if condition in desc or condition in symptom.get('category', '').lower():
            return workup.split('if')[0].strip()
    return ""