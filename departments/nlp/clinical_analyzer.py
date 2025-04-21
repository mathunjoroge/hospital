# departments/nlp/clinical_analyzer.py

from typing import List, Tuple, Dict, Set
import torch
import re
from departments.models.medicine import SOAPNote
from departments.nlp.logging_setup import logger
from departments.nlp.config import SIMILARITY_THRESHOLD, CONFIDENCE_THRESHOLD, EMBEDDING_DIM
from departments.nlp.knowledge_base import load_knowledge_base
from departments.nlp.nlp_utils import embed_text, preprocess_text, deduplicate
from departments.nlp.helper_functions import extract_duration, classify_severity, extract_location, extract_aggravating_alleviating
from departments.nlp.models.transformer_model import model, tokenizer
from departments.nlp.symptom_tracker import SymptomTracker  # Import SymptomTracker

class ClinicalAnalyzer:
    def __init__(self):
        self.model = model
        self.tokenizer = tokenizer
        self.knowledge = load_knowledge_base()
        self.medical_stop_words = self.knowledge.get("medical_stop_words", set())
        self.medical_terms = self.knowledge.get("medical_terms", set())
        self.synonyms = self.knowledge.get("synonyms", {})
        self.clinical_pathways = self.knowledge.get("clinical_pathways", {})
        self.history_diagnoses = self.knowledge.get("history_diagnoses", {})
        self.diagnosis_relevance = self.knowledge.get("diagnosis_relevance", {})
        self.management_config = self.knowledge.get("management_config", {})
        self.diagnosis_treatments = self.knowledge.get("diagnosis_treatments", {})
        
        # Cache diagnoses list
        self.diagnoses_list = set()
        if isinstance(self.clinical_pathways, dict):
            for category, pathways in self.clinical_pathways.items():
                if isinstance(pathways, dict):
                    for key, path in pathways.items():
                        if isinstance(path, dict):
                            differentials = path.get('differentials', [])
                            if isinstance(differentials, list):
                                self.diagnoses_list.update(d.lower() for d in differentials)
        
        # Initialize SymptomTracker
        self.common_symptoms = SymptomTracker()

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
            features['chief_complaint'] = note.situation.replace("Patient presents with", "").replace("Patient reports", "").replace("Patient experiencing", "").strip()
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
        logger.debug(f"Symptom candidates: {symptom_candidates}")
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

        # Embedding-based symptom validation
        clinical_embedding = embed_text("clinical symptom")
        expanded_candidates = set()
        for term in symptom_candidates:
            if not isinstance(term, str):
                logger.warning(f"Skipping invalid term (non-string): {term}")
                continue
            if term in negated_terms:
                logger.debug(f"Skipping negated term: {term}")
                continue
            if term not in self.common_symptoms.get_all_symptoms() and term not in self.medical_terms:
                continue
            expanded_candidates.add(term)
            if isinstance(self.synonyms, dict):
                for key, aliases in self.synonyms.items():
                    if not isinstance(aliases, list):
                        logger.warning(f"Invalid aliases for {key}: {aliases}")
                        continue
                    if term.lower() in [a.lower() for a in aliases if isinstance(a, str)]:
                        if key.lower() not in self.diagnoses_list:
                            expanded_candidates.add(key.lower())
                            expanded_candidates.update(a.lower() for a in aliases if isinstance(a, str))

        for term in expanded_candidates:
            if not isinstance(term, str):
                logger.warning(f"Non-string expanded candidate: {term}")
                continue
            if term in self.medical_terms and term not in self.diagnoses_list:
                try:
                    term_embedding = embed_text(term)
                    location = extract_location(term + " " + text)
                    context_term = f"{term} {location.lower()}" if location != "Unspecified" else term
                    context_embedding = embed_text(context_term)
                    similarity = torch.cosine_similarity(context_embedding.unsqueeze(0), clinical_embedding.unsqueeze(0)).item()
                    if similarity > SIMILARITY_THRESHOLD:
                        category, description = self.common_symptoms.search_symptom(term)
                        symptom_dict = {
                            'description': term,
                            'category': category or 'unknown',
                            'definition': description or 'No description available',
                            'duration': extract_duration(text),
                            'severity': classify_severity(text),
                            'location': location,
                            'aggravating': features['aggravating_factors'] or extract_aggravating_alleviating(text, "aggravating"),
                            'alleviating': features['alleviating_factors'] or extract_aggravating_alleviating(text, "alleviating")
                        }
                        features['symptoms'].append(symptom_dict)
                        logger.debug(f"Added embedding-based symptom: {symptom_dict}, similarity: {similarity}")
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
                logger.warning(f"Non-string description in deduplication: {desc}")
                continue
            desc_lower = desc.lower()
            if desc_lower not in seen:
                seen.add(desc_lower)
                for symptom in original_symptoms:
                    if not isinstance(symptom, dict):
                        logger.warning(f"Non-dict symptom in deduplication: {symptom}")
                        continue
                    if symptom.get('description', '').lower() == desc_lower:
                        features['symptoms'].append(symptom)
                        break
        logger.debug(f"Final symptoms: {features['symptoms']}")
        return features

    def generate_differential_dx(self, features: Dict) -> List[Tuple[str, float, str]]:
        """Generate ranked differential diagnoses."""
        logger.debug(f"Generating differentials for chief complaint: {features.get('chief_complaint')}, symptoms: {features.get('symptoms', [])}")
        dx_scores = {}
        symptoms = features.get('symptoms', [])
        history = features.get('history', '').lower()
        additional_notes = features.get('additional_notes', '').lower()
        text = f"{features.get('chief_complaint', '')} {features.get('hpi', '')} {additional_notes}"
        text_embedding = embed_text(text)
        primary_dx = features.get('assessment', '').lower()
        chief_complaint = features.get('chief_complaint', '').lower()

        # Assessment-based differential
        if primary_dx:
            clean_assessment = primary_dx.replace("possible ", "").strip()
            logger.debug(f"Primary diagnosis: {clean_assessment}")

        # Symptom and location matching with category consideration
        for symptom in symptoms:
            if not isinstance(symptom, dict):
                logger.warning(f"Invalid symptom format: {symptom}")
                continue
            symptom_type = symptom.get('description', '').lower()
            symptom_category = symptom.get('category', 'unknown').lower()
            location = symptom.get('location', '').lower()
            aggravating = symptom.get('aggravating', '').lower()
            alleviating = symptom.get('alleviating', '').lower()
            if not isinstance(self.clinical_pathways, dict):
                logger.error(f"clinical_pathways not a dict: {type(self.clinical_pathways)}")
                continue
            for category, pathways in self.clinical_pathways.items():
                if not isinstance(pathways, dict):
                    logger.error(f"pathways not a dict for {category}: {type(pathways)}")
                    continue
                for key, path in pathways.items():
                    if not isinstance(path, dict):
                        logger.error(f"path not a dict for {key}: {type(path)}")
                        continue
                    key_lower = key.lower()
                    synonyms = self.synonyms.get(symptom_type, [])
                    if (symptom_type == key_lower or location == key_lower or symptom_type in synonyms or
                        symptom_category == category.lower()):
                        differentials = path.get('differentials', [])
                        if not isinstance(differentials, list):
                            logger.error(f"differentials not a list for {key}: {type(differentials)}")
                            continue
                        for diff in differentials:
                            if not isinstance(diff, str):
                                logger.warning(f"Non-string differential for {key}: {diff}")
                                continue
                            if diff.lower() != primary_dx:
                                score = 0.7
                                if symptom_type in chief_complaint:
                                    score += 0.2
                                if symptom_category == category.lower():
                                    score += 0.1  # Bonus for category match
                                reasoning = f"Matches symptom: {symptom_type} (category: {symptom_category}) in {location}"
                                if aggravating and alleviating:
                                    reasoning += f"; influenced by {aggravating}/{alleviating}"
                                dx_scores[diff] = (score, reasoning)
                                logger.debug(f"Added symptom-based dx: {diff}")

        # History-based differentials
        if isinstance(self.history_diagnoses, dict):
            for condition, aliases in self.history_diagnoses.items():
                if not isinstance(aliases, list):
                    logger.error(f"aliases not a list for {condition}: {type(aliases)}")
                    continue
                if any(alias.lower() in history for alias in aliases):
                    if condition.lower() != primary_dx:
                        dx_scores[condition] = (0.75, f"Supported by medical history: {condition}")
                        logger.debug(f"Added history-based dx: {condition}")

        # Contextual clues with symptom descriptions
        for symptom in symptoms:
            if not isinstance(symptom, dict):
                continue
            symptom_type = symptom.get('description', '').lower()
            symptom_definition = symptom.get('definition', '').lower()
            if 'new pet' in additional_notes and 'cough' in symptom_type:
                dx_scores['Allergic cough'] = (0.75, f"Supported by new pet exposure and symptom: {symptom_type}")
            if 'new medication' in additional_notes and 'rash' in symptom_type:
                dx_scores['Drug reaction'] = (0.75, f"Suggested by new medication and symptom: {symptom_type}")
            if 'travel' in additional_notes and 'diarrhea' in symptom_type:
                dx_scores['Traveler’s diarrhea'] = (0.75, f"Suggested by recent travel and symptom: {symptom_type}")
            if 'sedentary job' in history and 'back pain' in symptom_type:
                dx_scores['Mechanical low back pain'] = (0.75, f"Supported by sedentary lifestyle and symptom: {symptom_type}")
            if 'eczema' in history and 'skin' in symptom_definition:
                dx_scores['Eczema flare'] = (0.75, f"Supported by eczema history and symptom: {symptom_type}")
            if 'lactose intolerance' in history and 'abdominal' in symptom_definition:
                dx_scores['Lactose intolerance'] = (0.75, f"Supported by lactose intolerance history and symptom: {symptom_type}")
        if "no weight loss" in text.lower():
            dx_scores.pop("Malignancy", None)
            logger.debug("Removed Malignancy due to no weight loss")

        # Embedding-based scoring
        for dx in dx_scores:
            try:
                dx_embedding = embed_text(dx)
                similarity = torch.cosine_similarity(text_embedding.unsqueeze(0), dx_embedding.unsqueeze(0)).item()
                old_score, reasoning = dx_scores[dx]
                dx_scores[dx] = (min(old_score + similarity * 0.1, 0.9), reasoning)
            except Exception as e:
                logger.warning(f"Similarity failed for dx {dx}: {str(e)}")

        # Filter irrelevant diagnoses
        def is_relevant(dx: str) -> bool:
            dx_lower = dx.lower()
            symptom_words = {s.get('description', '').lower() for s in symptoms if isinstance(s, dict)}
            locations = {s.get('location', '').lower() for s in symptoms if isinstance(s, dict)}
            categories = {s.get('category', '').lower() for s in symptoms if isinstance(s, dict)}
            if isinstance(self.diagnosis_relevance, dict):
                for condition, required in self.diagnosis_relevance.items():
                    if dx_lower == condition.lower():
                        matches = sum(1 for word in required if word in symptom_words or word in locations or word in categories)
                        return matches >= len(required) * 0.1 or any(s in chief_complaint for s in required)
            return True

        dx_scores = {dx: score for dx, score in dx_scores.items() if is_relevant(dx)}
        logger.debug(f"Filtered dx: {dx_scores.keys()}")

        # Normalize scores
        if dx_scores:
            total_score = sum(score for score, _ in dx_scores.values())
            if total_score > 0:
                dx_scores = {dx: (score / total_score * 0.9, reason) for dx, (score, reason) in dx_scores.items()}

        ranked_dx = []
        logger.debug(f"dx_scores before sorting: {dx_scores}")
        try:
            ranked_dx = [(dx, score, reason) for dx, (score, reason) in sorted(dx_scores.items(), key=lambda x: x[1][0], reverse=True)[:5]]
        except ValueError as e:
            logger.error(f"Error sorting differentials: {str(e)}")
            ranked_dx = []
            for dx, value in dx_scores.items():
                if not isinstance(value, tuple) or len(value) != 2:
                    logger.warning(f"Invalid dx_scores entry for {dx}: {value}")
                    continue
                ranked_dx.append((dx, value[0], value[1]))
            ranked_dx = sorted(ranked_dx, key=lambda x: x[1], reverse=True)[:5]

        if not ranked_dx:
            ranked_dx = [("Undetermined", 0.1, "Insufficient data")]
            logger.warning(f"No differentials generated for chief complaint: {features.get('chief_complaint', 'None')}, symptoms: {features.get('symptoms', [])}")
        logger.debug(f"Returning differentials: {ranked_dx}")
        return ranked_dx

    def generate_management_plan(self, features: Dict, differentials: List[Tuple[str, float, str]]) -> Dict:
        """Generate tailored management plan."""
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
        high_risk = False

        # Validate differentials
        validated_differentials = []
        for diff in differentials:
            if not isinstance(diff, tuple) or len(diff) != 3:
                logger.warning(f"Invalid differential format: {diff}")
                if isinstance(diff, str):
                    validated_differentials.append((diff, 0.5, "Unknown reasoning"))
                    filtered_dx.add(diff.lower())
                continue
            dx, score, reason = diff
            if not isinstance(dx, str) or not isinstance(score, (int, float)) or not isinstance(reason, str):
                logger.warning(f"Invalid differential components: {diff}")
                continue
            validated_differentials.append(diff)
            filtered_dx.add(dx.lower())
            if score >= CONFIDENCE_THRESHOLD:
                high_risk = True

        # Primary diagnosis-based plan
        if primary_dx and isinstance(self.clinical_pathways, dict):
            for category, pathways in self.clinical_pathways.items():
                if not isinstance(pathways, dict):
                    logger.error(f"pathways not a dict for {category}: {type(pathways)}")
                    continue
                for key, path in pathways.items():
                    if not isinstance(path, dict):
                        logger.error(f"path not a dict for {key}: {type(path)}")
                        continue
                    differentials = path.get('differentials', [])
                    if not isinstance(differentials, list):
                        logger.error(f"differentials not a list for {key}: {type(differentials)}")
                        continue
                    if any(d.lower() in primary_dx for d in differentials):
                        workup = path.get('workup', {})
                        if not isinstance(workup, dict):
                            logger.error(f"workup not a dict for {key}: {type(workup)}")
                            continue
                        for w in workup.get('urgent', []):
                            parsed = parse_conditional_workup(w, symptoms)
                            if parsed:
                                plan['workup']['urgent'].append(parsed)
                        for w in workup.get('routine', []):
                            parsed = parse_conditional_workup(w, symptoms)
                            if parsed:
                                plan['workup']['routine'].append(parsed)
                        management = path.get('management', {})
                        if not isinstance(management, dict):
                            logger.error(f"management not a dict for {key}: {type(management)}")
                            continue
                        plan['treatment']['symptomatic'].extend(management.get('symptomatic', []))
                        plan['treatment']['definitive'].extend(management.get('definitive', []))
                        logger.debug(f"Added primary dx-based plan for {key}")

        # Symptom-based pathways with category consideration
        for symptom in symptoms:
            if not isinstance(symptom, dict):
                logger.warning(f"Invalid symptom format: {symptom}")
                continue
            symptom_type = symptom.get('description', '').lower()
            symptom_category = symptom.get('category', 'unknown').lower()
            location = symptom.get('location', '').lower()
            if not isinstance(self.clinical_pathways, dict):
                logger.error(f"clinical_pathways not a dict: {type(self.clinical_pathways)}")
                continue
            for category, pathways in self.clinical_pathways.items():
                if not isinstance(pathways, dict):
                    logger.error(f"pathways not a dict for {category}: {type(pathways)}")
                    continue
                for key, path in pathways.items():
                    if not isinstance(path, dict):
                        logger.error(f"path not a dict for {key}: {type(path)}")
                        continue
                    if symptom_type == key.lower() or location == key.lower() or symptom_category == category.lower():
                        differentials = path.get('differentials', [])
                        if not isinstance(differentials, list):
                            logger.error(f"differentials not a list for {key}: {type(differentials)}")
                            continue
                        for diff in differentials:
                            if not isinstance(diff, str):
                                logger.warning(f"Non-string differential for {key}: {diff}")
                                continue
                            if diff.lower() == primary_dx or diff.lower() not in filtered_dx:
                                continue
                            workup = path.get('workup', {})
                            if not isinstance(workup, dict):
                                logger.error(f"workup not a dict for {key}: {type(workup)}")
                                continue
                            for w in workup.get('urgent', []):
                                parsed = parse_conditional_workup(w, symptoms)
                                if parsed:
                                    plan['workup']['urgent'].append(parsed)
                            for w in workup.get('routine', []):
                                parsed = parse_conditional_workup(w, symptoms)
                                if parsed:
                                    plan['workup']['routine'].append(parsed)
                            management = path.get('management', {})
                            if not isinstance(management, dict):
                                logger.error(f"management not a dict for {key}: {type(management)}")
                            plan['treatment']['symptomatic'].extend(management.get('symptomatic', []))
                            plan['treatment']['definitive'].extend(management.get('definitive', []))
                            logger.debug(f"Added differential-based plan for {key}")

        # Differential-based management
        if isinstance(self.diagnosis_treatments, dict):
            for diff in validated_differentials:
                dx, _, _ = diff
                if not isinstance(dx, str):
                    logger.warning(f"Non-string differential: {dx}")
                    continue
                for diag_key, mappings in self.diagnosis_treatments.items():
                    if not isinstance(mappings, dict):
                        logger.error(f"mappings not a dict for {diag_key}: {type(mappings)}")
                        continue
                    if diag_key.lower() in dx.lower():
                        workup = mappings.get('workup', {})
                        if not isinstance(workup, dict):
                            logger.error(f"workup not a dict for {diag_key}: {type(workup)}")
                            continue
                        for w in workup.get('urgent', []):
                            parsed = parse_conditional_workup(w, symptoms)
                            if parsed:
                                plan['workup']['urgent'].append(parsed)
                        for w in workup.get('routine', []):
                            parsed = parse_conditional_workup(w, symptoms)
                            if parsed:
                                plan['workup']['routine'].append(parsed)
                        treatment = mappings.get('treatment', {})
                        if not isinstance(treatment, dict):
                            logger.error(f"treatment not a dict for {diag_key}: {type(treatment)}")
                            continue
                        plan['treatment']['definitive'].extend(treatment.get('definitive', []))
                        logger.debug(f"Added dx-based plan for {dx}")

        # Contextual adjustments with category-based rules
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
        if high_risk:
            plan['follow_up'] = ["Follow-up in 3-5 days or sooner if symptoms worsen"]
        else:
            plan['follow_up'] = ["Follow-up in 1-2 weeks"]

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
    """Parse conditional workup requirements."""
    if not isinstance(workup, str):
        logger.warning(f"Invalid workup format: {workup}")
        return ""
    if 'if' in workup.lower():
        condition = workup.lower().split('if')[1].strip()
        for symptom in symptoms:
            if not isinstance(symptom, dict):
                logger.warning(f"Invalid symptom format: {symptom}")
                continue
            if (condition in symptom.get('description', '').lower() or
                condition in symptom.get('definition', '').lower()):
                return workup.split('if')[0].strip()
    return workup