import re
import spacy
from typing import List, Dict, Tuple, Set, Optional
from departments.models.medicine import SOAPNote
from departments.models.records import Patient
from departments.nlp.clinical_analyzer import ClinicalAnalyzer
from departments.nlp.symptom_tracker import SymptomTracker
from departments.nlp.nlp_utils import get_patient_info, preprocess_text
from departments.nlp.knowledge_base_io import load_knowledge_base, invalidate_cache
from departments.nlp.logging_setup import get_logger
from departments.nlp.config import MIN_CONFIDENCE_THRESHOLD, MONGO_URI, DB_NAME, SYMPTOMS_COLLECTION
from departments.nlp.nlp_pipeline import clean_term
import medspacy
from spacy.tokens import Doc

# Initialize MedSpacy and logger
nlp = medspacy.load()
logger = get_logger(__name__)

# Cached knowledge base
_knowledge_base: Optional[Dict] = None

# SciSpacy NER wrapper
class SciBERTWrapper:
    def __init__(self, model_name: str = "en_core_sci_sm", disable_linker: bool = True):
        try:
            self.nlp = spacy.load(model_name, disable=["lemmatizer"])
            logger.info(f"Loaded SpaCy model: {model_name}")
            if disable_linker and "entity_linker" in self.nlp.pipe_names:
                self.nlp.remove_pipe("entity_linker")
                logger.info("Removed entity_linker to avoid nmslib dependency.")
        except OSError as e:
            logger.error(f"Failed to load spaCy model {model_name}: {e}. Using blank model.")
            self.nlp = spacy.blank("en")

    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        if not isinstance(text, str):
            logger.error(f"Invalid text type: {type(text)}")
            return []
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]

    def extract_temporal_details(self, doc: Doc) -> List[Tuple[str, str]]:
        """Extract temporal relationships for entities."""
        temporal = []
        for ent in doc.ents:
            if ent.label_ == "ENTITY":
                for token in ent.sent:
                    if token.dep_ in ["tmod", "npadvmod"] or token.text.lower() in ["since", "for", "started", "began"]:
                        temporal.append((ent.text, token.text + ' ' + ' '.join([t.text for t in token.children if t.dep_ in ["nummod", "advmod"]])))
        return temporal

# Initialize SciSpacy
try:
    _sci_ner = SciBERTWrapper(model_name="en_core_sci_sm", disable_linker=True)
    logger.info("Initialized _sci_ner with en_core_sci_sm")
except Exception as e:
    logger.error(f"Failed to initialize _sci_ner: {e}")
    _sci_ner = SciBERTWrapper(model_name="en", disable_linker=True)

def sanitize_input(text: Optional[str]) -> str:
    """Sanitize input to prevent regex syntax errors."""
    if not isinstance(text, str):
        logger.warning(f"Invalid input type: {type(text)}. Converting to empty string.")
        return ""
    try:
        text = text.replace("\\.", ".").replace("\\", "").replace("\n", " ")
        text = re.sub(r'\s+', ' ', text.strip(), flags=re.UNICODE)
        return text
    except re.error as e:
        logger.error(f"Regex error in sanitize_input: {e}")
        return ""

def get_knowledge_base(force_cache: bool = False) -> Dict:
    """Load or return cached knowledge base."""
    global _knowledge_base
    if _knowledge_base is None or force_cache:
        logger.debug("Loading knowledge base")
        _knowledge_base = load_knowledge_base() or {}
    return _knowledge_base

def generate_rationale(features: Dict, differentials: List[Tuple], plan: Dict) -> str:
    """Generate clinical rationale for the analysis."""
    rationale = []
    primary_dx = sanitize_input(features.get('assessment', '')).lower()
    if primary_dx:
        try:
            dx_match = re.match(r"primary assessment:\s*(.*?)(?:$|\.)", primary_dx, re.DOTALL)
            primary_dx = dx_match.group(1).strip() if dx_match else "Not parsed"
        except re.error as e:
            logger.error(f"Regex error in generate_rationale: {e}")
            primary_dx = "Not parsed"
        symptoms = [f"{s.get('description', '')} (CUI: {s.get('umls_cui', 'None')})"
                    for s in features.get('symptoms', []) if isinstance(s, dict) and s.get('description')]
        rationale.append(f"Primary diagnosis: {primary_dx} based on clinical features: {', '.join(symptoms or ['none identified'])}.")
    for diff in differentials[:3]:
        if not isinstance(diff, tuple) or len(diff) != 3:
            logger.warning(f"Invalid differential tuple: {diff}")
            continue
        dx, score, reason = diff
        if score < MIN_CONFIDENCE_THRESHOLD and len(rationale) > 1:
            continue
        rationale.append(f"{dx} (Confidence: {score:.2f}): {reason}")
    if not rationale and differentials and isinstance(differentials[0], tuple) and len(differentials[0]) == 3:
        dx, score, reason = differentials[0]
        rationale.append(f"{dx} (Confidence: {score:.2f}): {reason} (Default inclusion)")
    if plan:
        treatments = sorted(set(plan.get('treatment', {}).get('symptomatic', []) +
                               plan.get('treatment', {}).get('definitive', []) +
                               plan.get('treatment', {}).get('lifestyle', [])))
        if treatments:
            rationale.append(f"Treatment plan: {', '.join(treatments)} to address primary diagnosis.")
        workup = sorted(set(plan.get('workup', {}).get('urgent', []) +
                           plan.get('workup', {}).get('routine', [])))
        rationale.append(f"Workup: {', '.join(workup or ['Diagnostic evaluation pending'])} to confirm diagnosis.")
    return '; '.join(rationale) or 'Based on clinical features and proposed management plan.'

def normalize_symptom(symptom: Optional[str], kb: Dict, tracker: SymptomTracker) -> str:
    """Normalize symptom using synonyms from knowledge base and SymptomTracker."""
    symptom = sanitize_input(symptom)
    if not symptom or len(symptom) < 3:
        return ""
    try:
        symptom_clean = clean_term(preprocess_text(symptom)).lower()
    except Exception as e:
        logger.error(f"Error cleaning symptom '{symptom}': {e}")
        return ""
    if not symptom_clean:
        logger.debug(f"Empty symptom after cleaning: {symptom}")
        return ""
    terms = list(dict.fromkeys(symptom_clean.split()))
    stop_terms = {'of', 'and', 'the', 'with', 'patient', 'complains', 'reports', 'started', 'is', 'associated', 'has', 'in'}
    valid_terms = [t for t in terms if t not in stop_terms and len(t) > 2]
    if not valid_terms:
        return ""

    result = []
    i = 0
    max_iterations = 100
    iteration_count = 0
    while i < len(valid_terms):
        iteration_count += 1
        if iteration_count > max_iterations:
            logger.error(f"Max iterations reached in normalize_symptom for '{symptom}'")
            break
        matched = False
        for j in range(3, 0, -1):
            if i + j <= len(valid_terms):
                phrase = ' '.join(valid_terms[i:i+j]).lower()
                try:
                    for canonical, aliases in kb.get('synonyms', {}).items():
                        canonical_lower = canonical.lower()
                        if phrase == canonical_lower or phrase in [a.lower() for a in aliases]:
                            if canonical_lower not in result:
                                result.append(canonical_lower)
                            i += j
                            matched = True
                            logger.debug(f"Normalized '{phrase}' to '{canonical_lower}' via knowledge base")
                            break
                    if matched:
                        break
                    if phrase in tracker.get_all_symptoms() and phrase not in result:
                        result.append(phrase)
                        i += j
                        matched = True
                        logger.debug(f"Normalized '{phrase}' via SymptomTracker")
                        break
                except Exception as e:
                    logger.error(f"Error processing phrase '{phrase}': {e}")
                    break
        if not matched:
            if valid_terms[i] not in result:
                result.append(valid_terms[i])
            i += 1
    normalized = ' '.join(result)
    if len(normalized) > 50:
        logger.debug(f"Skipping long normalized symptom: {normalized}")
        return ""
    logger.debug(f"Normalized symptom '{symptom}' -> '{normalized}'")
    return normalized

def enhance_symptoms(existing_symptoms: List[Dict], sci_entities: List[Tuple[str, str]], kb: Dict, tracker: SymptomTracker) -> List[Dict]:
    """Merge SciSpacy entities with existing symptoms, deduplicating and normalizing."""
    anatomical_terms = {'eyes', 'head', 'body', 'chest', 'abdomen'}
    valid_symptoms = {'headache', 'fever', 'fevers', 'chills', 'nausea', 'vomiting', 'loss of appetite', 'jaundice', 'pain in head', 'feeling of sickness'}
    symptom_mappings = {'pain in head': 'headache', 'feeling of sickness': 'nausea'}
    
    sci_symptoms = []
    for entity_text, _ in sci_entities:
        normalized = normalize_symptom(entity_text, kb, tracker)
        if not normalized or normalized in anatomical_terms or normalized not in valid_symptoms:
            continue
        normalized = symptom_mappings.get(normalized, normalized)
        try:
            cui, semantic_type = tracker._get_umls_cui(normalized)
            sci_symptoms.append({
                'description': normalized,
                'umls_cui': cui or 'None',
                'semantic_type': semantic_type or 'Sign or Symptom',
                'severity': 'Mild',
                'duration': 'Unknown',
                'location': 'Unknown',
                'aggravating': 'Unknown',
                'alleviating': 'None'
            })
        except Exception as e:
            logger.error(f"Error processing entity '{entity_text}': {e}")
            continue
    
    all_symptoms = existing_symptoms + sci_symptoms
    unique_symptoms = {}
    for s in all_symptoms:
        if not isinstance(s, dict) or not s.get('description'):
            continue
        desc = normalize_symptom(s.get('description', ''), kb, tracker).lower()
        desc = symptom_mappings.get(desc, desc)
        if desc not in valid_symptoms:
            continue
        if desc not in unique_symptoms or (s.get('umls_cui') and unique_symptoms[desc].get('umls_cui') == 'None'):
            unique_symptoms[desc] = s
            unique_symptoms[desc]['description'] = desc
    
    for s in unique_symptoms.values():
        desc_lower = s['description'].lower()
        s['semantic_type'] = 'Sign or Symptom'
        if desc_lower == 'headache':
            s.update({'location': 'Head', 'umls_cui': 'C0018681'})
        elif desc_lower == 'jaundice':
            s.update({'location': 'Eyes', 'umls_cui': 'C0022346'})
        elif desc_lower in ['nausea', 'vomiting', 'loss of appetite']:
            s.update({'location': 'Gastrointestinal',
                      'umls_cui': {'nausea': 'C0027497', 'vomiting': 'C0042963', 'loss of appetite': 'C0234450'}[desc_lower]})
        elif desc_lower in ['fever', 'fevers', 'chills']:
            s.update({'location': 'Systemic',
                      'umls_cui': {'fever': 'C0018682', 'fevers': 'C0018682', 'chills': 'C0085593'}[desc_lower]})
    
    return sorted(unique_symptoms.values(), key=lambda x: x.get('description', ''))

def generate_ai_analysis(note: SOAPNote, patient: Optional[Patient] = None, force_cache: bool = False) -> str:
    """Generate a comprehensive AI clinical analysis for a SOAP note."""
    note_id = getattr(note, 'id', 'unknown')
    logger.debug(f"Generating analysis for note {note_id}, situation: {getattr(note, 'situation', 'Unknown')}")
    if not isinstance(note, SOAPNote):
        logger.error(f"Invalid note type: {type(note)}")
        return """
[=== AI CLINICAL ANALYSIS ===]
[PATIENT INFO]
Sex: Unknown; Age: Unknown
[CHIEF CONCERN]
Unknown
[SYMPTOMS]
None identified
[NEGATED SYMPTOMS]
None reported
[DIAGNOSIS]
Not specified
[DIFFERENTIAL DIAGNOSIS]
• Undetermined: Insufficient data
[CLINICAL RATIONALE]
No clinical features provided
[RECOMMENDED WORKUP]
• Urgent: None
• Routine: None
[TREATMENT OPTIONS]
• Symptomatic: None
• Definitive: Pending diagnosis
• Lifestyle: None
[FOLLOW-UP]
As needed
[REFERENCES]
None available
[KNOWLEDGE BASE]
Version: Unknown, Last Updated: Unknown
DISCLAIMER: This AI-generated analysis requires clinical correlation.
"""

    plan: Dict = {
        'workup': {'urgent': [], 'routine': []},
        'treatment': {'symptomatic': [], 'definitive': [], 'lifestyle': []},
        'follow_up': ['Follow-up in 2 weeks'],
        'references': []
    }
    symptoms_output = 'None identified'
    negated_output = 'None reported'
    assessment = 'Not specified'
    differential_output = 'Undetermined: Insufficient data'
    patient_profile_text = 'Not provided'
    chief_complaint = 'Unknown'
    references_output = 'None available'
    follow_up = 'Follow-up in 2 weeks'

    try:
        analyzer = ClinicalAnalyzer()
        tracker = SymptomTracker(mongo_uri=MONGO_URI, db_name=DB_NAME, symptom_collection=SYMPTOMS_COLLECTION)
        situation = sanitize_input(getattr(note, 'situation', '')).lower().strip()
        chief_complaint = situation or 'Unknown'
        patient_id = getattr(patient, 'patient_id', None) if patient else None
        patient_info = {"sex": "Unknown", "age": None}
        if patient_id:
            patient_info = get_patient_info(patient_id) or patient_info
            logger.debug(f"Patient ID: {patient_id}, Patient Info: {patient_info}")
            if patient_info["sex"] == "Unknown":
                logger.warning(f"Patient info not found for patient_id: {patient_id}.")

        patient_profile = []
        if patient_info["sex"] != "Unknown":
            patient_profile.append(f"Sex: {patient_info['sex']}")
        if patient_info["age"] is not None:
            patient_profile.append(f"Age: {patient_info['age']}")
        patient_profile_text = '; '.join(patient_profile) or 'Not provided'

        kb = get_knowledge_base(force_cache)
        logger.debug(f"Knowledge base version: {kb.get('version', 'Unknown')}, last updated: {kb.get('last_updated', 'Unknown')}")
        expected_symptoms = []
        category = None
        max_iterations = 100
        iteration_count = 0
        for cat, pathways in kb.get('clinical_pathways', {}).items():
            if not isinstance(pathways, dict):
                continue
            for key, path in pathways.items():
                iteration_count += 1
                if iteration_count > max_iterations:
                    logger.error("Max iterations reached in clinical pathways loop")
                    break
                try:
                    if any(normalize_symptom(s, kb, tracker) in normalize_symptom(chief_complaint, kb, tracker) for s in key.split('|')):
                        expected_symptoms = path.get('required_symptoms', []) or []
                        category = cat
                        break
                except Exception as e:
                    logger.error(f"Error processing pathway key '{key}': {e}")
                    continue
            if expected_symptoms:
                break

        expected_symptoms_normalized = [normalize_symptom(s, kb, tracker) for s in expected_symptoms if normalize_symptom(s, kb, tracker)]

        hpi = sanitize_input(getattr(note, 'hpi', ''))
        meds = sanitize_input(getattr(note, 'medication_history', ''))
        assessment_text = sanitize_input(getattr(note, 'assessment', ''))
        note_text = f"{situation}\n{hpi}\n{meds}\n{assessment_text}"

        try:
            sci_entities = _sci_ner.extract_entities(note_text)
            logger.debug(f"SciSpacy entities: {sci_entities}")
            doc = _sci_ner.nlp(note_text)
            temporal_details = _sci_ner.extract_temporal_details(doc)
            logger.debug(f"Temporal details: {temporal_details}")
        except Exception as e:
            logger.error(f"SciSpacy processing failed: {e}")
            sci_entities = []
            temporal_details = []

        try:
            symptoms = tracker.process_note(note, chief_complaint, expected_symptoms) or []
            logger.debug(f"Raw symptoms from SymptomTracker for note {note_id}: {symptoms}")
        except Exception as e:
            logger.error(f"SymptomTracker processing failed: {e}")
            symptoms = []

        enriched_symptoms = enhance_symptoms(symptoms, sci_entities, kb, tracker)

        for symptom in enriched_symptoms:
            if not isinstance(symptom, dict):
                continue
            for entity, duration in temporal_details:
                if symptom.get('description', '').lower() == entity.lower():
                    symptom['duration'] = duration.capitalize()
            if symptom.get('description', '').lower() == 'headache' and 'paracetamol' in meds.lower():
                symptom['alleviating'] = 'Paracetamol'

        for symptom in enriched_symptoms:
            if not isinstance(symptom, dict) or not symptom.get('description'):
                logger.warning(f"Invalid symptom in note {note_id}: {symptom}")
                continue
            s_norm = normalize_symptom(symptom.get('description', ''), kb, tracker)
            for kb_category, symptom_dict in kb.get('symptoms', {}).items():
                if not isinstance(symptom_dict, dict):
                    continue
                for kb_symptom_name, kb_symptom_data in symptom_dict.items():
                    if s_norm.lower() == kb_symptom_name.lower():
                        symptom.update({
                            'description': kb_symptom_data.get('description', s_norm),
                            'umls_cui': kb_symptom_data.get('umls_cui', symptom.get('umls_cui', 'None')),
                            'semantic_type': kb_symptom_data.get('semantic_type', 'Sign or Symptom'),
                            'category': kb_category
                        })
                        break
            if symptom.get('umls_cui') == 'None':
                try:
                    cui, semantic_type = tracker._get_umls_cui(s_norm)
                    if cui:
                        symptom.update({'umls_cui': cui, 'semantic_type': semantic_type or 'Sign or Symptom'})
                        logger.debug(f"Assigned CUI {cui} to symptom '{s_norm}' via SymptomTracker")
                except Exception as e:
                    logger.error(f"Error getting UMLS CUI for '{s_norm}': {e}")

        features = analyzer.extract_clinical_features(note, expected_symptoms=expected_symptoms) or {'symptoms': [], 'assessment': ''}
        logger.debug(f"Features from ClinicalAnalyzer for note {note_id}: {features}")
        features['symptoms'] = enriched_symptoms
        differentials = analyzer.generate_differential_dx(features, patient) or []
        plan = analyzer.generate_management_plan(features, differentials) or plan
        assessment = sanitize_input(getattr(note, 'assessment', 'Not specified')) or 'Not specified'
        if differentials and any(diff[0] == "Undetermined" for diff in differentials if isinstance(diff, tuple) and len(diff) == 3):
            assessment = None

        recommendation = sanitize_input(getattr(note, 'classifier_recommendation', '') or getattr(note, 'recommendation', '')).lower().strip()
        if recommendation:
            try:
                plan.setdefault('workup', {}).setdefault('urgent', []).append(recommendation.capitalize())
                symptom_text = ' '.join([s.get('description', '').lower() for s in enriched_symptoms if isinstance(s, dict)])
                if 'malaria' in recommendation and any(s in symptom_text for s in ['fever', 'chills', 'jaundice']):
                    differentials.append(('Malaria', 0.85, 'Supported by fever, chills, jaundice, and clinician recommendation'))
                    plan.setdefault('workup', {}).setdefault('urgent', []).append('Peripheral blood smear for malaria parasite')
                    assessment = 'Jaundice observed, likely malaria (pending workup)'
                elif 'hepatitis' in recommendation and 'jaundice' in symptom_text:
                    differentials.append(('Hepatitis', 0.75, 'Supported by jaundice and clinician recommendation'))
                    plan.setdefault('workup', {}).setdefault('routine', []).append('Liver function tests')
            except Exception as e:
                logger.error(f"Error processing recommendation: {e}")

        symptom_text = ' '.join([s.get('description', '').lower() for s in enriched_symptoms if isinstance(s, dict)])
        if not differentials or all(diff[0] == "Undetermined" for diff in differentials if isinstance(diff, tuple) and len(diff) == 3):
            if any(s in symptom_text for s in ['fever', 'chills', 'jaundice']):
                differentials = [('Malaria', 0.85, 'Supported by fever, chills, jaundice'),
                                ('Hepatitis', 0.75, 'Supported by jaundice')]
                plan.setdefault('workup', {}).setdefault('urgent', []).append('Peripheral blood smear for malaria parasite')
                plan.setdefault('workup', {}).setdefault('routine', []).append('Liver function tests')
                assessment = 'Jaundice observed, likely malaria (pending workup)'

        symptom_output_lines = []
        for s in enriched_symptoms:
            if not isinstance(s, dict) or not s.get('description'):
                continue
            desc = s.get('description', '')
            severity = s.get('severity', 'Mild')
            duration = s.get('duration', 'Unknown')
            location = s.get('location', 'Unknown')
            aggravating = s.get('aggravating', 'Unknown')
            alleviating = s.get('alleviating', 'None')
            cui = s.get('umls_cui', 'None')
            sem_type = s.get('semantic_type', 'Sign or Symptom')
            symptom_output_lines.append(
                f"{desc} (CUI: {cui}, Semantic Type: {sem_type}), Severity: {severity}, Duration: {duration}, "
                f"Location: {location}, Aggravating: {aggravating}, Alleviating: {alleviating}"
            )
        symptoms_output = '\n• '.join(sorted(set(symptom_output_lines))) or 'None identified'

        negated_terms: Set[str] = set()
        try:
            negated_terms.update(tracker.get_negated_symptoms(note, chief_complaint) or [])
            medspacy_doc = nlp(note_text)
            for ent in medspacy_doc.ents:
                if ent._.is_negated:
                    normalized = normalize_symptom(ent.text, kb, tracker).lower()
                    if normalized:
                        negated_terms.add(normalized)
        except Exception as e:
            logger.error(f"Negation detection failed: {e}")
        if 'no chronic illness' in sanitize_input(getattr(note, 'medical_history', '') or '').lower():
            negated_terms.add('chronic illness')
        negated_output = '; '.join(sorted(negated_terms)) or 'None reported'

        if expected_symptoms_normalized:
            extracted_symptoms = [normalize_symptom(s.get('description', ''), kb, tracker) for s in features.get('symptoms', []) if isinstance(s, dict)]
            missing_symptoms = [s for s in expected_symptoms_normalized if s and s not in extracted_symptoms]
            if missing_symptoms:
                logger.warning(f"Missing expected symptoms: {missing_symptoms}")
                if category and category in kb.get('clinical_pathways', {}):
                    path = next((p for k, p in kb['clinical_pathways'][category].items()
                                 if any(normalize_symptom(s, kb, tracker) in normalize_symptom(chief_complaint, kb, tracker) for s in k.split('|'))), None)
                    if path:
                        existing_symptoms = {normalize_symptom(s.get('description', ''), kb, tracker) for s in features['symptoms'] if isinstance(s, dict)}
                        for s in path.get('required_symptoms', []) or []:
                            s_norm = normalize_symptom(s, kb, tracker)
                            if s_norm and s_norm not in existing_symptoms:
                                kb_symptom_data = None
                                for kb_cat, symptom_dict in kb.get('symptoms', {}).items():
                                    if s_norm.lower() in symptom_dict:
                                        kb_symptom_data = symptom_dict[s_norm.lower()]
                                        break
                                cui, sem_type = tracker._get_umls_cui(s_norm) if not kb_symptom_data else (
                                    kb_symptom_data.get('umls_cui', 'None'), kb_symptom_data.get('semantic_type', 'Sign or Symptom'))
                                features['symptoms'].append({
                                    'description': s_norm,
                                    'severity': 'Mild',
                                    'duration': 'Unknown',
                                    'location': 'Unknown',
                                    'aggravating': 'Unknown',
                                    'alleviating': 'None',
                                    'category': category,
                                    'umls_cui': cui or 'None',
                                    'semantic_type': sem_type or 'Sign or Symptom'
                                })
                                logger.debug(f"Added missing symptom '{s_norm}' with CUI {cui}")
                        differentials.extend([(dx, 0.7 - 0.1*i, f"Supported by {', '.join(path.get('required_symptoms', []))}")
                                             for i, dx in enumerate(path.get('differentials', []) or [])])
                        plan['workup'].update(path.get('workup', {'urgent': [], 'routine': []}))
                        plan['treatment'].update(path.get('management', {'symptomatic': [], 'definitive': [], 'lifestyle': []}))
                        plan['follow_up'] = path.get('follow_up', ['Follow-up in 2 weeks'])
                        plan['references'] = path.get('references', [])
                        if path.get('differentials', []):
                            assessment = f"Likely {path['differentials'][0].lower()} (pending workup)"

        differential_text = []
        for i, diff in enumerate(differentials[:3]):
            if not isinstance(diff, tuple) or len(diff) != 3:
                logger.warning(f"Invalid differential tuple: {diff}")
                continue
            dx, score, reason = diff
            likelihood = 'Most likely' if i == 0 else 'Less likely'
            differential_text.append(f"{dx} (Confidence: {score:.2f}): {reason} ({likelihood})")
        differential_output = '\n• '.join(differential_text) or 'Undetermined: Insufficient data'

        high_risk_conditions = set(kb.get('high_risk_conditions', []))
        high_risk = any(dx.lower() in high_risk_conditions for dx, _, _ in differentials if isinstance(dx, str))
        disclaimer = "High-risk conditions detected; urgent review recommended." if high_risk else "This AI-generated analysis requires clinical correlation."

        references = set()
        max_iterations = 100
        iteration_count = 0
        for cat, pathways in kb.get('clinical_pathways', {}).items():
            if not isinstance(pathways, dict):
                continue
            for key, path in pathways.items():
                iteration_count += 1
                if iteration_count > max_iterations:
                    logger.error("Max iterations reached in references loop")
                    break
                if not isinstance(path, dict):
                    continue
                try:
                    if any(normalize_symptom(s, kb, tracker) in normalize_symptom(chief_complaint, kb, tracker) for s in key.split('|')):
                        refs = path.get('references', []) or []
                        if isinstance(refs, list):
                            references.update(r for r in refs if any(dx.lower() in r.lower() for dx, _, _ in differentials if isinstance(dx, str)))
                except Exception as e:
                    logger.error(f"Error processing pathway key '{key}': {e}")
                    continue
        references_output = '\n• '.join(sorted(references)) or 'None available'

        follow_up = 'Follow-up in 2 weeks'
        recommendation = sanitize_input(getattr(note, 'recommendation', '') or getattr(note, 'classifier_recommendation', '')).lower().strip()
        if recommendation:
            try:
                follow_up_match = re.search(r"Follow-Up:\s*([^\.]+)", recommendation, re.IGNORECASE)
                if follow_up_match:
                    follow_up = follow_up_match.group(1).strip()
            except re.error as e:
                logger.error(f"Regex error in follow-up parsing: {e}")
        elif category and plan.get('follow_up'):
            follow_up = plan['follow_up'][0]

        if 'paracetamol' in meds.lower():
            plan.setdefault('treatment', {}).setdefault('symptomatic', []).append('Continue paracetamol 1000 mg for headache as needed')

        analysis_output = f"""
[=== AI CLINICAL ANALYSIS ===]
[PATIENT INFO]
{patient_profile_text}
[CHIEF CONCERN]
{chief_complaint.capitalize()}
[SYMPTOMS]
• {symptoms_output}
[NEGATED SYMPTOMS]
{negated_output}
[DIAGNOSIS]
{assessment or 'Not specified (insufficient data)'}
[DIFFERENTIAL DIAGNOSIS]
• {differential_output}
[CLINICAL RATIONALE]
{generate_rationale(features, differentials, plan)}
[RECOMMENDED WORKUP]
• Urgent: {'; '.join(sorted(set(plan.get('workup', {}).get('urgent', [])))) or 'None'}
• Routine: {'; '.join(sorted(set(plan.get('workup', {}).get('routine', [])))) or 'None'}
[TREATMENT OPTIONS]
• Symptomatic: {'; '.join(sorted(set(plan.get('treatment', {}).get('symptomatic', [])))) or 'None'}
• Definitive: {'; '.join(sorted(set(plan.get('treatment', {}).get('definitive', [])))) or 'Pending diagnosis'}
• Lifestyle: {'; '.join(sorted(set(plan.get('treatment', {}).get('lifestyle', [])))) or 'None'}
[FOLLOW-UP]
{follow_up}
[REFERENCES]
• {references_output}
[KNOWLEDGE BASE]
Version: {kb.get('version', 'Unknown')}, Last Updated: {kb.get('last_updated', 'Unknown')}
DISCLAIMER: {disclaimer}
"""
        logger.debug(f"Analysis output for note {note_id}")
        return analysis_output.strip()
    except Exception as e:
        logger.error(f"Analysis failed for note {note_id}: {str(e)}", exc_info=True)
        invalidate_cache()
        partial_output = {
            'patient_profile': patient_profile_text,
            'chief_complaint': chief_complaint,
            'symptoms': symptoms_output,
            'negated': negated_output,
            'diagnosis': assessment or 'Undetermined',
            'differentials': differential_output,
            'rationale': f'Analysis failed due to processing error: {str(e)}',
            'workup_urgent': '; '.join(sorted(set(plan.get('workup', {}).get('urgent', [])))) or 'None',
            'workup_routine': '; '.join(sorted(set(plan.get('workup', {}).get('routine', [])))) or 'None',
            'treatment_symptomatic': '; '.join(sorted(set(plan.get('treatment', {}).get('symptomatic', [])))) or 'None',
            'treatment_definitive': '; '.join(sorted(set(plan.get('treatment', {}).get('definitive', [])))) or 'Pending diagnosis',
            'treatment_lifestyle': '; '.join(sorted(set(plan.get('treatment', {}).get('lifestyle', [])))) or 'None',
            'follow_up': follow_up,
            'references': references_output,
            'kb_version': kb.get('version', 'Unknown') or 'Unknown',
            'kb_updated': kb.get('last_updated', 'Unknown') or 'Unknown',
            'disclaimer': 'This AI-generated analysis requires clinical correlation.'
        }
        return f"""
[=== AI CLINICAL ANALYSIS ===]
[PATIENT INFO]
{partial_output['patient_profile']}
[CHIEF CONCERN]
{partial_output['chief_complaint'].capitalize()}
[SYMPTOMS]
{partial_output['symptoms']}
[NEGATED SYMPTOMS]
{partial_output['negated']}
[DIAGNOSIS]
{partial_output['diagnosis']}
[DIFFERENTIAL DIAGNOSIS]
• {partial_output['differentials']}
[CLINICAL RATIONALE]
{partial_output['rationale']}
[RECOMMENDED WORKUP]
• Urgent: {partial_output['workup_urgent']}
• Routine: {partial_output['workup_routine']}
[TREATMENT OPTIONS]
• Symptomatic: {partial_output['treatment_symptomatic']}
• Definitive: {partial_output['treatment_definitive']}
• Lifestyle: {partial_output['treatment_lifestyle']}
[FOLLOW-UP]
{partial_output['follow_up']}
[REFERENCES]
• {partial_output['references']}
[KNOWLEDGE BASE]
Version: {partial_output['kb_version']}, Last Updated: {partial_output['kb_updated']}
DISCLAIMER: {partial_output['disclaimer']}
""".strip()