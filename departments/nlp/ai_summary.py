import re
from typing import List, Dict, Tuple, Optional
from departments.models.medicine import SOAPNote
from departments.models.records import Patient
from departments.nlp.clinical_analyzer import ClinicalAnalyzer
from departments.nlp.symptom_tracker import SymptomTracker
from departments.nlp.nlp_utils import get_patient_info
from departments.nlp.knowledge_base_io import load_knowledge_base, invalidate_cache
from departments.nlp.logging_setup import get_logger
from departments.nlp.config import MIN_CONFIDENCE_THRESHOLD, MONGO_URI, DB_NAME, KB_PREFIX, SYMPTOMS_COLLECTION
import medspacy

nlp = medspacy.load()
logger = get_logger()

# Cached knowledge base
_knowledge_base = None

def get_knowledge_base(force_cache: bool = False) -> Dict:
    """Load or return cached knowledge base."""
    global _knowledge_base
    if _knowledge_base is None or force_cache:
        logger.debug("Loading knowledge base")
        _knowledge_base = load_knowledge_base()
    return _knowledge_base

def generate_rationale(features: Dict, differentials: List[Tuple], plan: Dict) -> str:
    """Generate clinical rationale for the analysis."""
    rationale = []
    primary_dx = features.get('assessment', '').lower()
    if primary_dx:
        dx_name = re.search(r"primary assessment: (.*?)(?:\.|$)", primary_dx, re.DOTALL)
        if dx_name:
            primary_dx = dx_name.group(1).strip()
        symptoms = [f"{s.get('description', '')} (CUI: {s.get('umls_cui', 'None')})"
                    for s in features.get('symptoms', []) if isinstance(s, dict)]
        rationale.append(f"Primary diagnosis: {primary_dx} based on clinical features: {', '.join(symptoms or ['none identified'])}.")
    min_confidence = MIN_CONFIDENCE_THRESHOLD
    for diff in differentials[:3]:
        if not isinstance(diff, tuple) or len(diff) != 3:
            continue
        dx, score, reason = diff
        if score < min_confidence and len(rationale) > 1:
            continue
        rationale.append(f"{dx} (Confidence: {score:.2f}): {reason}")
    if not rationale and differentials:
        dx, score, reason = differentials[0]
        rationale.append(f"{dx} (Confidence: {score:.2f}): {reason} (Default inclusion due to limited matches)")
    if plan:
        symptomatic = plan.get('treatment', {}).get('symptomatic', [])
        definitive = plan.get('treatment', {}).get('definitive', [])
        lifestyle = plan.get('treatment', {}).get('lifestyle', [])
        treatments = symptomatic + definitive + lifestyle
        if treatments:
            rationale.append(f"Treatment plan: {', '.join(treatments)} to address primary diagnosis.")
        workup = plan.get('workup', {}).get('urgent', []) + plan.get('workup', {}).get('routine', [])
        if not workup:
            workup = ['Diagnostic evaluation pending']
        rationale.append(f"Workup: {', '.join(workup)} to confirm diagnosis.")
    return '; '.join(rationale) or 'Based on clinical features and proposed management plan.'

def normalize_symptom(symptom: str, kb: Dict) -> str:
    """Normalize symptom using synonyms from knowledge base."""
    symptom = re.sub(r'^(patient complains of |complains of |reports )\b', '', symptom.lower().strip(), flags=re.IGNORECASE)
    for canonical, aliases in kb.get('synonyms', {}).items():
        canonical_lower = canonical.lower()
        if symptom == canonical_lower or symptom in [a.lower() for a in aliases]:
            return canonical_lower
    return symptom

def generate_ai_analysis(note: SOAPNote, patient: Patient = None, force_cache: bool = False) -> str:
    """Generate a comprehensive AI clinical analysis for a SOAP note."""
    note_id = getattr(note, 'id', 'unknown')
    logger.debug(f"Generating analysis for note {note_id}, situation: {getattr(note, 'situation', 'Unknown')}")
    if not isinstance(note, SOAPNote):
        logger.error(f"Invalid note type: {type(note)}")
        return """
[=== AI CLINICAL ANALYSIS ===]
[PATIENT PROFILE]
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
DISCLAIMER: This AI-generated analysis requires clinical correlation.
"""

    # Initialize variables
    plan = {
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
        situation = getattr(note, 'situation', '').lower().strip()
        chief_complaint = situation or 'Unknown'
        patient_id = getattr(patient, 'patient_id', None) if patient else None
        patient_info = {"sex": "Unknown", "age": None}
        if patient_id:
            patient_info = get_patient_info(patient_id)
            logger.debug(f"Patient ID: {patient_id}, Patient Info: {patient_info}")
            if patient_info["sex"] == "Unknown":
                logger.warning(f"Patient info not found for patient_id: {patient_id}. Check patient records.")

        patient_profile = []
        if patient_info["sex"] != "Unknown":
            patient_profile.append(f"Sex: {patient_info['sex']}")
        if patient_info["age"] is not None:
            patient_profile.append(f"Age: {patient_info['age']}")
        patient_profile_text = '; '.join(patient_profile) or 'Not provided'

        # Load knowledge base
        kb = get_knowledge_base(force_cache)
        logger.debug(f"Knowledge base version: {kb.get('version', 'Unknown')}, last updated: {kb.get('last_updated', 'Unknown')}")
        expected_symptoms = []
        category = None
        for cat, pathways in kb.get('clinical_pathways', {}).items():
            for key, path in pathways.items():
                if any(normalize_symptom(s, kb) in normalize_symptom(chief_complaint, kb) for s in key.split('|')):
                    expected_symptoms = path.get('required_symptoms', [])
                    category = cat
                    break
            if expected_symptoms:
                break

        # Normalize expected symptoms
        expected_symptoms_normalized = [normalize_symptom(s, kb) for s in expected_symptoms]

        # Extract symptoms using SymptomTracker
        symptoms = tracker.process_note(note, chief_complaint, expected_symptoms)
        logger.debug(f"Raw symptoms from SymptomTracker for note {note_id}: {symptoms}")
        # Enrich symptoms with metadata from kb['symptoms']
        enriched_symptoms = []
        for symptom in symptoms:
            if not isinstance(symptom, dict):
                logger.warning(f"Invalid symptom type in note {note_id}: {type(symptom)}, value: {symptom}. Skipping.")
                continue
            logger.debug(f"Enriching symptom: {symptom.get('description', 'Unknown')}")
            original_symptom = symptom.copy()
            s_norm = normalize_symptom(symptom.get('description', ''), kb)
            found = False
            for kb_category, symptom_dict in kb.get('symptoms', {}).items():
                for kb_symptom_name, kb_symptom_data in symptom_dict.items():
                    if s_norm == kb_symptom_name.lower():
                        symptom.update({
                            'description': kb_symptom_data.get('description', s_norm),
                            'umls_cui': kb_symptom_data.get('umls_cui', symptom.get('umls_cui')),
                            'semantic_type': kb_symptom_data.get('semantic_type', symptom.get('semantic_type', 'Unknown')),
                            'category': kb_category
                        })
                        found = True
                        break
                if found:
                    break
            symptom['aggravating'] = original_symptom.get('aggravating', 'Unknown')
            symptom['alleviating'] = original_symptom.get('alleviating', 'Unknown')
            enriched_symptoms.append(symptom)
            logger.debug(f"Enriched symptom: {symptom.get('description', 'Unknown')}, aggravating: {symptom['aggravating'][:50]}, alleviating: {symptom['alleviating'][:50]}")

        features = analyzer.extract_clinical_features(note, expected_symptoms=expected_symptoms)
        logger.debug(f"Features from ClinicalAnalyzer for note {note_id}: {features}")
        features['symptoms'] = enriched_symptoms
        differentials = analyzer.generate_differential_dx(features, patient)
        plan = analyzer.generate_management_plan(features, differentials)
        assessment = getattr(note, 'assessment', 'Not specified') or 'Not specified'
        if differentials and differentials[0][0] == "Undetermined":
            assessment = "Not specified (insufficient data)"

        # Incorporate clinician recommendation
        recommendation = getattr(note, 'recommendation', '').lower().strip()
        if recommendation:
            plan['workup']['urgent'].append(recommendation.capitalize())
            symptom_text = ' '.join([s.get('description', '').lower() for s in enriched_symptoms if isinstance(s, dict)])
            if 'malaria' in recommendation and any(s in symptom_text for s in ['fever', 'chills', 'jaundice']):
                differentials.insert(0, ('Malaria', 0.85, 'Supported by fever, chills, jaundice, and clinician recommendation'))
            elif 'hepatitis' in recommendation and 'jaundice' in symptom_text:
                differentials.append(('Hepatitis', 0.75, 'Supported by jaundice and clinician recommendation'))

        # Symptom output with full metadata
        symptom_text = []
        for symptom in features.get('symptoms', []):
            if not isinstance(symptom, dict):
                logger.warning(f"Invalid symptom type in features for note {note_id}: {type(symptom)}, value: {symptom}. Skipping.")
                continue
            desc = symptom.get('description', '')
            severity = symptom.get('severity', 'Unknown')
            duration = symptom.get('duration', 'Unknown')
            location = symptom.get('location', 'Unknown')
            aggravating = symptom.get('aggravating', 'Unknown')
            alleviating = symptom.get('alleviating', 'Unknown')
            cui = symptom.get('umls_cui', 'None')
            sem_type = symptom.get('semantic_type', 'Unknown')
            if aggravating == 'Unknown' or alleviating == 'Unknown':
                logger.warning(f"Symptom '{desc}' has default aggravating/alleviating values: aggravating={aggravating}, alleviating={alleviating}")
            symptom_text.append(f"{desc} (CUI: {cui}, Semantic Type: {sem_type}), Severity: {severity}, Duration: {duration}, Location: {location}, Aggravating: {aggravating}, Alleviating: {alleviating}")
        symptoms_output = '\n• '.join(symptom_text) or 'None identified'

        # Negated symptoms from SymptomTracker
        negated_terms = tracker.get_negated_symptoms(note, chief_complaint)
        if 'no chronic illness' in (getattr(note, 'medical_history', '') or '').lower():
            negated_terms.add('chronic illness')
        negated_output = '; '.join(sorted(negated_terms)) or 'None reported'

        # Adjust features if symptoms don’t match expected
        if expected_symptoms_normalized:
            extracted_symptoms_normalized = [normalize_symptom(s.get('description', ''), kb) for s in features.get('symptoms', []) if isinstance(s, dict)]
            missing_symptoms = [s for s in expected_symptoms_normalized if s not in extracted_symptoms_normalized]
            if missing_symptoms:
                logger.warning(f"Extracted symptoms do not match expected symptoms: {missing_symptoms}. Using knowledge base fallback.")
                if category and category in kb['clinical_pathways']:
                    path = next((p for k, p in kb['clinical_pathways'][category].items()
                                if any(normalize_symptom(s, kb) in normalize_symptom(chief_complaint, kb) for s in k.split('|'))), None)
                    if path:
                        existing_symptoms = {normalize_symptom(s.get('description', ''), kb) for s in features['symptoms'] if isinstance(s, dict)}
                        for s in path.get('required_symptoms', []):
                            s_normalized = normalize_symptom(s, kb)
                            if s_normalized not in existing_symptoms:
                                kb_symptom_data = None
                                for kb_category, symptom_dict in kb.get('symptoms', {}).items():
                                    if s_normalized in symptom_dict:
                                        kb_symptom_data = symptom_dict[s_normalized]
                                        break
                                features['symptoms'].append({
                                    'description': s_normalized,
                                    'severity': 'Moderate',
                                    'duration': 'Unknown',
                                    'location': 'Unknown',
                                    'aggravating': 'Unknown',
                                    'alleviating': 'Unknown',
                                    'category': category,
                                    'umls_cui': kb_symptom_data.get('umls_cui') if kb_symptom_data else None,
                                    'semantic_type': kb_symptom_data.get('semantic_type', 'Unknown') if kb_symptom_data else 'Unknown'
                                })
                        differentials.extend([(dx, 0.7 - 0.1*i, f"Supported by {', '.join(path.get('required_symptoms', []))}")
                                            for i, dx in enumerate(path.get('differentials', []))])
                        plan['workup'].update(path.get('workup', {'urgent': [], 'routine': []}))
                        plan['treatment'].update(path.get('management', {'symptomatic': [], 'definitive': [], 'lifestyle': []}))
                        plan['follow_up'] = path.get('follow_up', ['Follow-up in 2 weeks'])
                        plan['references'] = path.get('references', [])
                        if path.get('differentials', []):
                            assessment = f"Likely {path['differentials'][0].lower()} (pending workup)"

        # Differential diagnosis output
        differential_text = []
        for i, diff in enumerate(differentials):
            if not isinstance(diff, tuple) or len(diff) != 3:
                continue
            dx, score, reason = diff
            likelihood = 'Most likely' if i == 0 else 'Less likely'
            differential_text.append(f"{dx} (Confidence: {score:.2f}): {reason} ({likelihood})")
        differential_output = '\n• '.join(differential_text) or 'Undetermined: Insufficient data'

        # High-risk conditions from knowledge base
        high_risk_conditions = set(kb.get('high_risk_conditions', []))
        high_risk = any(dx.lower() in high_risk_conditions for dx, _, _ in differentials if isinstance(dx, str))
        disclaimer = "High-risk conditions detected; urgent review recommended." if high_risk else "This AI-generated analysis requires clinical correlation."

        # References
        references = set()
        for cat, pathways in kb.get('clinical_pathways', {}).items():
            for key, path in pathways.items():
                if not isinstance(path, dict):
                    continue
                if any(normalize_symptom(s, kb) in normalize_symptom(chief_complaint, kb) for s in key.split('|')):
                    refs = path.get('references', [])
                    if isinstance(refs, list):
                        references.update(refs)
        references_output = '\n• '.join(sorted(references)) or 'None available'

        # Follow-up
        follow_up = 'Follow-up in 2 weeks'
        recommendation = getattr(note, 'recommendation', '')
        if recommendation:
            follow_up_match = re.search(r'Follow-Up:\s*([^\.]+)', recommendation, re.IGNORECASE)
            if follow_up_match:
                follow_up = follow_up_match.group(1).strip()
        elif category:
            follow_up = plan.get('follow_up', ['Follow-up in 2 weeks'])[0]

        # Add medication history to treatment
        medication_history = getattr(note, 'medication_history', '').lower().strip()
        if medication_history and 'paracetamol' in medication_history:
            plan['treatment']['symptomatic'].append('Continue paracetamol 1000 mg for headache as needed')

        analysis_output = f"""
[=== AI CLINICAL ANALYSIS ===]
[PATIENT PROFILE]
{patient_profile_text}
[CHIEF CONCERN]
{chief_complaint}
[SYMPTOMS]
• {symptoms_output}
[NEGATED SYMPTOMS]
{negated_output}
[DIAGNOSIS]
{assessment}
[DIFFERENTIAL DIAGNOSIS]
• {differential_output}
[CLINICAL RATIONALE]
{generate_rationale(features, differentials, plan)}
[RECOMMENDED WORKUP]
• Urgent: {'; '.join(sorted(set(plan['workup']['urgent']))) or 'None'}
• Routine: {'; '.join(sorted(set(plan['workup']['routine']))) or 'None'}
[TREATMENT OPTIONS]
• Symptomatic: {'; '.join(sorted(set(plan['treatment']['symptomatic']))) or 'None'}
• Definitive: {'; '.join(sorted(set(plan['treatment']['definitive']))) or 'Pending diagnosis'}
• Lifestyle: {'; '.join(sorted(set(plan['treatment'].get('lifestyle', [])))) or 'None'}
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
            'diagnosis': assessment,
            'differentials': differential_output,
            'rationale': 'Analysis failed due to processing error',
            'workup_urgent': '; '.join(sorted(set(plan['workup']['urgent']))) or 'None',
            'workup_routine': '; '.join(sorted(set(plan['workup']['routine']))) or 'None',
            'treatment_symptomatic': '; '.join(sorted(set(plan['treatment']['symptomatic']))) or 'None',
            'treatment_definitive': '; '.join(sorted(set(plan['treatment']['definitive']))) or 'Pending diagnosis',
            'treatment_lifestyle': '; '.join(sorted(set(plan['treatment'].get('lifestyle', [])))) or 'None',
            'follow_up': follow_up,
            'references': references_output,
            'kb_version': kb.get('version', 'Unknown') if kb else 'Unknown',
            'kb_updated': kb.get('last_updated', 'Unknown') if kb else 'Unknown',
            'disclaimer': 'This AI-generated analysis requires clinical correlation.'
        }
        return f"""
[=== AI CLINICAL ANALYSIS ===]
[PATIENT PROFILE]
{partial_output['patient_profile']}
[CHIEF CONCERN]
{partial_output['chief_complaint']}
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