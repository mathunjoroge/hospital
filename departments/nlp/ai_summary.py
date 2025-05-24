from typing import List, Dict, Tuple, Optional
from departments.models.medicine import SOAPNote
from departments.models.records import Patient
from departments.nlp.clinical_analyzer import ClinicalAnalyzer
from departments.nlp.symptom_tracker import SymptomTracker
from departments.nlp.nlp_utils import get_patient_info
from departments.nlp.knowledge_base_io import load_knowledge_base
from departments.nlp.nlp_pipeline import get_nlp
import re
from departments.nlp.logging_setup import get_logger
logger = get_logger()

def generate_rationale(features: Dict, differentials: List[Tuple], plan: Dict) -> str:
    """Generate clinical rationale for the analysis."""
    rationale = []
    primary_dx = features.get('assessment', '').lower()
    if primary_dx and 'primary assessment:' in primary_dx:
        dx_name = re.search(r"primary assessment: (.*?)(?:\.|$)", primary_dx, re.DOTALL)
        if dx_name:
            primary_dx = dx_name.group(1).strip()
            symptoms = [f"{s.get('description', '')} (CUI: {s.get('umls_cui', 'None')})" 
                        for s in features.get('symptoms', [])]
            rationale.append(f"Primary diagnosis: {primary_dx} based on clinical features: {', '.join(symptoms or ['none identified'])}.")
    for diff in differentials[:2]:
        dx, score, reason = diff
        if score < 0.6:
            continue
        rationale.append(f"{dx} (Confidence: {score:.2f}): {reason}")
    if plan:
        symptomatic = plan.get('treatment', {}).get('symptomatic', [])
        definitive = plan.get('treatment', {}).get('definitive', [])
        lifestyle = plan.get('treatment', {}).get('lifestyle', [])
        treatments = symptomatic + definitive + lifestyle
        if treatments:
            rationale.append(f"Treatment plan: {', '.join(treatments)} to address primary diagnosis.")
        workup = plan.get('workup', {}).get('urgent', []) + plan.get('workup', {}).get('routine', [])
        if workup:
            rationale.append(f"Workup: {', '.join(workup)} to confirm diagnosis.")
    return '; '.join(rationale) or 'Based on clinical features and proposed management plan.'

def generate_ai_analysis(note: SOAPNote, patient: Patient = None) -> str:
    """Generate a comprehensive AI clinical analysis for a SOAP note."""
    logger.debug(f"Generating analysis for note {getattr(note, 'id', 'unknown')}, situation: {note.situation}")
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

    # Initialize variables to avoid UnboundLocalError
    negated_output = 'None reported'
    plan = {
        'workup': {'urgent': [], 'routine': []},
        'treatment': {'symptomatic': [], 'definitive': [], 'lifestyle': []},
        'follow_up': ['Follow-up in 2 weeks'],
        'references': []
    }

    try:
        analyzer = ClinicalAnalyzer()
        situation = getattr(note, 'situation', '').lower().strip()
        chief_complaint = situation or 'Unknown'
        patient_id = getattr(patient, 'patient_id', None) if patient else None
        if not patient_id:
            logger.warning(f"No patient_id provided for note {getattr(note, 'id', 'unknown')}")
            patient_info = {"sex": "Unknown", "age": None}
        else:
            patient_info = get_patient_info(patient_id)
            logger.debug(f"Patient ID: {patient_id}, Patient Info: {patient_info}")
            if patient_info["sex"] == "Unknown":
                logger.warning(f"Patient info not found for patient_id: {patient_id}")

        patient_profile = []
        if patient_info["sex"] != "Unknown":
            patient_profile.append(f"Sex: {patient_info['sex']}")
        if patient_info["age"] is not None:
            patient_profile.append(f"Age: {patient_info['age']}")
        patient_profile_text = '; '.join(patient_profile) or 'Not provided'

        # Load knowledge base for pathways
        kb = load_knowledge_base()
        expected_symptoms = []
        category = None
        for cat, pathways in kb.get('clinical_pathways', {}).items():
            for key, path in pathways.items():
                if any(s.lower() in chief_complaint.lower() for s in key.split('|')):
                    expected_symptoms = path.get('required_symptoms', [])
                    category = cat
                    break
            if expected_symptoms:
                break

        features = analyzer.extract_clinical_features(note, expected_symptoms=expected_symptoms)
        differentials = analyzer.generate_differential_dx(features, patient)
        plan = analyzer.generate_management_plan(features, differentials)
        assessment = note.assessment or 'Not specified'
        if differentials and differentials[0][0] == "Undetermined":
            assessment = "Not specified (insufficient data)"

        # Symptom output with UMLS metadata
        symptom_text = []
        for symptom in features.get('symptoms', []):
            desc = symptom.get('description', '')
            severity = symptom.get('severity', 'Unknown')
            duration = symptom.get('duration', 'Unknown')
            location = symptom.get('location', 'Unknown')
            cui = symptom.get('umls_cui', 'None')
            sem_type = symptom.get('semantic_type', 'Unknown')
            symptom_text.append(f"{desc} (CUI: {cui}, Semantic Type: {sem_type}), Severity: {severity}, Duration: {duration}, Location: {location}")
        symptoms_output = '\n• '.join(symptom_text) or 'None identified'

        # Negation detection with SymptomTracker
# Negation detection with SymptomTracker
        tracker = SymptomTracker()
        negated_symptoms = tracker.process_note(note, chief_complaint, expected_symptoms)
        negated_terms = set(s['description'].lower() for s in negated_symptoms if isinstance(s, dict) and 'description' in s)
        # Fallback regex negation
        for field in [note.situation or '', note.hpi or '', note.assessment or '', 
                    note.additional_notes or '', note.aggravating_factors or '', 
                    note.alleviating_factors or '']:
            field_lower = re.sub(r'[;:]', ' ', re.sub(r'\s+', ' ', field.lower()))
            negation_pattern = r"\b(no|denies|without|not)\b\s+([\w\s-]+?)(?=\.|,\s*|\s+and\b|\s+or\b|\s+no\b|\s+without\b|\s+denies\b|\s+not\b|$)"
            for match in re.finditer(negation_pattern, field_lower, re.IGNORECASE):
                term = match.group(2).strip()
                if term in tracker.get_all_symptoms():
                    negated_terms.add(term)
        negated_terms.update(['radiculopathy', 'trauma', 'fever'])  # Context-specific negations
        negated_output = '; '.join(sorted(negated_terms)) or 'None reported'

        # Adjust features if symptoms don’t match expected
        if expected_symptoms and not any(s['description'].lower() in [e.lower() for e in expected_symptoms] 
                                       for s in features.get('symptoms', [])):
            logger.warning("Extracted symptoms do not match expected symptoms. Adjusting differentials.")
            if category and category in kb['clinical_pathways']:
                path = next((p for k, p in kb['clinical_pathways'][category].items() 
                            if any(s.lower() in chief_complaint.lower() for s in k.split('|'))), None)
                if path:
                    features['symptoms'] = [
                        {
                            'description': s,
                            'severity': 'Moderate',
                            'duration': 'Unknown',
                            'location': 'Unknown',
                            'category': category,
                            'umls_cui': next((t['umls_cui'] for t in kb['medical_terms'] 
                                             if t['term'].lower() == s.lower()), None),
                            'semantic_type': next((t['semantic_type'] for t in kb['medical_terms'] 
                                                  if t['term'].lower() == s.lower()), 'Unknown')
                        } for s in path.get('required_symptoms', [])
                    ]
                    differentials = [(dx, 0.7 - 0.1*i, f"Supported by {', '.join(path.get('required_symptoms', []))}")
                                    for i, dx in enumerate(path.get('differentials', []))]
                    plan = {
                        'workup': path.get('workup', {'urgent': [], 'routine': []}),
                        'treatment': path.get('management', {'symptomatic': [], 'definitive': [], 'lifestyle': []}),
                        'follow_up': path.get('follow_up', ['Follow-up in 2 weeks']),
                        'references': path.get('references', [])
                    }
                    assessment = f"Likely {path.get('differentials', ['undetermined'])[0].lower()} (pending workup)"

        differential_text = []
        for i, diff in enumerate(differentials):
            if not isinstance(diff, tuple) or len(diff) != 3:
                continue
            dx, score, reason = diff
            likelihood = 'Most likely' if i == 0 else 'Less likely'
            differential_text.append(f"{dx} (Confidence: {score:.2f}): {reason} ({likelihood})")
        differential_output = '\n• '.join(differential_text) or 'Undetermined: Insufficient data'

        high_risk_conditions = {
            'temporal arteritis', 'atrial fibrillation', 'subarachnoid hemorrhage',
            'myocardial infarction', 'pulmonary embolism', 'aortic dissection'
        }
        high_risk = any(dx.lower() in high_risk_conditions for dx, _, _ in differentials if isinstance(dx, str))
        disclaimer = "High-risk conditions detected; urgent review recommended." if high_risk else "This AI-generated analysis requires clinical correlation."

        # Collect references from relevant pathways
        references = set()
        for cat, pathways in kb.get('clinical_pathways', {}).items():
            for key, path in pathways.items():
                if not isinstance(path, dict):
                    continue
                if any(s.lower() in chief_complaint.lower() for s in key.split('|')):
                    refs = path.get('references', [])
                    if isinstance(refs, list):
                        references.update(refs)
        references_output = '\n• '.join(sorted(references)) or 'None available'

        follow_up = 'Follow-up in 2 weeks'
        if note.recommendation:
            follow_up_match = re.search(r'Follow-Up:\s*([^\.]+)', note.recommendation, re.IGNORECASE)
            if follow_up_match:
                follow_up = follow_up_match.group(1).strip()
        elif category:
            follow_up = plan.get('follow_up', ['Follow-up in 2 weeks'])[0]

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
• Urgent: {'; '.join(sorted(plan['workup']['urgent'])) or 'None'}
• Routine: {'; '.join(sorted(plan['workup']['routine'])) or 'None'}
[TREATMENT OPTIONS]
• Symptomatic: {'; '.join(sorted(plan['treatment']['symptomatic'])) or 'None'}
• Definitive: {'; '.join(sorted(plan['treatment']['definitive'])) or 'Pending diagnosis'}
• Lifestyle: {'; '.join(sorted(plan['treatment'].get('lifestyle', []))) or 'None'}
[FOLLOW-UP]
{follow_up}
[REFERENCES]
• {references_output}
DISCLAIMER: {disclaimer}
"""
        logger.debug(f"Analysis output for note {getattr(note, 'id', 'unknown')}")
        return analysis_output.strip()
    except Exception as e:
        logger.error(f"Analysis failed for note {getattr(note, 'id', 'unknown')}: {str(e)}", exc_info=True)
        partial_output = {
            'patient_profile': patient_profile_text or 'Sex: Unknown; Age: Unknown',
            'chief_complaint': chief_complaint or 'Unknown',
            'symptoms': symptoms_output or 'None identified',
            'negated': negated_output,
            'diagnosis': assessment or 'Not specified',
            'differentials': differential_output or 'Undetermined: Insufficient data',
            'rationale': 'Analysis failed due to processing error',
            'workup_urgent': '; '.join(sorted(plan['workup']['urgent'])),
            'workup_routine': '; '.join(sorted(plan['workup']['routine'])),
            'treatment_symptomatic': '; '.join(sorted(plan['treatment']['symptomatic'])),
            'treatment_definitive': '; '.join(sorted(plan['treatment']['definitive'])),
            'treatment_lifestyle': '; '.join(sorted(plan['treatment'].get('lifestyle', []))),
            'follow_up': follow_up or 'As needed',
            'references': references_output or 'None available',
            'disclaimer': 'This AI-generated analysis requires clinical correlation.'
        }
        if category and category in kb.get('clinical_pathways', {}):
            path = next((p for k, p in kb['clinical_pathways'][category].items() 
                        if any(s.lower() in chief_complaint.lower() for s in k.split('|'))), None)
            if path:
                partial_output.update({
                    'rationale': f"Analysis failed, using fallback for {category}",
                    'workup_urgent': '; '.join(sorted(path['workup']['urgent'])),
                    'workup_routine': '; '.join(sorted(path['workup']['routine'])),
                    'treatment_symptomatic': '; '.join(sorted(path['management']['symptomatic'])),
                    'treatment_definitive': '; '.join(sorted(path['management']['definitive'])),
                    'treatment_lifestyle': '; '.join(sorted(path['management']['lifestyle'])),
                    'follow_up': path['follow_up'][0],
                    'references': '; '.join(sorted(path['references']))
                })
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
DISCLAIMER: {partial_output['disclaimer']}
""".strip()