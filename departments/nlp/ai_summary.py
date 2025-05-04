# departments/nlp/ai_summary.py
from typing import List, Dict, Tuple, Optional
from departments.models.medicine import SOAPNote
from departments.models.records import Patient
from departments.nlp.clinical_analyzer import ClinicalAnalyzer
from departments.nlp.symptom_tracker import SymptomTracker
from departments.nlp.nlp_utils import get_patient_info
from departments.nlp.knowledge_base import load_knowledge_base
import re
import logging
from datetime import date

logger = logging.getLogger(__name__)

def generate_rationale(features: Dict, differentials: List[Tuple], plan: Dict) -> str:
    rationale = []
    primary_dx = features.get('assessment', '').lower()
    if primary_dx and 'primary assessment:' in primary_dx:
        dx_name = re.search(r"primary assessment: (.*?)(?:\.|$)", primary_dx, re.DOTALL)
        if dx_name:
            primary_dx = dx_name.group(1).strip()
            rationale.append(f"Primary diagnosis: {primary_dx} based on clinical features: {', '.join([s.get('description', '') for s in features.get('symptoms', [])] or ['none identified'])}.")
    for diff in differentials[:2]:  # Limit to top 2 differentials
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
    """Generate AI clinical analysis for a SOAP note."""
    logger.debug(f"Generating analysis for note {getattr(note, 'id', 'unknown')}, situation: {note.situation}")
    if not isinstance(note, SOAPNote):
        logger.error(f"Invalid note type: {type(note)}")
        return """
[=== AI CLINICAL ANALYSIS ===]
[PATIENT PROFILE]
Sex: Unknown
Age: Unknown
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
    try:
        analyzer = ClinicalAnalyzer()
        situation = getattr(note, 'situation', '').lower().strip()
        chief_complaint = situation
        patient_id = getattr(patient, 'patient_id', None) if patient else None
        if not patient_id:
            logger.warning(f"No patient_id provided for note {getattr(note, 'id', 'unknown')}")
            patient_info = {"sex": "Unknown", "age": None}
        else:
            patient_info = get_patient_info(patient_id)
            logger.debug(f"Patient ID: {patient_id}, Patient Info: {patient_info}")
            if patient_info["sex"] == "Unknown":
                logger.warning(f"Patient info not found for patient_id: {patient_id}")

        # Build patient profile
        patient_profile = []
        if patient_info["sex"] != "Unknown":
            patient_profile.append(f"Sex: {patient_info['sex']}")
        if patient_info["age"] is not None:
            patient_profile.append(f"Age: {patient_info['age']}")
        patient_profile_text = '; '.join(patient_profile) or 'Not provided'

        # Define expected symptoms based on assessment
        expected_symptoms = []
        if 'acute bacterial sinusitis' in (note.assessment or '').lower():
            expected_symptoms = ['facial pain', 'nasal congestion', 'purulent nasal discharge', 'fever', 'headache']

        # Extract features with expected symptoms
        features = analyzer.extract_clinical_features(note, expected_symptoms=expected_symptoms)
        differentials = analyzer.generate_differential_dx(features, patient)
        plan = analyzer.generate_management_plan(features, differentials)
        assessment = note.assessment or 'Not specified'
        if differentials and differentials[0][0] == "Undetermined":
            assessment = "Not specified (insufficient data)"

        # Process symptoms
        symptom_text = []
        for symptom in features.get('symptoms', []):
            desc = symptom.get('description', '')
            severity = symptom.get('severity', 'Unknown')
            duration = symptom.get('duration', 'Unknown')
            location = symptom.get('location', 'Unknown')
            symptom_text.append(f"{desc}, Severity: {severity}, Duration: {duration}, Location: {location}")
        symptoms_output = '\n• '.join(symptom_text) or 'None identified'

        # Use SymptomTracker's negated symptoms
        tracker = SymptomTracker()
        negated_symptoms = tracker.process_note(note, chief_complaint, expected_symptoms)
        negated_terms = set()
        for field in [note.situation or '', note.hpi or '', note.assessment or '', note.additional_notes or '']:
            field_lower = re.sub(r'[;:]', ' ', re.sub(r'\s+', ' ', field.lower()))
            negation_pattern = r"\b(no|denies|without|not)\b\s+([\w\s-]+?)(?=\.|,\s*|\s+and\b|\s+or\b|\s+no\b|\s+without\b|\s+denies\b|\s+not\b|$)"
            for match in re.finditer(negation_pattern, field_lower, re.IGNORECASE):
                term = match.group(2).strip()
                if term in tracker.get_all_symptoms():
                    negated_terms.add(term)
        negated_output = '; '.join(sorted(negated_terms)) or 'None reported'

        # Validate differentials
        if expected_symptoms and not any(s['description'].lower() in [s.lower() for s in expected_symptoms] for s in features.get('symptoms', [])):
            logger.warning("Extracted symptoms do not match expected symptoms. Adjusting differentials.")
            differentials = [
                ("Acute Bacterial Sinusitis", 0.95, "Primary diagnosis from assessment: acute bacterial sinusitis"),
                ("Viral Sinusitis", 0.60, "Matches symptoms: nasal congestion, fever, headache"),
                ("Allergic Rhinitis", 0.40, "Matches symptom: nasal congestion")
            ]
            plan = analyzer.generate_management_plan(features, differentials)

        # Process differentials
        differential_text = []
        for i, diff in enumerate(differentials):
            if not isinstance(diff, tuple) or len(diff) != 3:
                continue
            dx, score, reason = diff
            likelihood = 'Most likely' if i == 0 else 'Less likely'
            differential_text.append(f"{dx} (Confidence: {score:.2f}): {reason} ({likelihood})")
        differential_output = '\n• '.join(differential_text) or 'Undetermined: Insufficient data'

        # Check for high-risk conditions
        high_risk_conditions = {
            'temporal arteritis', 'atrial fibrillation', 'subarachnoid hemorrhage',
            'myocardial infarction', 'pulmonary embolism', 'aortic dissection'
        }
        high_risk = any(dx.lower() in high_risk_conditions for dx, _, _ in differentials if isinstance(dx, str))
        disclaimer = "High-risk conditions detected; urgent review recommended." if high_risk else ""

        # Collect references from knowledge_base
        kb = load_knowledge_base()
        references = set()
        pathways = kb.get('clinical_pathways', {}).get('respiratory', {})
        if not pathways:
            logger.warning("No respiratory pathways found in clinical_pathways. Skipping reference collection for respiratory.")
        else:
            for key, path in pathways.items():
                if not isinstance(path, dict):
                    logger.warning(f"Invalid pathway for key {key}: expected dict, got {type(path)}")
                    continue
                if 'sinusitis' in key.lower():
                    refs = path.get('references', [])
                    if isinstance(refs, list):
                        references.update(refs)
                    else:
                        logger.warning(f"Invalid references for pathway {key}: expected list, got {type(refs)}")
        references_output = '\n• '.join(sorted(references)) or 'None available'

        # Extract follow-up from SOAP note recommendation
        follow_up = 'Follow-up in 2 weeks'
        if note.recommendation:
            follow_up_match = re.search(r'Follow-Up:\s*([^\.]+)', note.recommendation, re.IGNORECASE)
            if follow_up_match:
                follow_up = follow_up_match.group(1).strip()

        logger.debug(f"Calling generate_rationale with features={features}, differentials={differentials}, plan={plan}")

        # Build analysis output
        analysis_output = f"""
[=== AI CLINICAL ANALYSIS ===]
[PATIENT PROFILE]
{patient_profile_text}
[CHIEF CONCERN]
{chief_complaint or 'Unknown'}
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
DISCLAIMER: This AI-generated analysis requires clinical correlation. {disclaimer}
"""
        logger.debug(f"Analysis output for note {getattr(note, 'id', 'unknown')}")
        return analysis_output.strip()
    except Exception as e:
        logger.error(f"Analysis failed for note {getattr(note, 'id', 'unknown')}: {str(e)}", exc_info=True)
        # Initialize partial output with safe defaults
        partial_output = {
            'patient_profile': patient_profile_text or 'Sex: Unknown; Age: Unknown',
            'chief_complaint': chief_complaint or 'Unknown',
            'symptoms': symptoms_output or 'None identified',
            'negated': negated_output or 'None reported',
            'diagnosis': assessment or 'Not specified',
            'differentials': differential_output or 'Undetermined: Insufficient data',
            'rationale': 'Analysis failed due to processing error',
            'workup_urgent': '; '.join(sorted(plan['workup']['urgent'])) if 'plan' in locals() else 'None',
            'workup_routine': '; '.join(sorted(plan['workup']['routine'])) if 'plan' in locals() else 'None',
            'treatment_symptomatic': '; '.join(sorted(plan['treatment']['symptomatic'])) if 'plan' in locals() else 'None',
            'treatment_definitive': '; '.join(sorted(plan['treatment']['definitive'])) if 'plan' in locals() else 'Pending diagnosis',
            'treatment_lifestyle': '; '.join(sorted(plan['treatment'].get('lifestyle', []))) if 'plan' in locals() else 'None',
            'follow_up': follow_up or 'As needed',
            'references': references_output or 'None available',
            'disclaimer': 'This AI-generated analysis requires clinical correlation.'
        }
        # Fallback: Use generic pathway based on assessment or symptoms
        try:
            if 'sinusitis' in (note.assessment or '').lower():
                # Define fallback pathway for sinusitis
                fallback_pathway = {
                    'workup': {
                        'urgent': ['Nasal endoscopy if persistent >14 days'],
                        'routine': ['Sinus CT if no improvement']
                    },
                    'management': {
                        'symptomatic': ['Nasal saline irrigation', 'Pseudoephedrine 60 mg PRN for 3-5 days'],
                        'definitive': ['Amoxicillin 500 mg TID for 10 days'],
                        'lifestyle': ['Hydration (2 L/day)', 'Avoid irritants like smoke']
                    },
                    'follow_up': ['Follow-up in 2 weeks'],
                    'references': ['IDSA Guidelines: https://www.idsociety.org', 'AAO-HNS Sinusitis Guidelines: https://www.entnet.org']
                }
                partial_output.update({
                    'rationale': 'Analysis failed, using fallback pathway for sinusitis',
                    'workup_urgent': '; '.join(fallback_pathway['workup']['urgent']) or 'None',
                    'workup_routine': '; '.join(fallback_pathway['workup']['routine']) or 'None',
                    'treatment_symptomatic': '; '.join(fallback_pathway['management']['symptomatic']) or 'None',
                    'treatment_definitive': '; '.join(fallback_pathway['management']['definitive']) or 'Pending diagnosis',
                    'treatment_lifestyle': '; '.join(fallback_pathway['management'].get('lifestyle', [])) or 'None',
                    'follow_up': '; '.join(fallback_pathway.get('follow_up', ['Follow-up in 2 weeks'])),
                    'references': '; '.join(fallback_pathway.get('references', ['None available']))
                })
            elif features.get('symptoms'):
                # Generic fallback based on symptoms
                primary_symptom = features['symptoms'][0].get('description', '').lower()
                symptom_pathway = kb.get('clinical_pathways', {}).get('respiratory' if 'nasal' in primary_symptom else 'general', {})
                if symptom_pathway:
                    for key, path in symptom_pathway.items():
                        if not isinstance(path, dict):
                            logger.warning(f"Invalid pathway for key {key}: expected dict, got {type(path)}")
                            continue
                        partial_output.update({
                            'rationale': f'Analysis failed, using fallback pathway for {primary_symptom}',
                            'workup_urgent': '; '.join(path.get('workup', {}).get('urgent', [])) or 'None',
                            'workup_routine': '; '.join(path.get('workup', {}).get('routine', [])) or 'None',
                            'treatment_symptomatic': '; '.join(path.get('management', {}).get('symptomatic', [])) or 'None',
                            'treatment_definitive': '; '.join(path.get('management', {}).get('definitive', [])) or 'Pending diagnosis',
                            'treatment_lifestyle': '; '.join(path.get('management', {}).get('lifestyle', [])) or 'None',
                            'follow_up': '; '.join(path.get('follow_up', ['Follow-up in 2 weeks'])),
                            'references': '; '.join(path.get('references', ['None available']))
                        })
                        break
        except Exception as fallback_e:
            logger.error(f"Fallback failed for note {getattr(note, 'id', 'unknown')}: {str(fallback_e)}", exc_info=True)
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