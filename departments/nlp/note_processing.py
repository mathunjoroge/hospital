from typing import Dict
from departments.models.medicine import SOAPNote
from departments.nlp.logging_setup import get_logger
from departments.nlp.clinical_analyzer import ClinicalAnalyzer
from departments.nlp.knowledge_base_io import load_knowledge_base, invalidate_cache
import re

logger = get_logger()

def normalize_symptom(symptom: str, kb: Dict) -> str:
    """Normalize symptom using synonyms from knowledge base."""
    symptom_lower = symptom.lower().strip()
    symptom_clean = re.sub(r'^(patient complains of |complains of |reports )\b', '', symptom_lower, flags=re.IGNORECASE)
    for canonical, aliases in kb.get('synonyms', {}).items():
        canonical_lower = canonical.lower()
        if symptom_clean == canonical_lower or symptom_clean in [a.lower() for a in aliases]:
            return canonical_lower
    return symptom_clean

def validate_symptom(symptom: Dict, kb: Dict) -> bool:
    """Validate symptom data against knowledge base."""
    desc = symptom.get('description', '').lower()
    category = symptom.get('category', '')
    cui = symptom.get('umls_cui', None)
    semantic_type = symptom.get('semantic_type', '')

    if not desc or len(desc) < 2:
        logger.warning(f"Invalid symptom description: {desc}")
        return False
    # Allow any category if kb['symptoms'] is empty or category is reasonable
    valid_categories = {s.get('category') for s in kb.get('symptoms', [])}
    if valid_categories and category not in valid_categories:
        logger.warning(f"Category '{category}' not in knowledge base for symptom '{desc}'")
        return False
    if cui and not cui.startswith('C'):
        logger.warning(f"Invalid UMLS CUI for symptom '{desc}': {cui}")
        return False
    if semantic_type and semantic_type not in ['Sign or Symptom', 'Disease or Syndrome', 'Finding', 'Unknown']:
        logger.warning(f"Unusual semantic type for symptom '{desc}': {semantic_type}")
    return True

def generate_ai_summary(note: SOAPNote, analyzer: ClinicalAnalyzer = None) -> str:
    """Generate an AI summary for a SOAP note, including symptoms with UMLS metadata and ICD-10."""
    logger.debug(f"Starting generate_ai_summary for note ID {getattr(note, 'id', 'unknown')}")
    
    if not isinstance(note, SOAPNote):
        logger.error(f"Invalid input: note must be a SOAPNote object, got {type(note)}")
        raise TypeError(f"Expected SOAPNote object, got {type(note)}")

    try:
        # Initialize ClinicalAnalyzer if not provided
        if analyzer is None:
            analyzer = ClinicalAnalyzer()
            logger.debug("Initialized new ClinicalAnalyzer instance")

        # Load knowledge base
        try:
            kb = load_knowledge_base()
            if not isinstance(kb.get('symptoms', []), list) or not isinstance(kb.get('synonyms', {}), dict):
                logger.error("Invalid knowledge base structure")
                invalidate_cache()
                return "Summary unavailable"
            logger.debug(f"Knowledge base version: {kb.get('version', 'Unknown')}, last updated: {kb.get('last_updated', 'Unknown')}")
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {str(e)}")
            invalidate_cache()
            return "Summary unavailable"

        summary_parts = []
        situation = getattr(note, 'situation', '') or ''
        if not isinstance(situation, str):
            logger.warning(f"Expected string for situation, got {type(situation)}")
            situation = str(situation)

        # Extract and normalize chief complaint
        demographic_pattern = r"^(male|female|man|woman|boy|girl),\s*(.*)"
        match = re.match(demographic_pattern, situation.lower().strip())
        chief_complaint = ""
        if match:
            _, chief_complaint = match.groups()
            chief_complaint = normalize_symptom(chief_complaint.strip(), kb)
            summary_parts.append(f"Chief Complaint: {chief_complaint}.")
        else:
            chief_complaint = normalize_symptom(situation.strip(), kb)
            summary_parts.append(f"Chief Complaint: {chief_complaint}.")

        # Extract clinical features using ClinicalAnalyzer
        try:
            features = analyzer.extract_clinical_features(note, expected_symptoms=[chief_complaint] if chief_complaint else [])
            valid_symptoms = features.get('symptoms', [])
            if not isinstance(valid_symptoms, list):
                logger.error(f"Expected list of symptoms, got {type(valid_symptoms)}")
                valid_symptoms = []
            logger.debug(f"Extracted symptoms: {[s.get('description', '') for s in valid_symptoms]}")
        except Exception as e:
            logger.error(f"Error extracting clinical features: {str(e)}")
            valid_symptoms = []

        # Enrich and validate symptoms with knowledge base metadata
        enriched_symptoms = []
        for symptom in valid_symptoms:
            if not isinstance(symptom, dict):
                logger.warning(f"Skipping invalid symptom format: {symptom}")
                continue
            s_norm = normalize_symptom(symptom.get('description', ''), kb)
            for kb_symptom in kb.get('symptoms', []):
                if s_norm == kb_symptom.get('symptom', '').lower():
                    symptom.update({
                        'description': kb_symptom.get('description', s_norm),
                        'umls_cui': kb_symptom.get('umls_cui', symptom.get('umls_cui')),
                        'semantic_type': kb_symptom.get('semantic_type', symptom.get('semantic_type', 'Unknown')),
                        'category': kb_symptom.get('category', symptom.get('category', '')),
                        'icd10': kb_symptom.get('icd10')
                    })
                    break
            if validate_symptom(symptom, kb):
                enriched_symptoms.append(symptom)
            else:
                logger.warning(f"Skipping invalid symptom: {symptom.get('description', 'Unknown')}")

        # Add HPI
        hpi = getattr(note, 'hpi', '') or ''
        if hpi:
            if not isinstance(hpi, str):
                hpi = str(hpi)
            summary_parts.append(f"HPI: {hpi.strip()}")

        # Add medication history
        medication_history = getattr(note, 'medication_history', '') or ''
        if medication_history:
            if not isinstance(medication_history, str):
                medication_history = str(medication_history)
            summary_parts.append(f"Medications: {medication_history.strip()}")

        # Add assessment
        assessment = getattr(note, 'assessment', '') or ''
        if assessment:
            if not isinstance(assessment, str):
                assessment = str(assessment)
            primary_dx = re.search(r"Primary Assessment: (.*?)(?:\.|$)", assessment, re.DOTALL)
            if primary_dx:
                summary_parts.append(f"Primary Diagnosis: {primary_dx.group(1).strip()}")
            else:
                summary_parts.append(f"Assessment: {assessment.strip()}")

        # Add symptoms with UMLS metadata and ICD-10
        if enriched_symptoms:
            symptom_text = "\nExtracted Symptoms:\n" + "\n".join(
                f"- {s['description']} (Category: {s['category']}"
                + (f", CUI: {s['umls_cui']}" if s.get('umls_cui') else "")
                + (f", Semantic Type: {s['semantic_type']}" if s.get('semantic_type') else "")
                + (f", ICD-10: {s['icd10']})" if s.get('icd10') else ")")
                for s in enriched_symptoms
            )
            summary_parts.append(symptom_text)

        if not summary_parts:
            logger.warning("No valid fields found for summary")
            return "Summary unavailable"

        # Add knowledge base metadata
        summary_parts.append(
            f"\nKnowledge Base: Version {kb.get('version', 'Unknown')}, Last Updated: {kb.get('last_updated', 'Unknown')}"
        )

        summary = "\n".join(summary_parts)
        logger.debug(f"Generated summary: {summary}")
        return summary
    except Exception as e:
        logger.error(f"Error in generate_ai_summary for note ID {getattr(note, 'id', 'unknown')}: {str(e)}")
        invalidate_cache()
        return "Summary unavailable"