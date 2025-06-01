from typing import Dict, List
from departments.models.medicine import SOAPNote
from departments.nlp.logging_setup import get_logger
from departments.nlp.clinical_analyzer import ClinicalAnalyzer
from departments.nlp.knowledge_base_io import load_knowledge_base, invalidate_cache
import re

logger = get_logger(__name__)

def normalize_symptom(symptom: str, kb: Dict) -> str:
    """Normalize symptom using synonyms from knowledge base."""
    symptom_lower = symptom.lower().strip()
    symptom_clean = re.sub(r'^(patient complains of |complains of |reports )\b', '', symptom_lower, flags=re.IGNORECASE)
    for canonical, aliases in kb.get('synonyms', {}).items():
        canonical_lower = canonical.lower()
        aliases_lower = [a.lower() for a in aliases]
        if symptom_clean == canonical_lower or symptom_clean in aliases_lower:
            logger.debug(f"Normalized '{symptom_clean}' to '{canonical_lower}'")
            return canonical_lower
    logger.debug(f"No normalization for '{symptom_clean}', using original")
    return symptom_clean

def validate_symptom(symptom: Dict, kb: Dict) -> bool:
    """Validate symptom data against knowledge base with flexible rules."""
    desc = symptom.get('description', '').lower().strip()
    category = symptom.get('category', '')
    cui = symptom.get('umls_cui', None)
    semantic_type = symptom.get('semantic_type', '')

    if not desc or len(desc) < 2:
        logger.warning(f"Invalid symptom description: {desc}")
        return False

    # Allow any non-empty category if knowledge base categories are empty
    valid_categories = {k for k in kb.get('symptoms', {}).keys()}
    if valid_categories and category and category not in valid_categories:
        logger.warning(f"Category '{category}' not in knowledge base for symptom '{desc}', defaulting to 'general'")
        symptom['category'] = 'general'

    # Relax CUI validation to allow None or valid CUIs
    if cui and not cui.startswith('C'):
        logger.warning(f"Invalid UMLS CUI for symptom '{desc}': {cui}")
        return False

    # Allow common semantic types or none
    valid_semantic_types = {'Sign or Symptom', 'Disease or Syndrome', 'Finding', 'Symptom', 'Unknown'}
    if semantic_type and semantic_type not in valid_semantic_types:
        logger.warning(f"Unusual semantic type for symptom '{desc}': {semantic_type}, setting to 'Unknown'")
        symptom['semantic_type'] = 'Unknown'

    logger.debug(f"Validated symptom: {desc}")
    return True

def generate_ai_summary(note: SOAPNote, analyzer: ClinicalAnalyzer = None) -> str:
    """Generate an AI summary for a SOAP note, including symptoms with UMLS metadata."""
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
            if not isinstance(kb.get('symptoms', {}), dict) or not isinstance(kb.get('synonyms', {}), dict):
                logger.error("Invalid knowledge base structure")
                invalidate_cache()
                return "Summary unavailable"
            logger.debug(f"Knowledge base version: {kb.get('version', 'unknown')}, last updated: {kb.get('last_updated', 'unknown')}")
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
            summary_parts.append(f"Chief Complaint: {chief_complaint}")
        else:
            chief_complaint = normalize_symptom(situation.strip(), kb)
            summary_parts.append(f"Chief Complaint: {chief_complaint}")

        # Extract clinical features
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

        # Enrich and validate symptoms
        enriched_symptoms = []
        for symptom in valid_symptoms:
            if not isinstance(symptom, dict):
                logger.warning(f"Skipping invalid symptom format: {symptom}")
                continue
            s_norm = normalize_symptom(symptom.get('description', ''), kb)
            symptom['description'] = s_norm

            # Enrich with knowledge base data
            for category, symptoms in kb.get('symptoms', {}).items():
                for kb_symptom, data in symptoms.items():
                    if s_norm == kb_symptom.lower():
                        symptom.update({
                            'description': data.get('description', s_norm),
                            'umls_cui': data.get('umls_cui', symptom.get('umls_cui')),
                            'semantic_type': data.get('semantic_type', symptom.get('semantic_type', 'Unknown')),
                            'category': data.get('category', symptom.get('category', 'general')),
                            'icd10': data.get('icd10', None)
                        })
                        break
                else:
                    continue
                break
            else:
                # Use analyzer-provided metadata if no match
                symptom.setdefault('category', 'general')
                symptom.setdefault('semantic_type', 'Unknown')
                if not symptom.get('umls_cui'):
                    logger.warning(f"No UMLS CUI for symptom '{s_norm}', using None")
                    symptom['umls_cui'] = None

            if validate_symptom(symptom, kb):
                enriched_symptoms.append(symptom)
            else:
                logger.warning(f"Invalid symptom: {symptom.get('description', 'unknown')}")

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
                summary_parts.append(f"Assessment: {primary_dx.group(1).strip()}")
            else:
                summary_parts.append(f"Assessment: {assessment.strip()}")

        # Add symptoms with UMLS metadata
        if enriched_symptoms:
            symptom_text = ["Extracted Symptoms:"]
            for s in enriched_symptoms:
                symptom_line = f"- {s.get('description', '')} (Category: {s.get('category', 'general')}"
                if s.get('umls_cui'):
                    symptom_line += f", CUI: {s['umls_cui']}"
                if s.get('semantic_type'):
                    symptom_line += f", Semantic Type: {s['semantic_type']}"
                if s.get('icd10'):
                    symptom_line += f", ICD-10: {s['icd10']}"
                symptom_line += ")"
                symptom_text.append(symptom_line)
            summary_parts.append('\n'.join(symptom_text))

        if not summary_parts:
            logger.warning("No valid fields found for summary")
            return "Summary unavailable"

        # Add knowledge base metadata
        summary_parts.append(
            f"\nKnowledge Base: Version {kb.get('version', 'unknown')}, Last Updated: {kb.get('last_updated', 'unknown')}"
        )

        summary = '\n'.join(summary_parts)
        logger.info(f"Generated summary for note ID {getattr(note, 'id', 'unknown')}: {len(summary)} characters")
        return summary
    except Exception as e:
        logger.error(f"Error in generate_ai_summary for note ID {getattr(note, 'id', 'unknown')}: {str(e)}")
        invalidate_cache()
        raise