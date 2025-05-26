from departments.models.medicine import SOAPNote
from departments.nlp.logging_setup import get_logger
from departments.nlp.clinical_analyzer import ClinicalAnalyzer
from departments.nlp.knowledge_base_io import load_knowledge_base, invalidate_cache
import re

logger = get_logger()

def normalize_symptom(symptom: str, kb: dict) -> str:
    """Normalize symptom using synonyms from knowledge base."""
    symptom_lower = symptom.lower().strip()
    symptom_clean = re.sub(r'^(patient complains of |complains of |reports )\b', '', symptom_lower, flags=re.IGNORECASE)
    for canonical, aliases in kb.get('synonyms', {}).items():
        if symptom_clean == canonical or symptom_clean in [a.lower() for a in aliases]:
            return canonical
    return symptom_clean

def validate_symptom(symptom: dict, kb: dict) -> bool:
    """Validate symptom data against knowledge base."""
    desc = symptom.get('description', '').lower()
    category = symptom.get('category', '')
    cui = symptom.get('umls_cui', None)
    semantic_type = symptom.get('semantic_type', '')
    
    if not desc or len(desc) < 2:
        logger.warning(f"Invalid symptom description: {desc}")
        return False
    if category not in kb.get('symptoms', {}):
        logger.warning(f"Invalid category for symptom '{desc}': {category}")
        return False
    if cui and not cui.startswith('C'):
        logger.warning(f"Invalid UMLS CUI for symptom '{desc}': {cui}")
        return False
    if semantic_type not in ['Sign or Symptom', 'Disease or Syndrome', 'Finding', 'Unknown']:
        logger.warning(f"Unusual semantic type for symptom '{desc}': {semantic_type}")
    return True

def generate_ai_summary(note: SOAPNote, analyzer: ClinicalAnalyzer = None) -> str:
    """Generate an AI summary for a SOAP note, including symptoms with UMLS metadata and ICD-10."""
    logger.debug(f"Starting generate_ai_summary for note ID {getattr(note, 'id', 'unknown')}")
    if not isinstance(note, SOAPNote):
        logger.error(f"Invalid input: note must be a SOAPNote object, got {type(note)}")
        return "Summary unavailable"

    try:
        # Initialize ClinicalAnalyzer if not provided
        if analyzer is None:
            analyzer = ClinicalAnalyzer()
            logger.debug("Initialized new ClinicalAnalyzer instance")

        # Load knowledge base
        try:
            kb = load_knowledge_base()
            logger.debug(f"Knowledge base version: {kb.get('version', 'Unknown')}, last updated: {kb.get('last_updated', 'Unknown')}")
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {str(e)}")
            invalidate_cache()
            return "Summary unavailable"

        summary_parts = []
        situation = getattr(note, 'situation', None) or ""

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
            logger.debug(f"Extracted symptoms: {[s['description'] for s in valid_symptoms]}")
        except Exception as e:
            logger.error(f"Error extracting clinical features: {str(e)}")
            valid_symptoms = []

        # Enrich and validate symptoms with knowledge base metadata
        enriched_symptoms = []
        for symptom in valid_symptoms:
            s_norm = normalize_symptom(symptom['description'], kb)
            for cat, sym_dict in kb.get('symptoms', {}).items():
                if s_norm in sym_dict:
                    symptom.update({
                        'description': sym_dict[s_norm]['description'],
                        'umls_cui': sym_dict[s_norm].get('umls_cui', symptom.get('umls_cui', None)),
                        'semantic_type': sym_dict[s_norm].get('semantic_type', symptom.get('semantic_type', 'Unknown')),
                        'category': cat,
                        'icd10': sym_dict[s_norm].get('icd10', None)
                    })
                    break
            if validate_symptom(symptom, kb):
                enriched_symptoms.append(symptom)
            else:
                logger.warning(f"Skipping invalid symptom: {symptom.get('description', 'Unknown')}")

        # Add HPI
        hpi = getattr(note, 'hpi', None)
        if hpi:
            summary_parts.append(f"HPI: {hpi.strip()}")

        # Add medication history
        medication_history = getattr(note, 'medication_history', None)
        if medication_history:
            summary_parts.append(f"Medications: {medication_history.strip()}")

        # Add assessment
        assessment = getattr(note, 'assessment', None)
        if assessment:
            primary_dx = re.search(r"Primary Assessment: (.*?)(?:\.|$)", assessment, re.DOTALL)
            if primary_dx:
                summary_parts.append(f"Primary Diagnosis: {primary_dx.group(1).strip()}")
            else:
                summary_parts.append(f"Assessment: {assessment.strip()}")

        # Add symptoms with UMLS metadata and ICD-10
        if enriched_symptoms:
            symptom_text = "\nExtracted Symptoms:\n" + "\n".join(
                f"- {s['description']} (Category: {s['category']}"
                + (f", CUI: {s['umls_cui']}, Semantic Type: {s['semantic_type']}"
                   + (f", ICD-10: {s['icd10']})" if s.get('icd10') else ")")
                ) for s in enriched_symptoms
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