import os
import json
from datetime import datetime
from typing import Dict, Optional
from departments.nlp.logging_setup import get_logger
from departments.nlp.knowledge_base_init import initialize_knowledge_files

logger = get_logger()

# Singleton for caching knowledge base
_knowledge_base_cache = None

def load_knowledge_base(knowledge_base_path: str = None) -> Dict:
    """Load knowledge base from JSON files with validation and UMLS metadata."""
    global _knowledge_base_cache
    if _knowledge_base_cache is not None:
        logger.debug("Returning cached knowledge base")
        return _knowledge_base_cache

    initialize_knowledge_files()
    knowledge_base_dir = knowledge_base_path or os.path.join(os.path.dirname(__file__), "knowledge_base")
    resources = {
        "medical_stop_words": "medical_stop_words.json",
        "medical_terms": "medical_terms.json",
        "synonyms": "synonyms.json",
        "clinical_pathways": "clinical_pathways.json",
        "history_diagnoses": "history_diagnoses.json",
        "diagnosis_relevance": "diagnosis_relevance.json",
        "management_config": "management_config.json"
    }
    knowledge = {
        "version": "1.0.0",  # Added version metadata
        "medical_stop_words": [],
        "medical_terms": [],
        "synonyms": {},
        "clinical_pathways": {},
        "history_diagnoses": {},
        "diagnosis_relevance": {},
        "management_config": {}
    }
    required_categories = {'respiratory', 'neurological', 'cardiovascular', 'gastrointestinal', 'musculoskeletal', 'infectious'}
    high_risk_conditions = {'pulmonary embolism', 'myocardial infarction', 'meningitis', 'malaria', 'dengue'}

    for key, filename in resources.items():
        file_path = os.path.join(knowledge_base_dir, filename)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                logger.debug(f"Loaded raw data for {key}")
                if key == "medical_stop_words":
                    if not isinstance(data, list) or not all(isinstance(w, str) for w in data):
                        logger.error(f"Expected list of strings for {filename}, got {type(data)}")
                        data = []
                    data = set(data)
                elif key == "medical_terms":
                    if not isinstance(data, list) or not all(isinstance(t, dict) and "term" in t for t in data):
                        logger.error(f"Expected list of dicts with 'term' for {filename}, got {type(data)}")
                        data = []
                    else:
                        data = [
                            {
                                "term": t.get("term", ""),
                                "category": t.get("category", "unknown"),
                                "umls_cui": t.get("umls_cui"),
                                "semantic_type": t.get("semantic_type", "Unknown")
                            } for t in data if isinstance(t, dict) and t.get("term")
                        ]
                elif key == "clinical_pathways":
                    if not isinstance(data, dict):
                        logger.error(f"Expected dict for {filename}, got {type(data)}")
                        data = {}
                    else:
                        valid_data = {}
                        for cat, paths in data.items():
                            if not isinstance(paths, dict):
                                logger.warning(f"Invalid category {cat}: expected dict, got {type(paths)}")
                                continue
                            valid_paths = {}
                            for pkey, path in paths.items():
                                if not isinstance(path, dict):
                                    logger.warning(f"Invalid path {pkey}: expected dict, got {type(path)}")
                                    continue
                                # Soften validation
                                differentials = path.get("differentials", [])
                                if not differentials or not isinstance(differentials, list):
                                    logger.warning(f"Path {pkey}: differentials empty or invalid")
                                    differentials = ["Undetermined"]
                                    path["differentials"] = differentials
                                diagnosis_relevance = knowledge.get('diagnosis_relevance', {})
                                for dx in differentials:
                                    if dx.lower() not in diagnosis_relevance:
                                        logger.debug(f"Path {pkey}: differential {dx} lacks required symptoms in diagnosis_relevance")
                                if any(dx.lower() in high_risk_conditions for dx in differentials) and not path.get("contextual_triggers", []):
                                    logger.debug(f"Path {pkey}: high-risk differentials missing contextual_triggers")
                                management = path.get("management", {})
                                if not management.get("symptomatic") and not management.get("definitive"):
                                    logger.debug(f"Path {pkey}: no management options, using default")
                                    management["symptomatic"] = ["Symptomatic relief pending"]
                                    path["management"] = management
                                workup = path.get("workup", {})
                                if not workup.get("urgent") and not workup.get("routine"):
                                    logger.debug(f"Path {pkey}: no workup options, using default")
                                    workup["routine"] = ["Diagnostic evaluation pending"]
                                    path["workup"] = workup
                                references = path.get("references", [])
                                if not references:
                                    logger.debug(f"Path {pkey}: missing references, setting to default")
                                    path["references"] = ["None specified"]
                                metadata = path.get("metadata", {})
                                if not metadata.get("source"):
                                    logger.debug(f"Path {pkey}: missing metadata.source, setting to Unknown")
                                    metadata["source"] = ["Unknown"]
                                if not metadata.get("last_updated"):
                                    metadata["last_updated"] = datetime.now().strftime("%Y-%m-%d")
                                    logger.debug(f"Added default last_updated for {pkey}")
                                if not path.get("follow_up", []):
                                    path["follow_up"] = ["Follow-up in 2 weeks"]
                                    logger.debug(f"Added default follow-up for {pkey}")
                                metadata["umls_cui"] = metadata.get("umls_cui")
                                valid_paths[pkey] = path
                            if valid_paths:
                                valid_data[cat] = valid_paths
                        data = valid_data
                        missing_categories = required_categories - set(valid_data.keys())
                        if missing_categories:
                            logger.warning(f"Missing required categories in clinical_pathways: {missing_categories}")
                        logger.info(f"Loaded {len(valid_data)} clinical pathway categories with {sum(len(paths) for paths in valid_data.values())} total pathways")
                elif key == "synonyms":
                    if not isinstance(data, dict) or not all(isinstance(k, str) and isinstance(v, list) for k, v in data.items()):
                        logger.error(f"Expected dict of string keys and list values for {filename}, got {type(data)}")
                        data = {}
                    else:
                        data = {k: v for k, v in data.items() if all(isinstance(a, str) for a in v)}
                elif key == "diagnosis_relevance":
                    if isinstance(data, list):
                        # Convert list to dict, e.g., [{'diagnosis': 'x', 'relevance': [...], 'category': 'y'}, ...]
                        try:
                            data = {
                                item['diagnosis']: {
                                    'relevance': item.get('relevance', []),
                                    'category': item.get('category', 'unknown')
                                } for item in data
                                if isinstance(item, dict) and 'diagnosis' in item
                            }
                            logger.info(f"Converted diagnosis_relevance list to dictionary with {len(data)} entries")
                        except Exception as e:
                            logger.error(f"Failed to convert diagnosis_relevance list to dict: {str(e)}")
                            data = {}
                    elif not isinstance(data, dict):
                        logger.error(f"Expected dict or convertible list for {filename}, got {type(data)}")
                        data = {}
                    else:
                        data = {
                            k: v for k, v in data.items()
                            if isinstance(v, dict) and 'relevance' in v and isinstance(v['relevance'], list)
                            and all(isinstance(r, dict) and 'symptom' in r and 'weight' in r for r in v['relevance'])
                        }
                else:
                    if not isinstance(data, dict):
                        logger.error(f"Expected dict for {filename}, got {type(data)}")
                        data = {}
                knowledge[key] = data
                if not data:
                    logger.warning(f"Empty resource loaded: {filename}")
                else:
                    logger.info(f"Loaded {len(data)} entries for {key}")
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}. Using default data.")
            knowledge[key] = knowledge[key]  # Use initialized default
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {file_path}: {str(e)}. Using default data.")
            knowledge[key] = knowledge[key]
    _knowledge_base_cache = knowledge
    return knowledge

def save_knowledge_base(kb: Dict) -> bool:
    """Save knowledge base to respective JSON files."""
    global _knowledge_base_cache
    knowledge_base_dir = os.path.join(os.path.dirname(__file__), "knowledge_base")
    os.makedirs(knowledge_base_dir, exist_ok=True)
    resources = {
        "medical_stop_words": "medical_stop_words.json",
        "medical_terms": "medical_terms.json",
        "synonyms": "synonyms.json",
        "clinical_pathways": "clinical_pathways.json",
        "history_diagnoses": "history_diagnoses.json",
        "diagnosis_relevance": "diagnosis_relevance.json",
        "management_config": "management_config.json"
    }
    try:
        for key, filename in resources.items():
            file_path = os.path.join(knowledge_base_dir, filename)
            data = kb.get(key, {})
            if key == "medical_stop_words":
                data = list(data) if isinstance(data, set) else data
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {key} to {file_path}")
        _knowledge_base_cache = kb  # Update cache
        logger.info("Knowledge base saved to JSON files.")
        return True
    except Exception as e:
        logger.error(f"Error saving knowledge base: {str(e)}")
        return False