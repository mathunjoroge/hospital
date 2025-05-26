import os
import json
from datetime import datetime
from typing import Dict, Optional
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from departments.nlp.logging_setup import get_logger
from departments.nlp.knowledge_base_init import initialize_knowledge_files
from departments.nlp.config import MONGO_URI, DB_NAME

logger = get_logger()

# Singleton for caching knowledge base
_knowledge_base_cache = None
_cache_mtime = None

def load_knowledge_base(knowledge_base_path: str = None, force_reload: bool = False) -> Dict:
    """Load knowledge base from MongoDB (primary) or JSON files (fallback) with validation and UMLS metadata."""
    global _knowledge_base_cache, _cache_mtime
    knowledge_base_dir = knowledge_base_path or os.path.join(os.path.dirname(__file__), "knowledge_base")

    # Check if cache is valid
    if _knowledge_base_cache is not None and not force_reload:
        if os.path.exists(knowledge_base_dir):
            mtime = max(os.path.getmtime(os.path.join(knowledge_base_dir, f)) for f in os.listdir(knowledge_base_dir) if f.endswith('.json'))
            if _cache_mtime is None or mtime <= _cache_mtime:
                logger.debug("Returning cached knowledge base")
                return _knowledge_base_cache
        else:
            logger.debug("No JSON files found, using cached knowledge base")
            return _knowledge_base_cache

    initialize_knowledge_files()
    resources = {
        "symptoms": "symptoms.json",
        "medical_stop_words": "medical_stop_words.json",
        "medical_terms": "medical_terms.json",
        "synonyms": "synonyms.json",
        "clinical_pathways": "clinical_pathways.json",
        "history_diagnoses": "history_diagnoses.json",
        "diagnosis_relevance": "diagnosis_relevance.json",
        "management_config": "management_config.json"
    }
    knowledge = {
        "version": "1.1.0",
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "MongoDB" if MONGO_URI else "JSON",
        "symptoms": {},
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
    strict_validation = os.getenv('STRICT_KB_VALIDATION', 'false').lower() == 'true'

    # Try MongoDB first
    try:
        client = MongoClient(MONGO_URI)
        client.admin.command('ping')
        db = client[DB_NAME]
        logger.info("Loading knowledge base from MongoDB")
        for key in resources:
            collection = db[key]
            data = list(collection.find())
            if not data:
                logger.warning(f"No data found in MongoDB collection {key}. Falling back to JSON.")
                continue
            if key == "symptoms":
                valid_data = {}
                for doc in data:
                    category = doc.get('category')
                    symptom = doc.get('symptom')
                    if not category or not symptom:
                        logger.warning(f"Invalid symptom document: {doc}")
                        continue
                    if category not in valid_data:
                        valid_data[category] = {}
                    valid_data[category][symptom] = {
                        'description': doc.get('description', f"UMLS-derived: {symptom}"),
                        'umls_cui': doc.get('umls_cui'),
                        'semantic_type': doc.get('semantic_type', 'Unknown')
                    }
                knowledge[key] = valid_data
            elif key == "medical_stop_words":
                knowledge[key] = set(doc['word'] for doc in data if 'word' in doc)
            elif key == "medical_terms":
                knowledge[key] = [
                    {
                        "term": doc.get("term", ""),
                        "category": doc.get("category", "unknown"),
                        "umls_cui": doc.get("umls_cui"),
                        "semantic_type": doc.get("semantic_type", "Unknown")
                    } for doc in data if doc.get("term")
                ]
            elif key == "synonyms":
                knowledge[key] = {doc['term']: doc['aliases'] for doc in data if 'term' in doc and isinstance(doc.get('aliases'), list)}
            elif key == "clinical_pathways":
                valid_data = {}
                for doc in data:
                    category = doc.get('category')
                    paths = doc.get('paths', {})
                    if not category or not isinstance(paths, dict):
                        logger.warning(f"Invalid clinical pathway document: {doc}")
                        continue
                    valid_paths = {}
                    for pkey, path in paths.items():
                        if not isinstance(path, dict):
                            logger.warning(f"Invalid path {pkey}: {path}")
                            continue
                        if strict_validation:
                            if not path.get('differentials'):
                                logger.error(f"Path {pkey}: missing differentials")
                                continue
                            if any(dx.lower() in high_risk_conditions for dx in path.get('differentials', [])) and not path.get('contextual_triggers'):
                                logger.error(f"Path {pkey}: high-risk differentials missing contextual_triggers")
                                continue
                        else:
                            path['differentials'] = path.get('differentials', ['Undetermined'])
                            if any(dx.lower() in high_risk_conditions for dx in path['differentials']) and not path.get('contextual_triggers'):
                                path['contextual_triggers'] = ['High-risk condition detected']
                            management = path.setdefault('management', {})
                            management.setdefault('symptomatic', ['Symptomatic relief pending'])
                            workup = path.setdefault('workup', {})
                            workup.setdefault('routine', ['Diagnostic evaluation pending'])
                            path.setdefault('references', ['None specified'])
                            metadata = path.setdefault('metadata', {})
                            metadata.setdefault('source', 'Unknown')
                            metadata.setdefault('last_updated', datetime.now().strftime("%Y-%m-%d"))
                            path.setdefault('follow_up', ['Follow-up in 2 weeks'])
                        valid_paths[pkey] = path
                    if valid_paths:
                        valid_data[category] = valid_paths
                knowledge[key] = valid_data
                if required_categories - set(valid_data.keys()):
                    logger.warning(f"Missing required categories in clinical_pathways: {required_categories - set(valid_data.keys())}")
            else:
                knowledge[key] = {doc['key']: doc['value'] for doc in data if 'key' in doc and 'value' in doc}
            logger.info(f"Loaded {len(data)} entries for {key} from MongoDB")
        client.close()
    except ConnectionFailure as e:
        logger.error(f"MongoDB connection failed: {str(e)}. Falling back to JSON.")
        knowledge['source'] = "JSON"
    except Exception as e:
        logger.error(f"Error loading from MongoDB: {str(e)}. Falling back to JSON.")
        knowledge['source'] = "JSON"

    # Load from JSON if MongoDB fails or data is missing
    if knowledge['source'] == "JSON" or not all(knowledge[key] for key in resources):
        for key, filename in resources.items():
            if knowledge[key] and key != 'medical_stop_words':  # Skip if already loaded from MongoDB (except stop words, which are sets)
                continue
            file_path = os.path.join(knowledge_base_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    logger.debug(f"Loaded raw JSON data for {key}")
                    if key == "symptoms":
                        valid_data = {}
                        for category, symptoms in data.items():
                            if not isinstance(symptoms, dict):
                                logger.warning(f"Invalid category {category}: {type(symptoms)}")
                                continue
                            valid_data[category] = {}
                            for symptom, info in symptoms.items():
                                if not isinstance(info, dict) or not info.get('description'):
                                    logger.warning(f"Invalid symptom {symptom}: {info}")
                                    continue
                                valid_data[category][symptom] = {
                                    'description': info['description'],
                                    'umls_cui': info.get('umls_cui'),
                                    'semantic_type': info.get('semantic_type', 'Unknown')
                                }
                        knowledge[key] = valid_data
                    elif key == "medical_stop_words":
                        knowledge[key] = set(data) if isinstance(data, list) and all(isinstance(w, str) for w in data) else set()
                    elif key == "medical_terms":
                        knowledge[key] = [
                            {
                                "term": t.get("term", ""),
                                "category": t.get("category", "unknown"),
                                "umls_cui": t.get("umls_cui"),
                                "semantic_type": t.get("semantic_type", "Unknown")
                            } for t in data if isinstance(t, dict) and t.get("term")
                        ]
                    elif key == "synonyms":
                        knowledge[key] = {k: v for k, v in data.items() if isinstance(v, list) and all(isinstance(a, str) for a in v)}
                    elif key == "clinical_pathways":
                        valid_data = {}
                        for cat, paths in data.items():
                            if not isinstance(paths, dict):
                                logger.warning(f"Invalid category {cat}: {type(paths)}")
                                continue
                            valid_paths = {}
                            for pkey, path in paths.items():
                                if not isinstance(path, dict):
                                    logger.warning(f"Invalid path {pkey}: {type(path)}")
                                    continue
                                if strict_validation:
                                    if not path.get('differentials'):
                                        logger.error(f"Path {pkey}: missing differentials")
                                        continue
                                    if any(dx.lower() in high_risk_conditions for dx in path.get('differentials', [])) and not path.get('contextual_triggers'):
                                        logger.error(f"Path {pkey}: high-risk differentials missing contextual_triggers")
                                        continue
                                else:
                                    path['differentials'] = path.get('differentials', ['Undetermined'])
                                    if any(dx.lower() in high_risk_conditions for dx in path['differentials']) and not path.get('contextual_triggers'):
                                        path['contextual_triggers'] = ['High-risk condition detected']
                                    management = path.setdefault('management', {})
                                    management.setdefault('symptomatic', ['Symptomatic relief pending'])
                                    workup = path.setdefault('workup', {})
                                    workup.setdefault('routine', ['Diagnostic evaluation pending'])
                                    path.setdefault('references', ['None specified'])
                                    metadata = path.setdefault('metadata', {})
                                    metadata.setdefault('source', 'Unknown')
                                    metadata.setdefault('last_updated', datetime.now().strftime("%Y-%m-%d"))
                                    path.setdefault('follow_up', ['Follow-up in 2 weeks'])
                                valid_paths[pkey] = path
                            if valid_paths:
                                valid_data[cat] = valid_paths
                        knowledge[key] = valid_data
                        if required_categories - set(valid_data.keys()):
                            logger.warning(f"Missing required categories in clinical_pathways: {required_categories - set(valid_data.keys())}")
                    elif key == "diagnosis_relevance":
                        if isinstance(data, list):
                            data = {
                                item['diagnosis']: {
                                    'relevance': item.get('relevance', []),
                                    'category': item.get('category', 'unknown')
                                } for item in data if isinstance(item, dict) and 'diagnosis' in item
                            }
                        knowledge[key] = {
                            k: v for k, v in data.items()
                            if isinstance(v, dict) and 'relevance' in v and isinstance(v['relevance'], list)
                        }
                    else:
                        knowledge[key] = data if isinstance(data, dict) else {}
                    logger.info(f"Loaded {len(data)} entries for {key} from JSON")
            except FileNotFoundError:
                logger.error(f"File not found: {file_path}. Using default data.")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in {file_path}: {str(e)}. Using default data.")

    # Cross-reference UMLS CUIs
    for term in knowledge['medical_terms']:
        term_lower = term['term'].lower()
        if term_lower in knowledge['synonyms'] and term.get('umls_cui'):
            for synonym in knowledge['synonyms'][term_lower]:
                if not any(t['term'].lower() == synonym.lower() and t.get('umls_cui') for t in knowledge['medical_terms']):
                    knowledge['medical_terms'].append({
                        'term': synonym,
                        'category': term['category'],
                        'umls_cui': term['umls_cui'],
                        'semantic_type': term['semantic_type']
                    })
                    logger.debug(f"Added UMLS-linked synonym {synonym} for {term_lower}")

    _knowledge_base_cache = knowledge
    if os.path.exists(knowledge_base_dir):
        _cache_mtime = max(os.path.getmtime(os.path.join(knowledge_base_dir, f)) for f in os.listdir(knowledge_base_dir) if f.endswith('.json'))
    return knowledge

def save_knowledge_base(kb: Dict, knowledge_base_path: str = None) -> bool:
    """Save knowledge base to MongoDB (primary) and JSON files (fallback)."""
    global _knowledge_base_cache, _cache_mtime
    knowledge_base_dir = knowledge_base_path or os.path.join(os.path.dirname(__file__), "knowledge_base")
    os.makedirs(knowledge_base_dir, exist_ok=True)
    resources = {
        "symptoms": "symptoms.json",
        "medical_stop_words": "medical_stop_words.json",
        "medical_terms": "medical_terms.json",
        "synonyms": "synonyms.json",
        "clinical_pathways": "clinical_pathways.json",
        "history_diagnoses": "history_diagnoses.json",
        "diagnosis_relevance": "diagnosis_relevance.json",
        "management_config": "management_config.json"
    }

    # Save to MongoDB
    try:
        client = MongoClient(MONGO_URI)
        client.admin.command('ping')
        db = client[DB_NAME]
        for key, data in kb.items():
            if key not in resources:
                continue
            collection = db[key]
            collection.drop()  # Clear existing data
            if key == "symptoms":
                for category, symptoms in data.items():
                    for symptom, info in symptoms.items():
                        collection.insert_one({
                            'category': category,
                            'symptom': symptom,
                            'description': info['description'],
                            'umls_cui': info.get('umls_cui'),
                            'semantic_type': info.get('semantic_type', 'Unknown')
                        })
            elif key == "medical_stop_words":
                for word in data:
                    collection.insert_one({'word': word})
            elif key == "medical_terms":
                for term in data:
                    collection.insert_one(term)
            elif key == "synonyms":
                for term, aliases in data.items():
                    collection.insert_one({'term': term, 'aliases': aliases})
            elif key == "clinical_pathways":
                for category, paths in data.items():
                    collection.insert_one({'category': category, 'paths': paths})
            else:
                for k, v in data.items():
                    collection.insert_one({'key': k, 'value': v})
            logger.info(f"Saved {key} to MongoDB collection")
        client.close()
    except ConnectionFailure as e:
        logger.error(f"MongoDB connection failed: {str(e)}. Saving to JSON only.")
    except Exception as e:
        logger.error(f"Error saving to MongoDB: {str(e)}. Saving to JSON only.")

    # Save to JSON
    try:
        for key, filename in resources.items():
            file_path = os.path.join(knowledge_base_dir, filename)
            data = kb.get(key, {})
            if key == "medical_stop_words":
                data = list(data) if isinstance(data, set) else data
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {key} to {file_path}")
        _knowledge_base_cache = kb
        _cache_mtime = max(os.path.getmtime(os.path.join(knowledge_base_dir, f)) for f in os.listdir(knowledge_base_dir) if f.endswith('.json'))
        logger.info("Knowledge base saved to MongoDB and JSON files.")
        return True
    except Exception as e:
        logger.error(f"Error saving knowledge base to JSON: {str(e)}")
        return False

def invalidate_cache():
    """Invalidate the knowledge base cache."""
    global _knowledge_base_cache, _cache_mtime
    _knowledge_base_cache = None
    _cache_mtime = None
    logger.info("Knowledge base cache invalidated")