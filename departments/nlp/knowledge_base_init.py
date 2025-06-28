from datetime import datetime
from typing import Dict, List
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from psycopg2.extras import RealDictCursor, execute_batch
from departments.nlp.logging_setup import get_logger
from departments.nlp.config import (
    MONGO_URI, DB_NAME, KB_PREFIX
)
from departments.nlp.nlp_pipeline import get_postgres_connection
from departments.nlp.nlp_utils import parse_date
from departments.nlp.nlp_common import FALLBACK_CUI_MAP

logger = get_logger(__name__)

# Default stop words
default_stop_words: List[str] = [
    "patient", "history", "present", "illness", "denies", "reports",
    "without", "with", "this", "that", "these", "those", "they", "them",
    "their", "have", "has", "had", "been", "being", "other", "associated",
    "complains", "noted", "states", "observed", "left", "right", "ago",
    "since", "recently", "following", "during", "upon", "after"
]

# Default medical terms and symptoms derived from FALLBACK_CUI_MAP
default_medical_terms: List[Dict] = [
    {"term": term, "category": data.get('category', 'unknown'), "umls_cui": data['umls_cui'], "semantic_type": data['semantic_type']}
    for term, data in FALLBACK_CUI_MAP.items()
]

default_symptoms: Dict[str, Dict[str, Dict]] = {}
for term, data in FALLBACK_CUI_MAP.items():
    category = data.get('category', 'unknown')
    if category not in default_symptoms:
        default_symptoms[category] = {}
    default_symptoms[category][term] = {
        "description": f"UMLS-derived: {term}",
        "semantic_type": data['semantic_type'],
        "umls_cui": data['umls_cui']
    }

# Default synonyms
default_synonyms: Dict[str, List[str]] = {
    "facial pain": ["sinus pain", "sinus pressure", "facial pressure"],
    "nasal congestion": ["stuffy nose", "blocked nose", "nasal obstruction"],
    "purulent nasal discharge": ["yellow discharge", "green discharge", "pus from nose"],
    "fever": ["elevated temperature", "pyrexia", "febrile"],
    "headache": ["cephalalgia", "head pain", "cranial pain"],
    "cough": ["hacking cough", "persistent cough", "dry cough"],
    "chest pain": ["thoracic pain", "chest discomfort", "sternal pain"],
    "shortness of breath": ["dyspnea", "breathlessness", "difficulty breathing"],
    "photophobia": ["light sensitivity", "eye discomfort"],
    "neck stiffness": ["nuchal rigidity", "neck pain"],
    "rash": ["skin eruption", "dermatitis", "skin rash"],
    "back pain": ["lower back pain", "lumbar pain", "backache", "back stiffness"],
    "knee pain": ["knee discomfort", "patellar pain"],
    "chest tightness": ["chest pressure", "tightness in chest", "chest discomfort"],
    "nausea": ["queasiness", "sickness", "inclination to vomit"],
    "obesity": ["excess weight", "overweight", "high BMI"],
    "joint pain": ["arthralgia", "joint discomfort", "joint ache"],
    "pain on movement": ["motion-induced pain", "movement pain", "activity-related pain"]
}

# Default diagnosis relevance
default_diagnosis_relevance: List[Dict] = [
    {
        "diagnosis": "acute bacterial sinusitis",
        "relevance": [
            {"symptom": "facial pain", "weight": 0.4},
            {"symptom": "nasal congestion", "weight": 0.3},
            {"symptom": "purulent nasal discharge", "weight": 0.2},
            {"symptom": "fever", "weight": 0.05},
            {"symptom": "headache", "weight": 0.05}
        ],
        "category": "respiratory"
    },
    # Add other diagnosis relevance entries as needed
]

# Default clinical pathways
default_clinical_pathways: Dict[str, Dict] = {
    "respiratory": {
        "facial pain|nasal congestion|purulent nasal discharge": {
            "differentials": ["Acute Bacterial Sinusitis", "Viral Sinusitis", "Allergic Rhinitis"],
            "contextual_triggers": ["recent viral infection"],
            "required_symptoms": ["facial pain", "nasal congestion", "purulent nasal discharge"],
            "exclusion_criteria": ["photophobia", "nausea"],
            "workup": {
                "urgent": ["Nasal endoscopy if persistent >14 days"],
                "routine": ["Sinus CT if no improvement"]
            },
            "management": {
                "symptomatic": ["Nasal saline irrigation", "Pseudoephedrine 60 mg PRN for 3-5 days"],
                "definitive": ["Amoxicillin 500 mg TID for 10 days", "Amoxicillin-clavulanate 875 mg BID if resistant"],
                "lifestyle": ["Hydration (2 L/day)", "Avoid irritants like smoke"]
            },
            "follow_up": ["Follow-up in 2 weeks"],
            "references": [
                "IDSA Guidelines: https://www.idsociety.org",
                "AAO-HNS Sinusitis Guidelines: https://www.entnet.org"
            ],
            "metadata": {
                "source": ["IDSA", "AAO-HNS"],
                "last_updated": datetime.now(),
                "umls_cui": "C0234450"
            }
        },
        "cough|dyspnea|fever": {
            "differentials": ["Pneumonia", "Bronchitis", "COVID-19", "Influenza"],
            "contextual_triggers": ["recent travel", "close contact with sick individuals"],
            "required_symptoms": ["cough", "fever", "dyspnea"],
            "exclusion_criteria": ["chest pain without respiratory findings", "hemoptysis"],
            "workup": {
                "urgent": ["Chest X-ray", "Pulse oximetry", "COVID-19 PCR/Antigen test"],
                "routine": ["CBC", "CRP", "Sputum culture if productive"]
            },
            "management": {
                "symptomatic": ["Antipyretics", "Hydration", "Cough suppressants PRN"],
                "definitive": ["Azithromycin 500 mg day 1, then 250 mg daily x4 days if bacterial"],
                "lifestyle": ["Rest", "Isolation if contagious", "Smoking cessation"]
            },
            "follow_up": ["Follow-up in 3–5 days or sooner if worsening"],
            "references": [
                "ATS/IDSA Guidelines: https://www.thoracic.org",
                "CDC COVID-19 Guidance: https://www.cdc.gov"
            ],
            "metadata": {
                "source": ["ATS", "CDC"],
                "last_updated": datetime.now(),
                "umls_cui": "C0032285"
            }
        },
    },
    "hepatic": {
        "jaundice|abdominal pain": {
            "differentials": ["Hepatitis", "Cholecystitis", "Cirrhosis"],
            "contextual_triggers": ["recent alcohol consumption", "viral exposure"],
            "required_symptoms": ["jaundice", "abdominal pain"],
            "exclusion_criteria": ["chest pain", "shortness of breath"],
            "workup": {
                "urgent": ["Liver function tests", "Abdominal ultrasound"],
                "routine": ["Viral hepatitis panel"]
            },
            "management": {
                "symptomatic": ["Hydration", "Antiemetics PRN"],
                "definitive": ["Antiviral therapy if hepatitis confirmed"],
                "lifestyle": ["Avoid alcohol", "Low-fat diet"]
            },
            "follow_up": ["Follow-up in 1 week"],
            "references": [
                "AASLD Guidelines: https://www.aasld.org"
            ],
            "metadata": {
                "source": ["AASLD"],
                "last_updated": datetime.now(),
                "umls_cui": "C0022346"
            }
        },
    }
}

default_history_diagnoses: Dict[str, Dict] = {
    "hypertension": {
        "synonyms": ["high blood pressure", "HTN"],
        "umls_cui": "C0020538",
        "semantic_type": "Disease or Syndrome"
    },
    # Add other history diagnoses as needed
}

default_diagnosis_treatments: Dict[str, Dict] = {
    "acute bacterial sinusitis": {
        "treatments": {
            "symptomatic": ["Nasal saline irrigation", "Pseudoephedrine 60 mg PRN"],
            "definitive": ["Amoxicillin 500 mg TID for 10 days"],
            "lifestyle": ["Hydration (2 L/day)"]
        }
    },
    # Add other diagnosis treatments as needed
}

default_management_config: Dict = {
    "follow_up_default": "Follow-up in 2 weeks",
    "follow_up_urgent": "Follow-up in 3-5 days or sooner if symptoms worsen",
    "urgent_threshold": 0.9,
    "min_symptom_match": 0.7
}

# Resource configuration
resources = {
    "symptoms": (default_symptoms, lambda x: x, "postgresql"),
    "medical_stop_words": (default_stop_words, lambda x: list(x), "postgresql"),
    "medical_terms": (default_medical_terms, lambda x: x, "postgresql"),
    "synonyms": (default_synonyms, lambda x: x, "mongodb"),
    "clinical_pathways": (default_clinical_pathways, lambda x: x, "mongodb"),
    "history_diagnoses": (default_history_diagnoses, lambda x: x, "mongodb"),
    "diagnosis_relevance": (default_diagnosis_relevance, lambda x: x, "mongodb"),
    "management_config": (default_management_config, lambda x: x, "mongodb"),
    "diagnosis_treatments": (default_diagnosis_treatments, lambda x: x, "mongodb")
}

def validate_umls_cui(symptom, cui, semantic_type):
    from departments.nlp.nlp_pipeline import get_postgres_connection
    from departments.nlp.logging_setup import get_logger
    logger = get_logger(__name__)

    with get_postgres_connection() as cursor:
        if not cursor:
            logger.error(f"No database cursor available to validate CUI {cui} for symptom {symptom}")
            return False
        try:
            query = """
                SELECT COUNT(*) 
                FROM umls.MRCONSO 
                WHERE CUI = %s AND STR ILIKE %s
            """
            cursor.execute(query, (cui, f"%{symptom}%"))
            result = cursor.fetchone()
            return result['count'] > 0 if result else False
        except Exception as e:
            logger.error(f"Failed to validate CUI {cui} for symptom {symptom}: {e}")
            return False

def initialize_knowledge_files() -> None:
    """Initialize PostgreSQL and MongoDB with default knowledge base resources."""
    current_date = datetime.now()

    # Validate data
    validated_resources = {}
    for key, (default_data, transform, storage) in resources.items():
        validated_data = default_data
        if key == "symptoms" and storage == "postgresql":
            validated_data = {}
            for category, symptoms in default_data.items():
                if not isinstance(symptoms, dict):
                    logger.warning(f"Invalid symptoms structure for category {category}: {type(symptoms)}")
                    continue
                validated_data[category] = {}
                for symptom, info in symptoms.items():
                    if not isinstance(info, dict) or not info.get('description') or not info.get('umls_cui'):
                        logger.warning(f"Invalid symptom {symptom} in {category}: {info}")
                        continue
                    if validate_umls_cui(symptom, info['umls_cui'], info['semantic_type']):
                        validated_data[category][symptom] = info
        elif key == "medical_stop_words" and storage == "postgresql":
            if not isinstance(default_data, list) or not all(isinstance(w, str) for w in default_data):
                logger.warning(f"Invalid medical_stop_words: {type(default_data)}")
                validated_data = []
            else:
                validated_data = default_data
        elif key == "medical_terms" and storage == "postgresql":
            validated_data = [
                t for t in default_data
                if isinstance(t, dict) and t.get('term') and t.get('umls_cui') and
                validate_umls_cui(t['term'], t['umls_cui'], t['semantic_type'])
            ]
        elif key == "clinical_pathways" and storage == "mongodb":
            validated_data = {}
            for category, paths in default_data.items():
                if not isinstance(paths, dict):
                    logger.warning(f"Invalid clinical_pathways for category {category}: {type(paths)}")
                    continue
                validated_data[category] = {}
                for path_key, path in paths.items():
                    if not isinstance(path, dict) or 'metadata' not in path:
                        logger.warning(f"Invalid path '{path_key}' in category '{category}'")
                        continue
                    validated_path = path.copy()
                    if 'last_updated' in validated_path['metadata']:
                        validated_path['metadata']['last_updated'] = parse_date(validated_path['metadata']['last_updated'])
                    validated_data[category][path_key] = validated_path
        elif key == "diagnosis_relevance" and storage == "mongodb":
            validated_data = [
                item for item in default_data
                if isinstance(item, dict) and item.get('diagnosis') and isinstance(item.get('relevance'), list)
            ]
            for item in validated_data:
                weights = sum(r['weight'] for r in item['relevance'])
                if abs(weights - 1.0) > 0.01:
                    logger.warning(f"Diagnosis {item['diagnosis']} relevance weights do not sum to 1.0: {weights}")
        elif key == "synonyms" and storage == "mongodb":
            validated_data = {
                term: aliases for term, aliases in default_data.items()
                if isinstance(aliases, list) and all(isinstance(a, str) for a in aliases)
            }
        elif key == "history_diagnoses" and storage == "mongodb":
            validated_data = {
                k: v for k, v in default_data.items()
                if isinstance(v, dict) and v.get('synonyms') and v.get('umls_cui')
            }
        elif key == "diagnosis_treatments" and storage == "mongodb":
            validated_data = {
                k: v for k, v in default_data.items()
                if isinstance(v, dict) and isinstance(v.get('treatments'), dict)
            }
        elif key == "management_config" and storage == "mongodb":
            if not isinstance(default_data, dict):
                logger.warning(f"Invalid management_config: {type(default_data)}")
                validated_data = {}
        validated_resources[key] = validated_data

    # Initialize PostgreSQL
    with get_postgres_connection() as cursor:
        if not cursor:
            logger.error("Failed to connect to PostgreSQL")
            return
        try:
            cursor.execute("SELECT COUNT(*) AS count FROM medical_stop_words")
            if cursor.fetchone()['count'] == 0:
                execute_batch(cursor, "INSERT INTO medical_stop_words (word) VALUES (%s) ON CONFLICT DO NOTHING",
                              [(word,) for word in validated_resources['medical_stop_words']])
                logger.info(f"Initialized {len(validated_resources['medical_stop_words'])} medical_stop_words in PostgreSQL")

            cursor.execute("SELECT COUNT(*) AS count FROM medical_terms")
            if cursor.fetchone()['count'] == 0:
                execute_batch(cursor, """
                    INSERT INTO medical_terms (term, category, umls_cui, semantic_type)
                    VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING
                """, [(t['term'], t['category'], t['umls_cui'], t['semantic_type']) for t in validated_resources['medical_terms']])
                logger.info(f"Initialized {len(validated_resources['medical_terms'])} medical_terms in PostgreSQL")

            cursor.execute("SELECT COUNT(*) AS count FROM symptoms")
            if cursor.fetchone()['count'] == 0:
                execute_batch(cursor, """
                    INSERT INTO symptoms (symptom, category, description, umls_cui, semantic_type)
                    VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING
                """, [(s, cat, info['description'], info['umls_cui'], info['semantic_type'])
                      for cat, symptoms in validated_resources['symptoms'].items()
                      for s, info in symptoms.items()])
                logger.info(f"Initialized {sum(len(s) for s in validated_resources['symptoms'].values())} symptoms in PostgreSQL")

            cursor.execute("""
                INSERT INTO knowledge_base_metadata (key, version, last_updated)
                VALUES (%s, %s, %s) ON CONFLICT (key) DO UPDATE
                SET version = EXCLUDED.version, last_updated = EXCLUDED.last_updated
            """, ('knowledge_base', '1.1.0', current_date.strftime("%Y-%m-%d %H:%M:%S")))
            cursor.connection.commit()
        except Exception as e:
            logger.error(f"PostgreSQL initialization failed: {str(e)}")
            cursor.connection.rollback()

    # Initialize MongoDB
    try:
        client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=10000,
            socketTimeoutMS=10000
        )
        client.admin.command('ping')
        db = client.get_database(DB_NAME)
        kb_prefix = KB_PREFIX or 'kb_'
        for key, (default_data, _, storage) in resources.items():
            if storage != "mongodb":
                continue
            collection = db[f'{kb_prefix}{key}']
            if collection.count_documents({}) > 0:
                logger.info(f"Collection {key} already contains data, skipping initialization")
                continue
            validated_data = validated_resources[key]
            if key == "clinical_pathways":
                formatted_data = [
                    {
                        'category': category,
                        'paths': {
                            path_key: {
                                **path,
                                'metadata': {
                                    **path['metadata'],
                                    'last_updated': parse_date(path['metadata']['last_updated']).strftime("%Y-%m-%d %H:%M:%S")
                                }
                            }
                            for path_key, path in paths.items()
                        }
                    }
                    for category, paths in validated_data.items()
                ]
                collection.insert_many(formatted_data)
                logger.info(f"Initialized {len(validated_data)} clinical_pathways in MongoDB")
            elif key == "diagnosis_relevance":
                collection.insert_many(validated_data)
                logger.info(f"Initialized {len(validated_data)} diagnosis_relevance entries in MongoDB")
            elif key == "synonyms":
                collection.insert_many([{'term': term, 'aliases': aliases} for term, aliases in validated_data.items()])
                logger.info(f"Initialized {len(validated_data)} synonyms in MongoDB")
            elif key in ["history_diagnoses", "diagnosis_treatments", "management_config"]:
                collection.insert_many([{'key': k, 'value': v} for k, v in validated_data.items()])
                logger.info(f"Initialized {len(validated_data)} {key} entries in MongoDB")
        client.close()
    except ConnectionFailure as e:
        logger.error(f"MongoDB initialization failed: {str(e)}")

def load_knowledge_base() -> Dict:
    """Load knowledge base for use in nlp_pipeline.py and clinical_analyzer.py."""
    return {
        "medical_stop_words": set(default_stop_words),
        "medical_terms": default_medical_terms,
        "symptoms": default_symptoms,
        "synonyms": default_synonyms,
        "clinical_pathways": default_clinical_pathways,
        "history_diagnoses": default_history_diagnoses,
        "diagnosis_relevance": default_diagnosis_relevance,
        "management_config": default_management_config,
        "diagnosis_treatments": default_diagnosis_treatments,
        "version": "1.1.0"
    }