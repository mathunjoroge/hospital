from datetime import datetime
from typing import Dict, List
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from psycopg2.extras import RealDictCursor, execute_batch
from psycopg2.pool import SimpleConnectionPool
from departments.nlp.logging_setup import get_logger
from departments.nlp.config import (
    MONGO_URI, DB_NAME, POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
)

logger = get_logger(__name__)

# PostgreSQL connection pool
pool = SimpleConnectionPool(
    minconn=1, maxconn=10, host=POSTGRES_HOST, port=POSTGRES_PORT,
    dbname=POSTGRES_DB, user=POSTGRES_USER, password=POSTGRES_PASSWORD,
    cursor_factory=RealDictCursor
)

# Default stop words
default_stop_words: List[str] = [
    "patient", "history", "present", "illness", "denies", "reports",
    "without", "with", "this", "that", "these", "those", "they", "them",
    "their", "have", "has", "had", "been", "being", "other", "associated",
    "complains", "noted", "states", "observed", "left", "right", "ago",
    "since", "recently", "following", "during", "upon", "after"
]

fallback_medical_terms: List[Dict] = [
    {"term": "facial pain", "category": "respiratory", "umls_cui": "C0234450", "semantic_type": "Sign or Symptom"},
    {"term": "nasal congestion", "category": "respiratory", "umls_cui": "C0027424", "semantic_type": "Sign or Symptom"},
    {"term": "purulent nasal discharge", "category": "respiratory", "umls_cui": "C0242209", "semantic_type": "Sign or Symptom"},
    {"term": "fever", "category": "infectious", "umls_cui": "C0015967", "semantic_type": "Sign or Symptom"},
    {"term": "headache", "category": "neurological", "umls_cui": "C0018681", "semantic_type": "Sign or Symptom"},
    {"term": "cough", "category": "respiratory", "umls_cui": "C0010200", "semantic_type": "Sign or Symptom"},
    {"term": "chest pain", "category": "cardiovascular", "umls_cui": "C0008031", "semantic_type": "Sign or Symptom"},
    {"term": "shortness of breath", "category": "cardiovascular", "umls_cui": "C0013404", "semantic_type": "Sign or Symptom"},
    {"term": "photophobia", "category": "neurological", "umls_cui": "C0085636", "semantic_type": "Sign or Symptom"},
    {"term": "neck stiffness", "category": "neurological", "umls_cui": "C0029101", "semantic_type": "Sign or Symptom"},
    {"term": "rash", "category": "dermatological", "umls_cui": "C0015230", "semantic_type": "Sign or Symptom"},
    {"term": "back pain", "category": "musculoskeletal", "umls_cui": "C0004604", "semantic_type": "Sign or Symptom"},
    {"term": "knee pain", "category": "musculoskeletal", "umls_cui": "C0231749", "semantic_type": "Sign or Symptom"},
    {"term": "epigastric pain", "category": "gastrointestinal", "umls_cui": "C0234451", "semantic_type": "Sign or Symptom"},
    {"term": "fatigue", "category": "general", "umls_cui": "C0013144", "semantic_type": "Sign or Symptom"},
    {"term": "chest tightness", "category": "cardiovascular", "umls_cui": "C0232292", "semantic_type": "Sign or Symptom"},
    {"term": "nausea", "category": "gastrointestinal", "umls_cui": "C0027497", "semantic_type": "Sign or Symptom"},
    {"term": "obesity", "category": "general", "umls_cui": "C0028754", "semantic_type": "Disease or Syndrome"},
    {"term": "joint pain", "category": "musculoskeletal", "umls_cui": "C0003862", "semantic_type": "Sign or Symptom"},
    {"term": "pain on movement", "category": "musculoskeletal", "umls_cui": "C0234452", "semantic_type": "Sign or Symptom"},
    {"term": "jaundice", "category": "hepatic", "umls_cui": "C0022346", "semantic_type": "Sign or Symptom"},
    {"term": "abdominal pain", "category": "hepatic", "umls_cui": "C0000737", "semantic_type": "Sign or Symptom"},
]

# Fetch medical terms from PostgreSQL
def fetch_medical_terms_from_postgres() -> List[Dict]:
    conn = pool.getconn()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT term, category, umls_cui, semantic_type
            FROM medical_terms
        """)
        results = cursor.fetchall()
        medical_terms = [
            {
                "term": row['term'],
                "category": row['category'],
                "umls_cui": row['umls_cui'],
                "semantic_type": row['semantic_type']
            }
            for row in results
        ]
        logger.info(f"Fetched {len(medical_terms)} medical terms from PostgreSQL")
        return medical_terms
    except Exception as e:
        logger.error(f"Failed to fetch medical terms from PostgreSQL: {str(e)}")
        return []
    finally:
        cursor.close()
        pool.putconn(conn)

# Load default_medical_terms from PostgreSQL or use fallback
default_medical_terms = fetch_medical_terms_from_postgres()
if not default_medical_terms:
    logger.warning("No medical terms fetched from PostgreSQL, using fallback_medical_terms")
    default_medical_terms = fallback_medical_terms

# Default symptoms
default_symptoms: Dict[str, Dict[str, Dict]] = {
    "cardiovascular": {
        "chest pain": {"description": "UMLS-derived: chest pain", "semantic_type": "Sign or Symptom", "umls_cui": "C0008031"},
        "chest tightness": {"description": "UMLS-derived: chest tightness", "semantic_type": "Sign or Symptom", "umls_cui": "C0232292"},
        "shortness of breath": {"description": "UMLS-derived: shortness of breath", "semantic_type": "Sign or Symptom", "umls_cui": "C0013404"},
    },
    "dermatological": {
        "rash": {"description": "UMLS-derived: rash", "semantic_type": "Sign or Symptom", "umls_cui": "C0015230"},
    },
    "gastrointestinal": {
        "epigastric pain": {"description": "UMLS-derived: epigastric pain", "semantic_type": "Sign or Symptom", "umls_cui": "C0234451"},
        "nausea": {"description": "UMLS-derived: nausea", "semantic_type": "Sign or Symptom", "umls_cui": "C0027497"},
    },
    "general": {
        "fatigue": {"description": "UMLS-derived: fatigue", "semantic_type": "Sign or Symptom", "umls_cui": "C0013144"},
        "obesity": {"description": "UMLS-derived: obesity", "semantic_type": "Disease or Syndrome", "umls_cui": "C0028754"},
    },
    "hepatic": {
        "jaundice": {"description": "UMLS-derived: jaundice", "semantic_type": "Sign or Symptom", "umls_cui": "C0022346"},
        "abdominal pain": {"description": "UMLS-derived: abdominal pain", "semantic_type": "Sign or Symptom", "umls_cui": "C0000737"},
    },
    "infectious": {
        "fever": {"description": "UMLS-derived: fever", "semantic_type": "Sign or Symptom", "umls_cui": "C0015967"},
    },
    "musculoskeletal": {
        "back pain": {"description": "UMLS-derived: back pain", "semantic_type": "Sign or Symptom", "umls_cui": "C0004604"},
        "joint pain": {"description": "UMLS-derived: joint pain", "semantic_type": "Sign or Symptom", "umls_cuisport": "C0003862"},
        "knee pain": {"description": "UMLS-derived: knee pain", "semantic_type": "Sign or Symptom", "umls_cui": "C0231749"},
        "pain on movement": {"description": "UMLS-derived: pain on movement", "semantic_type": "Sign or Symptom", "umls_cui": "C0234452"},
    },
    "neurological": {
        "headache": {"description": "UMLS-derived: headache", "semantic_type": "Sign or Symptom", "umls_cui": "C0018681"},
        "neck stiffness": {"description": "UMLS-derived: neck stiffness", "semantic_type": "Sign or Symptom", "umls_cui": "C0029101"},
        "photophobia": {"description": "UMLS-derived: photophobia", "semantic_type": "Sign or Symptom", "umls_cui": "C0085636"},
    },
    "respiratory": {
        "cough": {"description": "UMLS-derived: cough", "semantic_type": "Sign or Symptom", "umls_cui": "C0010200"},
        "facial pain": {"description": "UMLS-derived: facial pain", "semantic_type": "Sign or Symptom", "umls_cui": "C0234450"},
        "nasal congestion": {"description": "UMLS-derived: nasal congestion", "semantic_type": "Sign or Symptom", "umls_cui": "C0027424"},
        "purulent nasal discharge": {"description": "UMLS-derived: purulent nasal discharge", "semantic_type": "Sign or Symptom", "umls_cui": "C0242209"},
    }
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
    # ... (rest of diagnosis_relevance unchanged)
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
    # ... (rest unchanged)
}

# Default management config
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
    "management_config": (default_management_config, lambda x: x, "mongodb")
}

def validate_umls_cui(term: str, cui: str, semantic_type: str) -> bool:
    """Validate term against PostgreSQL UMLS."""
    conn = pool.getconn()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT c.CUI, sty.STY
            FROM umls.MRCONSO c
            JOIN umls.MRSTY sty ON c.CUI = sty.CUI
            WHERE c.CUI = %s AND LOWER(c.STR) = %s AND c.SAB = 'SNOMEDCT_US'
        """, (cui, term.lower()))
        result = cursor.fetchone()
        if result and result['sty'] == semantic_type:
            return True
        logger.warning(f"UMLS validation failed for term '{term}', CUI: {cui}, Semantic Type: {semantic_type}")
        return False
    except Exception as e:
        logger.error(f"UMLS validation error for '{term}': {str(e)}")
        return False
    finally:
        cursor.close()
        pool.putconn(conn)

def initialize_knowledge_files() -> None:
    """Initialize PostgreSQL and MongoDB with default knowledge base resources."""
    current_date = datetime.now()

    # Validate data
    for key, (default_data, _, storage) in resources.items():
        if key == "symptoms" and storage == "postgresql":
            for category, symptoms in default_data.items():
                if not isinstance(symptoms, dict):
                    logger.error(f"Invalid symptoms structure for category {category}: {type(symptoms)}")
                    return
                for symptom, info in symptoms.items():
                    if not isinstance(info, dict) or not info.get('description') or not info.get('umls_cui'):
                        logger.error(f"Invalid symptom {symptom} in {category}: {info}")
                        return
        elif key == "medical_stop_words" and storage == "postgresql":
            if not isinstance(default_data, list) or not all(isinstance(w, str) for w in default_data):
                logger.error(f"Invalid medical_stop_words: {type(default_data)}")
                return
        elif key == "medical_terms" and storage == "postgresql":
            for term in default_data:
                if not isinstance(term, dict) or not term.get('term') or not term.get('umls_cui'):
                    logger.error(f"Invalid medical_term: {term}")
                    return
        elif key == "clinical_pathways" and storage == "mongodb":
            for category, paths in default_data.items():
                for path_key, path in paths.items():
                    if 'metadata' in path and 'last_updated' in path['metadata']:
                        last_updated = path['metadata']['last_updated']
                        if isinstance(last_updated, datetime):
                            continue
                        try:
                            path['metadata']['last_updated'] = datetime.strptime(last_updated, "%Y-%m-%d %H:%M:%S")
                        except (ValueError, TypeError):
                            try:
                                path['metadata']['last_updated'] = datetime.strptime(last_updated, "%Y-%m-%d")
                            except (ValueError, TypeError):
                                logger.warning(f"Invalid last_updated format for clinical path '{path_key}' in category '{category}'. Setting to current date.")
                                path['metadata']['last_updated'] = current_date
        elif key == "diagnosis_relevance" and storage == "mongodb":
            for item in default_data:
                if not isinstance(item, dict) or not item.get('diagnosis') or not item.get('relevance'):
                    logger.error(f"Invalid diagnosis_relevance: {item}")
                    return
                weights = sum(r['weight'] for r in item['relevance'])
                if abs(weights - 1.0) > 0.01:
                    logger.warning(f"Diagnosis {item['diagnosis']} relevance weights do not sum to 1.0: {weights}")

    # Validate medical_terms and symptoms against UMLS
    valid_medical_terms = [t for t in default_medical_terms if validate_umls_cui(t['term'], t['umls_cui'], t['semantic_type'])]
    valid_symptoms = {}
    for cat, symptoms in default_symptoms.items():
        valid_symptoms[cat] = {
            s: info for s, info in symptoms.items()
            if validate_umls_cui(s, info['umls_cui'], info['semantic_type'])
        }
    logger.info(f"Validated {len(valid_medical_terms)}/{len(default_medical_terms)} medical_terms and {sum(len(s) for s in valid_symptoms.values())} symptoms")

    # Initialize PostgreSQL
    conn = pool.getconn()
    if not conn:
        logger.error("Failed to connect to PostgreSQL")
        return
    try:
        cursor = conn.cursor()
        # Debug logging for existing data
        cursor.execute("SELECT key, version, last_updated FROM knowledge_base_metadata")
        for row in cursor.fetchall():
            logger.debug(f"knowledge_base_metadata: key={row['key']}, version={row['version']}, last_updated={row['last_updated']}, type={type(row['last_updated'])}")

        cursor.execute("SELECT COUNT(*) AS count FROM medical_stop_words")
        if cursor.fetchone()['count'] == 0:
            execute_batch(cursor, "INSERT INTO medical_stop_words (word) VALUES (%s) ON CONFLICT DO NOTHING",
                          [(word,) for word in default_stop_words])
            logger.info(f"Initialized {len(default_stop_words)} medical_stop_words in PostgreSQL")

        cursor.execute("SELECT COUNT(*) AS count FROM medical_terms")
        if cursor.fetchone()['count'] == 0:
            execute_batch(cursor, """
                INSERT INTO medical_terms (term, category, umls_cui, semantic_type)
                VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING
            """, [(t['term'], t['category'], t['umls_cui'], t['semantic_type']) for t in valid_medical_terms])
            logger.info(f"Initialized {len(valid_medical_terms)} medical_terms in PostgreSQL")

        cursor.execute("SELECT COUNT(*) AS count FROM symptoms")
        if cursor.fetchone()['count'] == 0:
            execute_batch(cursor, """
                INSERT INTO symptoms (symptom, category, description, umls_cui, semantic_type)
                VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING
            """, [(s, cat, info['description'], info['umls_cui'], info['semantic_type'])
                  for cat, symptoms in valid_symptoms.items()
                  for s, info in symptoms.items()])
            logger.info(f"Initialized {sum(len(s) for s in valid_symptoms.values())} symptoms in PostgreSQL")

        cursor.execute("""
            INSERT INTO knowledge_base_metadata (key, version, last_updated)
            VALUES (%s, %s, %s) ON CONFLICT (key) DO UPDATE
            SET version = EXCLUDED.version, last_updated = EXCLUDED.last_updated
        """, ('knowledge_base', '1.1.0', current_date.strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
    except Exception as e:
        logger.error(f"PostgreSQL initialization failed: {str(e)}")
        conn.rollback()
    finally:
        cursor.close()
        pool.putconn(conn)

    # Initialize MongoDB
    try:
        client = MongoClient(MONGO_URI)
        client.admin.command('ping')
        db = client.get_database(DB_NAME)
        for key, (default_data, _, storage) in resources.items():
            if storage != "mongodb":
                continue
            collection = db[key]
            collection.drop()  # Reset for clean initialization
            if key == "clinical_pathways":
                formatted_data = []
                for category, paths in default_data.items():
                    formatted_paths = {}
                    for path_key, path in paths.items():
                        formatted_path = path.copy()
                        if 'metadata' in formatted_path and 'last_updated' in formatted_path['metadata']:
                            last_updated = formatted_path['metadata']['last_updated']
                            logger.debug(f"Processing clinical path '{path_key}' in category '{category}': last_updated={last_updated}, type={type(last_updated)}")
                            if isinstance(last_updated, str):
                                try:
                                    formatted_path['metadata']['last_updated'] = datetime.strptime(last_updated, "%Y-%m-%d %H:%M:%S")
                                except ValueError:
                                    try:
                                        formatted_path['metadata']['last_updated'] = datetime.strptime(last_updated, "%Y-%m-%d")
                                    except ValueError:
                                        logger.warning(f"Invalid last_updated format in clinical path '{path_key}' in category '{category}': {last_updated}. Using current date.")
                                        formatted_path['metadata']['last_updated'] = current_date
                            formatted_path['metadata']['last_updated'] = formatted_path['metadata']['last_updated'].strftime("%Y-%m-%d %H:%M:%S")
                        formatted_paths[path_key] = formatted_path
                    formatted_data.append({'category': category, 'paths': formatted_paths})
                collection.insert_many(formatted_data)
                logger.info(f"Initialized {len(default_data)} clinical_pathways in MongoDB")
            elif key == "diagnosis_relevance":
                collection.insert_many(default_data)
                logger.info(f"Initialized {len(default_data)} diagnosis_relevance entries in MongoDB")
            elif key == "synonyms":
                collection.insert_many([{'term': term, 'aliases': aliases} for term, aliases in default_data.items()])
                logger.info(f"Initialized {len(default_data)} synonyms in MongoDB")
            else:
                collection.insert_many([{'key': k, 'value': v} for k, v in default_data.items()])
                logger.info(f"Initialized {len(default_data)} {key} entries in MongoDB")
        client.close()
    except ConnectionFailure as e:
        logger.error(f"MongoDB initialization failed: {str(e)}")

def load_knowledge_base() -> Dict:
    """Load knowledge base for use in nlp_pipeline.py."""
    return {
        "medical_stop_words": set(default_stop_words),
        "version": "1.1.0",
        "symptoms": default_symptoms,
        "clinical_pathways": default_clinical_pathways
    }