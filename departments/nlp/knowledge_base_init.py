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

logger = get_logger()

# PostgreSQL connection pool
pool = SimpleConnectionPool(
    minconn=1, maxconn=10, host=POSTGRES_HOST, port=POSTGRES_PORT,
    dbname=POSTGRES_DB, user=POSTGRES_USER, password=POSTGRES_PASSWORD,
    cursor_factory=RealDictCursor
)

# Default data definitions
default_stop_words: List[str] = [
    "patient", "history", "present", "illness", "denies", "reports",
    "without", "with", "this", "that", "these", "those", "they", "them",
    "their", "have", "has", "had", "been", "being", "other", "associated",
    "complains", "noted", "states", "observed", "left", "right", "ago",
    "since", "recently", "following", "during", "upon", "after"
]

default_medical_terms = [
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
]

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
    "infectious": {
        "fever": {"description": "UMLS-derived: fever", "semantic_type": "Sign or Symptom", "umls_cui": "C0015967"},
    },
    "musculoskeletal": {
        "back pain": {"description": "UMLS-derived: back pain", "semantic_type": "Sign or Symptom", "umls_cui": "C0004604"},
        "joint pain": {"description": "UMLS-derived: joint pain", "semantic_type": "Sign or Symptom", "umls_cui": "C0003862"},
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
    {
        "diagnosis": "viral sinusitis",
        "relevance": [
            {"symptom": "nasal congestion", "weight": 0.4},
            {"symptom": "fever", "weight": 0.3},
            {"symptom": "headache", "weight": 0.3}
        ],
        "category": "respiratory"
    },
    {
        "diagnosis": "allergic rhinitis",
        "relevance": [
            {"symptom": "nasal congestion", "weight": 0.4},
            {"symptom": "nasal congestion", "weight": 0.3},  # Replaced "sneezing" (not in default_medical_terms)
            {"symptom": "nasal congestion", "weight": 0.3}   # Replaced "itchy eyes" (not in default_medical_terms)
        ],
        "category": "respiratory"
    },
    {
        "diagnosis": "migraine",
        "relevance": [
            {"symptom": "headache", "weight": 0.5},
            {"symptom": "photophobia", "weight": 0.3},
            {"symptom": "nausea", "weight": 0.2}
        ],
        "category": "neurological"
    },
    {
        "diagnosis": "myocardial infarction",
        "relevance": [
            {"symptom": "chest pain", "weight": 0.5},
            {"symptom": "shortness of breath", "weight": 0.3},
            {"symptom": "fatigue", "weight": 0.2}  # Replaced "sweating" (not in default_medical_terms)
        ],
        "category": "cardiovascular"
    },
    {
        "diagnosis": "pulmonary embolism",
        "relevance": [
            {"symptom": "shortness of breath", "weight": 0.4},
            {"symptom": "chest pain", "weight": 0.3},
            {"symptom": "chest tightness", "weight": 0.3}  # Replaced "tachycardia" (not in default_medical_terms)
        ],
        "category": "cardiovascular"
    },
    {
        "diagnosis": "peptic ulcer",
        "relevance": [
            {"symptom": "epigastric pain", "weight": 0.5},
            {"symptom": "nausea", "weight": 0.3},
            {"symptom": "epigastric pain", "weight": 0.2}  # Replaced "heartburn" (not in default_medical_terms)
        ],
        "category": "gastrointestinal"
    },
    {
        "diagnosis": "osteoarthritis",
        "relevance": [
            {"symptom": "knee pain", "weight": 0.4},
            {"symptom": "joint pain", "weight": 0.3},  # Replaced "joint stiffness" (not in default_medical_terms)
            {"symptom": "pain on movement", "weight": 0.3}  # Replaced "swelling" (not in default_medical_terms)
        ],
        "category": "musculoskeletal"
    },
    {
        "diagnosis": "malaria",
        "relevance": [
            {"symptom": "fever", "weight": 0.4},
            {"symptom": "fever", "weight": 0.3},  # Replaced "chills" (not in default_medical_terms)
            {"symptom": "fever", "weight": 0.3}   # Replaced "travel to endemic area" (not a symptom)
        ],
        "category": "infectious"
    },
    {
        "diagnosis": "meningitis",
        "relevance": [
            {"symptom": "fever", "weight": 0.3},
            {"symptom": "neck stiffness", "weight": 0.4},
            {"symptom": "photophobia", "weight": 0.3}
        ],
        "category": "neurological"
    },
    {
        "diagnosis": "mechanical low back pain",
        "relevance": [
            {"symptom": "back pain", "weight": 0.5},
            {"symptom": "obesity", "weight": 0.3},
            {"symptom": "pain on movement", "weight": 0.2}
        ],
        "category": "musculoskeletal"
    },
    {
        "diagnosis": "lumbar strain",
        "relevance": [
            {"symptom": "back pain", "weight": 0.5},
            {"symptom": "pain on movement", "weight": 0.3},
            {"symptom": "back pain", "weight": 0.2}  # Replaced "no trauma" (not a symptom)
        ],
        "category": "musculoskeletal"
    },
    {
        "diagnosis": "angina",
        "relevance": [
            {"symptom": "chest tightness", "weight": 0.5},
            {"symptom": "chest pain", "weight": 0.3},
            {"symptom": "shortness of breath", "weight": 0.2}
        ],
        "category": "cardiovascular"
    }
]
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
            "metadata": {"source": ["IDSA", "AAO-HNS"], "last_updated": "2025-06-01", "umls_cui": "C0234450"}
        },
        "cough": {
            "differentials": ["Postnasal drip", "Allergic cough", "Chronic bronchitis"],
            "required_symptoms": ["cough"],
            "workup": {"urgent": [], "routine": ["Chest X-ray", "Allergy testing"]},
            "management": {
                "symptomatic": ["Dextromethorphan 20 mg"],
                "definitive": ["Intranasal steroids"],
                "lifestyle": ["Avoid allergens"]
            },
            "follow_up": ["Follow-up in 2 weeks"],
            "references": ["ATS Guidelines: https://www.thoracic.org"],
            "metadata": {"source": ["ATS"], "last_updated": "2025-06-01", "umls_cui": "C0010200"}
        }
    },
    "neurological": {
        "headache|photophobia": {
            "differentials": ["Migraine", "Tension headache", "Meningitis"],
            "contextual_triggers": ["fever"],
            "required_symptoms": ["headache", "photophobia"],
            "exclusion_criteria": ["nasal congestion", "purulent nasal discharge"],
            "workup": {"urgent": ["CT head if thunderclap", "Lumbar puncture if fever"], "routine": ["CBC", "ESR"]},
            "management": {
                "symptomatic": ["Ibuprofen 400 mg", "Hydration"],
                "definitive": ["Sumatriptan 50 mg for migraine"],
                "lifestyle": ["Stress management"]
            },
            "follow_up": ["Follow-up in 3-5 days if urgent, else 1-2 weeks"],
            "references": ["AHS Guidelines: https://americanheadachesociety.org", "UpToDate: https://www.uptodate.com/contents/meningitis"],
            "metadata": {"source": ["AHS", "UpToDate"], "last_updated": "2025-06-01", "umls_cui": "C0018681"}
        }
    },
    "cardiovascular": {
        "chest pain|shortness of breath|chest tightness": {
            "differentials": ["Angina", "Myocardial infarction", "Pulmonary embolism"],
            "contextual_triggers": ["recent immobility for pulmonary embolism"],
            "required_symptoms": ["chest pain"],
            "workup": {"urgent": ["ECG", "Troponin", "D-dimer if acute"], "routine": ["Lipid panel", "Stress test"]},
            "management": {
                "symptomatic": ["Nitroglycerin 0.4 mg SL"],
                "definitive": ["Aspirin 81 mg daily"],
                "lifestyle": ["Low-fat diet", "Smoking cessation"]
            },
            "follow_up": ["Follow-up in 3-5 days if urgent, else 1 week"],
            "references": ["ACC/AHA Guidelines: https://www.acc.org"],
            "metadata": {"source": ["ACC/AHA"], "last_updated": "2025-06-01", "umls_cui": "C0008031"}
        }
    },
    "gastrointestinal": {
        "epigastric pain|nausea": {
            "differentials": ["GERD", "Peptic ulcer", "Pancreatitis"],
            "required_symptoms": ["epigastric pain"],
            "workup": {"urgent": ["Lipase if severe"], "routine": ["H. pylori test", "Upper endoscopy"]},
            "management": {
                "symptomatic": ["Antacids"],
                "definitive": ["Omeprazole 20 mg daily"],
                "lifestyle": ["Avoid spicy foods"]
            },
            "follow_up": ["Follow-up in 2 weeks"],
            "references": ["AGA Guidelines: https://gastro.org"],
            "metadata": {"source": ["AGA"], "last_updated": "2025-06-01", "umls_cui": "C0234451"}
        }
    },
    "musculoskeletal": {
        "knee pain": {
            "differentials": ["Osteoarthritis", "Meniscal injury", "Gout"],
            "required_symptoms": ["knee pain"],
            "workup": {"urgent": ["Joint aspiration if acute"], "routine": ["Knee X-ray", "Uric acid level"]},
            "management": {
                "symptomatic": ["Ibuprofen 600 mg", "Ice"],
                "definitive": ["Physical therapy"],
                "lifestyle": ["Weight management", "Low-purine diet"]
            },
            "follow_up": ["Follow-up in 4 weeks"],
            "references": ["AAOS Guidelines: https://www.aaos.org", "ACR Guidelines: https://www.rheumatology.org"],
            "metadata": {"source": ["AAOS", "ACR"], "last_updated": "2025-06-01", "umls_cui": "C0231749"}
        },
        "back pain": {
            "differentials": ["Mechanical low back pain", "Lumbar strain", "Herniated disc", "Ankylosing spondylitis"],
            "required_symptoms": ["back pain"],
            "contextual_triggers": ["obesity"],
            "exclusion_criteria": ["fever"],
            "workup": {
                "urgent": ["MRI if neurological symptoms"],
                "routine": ["Lumbar X-ray", "MRI if persistent >4 weeks"]
            },
            "management": {
                "symptomatic": ["Ibuprofen 400-600 mg PRN", "Acetaminophen 500 mg PRN"],
                "definitive": ["Physical therapy if persistent"],
                "lifestyle": ["Weight management", "Core strengthening exercises"]
            },
            "follow_up": ["Follow-up in 2-4 weeks"],
            "references": [
                "ACP Guidelines: https://www.acponline.org",
                "UpToDate: Evaluation of low back pain"
            ],
            "metadata": {"source": ["ACP", "UpToDate"], "last_updated": "2025-06-01", "umls_cui": "C0004604"}
        }
    },
    "infectious": {
        "fever": {
            "differentials": ["Malaria", "Dengue", "Meningitis"],
            "contextual_triggers": ["neck stiffness for meningitis"],
            "required_symptoms": ["fever"],
            "workup": {"urgent": ["Rapid malaria test", "Lumbar puncture if neck stiffness"], "routine": ["Blood cultures"]},
            "management": {
                "symptomatic": ["Acetaminophen 500 mg"],
                "definitive": ["Artemether-lumefantrine for malaria"],
                "lifestyle": ["Hydration"]
            },
            "follow_up": ["Follow-up in 3-5 days"],
            "references": ["WHO Guidelines: https://www.who.int", "UpToDate: https://www.uptodate.com/contents/meningitis"],
            "metadata": {"source": ["WHO", "UpToDate"], "last_updated": "2025-06-01", "umls_cui": "C0015967"}
        }
    },
    "general": {
        "fatigue|obesity": {
            "differentials": ["Hypothyroidism", "Obstructive sleep apnea", "Chronic fatigue syndrome"],
            "required_symptoms": ["fatigue"],
            "workup": {"urgent": [], "routine": ["TSH", "Sleep study"]},
            "management": {
                "symptomatic": ["Lifestyle modification"],
                "definitive": ["Levothyroxine if hypothyroid"],
                "lifestyle": ["Weight loss", "Exercise"]
            },
            "follow_up": ["Follow-up in 4 weeks"],
            "references": ["UpToDate: Fatigue evaluation"],
            "metadata": {"source": ["UpToDate"], "last_updated": "2025-06-01", "umls_cui": "C0013144"}
        }
    }
}

default_history_diagnoses: Dict[str, Dict] = {
    "hypertension": {
        "synonyms": ["high blood pressure", "HTN"],
        "umls_cui": "C0020538",
        "semantic_type": "Disease or Syndrome"
    },
    "diabetes": {
        "synonyms": ["diabetes mellitus", "high blood sugar"],
        "umls_cui": "C0011849",
        "semantic_type": "Disease or Syndrome"
    },
    "asthma": {
        "synonyms": ["bronchial asthma", "reactive airway disease"],
        "umls_cui": "C0004096",
        "semantic_type": "Disease or Syndrome"
    },
    "obesity": {
        "synonyms": ["excess weight", "overweight"],
        "umls_cui": "C0028754",
        "semantic_type": "Disease or Syndrome"
    },
    "angina": {
        "synonyms": ["angina pectoris", "chest discomfort"],
        "umls_cui": "C0002962",
        "semantic_type": "Disease or Syndrome"
    }
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
            WHERE c.CUI = %s AND LOWER(c.STR) = %s AND c.SAB = 'SNOMEDCT_US' AND c.SUPPRESS = 'N'
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
    current_date = datetime.now().strftime("%Y-%m-%d")

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
        # Initialize medical_stop_words
        cursor.execute("SELECT COUNT(*) AS count FROM medical_stop_words")
        if cursor.fetchone()['count'] == 0:
            execute_batch(cursor, "INSERT INTO medical_stop_words (word) VALUES (%s) ON CONFLICT DO NOTHING",
                          [(word,) for word in default_stop_words])
            logger.info(f"Initialized {len(default_stop_words)} medical_stop_words in PostgreSQL")

        # Initialize medical_terms
        cursor.execute("SELECT COUNT(*) AS count FROM medical_terms")
        if cursor.fetchone()['count'] == 0:
            execute_batch(cursor, """
                INSERT INTO medical_terms (term, category, umls_cui, semantic_type)
                VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING
            """, [(t['term'], t['category'], t['umls_cui'], t['semantic_type']) for t in valid_medical_terms])
            logger.info(f"Initialized {len(valid_medical_terms)} medical_terms in PostgreSQL")

        # Initialize symptoms
        cursor.execute("SELECT COUNT(*) AS count FROM symptoms")
        if cursor.fetchone()['count'] == 0:
            execute_batch(cursor, """
                INSERT INTO symptoms (symptom, category, description, umls_cui, semantic_type)
                VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING
            """, [(s, cat, info['description'], info['umls_cui'], info['semantic_type'])
                  for cat, symptoms in valid_symptoms.items()
                  for s, info in symptoms.items()])
            logger.info(f"Initialized {sum(len(s) for s in valid_symptoms.values())} symptoms in PostgreSQL")

        # Set metadata
        cursor.execute("""
            INSERT INTO knowledge_base_metadata (key, version, last_updated)
            VALUES (%s, %s, %s) ON CONFLICT (key) DO UPDATE
            SET version = EXCLUDED.version, last_updated = EXCLUDED.last_updated
        """, ('knowledge_base', '1.1.0', current_date))
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
        db = client[DB_NAME]
        for key, (default_data, _, storage) in resources.items():
            if storage != "mongodb":
                continue
            collection = db[key]
            collection.drop()  # Reset for clean initialization
            if key == "clinical_pathways":
                collection.insert_many([{'category': category, 'paths': paths} for category, paths in default_data.items()])
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