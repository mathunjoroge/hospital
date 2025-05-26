import os
import json
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from departments.nlp.logging_setup import get_logger
from departments.nlp.config import MONGO_URI, DB_NAME

logger = get_logger()

def initialize_knowledge_files() -> None:
    """Initialize default JSON files and MongoDB collections for knowledge base resources with UMLS metadata."""
    knowledge_base_dir = os.path.join(os.path.dirname(__file__), "knowledge_base")
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Check directory permissions
    try:
        if not os.path.exists(knowledge_base_dir):
            os.makedirs(knowledge_base_dir, exist_ok=True)
            logger.info(f"Created knowledge base directory: {knowledge_base_dir}")
        if not os.access(knowledge_base_dir, os.W_OK):
            logger.error(f"No write permission for {knowledge_base_dir}")
            return
    except OSError as e:
        logger.error(f"Failed to create or access {knowledge_base_dir}: {str(e)}")
        return

    default_stop_words = [
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
        {"term": "shortness of breath", "category": "cardiovascular", "umls_cui": "C0013144", "semantic_type": "Sign or Symptom"},
        {"term": "photophobia", "category": "neurological", "umls_cui": "C0085636", "semantic_type": "Sign or Symptom"},
        {"term": "neck stiffness", "category": "neurological", "umls_cui": "C0029101", "semantic_type": "Sign or Symptom"},
        {"term": "rash", "category": "dermatological", "umls_cui": "C0015230", "semantic_type": "Sign or Symptom"},
        {"term": "back pain", "category": "musculoskeletal", "umls_cui": "C0004604", "semantic_type": "Sign or Symptom"},
        {"term": "knee pain", "category": "musculoskeletal", "umls_cui": "C0231749", "semantic_type": "Sign or Symptom"},
        {"term": "epigastric pain", "category": "gastrointestinal", "umls_cui": "C0234451", "semantic_type": "Sign or Symptom"},
        {"term": "fatigue", "category": "general", "umls_cui": "C0013144", "semantic_type": "Sign or Symptom"},
        {"term": "chest tightness", "category": "cardiovascular", "umls_cui": "C0242209", "semantic_type": "Sign or Symptom"},
        {"term": "nausea", "category": "gastrointestinal", "umls_cui": "C0027497", "semantic_type": "Sign or Symptom"},
        {"term": "obesity", "category": "general", "umls_cui": "C0028754", "semantic_type": "Disease or Syndrome"},
        {"term": "joint pain", "category": "musculoskeletal", "umls_cui": "C0003862", "semantic_type": "Sign or Symptom"},
        {"term": "pain on movement", "category": "musculoskeletal", "umls_cui": "C0234452", "semantic_type": "Sign or Symptom"}
    ]

    default_symptoms = {
        "respiratory": {
            "facial pain": {"description": "UMLS-derived: facial pain", "umls_cui": "C0234450", "semantic_type": "Sign or Symptom"},
            "nasal congestion": {"description": "UMLS-derived: nasal congestion", "umls_cui": "C0027424", "semantic_type": "Sign or Symptom"},
            "purulent nasal discharge": {"description": "UMLS-derived: purulent nasal discharge", "umls_cui": "C0242209", "semantic_type": "Sign or Symptom"},
            "cough": {"description": "UMLS-derived: cough", "umls_cui": "C0010200", "semantic_type": "Sign or Symptom"}
        },
        "infectious": {
            "fever": {"description": "UMLS-derived: fever", "umls_cui": "C0015967", "semantic_type": "Sign or Symptom"}
        },
        "neurological": {
            "headache": {"description": "UMLS-derived: headache", "umls_cui": "C0018681", "semantic_type": "Sign or Symptom"},
            "photophobia": {"description": "UMLS-derived: photophobia", "umls_cui": "C0085636", "semantic_type": "Sign or Symptom"},
            "neck stiffness": {"description": "UMLS-derived: neck stiffness", "umls_cui": "C0029101", "semantic_type": "Sign or Symptom"}
        },
        "cardiovascular": {
            "chest pain": {"description": "UMLS-derived: chest pain", "umls_cui": "C0008031", "semantic_type": "Sign or Symptom"},
            "shortness of breath": {"description": "UMLS-derived: shortness of breath", "umls_cui": "C0013144", "semantic_type": "Sign or Symptom"},
            "chest tightness": {"description": "UMLS-derived: chest tightness", "umls_cui": "C0242209", "semantic_type": "Sign or Symptom"}
        },
        "dermatological": {
            "rash": {"description": "UMLS-derived: rash", "umls_cui": "C0015230", "semantic_type": "Sign or Symptom"}
        },
        "musculoskeletal": {
            "back pain": {"description": "UMLS-derived: back pain", "umls_cui": "C0004604", "semantic_type": "Sign or Symptom"},
            "knee pain": {"description": "UMLS-derived: knee pain", "umls_cui": "C0231749", "semantic_type": "Sign or Symptom"},
            "joint pain": {"description": "UMLS-derived: joint pain", "umls_cui": "C0003862", "semantic_type": "Sign or Symptom"},
            "pain on movement": {"description": "UMLS-derived: pain on movement", "umls_cui": "C0234452", "semantic_type": "Sign or Symptom"}
        },
        "gastrointestinal": {
            "epigastric pain": {"description": "UMLS-derived: epigastric pain", "umls_cui": "C0234451", "semantic_type": "Sign or Symptom"},
            "nausea": {"description": "UMLS-derived: nausea", "umls_cui": "C0027497", "semantic_type": "Sign or Symptom"}
        },
        "general": {
            "fatigue": {"description": "UMLS-derived: fatigue", "umls_cui": "C0013144", "semantic_type": "Sign or Symptom"},
            "obesity": {"description": "UMLS-derived: obesity", "umls_cui": "C0028754", "semantic_type": "Disease or Syndrome"}
        }
    }

    default_synonyms = {
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

    default_diagnosis_relevance = [
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
                {"symptom": "sneezing", "weight": 0.3},
                {"symptom": "itchy eyes", "weight": 0.3}
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
                {"symptom": "sweating", "weight": 0.2}
            ],
            "category": "cardiovascular"
        },
        {
            "diagnosis": "pulmonary embolism",
            "relevance": [
                {"symptom": "shortness of breath", "weight": 0.4},
                {"symptom": "chest pain", "weight": 0.3},
                {"symptom": "tachycardia", "weight": 0.3}
            ],
            "category": "cardiovascular"
        },
        {
            "diagnosis": "peptic ulcer",
            "relevance": [
                {"symptom": "epigastric pain", "weight": 0.5},
                {"symptom": "nausea", "weight": 0.3},
                {"symptom": "heartburn", "weight": 0.2}
            ],
            "category": "gastrointestinal"
        },
        {
            "diagnosis": "osteoarthritis",
            "relevance": [
                {"symptom": "knee pain", "weight": 0.4},
                {"symptom": "joint stiffness", "weight": 0.3},
                {"symptom": "swelling", "weight": 0.3}
            ],
            "category": "musculoskeletal"
        },
        {
            "diagnosis": "malaria",
            "relevance": [
                {"symptom": "fever", "weight": 0.4},
                {"symptom": "chills", "weight": 0.3},
                {"symptom": "travel to endemic area", "weight": 0.3}
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
                {"symptom": "no trauma", "weight": 0.2}
            ],
            "category": "musculoskeletal"
        },
        {
            "diagnosis": "angina",
            "relevance": [
                {"symptom": "chest tightness", "weight": 0.5},
                {"symptom": "chest pain", "weight": 0.3}
            ],
            "category": "cardiovascular"
        }
    ]

    default_clinical_pathways = {
        "respiratory": {
            "facial pain|nasal congestion|sinusitis|sinus pain|purulent nasal discharge": {
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
                "metadata": {"source": ["IDSA", "AAO-HNS"], "last_updated": current_date, "umls_cui": "C0234450"}
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
                "metadata": {"source": ["ATS"], "last_updated": current_date, "umls_cui": "C0010200"}
            }
        },
        "neurological": {
            "headache|photophobia": {
                "differentials": ["Migraine", "Tension headache", "Meningitis"],
                "contextual_triggers": ["fever for meningitis"],
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
                "metadata": {"source": ["AHS", "UpToDate"], "last_updated": current_date, "umls_cui": "C0018681"}
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
                "metadata": {"source": ["ACC/AHA"], "last_updated": current_date, "umls_cui": "C0008031"}
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
                "metadata": {"source": ["AGA"], "last_updated": current_date, "umls_cui": "C0234451"}
            }
        },
        "musculoskeletal": {
            "knee pain|swelling": {
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
                "metadata": {"source": ["AAOS", "ACR"], "last_updated": current_date, "umls_cui": "C0231749"}
            },
            "back pain|lower back pain|backache": {
                "differentials": ["Mechanical low back pain", "Lumbar strain", "Herniated disc", "Ankylosing spondylitis"],
                "required_symptoms": ["back pain"],
                "contextual_triggers": ["obesity", "sedentary lifestyle"],
                "exclusion_criteria": ["fever", "weight loss", "trauma"],
                "workup": {
                    "urgent": ["MRI if neurological symptoms (e.g., radiating pain, weakness)"],
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
                "metadata": {"source": ["ACP", "UpToDate"], "last_updated": current_date, "umls_cui": "C0004604"}
            }
        },
        "infectious": {
            "fever|chills": {
                "differentials": ["Malaria", "Dengue", "Meningitis"],
                "contextual_triggers": ["travel to endemic area for malaria or dengue", "neck stiffness for meningitis"],
                "required_symptoms": ["fever", "chills"],
                "workup": {"urgent": ["Rapid malaria test", "Lumbar puncture if neck stiffness"], "routine": ["Blood cultures"]},
                "management": {
                    "symptomatic": ["Acetaminophen 500 mg"],
                    "definitive": ["Artemether-lumefantrine for malaria"],
                    "lifestyle": ["Hydration"]
                },
                "follow_up": ["Follow-up in 3-5 days"],
                "references": ["WHO Guidelines: https://www.who.int", "UpToDate: https://www.uptodate.com/contents/meningitis"],
                "metadata": {"source": ["WHO", "UpToDate"], "last_updated": current_date, "umls_cui": "C0015967"}
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
                "metadata": {"source": ["UpToDate"], "last_updated": current_date, "umls_cui": "C0013144"}
            }
        }
    }

    default_history_diagnoses = {
        "hypertension": ["high blood pressure", "HTN"],
        "diabetes": ["diabetes mellitus", "high blood sugar"],
        "asthma": ["bronchial asthma", "reactive airway disease"]
    }

    default_management_config = {
        "follow_up_default": "Follow-up in 2 weeks",
        "follow_up_urgent": "Follow-up in 3-5 days or sooner if symptoms worsen",
        "urgent_threshold": 0.9,
        "min_symptom_match": 0.7
    }

    resources = {
        "symptoms": (default_symptoms, lambda x: x),
        "medical_stop_words": (default_stop_words, lambda x: list(x)),
        "medical_terms": (default_medical_terms, lambda x: x),
        "synonyms": (default_synonyms, lambda x: x),
        "clinical_pathways": (default_clinical_pathways, lambda x: x),
        "history_diagnoses": (default_history_diagnoses, lambda x: x),
        "diagnosis_relevance": (default_diagnosis_relevance, lambda x: x),
        "management_config": (default_management_config, lambda x: x)
    }

    # Validate default data
    for key, (default_data, _) in resources.items():
        if key == "symptoms":
            for category, symptoms in default_data.items():
                if not isinstance(symptoms, dict):
                    logger.error(f"Invalid symptoms structure for category {category}: {type(symptoms)}")
                    return
                for symptom, info in symptoms.items():
                    if not isinstance(info, dict) or not info.get('description') or not info.get('umls_cui'):
                        logger.error(f"Invalid symptom {symptom} in {category}: {info}")
                        return
        elif key == "medical_stop_words":
            if not isinstance(default_data, list) or not all(isinstance(w, str) for w in default_data):
                logger.error(f"Invalid medical_stop_words: {type(default_data)}")
                return
        elif key == "medical_terms":
            for term in default_data:
                if not isinstance(term, dict) or not term.get('term') or not term.get('umls_cui'):
                    logger.error(f"Invalid medical_term: {term}")
                    return
        elif key == "synonyms":
            for term, aliases in default_data.items():
                if not isinstance(aliases, list) or not all(isinstance(a, str) for a in aliases):
                    logger.error(f"Invalid synonyms for {term}: {aliases}")
                    return
        elif key == "clinical_pathways":
            for category, paths in default_data.items():
                for pkey, path in paths.items():
                    if not isinstance(path, dict) or not path.get('differentials') or not path.get('required_symptoms'):
                        logger.error(f"Invalid clinical pathway {pkey}: {path}")
                        return
                    weights = sum(r['weight'] for r in default_diagnosis_relevance if r['diagnosis'] in path['differentials'] for r in r.get('relevance', []))
                    if abs(weights - 1.0) > 0.01:
                        logger.warning(f"Diagnosis relevance weights for {pkey} do not sum to 1.0: {weights}")
        elif key == "diagnosis_relevance":
            for item in default_data:
                if not isinstance(item, dict) or not item.get('diagnosis') or not item.get('relevance'):
                    logger.error(f"Invalid diagnosis_relevance: {item}")
                    return
                weights = sum(r['weight'] for r in item['relevance'])
                if abs(weights - 1.0) > 0.01:
                    logger.warning(f"Diagnosis {item['diagnosis']} relevance weights do not sum to 1.0: {weights}")

    # Initialize MongoDB collections
    try:
        client = MongoClient(MONGO_URI)
        client.admin.command('ping')
        db = client[DB_NAME]
        for key, (default_data, _) in resources.items():
            collection = db[key]
            if collection.count_documents({}) == 0:
                if key == "symptoms":
                    for category, symptoms in default_data.items():
                        for symptom, info in symptoms.items():
                            collection.insert_one({
                                'category': category,
                                'symptom': symptom,
                                'description': info['description'],
                                'umls_cui': info.get('umls_cui'),
                                'semantic_type': info.get('semantic_type', 'Unknown')
                            })
                elif key == "medical_stop_words":
                    for word in default_data:
                        collection.insert_one({'word': word})
                elif key == "medical_terms":
                    for term in default_data:
                        collection.insert_one(term)
                elif key == "synonyms":
                    for term, aliases in default_data.items():
                        collection.insert_one({'term': term, 'aliases': aliases})
                elif key == "clinical_pathways":
                    for category, paths in default_data.items():
                        collection.insert_one({'category': category, 'paths': paths})
                elif key == "diagnosis_relevance":
                    for item in default_data:
                        collection.insert_one(item)
                else:
                    for k, v in default_data.items():
                        collection.insert_one({'key': k, 'value': v})
                logger.info(f"Initialized MongoDB collection {key} with default data")
        client.close()
    except ConnectionFailure as e:
        logger.error(f"MongoDB connection failed: {str(e)}. Initializing JSON only.")
    except Exception as e:
        logger.error(f"Error initializing MongoDB: {str(e)}. Initializing JSON only.")

    # Initialize JSON files
    for key, (default_data, transform) in resources.items():
        file_path = os.path.join(knowledge_base_dir, f"{key}.json")
        if not os.path.exists(file_path):
            try:
                data = transform(default_data)
                if key in ["symptoms", "clinical_pathways", "management_config"]:
                    data = {"version": "1.1.0", "last_updated": current_date, "data": data}
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
                logger.info(f"Created default file: {file_path}")
            except (OSError, json.JSONEncodeError) as e:
                logger.error(f"Failed to create {file_path}: {str(e)}")