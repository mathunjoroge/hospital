import os
import json
from departments.nlp.logging_setup import get_logger

logger = get_logger()

def initialize_knowledge_files() -> None:
    """Initialize default JSON files for knowledge base resources with UMLS metadata."""
    knowledge_base_dir = os.path.join(os.path.dirname(__file__), "knowledge_base")
    
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
        {"term": "facial pain", "category": "respiratory", "umls_cui": None, "semantic_type": "Unknown"},
        {"term": "nasal congestion", "category": "respiratory", "umls_cui": None, "semantic_type": "Unknown"},
        {"term": "purulent nasal discharge", "category": "respiratory", "umls_cui": None, "semantic_type": "Unknown"},
        {"term": "fever", "category": "infectious", "umls_cui": "C0015967", "semantic_type": "Sign or Symptom"},
        {"term": "headache", "category": "neurological", "umls_cui": "C0018681", "semantic_type": "Sign or Symptom"},
        {"term": "cough", "category": "respiratory", "umls_cui": "C0010200", "semantic_type": "Sign or Symptom"},
        {"term": "chest pain", "category": "cardiovascular", "umls_cui": None, "semantic_type": "Unknown"},
        {"term": "shortness of breath", "category": "cardiovascular", "umls_cui": None, "semantic_type": "Unknown"},
        {"term": "photophobia", "category": "neurological", "umls_cui": None, "semantic_type": "Unknown"},
        {"term": "neck stiffness", "category": "neurological", "umls_cui": None, "semantic_type": "Unknown"},
        {"term": "rash", "category": "dermatological", "umls_cui": None, "semantic_type": "Unknown"},
        {"term": "back pain", "category": "musculoskeletal", "umls_cui": "C0004604", "semantic_type": "Sign or Symptom"},
        {"term": "knee pain", "category": "musculoskeletal", "umls_cui": None, "semantic_type": "Unknown"},
        {"term": "epigastric pain", "category": "gastrointestinal", "umls_cui": None, "semantic_type": "Unknown"},
        {"term": "fatigue", "category": "general", "umls_cui": "C0013144", "semantic_type": "Sign or Symptom"},
        {"term": "chest tightness", "category": "cardiovascular", "umls_cui": "C0242209", "semantic_type": "Sign or Symptom"},
        {"term": "nausea", "category": "gastrointestinal", "umls_cui": "C0027497", "semantic_type": "Sign or Symptom"},
        {"term": "obesity", "category": "musculoskeletal", "umls_cui": "C0028754", "semantic_type": "Disease or Syndrome"},
        {"term": "joint pain", "category": "musculoskeletal", "umls_cui": None, "semantic_type": "Unknown"},
        {"term": "pain on movement", "category": "musculoskeletal", "umls_cui": None, "semantic_type": "Unknown"}
    ]

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
                {"symptom": "purulent nasal discharge", "weight": 0.3},
                {"symptom": "fever", "weight": 0.1},
                {"symptom": "headache", "weight": 0.05}
            ],
            "category": "respiratory"
        },
        {
            "diagnosis": "viral sinusitis",
            "relevance": [
                {"symptom": "nasal congestion", "weight": 0.4},
                {"symptom": "fever", "weight": 0.3},
                {"symptom": "headache", "weight": 0.2}
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
                "metadata": {"source": ["IDSA", "AAO-HNS"], "last_updated": "2025-05-24", "umls_cui": None}
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
                "metadata": {"source": ["ATS"], "last_updated": "2025-05-24", "umls_cui": "C0010200"}
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
                "metadata": {"source": ["AHS", "UpToDate"], "last_updated": "2025-05-24", "umls_cui": "C0018681"}
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
                "metadata": {"source": ["ACC/AHA"], "last_updated": "2025-05-24", "umls_cui": "C0242209"}
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
                "metadata": {"source": ["AGA"], "last_updated": "2025-05-24", "umls_cui": None}
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
                "metadata": {"source": ["AAOS", "ACR"], "last_updated": "2025-05-24", "umls_cui": None}
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
                "metadata": {"source": ["ACP", "UpToDate"], "last_updated": "2025-05-24", "umls_cui": "C0004604"}
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
                "metadata": {"source": ["WHO", "UpToDate"], "last_updated": "2025-05-24", "umls_cui": "C0015967"}
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
        "medical_stop_words": (default_stop_words, lambda x: list(x)),
        "medical_terms": (default_medical_terms, lambda x: x),
        "synonyms": (default_synonyms, lambda x: x),
        "clinical_pathways": (default_clinical_pathways, lambda x: x),
        "history_diagnoses": (default_history_diagnoses, lambda x: x),
        "diagnosis_relevance": (default_diagnosis_relevance, lambda x: x),
        "management_config": (default_management_config, lambda x: x)
    }

    for key, (default_data, transform) in resources.items():
        file_path = os.path.join(knowledge_base_dir, f"{key}.json")
        if not os.path.exists(file_path):
            try:
                with open(file_path, 'w') as f:
                    json.dump(transform(default_data), f, indent=2)
                logger.info(f"Created default file: {file_path}")
            except (OSError, json.JSONEncodeError) as e:
                logger.error(f"Failed to create {file_path}: {str(e)}")