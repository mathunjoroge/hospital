# departments/nlp/knowledge_base.py
import os
import json
from datetime import datetime
from departments.nlp.logging_setup import logger
from typing import Dict, Set, List

def initialize_knowledge_files() -> None:
    """Initialize default JSON files for knowledge base resources if they don't exist."""
    knowledge_base_dir = os.path.join(os.path.dirname(__file__), "knowledge_base")
    os.makedirs(knowledge_base_dir, exist_ok=True)

    default_stop_words = [
        "patient", "history", "present", "illness", "denies", "reports",
        "without", "with", "this", "that", "these", "those", "they", "them",
        "their", "have", "has", "had", "been", "being", "other", "associated",
        "complains", "noted", "states", "observed", "left", "right", "ago",
        "since", "recently", "following", "during", "upon", "after"
    ]

    default_synonyms = {
        "facial pain": ["sinus pain", "sinus pressure", "facial pressure"],
        "nasal congestion": ["stuffy nose", "blocked nose", "nasal obstruction"],
        "purulent nasal discharge": ["yellow discharge", "green discharge", "pus from nose"],
        "fever": ["elevated temperature", "pyrexia", "febrile"],
        "headache": ["cephalalgia", "head pain", "cranial pain"],
        "cough": ["hacking cough", "persistent cough", "dry cough"],
        "chest pain": ["thoracic pain", "chest discomfort", "sternal pain"],
        "diarrhea": ["loose stools", "frequent bowel movements", "watery stools"],
        "shortness of breath": ["dyspnea", "breathlessness"],
        "photophobia": ["light sensitivity", "eye discomfort"],
        "neck stiffness": ["nuchal rigidity", "neck pain"],
        "rash": ["skin eruption", "dermatitis"]
    }

    default_diagnosis_relevance = {
        "acute bacterial sinusitis": [
            {"symptom": "facial pain", "weight": 0.4},
            {"symptom": "nasal congestion", "weight": 0.3},
            {"symptom": "purulent nasal discharge", "weight": 0.3},
            {"symptom": "fever", "weight": 0.1},
            {"symptom": "headache", "weight": 0.05}
        ],
        "viral sinusitis": [
            {"symptom": "nasal congestion", "weight": 0.4},
            {"symptom": "fever", "weight": 0.3},
            {"symptom": "headache", "weight": 0.2}
        ],
        "allergic rhinitis": [
            {"symptom": "nasal congestion", "weight": 0.4},
            {"symptom": "sneezing", "weight": 0.3},
            {"symptom": "itchy eyes", "weight": 0.3}
        ],
        "migraine": [
            {"symptom": "headache", "weight": 0.5},
            {"symptom": "photophobia", "weight": 0.3},
            {"symptom": "nausea", "weight": 0.2}
        ],
        "myocardial infarction": [
            {"symptom": "chest pain", "weight": 0.5},
            {"symptom": "shortness of breath", "weight": 0.3},
            {"symptom": "sweating", "weight": 0.2}
        ],
        "pulmonary embolism": [
            {"symptom": "shortness of breath", "weight": 0.4},
            {"symptom": "chest pain", "weight": 0.3},
            {"symptom": "tachycardia", "weight": 0.3}
        ],
        "peptic ulcer": [
            {"symptom": "epigastric pain", "weight": 0.5},
            {"symptom": "nausea", "weight": 0.3},
            {"symptom": "heartburn", "weight": 0.2}
        ],
        "osteoarthritis": [
            {"symptom": "knee pain", "weight": 0.4},
            {"symptom": "joint stiffness", "weight": 0.3},
            {"symptom": "swelling", "weight": 0.3}
        ],
        "malaria": [
            {"symptom": "fever", "weight": 0.4},
            {"symptom": "chills", "weight": 0.3},
            {"symptom": "travel to endemic area", "weight": 0.3}
        ],
        "meningitis": [
            {"symptom": "fever", "weight": 0.3},
            {"symptom": "neck stiffness", "weight": 0.4},
            {"symptom": "photophobia", "weight": 0.3}
        ]
    }

    default_clinical_pathways = {
        "respiratory": {
            "facial pain|nasal congestion|sinusitis|sinus pain|purulent nasal discharge": {
                "differentials": ["Acute Bacterial Sinusitis", "Viral Sinusitis", "Allergic Rhinitis"],
                "contextual_triggers": ["recent viral infection"],
                "required_symptoms": ["facial pain", "nasal congestion", "purulent nasal discharge"],
                "exclusion_criteria": ["photophobia", "nausea"],  # Prevent migraine confusion
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
                "metadata": {"source": ["IDSA", "AAO-HNS"], "last_updated": "2025-05-04"}
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
                "metadata": {"source": ["ATS"], "last_updated": "2025-05-04"}
            }
        },
        "neurological": {
            "headache|photophobia": {
                "differentials": ["Migraine", "Tension headache", "Meningitis"],
                "contextual_triggers": ["fever for meningitis"],
                "required_symptoms": ["headache", "photophobia"],
                "exclusion_criteria": ["nasal congestion", "purulent nasal discharge"],  # Prevent sinusitis confusion
                "workup": {"urgent": ["CT head if thunderclap", "Lumbar puncture if fever"], "routine": ["CBC", "ESR"]},
                "management": {
                    "symptomatic": ["Ibuprofen 400 mg", "Hydration"],
                    "definitive": ["Sumatriptan 50 mg for migraine"],
                    "lifestyle": ["Stress management"]
                },
                "follow_up": ["Follow-up in 3-5 days if urgent, else 1-2 weeks"],
                "references": ["AHS Guidelines: https://americanheadachesociety.org", "UpToDate: https://www.uptodate.com/contents/meningitis"],
                "metadata": {"source": ["AHS", "UpToDate"], "last_updated": "2025-05-04"}
            }
        },
        "cardiovascular": {
            "chest pain|shortness of breath": {
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
                "metadata": {"source": ["ACC/AHA"], "last_updated": "2025-05-04"}
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
                "metadata": {"source": ["AGA"], "last_updated": "2025-05-04"}
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
                "metadata": {"source": ["AAOS", "ACR"], "last_updated": "2025-05-04"}
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
                "metadata": {"source": ["WHO", "UpToDate"], "last_updated": "2025-05-04"}
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
        "urgent_threshold": "0.9",
        "min_symptom_match": 0.7  # Minimum score for pathway matching
    }

    resources = {
        "medical_stop_words": (default_stop_words, lambda x: set(x)),
        "medical_terms": ([], lambda x: set(x)),
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
                    json.dump(default_data, f, indent=2)
                logger.info(f"Created default file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to create {file_path}: {str(e)}")

def load_knowledge_base() -> Dict:
    """Load knowledge base resources with enhanced validation and fallback."""
    initialize_knowledge_files()
    knowledge_base_dir = os.path.join(os.path.dirname(__file__), "knowledge_base")
    resources = {
        "medical_stop_words": "medical_stop_words.json",
        "medical_terms": "medical_terms.json",
        "synonyms": "synonyms.json",
        "clinical_pathways": "clinical_pathways.json",
        "history_diagnoses": "history_diagnoses.json",
        "diagnosis_relevance": "diagnosis_relevance.json",
        "management_config": "management_config.json"
    }
    knowledge = {}
    required_categories = {'respiratory', 'neurological', 'cardiovascular', 'gastrointestinal', 'musculoskeletal', 'infectious'}
    high_risk_conditions = {'pulmonary embolism', 'myocardial infarction', 'meningitis', 'malaria', 'dengue'}

    for key, filename in resources.items():
        file_path = os.path.join(knowledge_base_dir, filename)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                logger.debug(f"Loaded raw data for {key}: {data}")
                if key == "medical_stop_words":
                    if not isinstance(data, list) or not all(isinstance(w, str) for w in data):
                        logger.error(f"Expected list of strings for {filename}, got {type(data)}")
                        data = resources[key][0]
                    data = set(data).union(resources["medical_stop_words"][0])
                elif key == "clinical_pathways":
                    if not isinstance(data, dict):
                        logger.error(f"Expected dict for {filename}, got {type(data)}")
                        data = resources[key][0]
                    else:
                        valid_data = {}
                        for cat, paths in data.items():
                            if not isinstance(paths, dict):
                                logger.warning(f"Skipping invalid category {cat}: expected dict, got {type(paths)}")
                                continue
                            valid_paths = {}
                            for pkey, path in paths.items():
                                if not isinstance(path, dict):
                                    logger.warning(f"Skipping invalid path {pkey}: expected dict, got {type(path)}")
                                    continue
                                differentials = path.get("differentials", [])
                                if not differentials or not isinstance(differentials, list) or len(differentials) < 2:
                                    logger.warning(f"Skipping path {pkey}: differentials empty or insufficient (got {differentials})")
                                    continue
                                diagnosis_relevance = knowledge.get('diagnosis_relevance', resources["diagnosis_relevance"][0])
                                for dx in differentials:
                                    if dx.lower() not in diagnosis_relevance:
                                        logger.warning(f"Path {pkey}: differential {dx} lacks required symptoms in diagnosis_relevance, including for flexibility")
                                if any(dx.lower() in high_risk_conditions for dx in differentials):
                                    if not path.get("contextual_triggers", []):
                                        logger.warning(f"Path {pkey}: high-risk differentials require contextual_triggers, including for flexibility")
                                management = path.get("management", {})
                                if not management.get("symptomatic") and not management.get("definitive"):
                                    logger.warning(f"Skipping path {pkey}: no management options")
                                    continue
                                workup = path.get("workup", {})
                                if not workup.get("urgent") and not workup.get("routine"):
                                    logger.warning(f"Skipping path {pkey}: no workup options")
                                    continue
                                references = path.get("references", [])
                                if not references:
                                    logger.warning(f"Path {pkey}: missing references, setting to default")
                                    path["references"] = ["None specified"]
                                metadata = path.get("metadata", {})
                                if not metadata.get("source"):
                                    logger.warning(f"Path {pkey}: missing metadata.source, setting to Unknown")
                                    metadata["source"] = ["Unknown"]
                                if not metadata.get("last_updated"):
                                    metadata["last_updated"] = datetime.now().strftime("%Y-%m-%d")
                                    logger.info(f"Added default last_updated for {pkey}")
                                if not path.get("follow_up", []):
                                    path["follow_up"] = ["Follow-up in 2 weeks"]
                                    logger.info(f"Added default follow-up for {pkey}")
                                valid_paths[pkey] = path
                            if valid_paths:
                                valid_data[cat] = valid_paths
                        data = valid_data or resources[key][0]
                        missing_categories = required_categories - set(valid_data.keys())
                        if missing_categories:
                            logger.warning(f"Missing required categories in clinical_pathways: {missing_categories}")
                        if 'respiratory' not in valid_data:
                            logger.error(f"Critical category 'respiratory' missing after validation")
                        logger.info(f"Loaded {len(valid_data)} clinical pathway categories with {sum(len(paths) for paths in valid_data.values())} total pathways")
                elif key == "synonyms":
                    if not isinstance(data, dict) or not all(isinstance(k, str) and isinstance(v, list) for k, v in data.items()):
                        logger.error(f"Expected dict of string keys and list values for {filename}, got {type(data)}")
                        data = resources[key][0]
                    else:
                        data = {k: v for k, v in data.items() if all(isinstance(a, str) for a in v)}
                elif key == "diagnosis_relevance":
                    if not isinstance(data, dict):
                        logger.error(f"Expected dict for {filename}, got {type(data)}")
                        data = resources[key][0]
                    else:
                        data = {k: v for k, v in data.items() if isinstance(v, list) and all(isinstance(r, dict) and "symptom" in r and "weight" in r for r in v)}
                elif key == "medical_terms":
                    if not isinstance(data, list) or not all(isinstance(t, str) for t in data):
                        logger.error(f"Expected list of strings for {filename}, got {type(data)}")
                        data = resources[key][0]
                    data = set(data)
                else:
                    if not isinstance(data, dict):
                        logger.error(f"Expected dict for {filename}, got {type(data)}")
                        data = resources[key][0]
                knowledge[key] = data
                if not data:
                    logger.warning(f"Empty resource loaded: {filename}")
                else:
                    logger.info(f"Loaded {len(data)} entries for {key}")
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}. Using default data.")
            knowledge[key] = resources[key][0]
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {file_path}: {str(e)}. Using default data.")
            knowledge[key] = resources[key][0]
    return knowledge