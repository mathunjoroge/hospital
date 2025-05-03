# departments/nlp/knowledge_base.py
import os
import json
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
        "fever": ["elevated temperature", "pyrexia"],
        "headache": ["cephalalgia", "head pain"],
        "cough": ["hacking cough", "persistent cough"],
        "chest pain": ["thoracic pain", "chest discomfort"],
        "diarrhea": ["loose stools", "frequent bowel movements"]
    }
    
    default_diagnosis_relevance = {
        "acute bacterial sinusitis": ["facial pain", "nasal congestion", "purulent nasal discharge", "fever", "headache"],
        "viral sinusitis": ["nasal congestion", "fever", "headache"],
        "allergic rhinitis": ["nasal congestion", "sneezing", "itchy eyes"],
        "migraine": ["headache", "photophobia", "nausea"],
        "myocardial infarction": ["chest pain", "shortness of breath", "sweating"],
        "peptic ulcer": ["epigastric pain", "nausea", "heartburn"],
        "osteoarthritis": ["knee pain", "joint stiffness", "swelling"]
    }
    
    default_clinical_pathways = {
        "neurological": {
            "headache": {
                "differentials": ["Migraine", "Tension headache", "Cluster headache"],
                "workup": {"urgent": ["CT head if thunderclap"], "routine": ["CBC", "ESR"]},
                "management": {
                    "symptomatic": ["Ibuprofen 400 mg", "Hydration"],
                    "definitive": ["Sumatriptan 50 mg"],
                    "lifestyle": ["Stress management"]
                },
                "follow_up": ["Follow-up in 1-2 weeks"],
                "references": ["AHS Guidelines: https://americanheadachesociety.org"],
                "metadata": {"source": ["AHS"]}
            },
            "photophobia": {
                "differentials": ["Migraine", "Meningitis"],
                "workup": {"urgent": ["Lumbar puncture if fever"], "routine": []},
                "management": {
                    "symptomatic": ["Dark room rest"],
                    "definitive": [],
                    "lifestyle": []
                },
                "follow_up": ["Follow-up in 3-5 days if persistent"],
                "references": ["Guideline: https://www.uptodate.com/contents/meningitis"],
                "metadata": {"source": ["UpToDate"]}
            }
        },
        "cardiovascular": {
            "chest pain": {
                "differentials": ["Angina", "Myocardial infarction", "Musculoskeletal pain"],
                "workup": {"urgent": ["ECG", "Troponin"], "routine": ["Lipid panel", "Stress test"]},
                "management": {
                    "symptomatic": ["Nitroglycerin 0.4 mg SL"],
                    "definitive": ["Aspirin 81 mg daily"],
                    "lifestyle": ["Low-fat diet"]
                },
                "follow_up": ["Follow-up in 1 week"],
                "references": ["ACC/AHA Guidelines: https://www.acc.org"],
                "metadata": {"source": ["ACC/AHA"]}
            },
            "shortness of breath": {
                "differentials": ["Angina", "Pulmonary embolism", "COPD"],
                "workup": {"urgent": ["D-dimer if acute"], "routine": ["Pulmonary function test"]},
                "management": {
                    "symptomatic": ["Oxygen therapy"],
                    "definitive": [],
                    "lifestyle": ["Smoking cessation"]
                },
                "follow_up": ["Follow-up in 2 weeks"],
                "references": ["GOLD Guidelines: https://goldcopd.org"],
                "metadata": {"source": ["GOLD"]}
            },
            "edema": {
                "differentials": ["Heart failure", "Venous insufficiency", "Nephrotic syndrome"],
                "workup": {"urgent": ["BNP"], "routine": ["Renal panel", "Ultrasound Doppler"]},
                "management": {
                    "symptomatic": ["Leg elevation"],
                    "definitive": ["Furosemide 20 mg"],
                    "lifestyle": ["Low-sodium diet"]
                },
                "follow_up": ["Follow-up in 1-2 weeks"],
                "references": ["Guideline: https://www.uptodate.com/contents/heart-failure"],
                "metadata": {"source": ["UpToDate"]}
            }
        },
        "gastrointestinal": {
            "epigastric pain": {
                "differentials": ["GERD", "Peptic ulcer", "Pancreatitis"],
                "workup": {"urgent": ["Lipase if severe"], "routine": ["H. pylori test"]},
                "management": {
                    "symptomatic": ["Antacids"],
                    "definitive": ["Omeprazole 20 mg daily"],
                    "lifestyle": ["Avoid spicy foods"]
                },
                "follow_up": ["Follow-up in 2 weeks"],
                "references": ["AGA Guidelines: https://gastro.org"],
                "metadata": {"source": ["AGA"]}
            },
            "nausea": {
                "differentials": ["GERD", "Gastritis"],
                "workup": {"urgent": [], "routine": ["Upper endoscopy if persistent"]},
                "management": {
                    "symptomatic": ["Ondansetron 4 mg"],
                    "definitive": [],
                    "lifestyle": []
                },
                "follow_up": ["Follow-up in 1-2 weeks"],
                "references": ["Guideline: https://www.uptodate.com/contents/nausea"],
                "metadata": {"source": ["UpToDate"]}
            },
            "diarrhea": {
                "differentials": ["Viral gastroenteritis", "Lactose intolerance", "Traveler’s diarrhea"],
                "workup": {"urgent": ["Stool culture if bloody"], "routine": ["Electrolytes"]},
                "management": {
                    "symptomatic": ["Loperamide 2 mg"],
                    "definitive": ["Hydration"],
                    "lifestyle": ["Probiotics"]
                },
                "follow_up": ["Follow-up in 1 week"],
                "references": ["IDSA Guidelines: https://www.idsociety.org"],
                "metadata": {"source": ["IDSA"]}
            }
        },
        "musculoskeletal": {
            "knee pain": {
                "differentials": ["Osteoarthritis", "Meniscal injury", "Bursitis"],
                "workup": {"urgent": [], "routine": ["Knee X-ray", "MRI if locking"]},
                "management": {
                    "symptomatic": ["Ibuprofen 600 mg", "Ice"],
                    "definitive": ["Physical therapy"],
                    "lifestyle": ["Weight management"]
                },
                "follow_up": ["Follow-up in 4 weeks"],
                "references": ["AAOS Guidelines: https://www.aaos.org"],
                "metadata": {"source": ["AAOS"]}
            },
            "swelling": {
                "differentials": ["Osteoarthritis", "Gout"],
                "workup": {"urgent": ["Joint aspiration if acute"], "routine": ["Uric acid level"]},
                "management": {
                    "symptomatic": ["Elevation"],
                    "definitive": [],
                    "lifestyle": ["Low-purine diet"]
                },
                "follow_up": ["Follow-up in 2 weeks"],
                "references": ["ACR Guidelines: https://www.rheumatology.org"],
                "metadata": {"source": ["ACR"]}
            },
            "back pain": {
                "differentials": ["Mechanical low back pain", "Herniated disc", "Spondylosis"],
                "workup": {"urgent": ["MRI if neurological symptoms"], "routine": ["Lumbar X-ray"]},
                "management": {
                    "symptomatic": ["Acetaminophen 500 mg", "Heat"],
                    "definitive": ["Physical therapy"],
                    "lifestyle": ["Core strengthening"]
                },
                "follow_up": ["Follow-up in 4 weeks"],
                "references": ["ACP Guidelines: https://www.acponline.org"],
                "metadata": {"source": ["ACP"]}
            }
        },
        "respiratory": {
            "cough": {
                "differentials": ["Postnasal drip", "Allergic cough", "Chronic bronchitis"],
                "workup": {"urgent": [], "routine": ["Chest X-ray", "Allergy testing"]},
                "management": {
                    "symptomatic": ["Dextromethorphan 20 mg"],
                    "definitive": ["Intranasal steroids"],
                    "lifestyle": ["Avoid allergens"]
                },
                "follow_up": ["Follow-up in 2 weeks"],
                "references": ["ATS Guidelines: https://www.thoracic.org"],
                "metadata": {"source": ["ATS"]}
            },
            "facial pain|nasal congestion|sinusitis|sinus pain|purulent nasal discharge": {
                "differentials": ["Acute Bacterial Sinusitis", "Viral Sinusitis", "Allergic Rhinitis"],
                "workup": {
                    "urgent": ["Nasal endoscopy if persistent"],
                    "routine": ["Sinus CT if no improvement"]
                },
                "management": {
                    "symptomatic": ["Nasal saline irrigation", "Decongestants"],
                    "definitive": ["Amoxicillin 500 mg TID for 10 days"],
                    "lifestyle": ["Hydration", "Avoid irritants like smoke"]
                },
                "follow_up": ["Follow-up in 2 weeks"],
                "references": [
                    "Guideline: https://www.uptodate.com/contents/acute-sinusitis",
                    "AAO-HNS Sinusitis Guidelines: https://www.entnet.org"
                ],
                "metadata": {"source": ["UpToDate", "AAO-HNS"]}
            }
        },
        "dermatological": {
            "rash": {
                "differentials": ["Eczema flare", "Contact dermatitis", "Drug reaction"],
                "workup": {"urgent": [], "routine": ["Skin patch test"]},
                "management": {
                    "symptomatic": ["Hydrocortisone 1% cream"],
                    "definitive": ["Avoid triggers"],
                    "lifestyle": ["Moisturize regularly"]
                },
                "follow_up": ["Follow-up in 2 weeks"],
                "references": ["AAD Guidelines: https://www.aad.org"],
                "metadata": {"source": ["AAD"]}
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
        "urgent_threshold": "0.9"
    }
    
    default_diagnosis_treatments = {
        "acute bacterial sinusitis": {
            "workup": {
                "urgent": ["Nasal endoscopy if persistent"],
                "routine": ["Sinus CT if no improvement"]
            },
            "treatment": {
                "symptomatic": ["Nasal saline irrigation", "Decongestants"],
                "definitive": ["Amoxicillin 500 mg TID for 10 days"],
                "lifestyle": ["Hydration", "Avoid irritants like smoke"]
            }
        },
        "migraine": {
            "workup": {
                "urgent": ["CT head if thunderclap"],
                "routine": ["CBC", "ESR"]
            },
            "treatment": {
                "symptomatic": ["Ibuprofen 400 mg", "Hydration"],
                "definitive": ["Sumatriptan 50 mg"],
                "lifestyle": ["Stress management"]
            }
        }
    }
    
    resources = {
        "medical_stop_words": (default_stop_words, lambda x: set(x)),
        "medical_terms": ([], lambda x: set(x)),
        "synonyms": (default_synonyms, lambda x: x),
        "clinical_pathways": (default_clinical_pathways, lambda x: x),
        "history_diagnoses": (default_history_diagnoses, lambda x: x),
        "diagnosis_relevance": (default_diagnosis_relevance, lambda x: x),
        "management_config": (default_management_config, lambda x: x),
        "diagnosis_treatments": (default_diagnosis_treatments, lambda x: x)
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
    """Load knowledge base resources with validation and fallback."""
    initialize_knowledge_files()  # Ensure files exist
    knowledge_base_dir = os.path.join(os.path.dirname(__file__), "knowledge_base")
    resources = {
        "medical_stop_words": "medical_stop_words.json",
        "medical_terms": "medical_terms.json",
        "synonyms": "synonyms.json",
        "clinical_pathways": "clinical_pathways.json",
        "history_diagnoses": "history_diagnoses.json",
        "diagnosis_relevance": "diagnosis_relevance.json",
        "management_config": "management_config.json",
        "diagnosis_treatments": "diagnosis_treatments.json"
    }
    knowledge = {}
    for key, filename in resources.items():
        file_path = os.path.join(knowledge_base_dir, filename)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
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
                                logger.warning(f"Skipping invalid category {cat}: {type(paths)}")
                                continue
                            valid_paths = {}
                            for pkey, path in paths.items():
                                if not isinstance(path, dict):
                                    logger.warning(f"Skipping invalid path {pkey}: {type(path)}")
                                    continue
                                if not path.get("differentials", []) or not isinstance(path.get("differentials"), list):
                                    logger.warning(f"Skipping path {pkey}: empty or invalid differentials")
                                    continue
                                if not path.get("management", {}).get("symptomatic") and not path.get("management", {}).get("definitive"):
                                    logger.warning(f"Skipping path {pkey}: no management options")
                                    continue
                                if not path.get("workup", {}).get("urgent") and not path.get("workup", {}).get("routine"):
                                    logger.warning(f"Skipping path {pkey}: no workup options")
                                    continue
                                if not path.get("follow_up", []):
                                    path["follow_up"] = ["Follow-up in 2 weeks"]
                                    logger.info(f"Added default follow-up for {pkey}")
                                valid_paths[pkey] = path
                            if valid_paths:
                                valid_data[cat] = valid_paths
                        data = valid_data or resources[key][0]
                        logger.info(f"Loaded {len(data)} clinical pathway categories with {sum(len(paths) for paths in data.values())} total pathways")
                elif key == "synonyms":
                    if not isinstance(data, dict) or not all(isinstance(k, str) and isinstance(v, list) for k, v in data.items()):
                        logger.error(f"Expected dict of string keys and list values for {filename}, got {type(data)}")
                        data = resources[key][0]
                    else:
                        data = {k: v for k, v in data.items() if all(isinstance(a, str) for a in v)}
                elif key == "diagnosis_relevance":
                    if not isinstance(data, dict) or not all(isinstance(k, str) and isinstance(v, list) for k, v in data.items()):
                        logger.error(f"Expected dict of string keys and list values for {filename}, got {type(data)}")
                        data = resources[key][0]
                    else:
                        data = {k: v for k, v in data.items() if all(isinstance(r, str) for r in v)}
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