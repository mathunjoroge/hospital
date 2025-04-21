import os
import json
from departments.nlp.logging_setup import logger
from typing import Dict

def load_knowledge_base() -> Dict:
    """Load knowledge base resources with validation and fallback."""
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
    default_stop_words = [
        "patient", "history", "present", "illness", "denies", "reports",
        "without", "with", "this", "that", "these", "those", "they", "them",
        "their", "have", "has", "had", "been", "being", "other", "associated",
        "complains", "noted", "states", "observed", "left", "right", "ago",
        "since", "recently", "following", "during", "upon", "after"
    ]
    fallback_clinical_pathways = {
        "neurological": {
            "headache": {
                "differentials": ["Migraine", "Tension headache", "Cluster headache"],
                "workup": {"urgent": ["CT head if thunderclap"], "routine": ["CBC", "ESR"]},
                "management": {"symptomatic": ["Ibuprofen 400 mg", "Hydration"], "definitive": ["Sumatriptan 50 mg"]}
            },
            "photophobia": {
                "differentials": ["Migraine", "Meningitis"],
                "workup": {"urgent": ["Lumbar puncture if fever"], "routine": []},
                "management": {"symptomatic": ["Dark room rest"], "definitive": []}
            }
        },
        "cardiovascular": {
            "chest pain": {
                "differentials": ["Angina", "Myocardial infarction", "Musculoskeletal pain"],
                "workup": {"urgent": ["ECG", "Troponin"], "routine": ["Lipid panel", "Stress test"]},
                "management": {"symptomatic": ["Nitroglycerin 0.4 mg SL"], "definitive": ["Aspirin 81 mg daily"]}
            },
            "shortness of breath": {
                "differentials": ["Angina", "Pulmonary embolism", "COPD"],
                "workup": {"urgent": ["D-dimer if acute"], "routine": ["Pulmonary function test"]},
                "management": {"symptomatic": ["Oxygen therapy"], "definitive": []}
            },
            "edema": {
                "differentials": ["Heart failure", "Venous insufficiency", "Nephrotic syndrome"],
                "workup": {"urgent": ["BNP"], "routine": ["Renal panel", "Ultrasound Doppler"]},
                "management": {"symptomatic": ["Leg elevation"], "definitive": ["Furosemide 20 mg"]}
            }
        },
        "gastrointestinal": {
            "epigastric pain": {
                "differentials": ["GERD", "Peptic ulcer", "Pancreatitis"],
                "workup": {"urgent": ["Lipase if severe"], "routine": ["H. pylori test"]},
                "management": {"symptomatic": ["Antacids"], "definitive": ["Omeprazole 20 mg daily"]}
            },
            "nausea": {
                "differentials": ["GERD", "Gastritis"],
                "workup": {"urgent": [], "routine": ["Upper endoscopy if persistent"]},
                "management": {"symptomatic": ["Ondansetron 4 mg"], "definitive": []}
            },
            "diarrhea": {
                "differentials": ["Viral gastroenteritis", "Lactose intolerance", "Traveler’s diarrhea"],
                "workup": {"urgent": ["Stool culture if bloody"], "routine": ["Electrolytes"]},
                "management": {"symptomatic": ["Loperamide 2 mg"], "definitive": ["Hydration"]}
            }
        },
        "musculoskeletal": {
            "knee pain": {
                "differentials": ["Osteoarthritis", "Meniscal injury", "Bursitis"],
                "workup": {"urgent": [], "routine": ["Knee X-ray", "MRI if locking"]},
                "management": {"symptomatic": ["Ibuprofen 600 mg", "Ice"], "definitive": ["Physical therapy"]}
            },
            "swelling": {
                "differentials": ["Osteoarthritis", "Gout"],
                "workup": {"urgent": ["Joint aspiration if acute"], "routine": ["Uric acid level"]},
                "management": {"symptomatic": ["Elevation"], "definitive": []}
            },
            "back pain": {
                "differentials": ["Mechanical low back pain", "Herniated disc", "Spondylosis"],
                "workup": {"urgent": ["MRI if neurological symptoms"], "routine": ["Lumbar X-ray"]},
                "management": {"symptomatic": ["Acetaminophen 500 mg", "Heat"], "definitive": ["Physical therapy"]}
            }
        },
        "respiratory": {
            "cough": {
                "differentials": ["Postnasal drip", "Allergic cough", "Chronic bronchitis"],
                "workup": {"urgent": [], "routine": ["Chest X-ray", "Allergy testing"]},
                "management": {"symptomatic": ["Dextromethorphan 20 mg"], "definitive": ["Intranasal steroids"]}
            }
        },
        "dermatological": {
            "rash": {
                "differentials": ["Eczema flare", "Contact dermatitis", "Drug reaction"],
                "workup": {"urgent": [], "routine": ["Skin patch test"]},
                "management": {"symptomatic": ["Hydrocortisone 1% cream"], "definitive": ["Avoid triggers"]}
            }
        }
    }

    for key, filename in resources.items():
        file_path = os.path.join(knowledge_base_dir, filename)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if key == "medical_stop_words":
                    if not isinstance(data, list):
                        logger.error(f"Expected list for {filename}, got {type(data)}")
                        data = default_stop_words
                    data = set(data).union(default_stop_words)
                elif key == "clinical_pathways":
                    if not isinstance(data, dict):
                        logger.error(f"Expected dict for {filename}, got {type(data)}")
                        data = fallback_clinical_pathways
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
                                valid_paths[pkey] = path
                            if valid_paths:
                                valid_data[cat] = valid_paths
                        data = valid_data or fallback_clinical_pathways
                        logger.info(f"Loaded {len(data)} clinical pathway categories with {sum(len(paths) for paths in data.values())} total pathways")
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
            logger.error(f"File not found: {file_path}")
            if key == "medical_stop_words":
                knowledge[key] = set(default_stop_words)
            elif key == "clinical_pathways":
                knowledge[key] = fallback_clinical_pathways
            else:
                knowledge[key] = {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {file_path}: {str(e)}")
            if key == "medical_stop_words":
                knowledge[key] = set(default_stop_words)
            elif key == "clinical_pathways":
                knowledge[key] = fallback_clinical_pathways
            else:
                knowledge[key] = {}
    return knowledge